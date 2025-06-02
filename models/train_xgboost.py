import xgboost as xgb
import pandas as pd
import numpy as np
from sklearn.preprocessing import RobustScaler, StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from pathlib import Path
import glob
import os
from scipy.stats import spearmanr # For Rank IC

class XGBoostAlphaTrainer:
    """
    Trainer class for an XGBoost model to predict alpha signals.
    Mirrors the structure of AlphaTrainer for neural networks.
    """
    def __init__(self, model_config=None):
        self.device = 'cpu' # XGBoost typically runs on CPU, GPU support can be enabled via tree_method='gpu_hist'
        
        # Default XGBoost configuration
        self.config = {
            'n_estimators': 500,        # Number of boosting rounds
            'learning_rate': 0.05,      # Step size shrinkage
            'max_depth': 5,             # Maximum depth of a tree
            'subsample': 0.8,           # Subsample ratio of the training instance
            'colsample_bytree': 0.8,    # Subsample ratio of columns when constructing each tree
            'gamma': 0,                 # Minimum loss reduction required to make a further partition
            'reg_alpha': 0,             # L1 regularization term on weights
            'reg_lambda': 1,            # L2 regularization term on weights
            'objective': 'reg:squarederror', # Learning task and objective
            'random_state': 42,         # Seed for reproducibility
            'n_jobs': -1,               # Use all available cores
            # Early stopping parameters
            'early_stopping_rounds': 50, # Activates early stopping
            # Data related configuration (from AlphaTrainer logic)
            'test_size': 0.2,
        }
        if model_config:
            self.config.update(model_config)

        # Feature names to be used for training
        self.feature_names = [
            'daily_return', 'FEAT_DebtEquity_quarterly', 'FEAT_PE_quarterly',
            'FEAT_EVEBITDA_quarterly', 'FEAT_ROE_quarterly', 'price_mom_5d',
            'price_mom_10d', 'price_mom_20d', 'vol_w_mom_5d', 'vol_w_mom_10d',
            'vol_w_mom_20d', 'volatility_10d', 'skewness_20d', 'rsi_14d',
            'zscore_mom_10d_60w'
        ]

        # Scalers for features and target
        self.feature_scaler = RobustScaler()
        self.target_scaler = StandardScaler()

        self.model = None
        # Training history (e.g., to store evaluation metrics from XGBoost)
        self.history = {'eval_metric_history': []} 

    def prepare_data(self, stock_df_paths, market_df_paths):
        """
        Prepares data for XGBoost training.
        Loads stock and market data, merges them, calculates alpha,
        splits into train/test, and scales features/target.
        """
        features_cols = self.feature_names
        
        # Load and combine individual stock feature data
        dfs_stock = []
        for path_str in stock_df_paths:
            p = Path(path_str).resolve()
            exchange = p.parts[-2] # Assumes exchange is the parent directory name (e.g., HOSE, HNX)
            try:
                df = pd.read_csv(p)
                df['exchange'] = exchange
                # Ensure 'time' column is datetime
                if 'time' in df.columns and not pd.api.types.is_datetime64_any_dtype(df['time']):
                    df['time'] = pd.to_datetime(df['time'], errors='coerce')
                dfs_stock.append(df)
            except Exception as e:
                print(f"Warning: Could not load or process stock file {p}: {e}")
        if not dfs_stock:
            raise ValueError("No stock data could be loaded. Please check paths and file contents.")
        df_stock_all = pd.concat(dfs_stock, ignore_index=True)

        # Load and combine market index data
        dfs_market = []
        for path_str in market_df_paths:
            p = Path(path_str).resolve()
            name = p.stem.upper() # E.g., VNINDEX, HNXINDEX
            exchange = "HNX" if "HNX" in name else "HOSE" # Infer exchange from file name
            try:
                df = pd.read_csv(p)
                df['exchange'] = exchange
                # Ensure 'time' column is datetime
                if 'time' in df.columns and not pd.api.types.is_datetime64_any_dtype(df['time']):
                    df['time'] = pd.to_datetime(df['time'], errors='coerce')
                dfs_market.append(df)
            except Exception as e:
                print(f"Warning: Could not load or process market file {p}: {e}")
        if not dfs_market:
            raise ValueError("No market data could be loaded. Please check paths and file contents.")
        df_market_all = pd.concat(dfs_market, ignore_index=True)
        
        # Select relevant columns from market data
        df_market_subset = df_market_all[['time', 'exchange', 'FUT_RET_10D_market']]        
        
        # Merge stock data with market data
        # Ensure 'time' and 'exchange' are common types before merging
        merged_df = pd.merge(df_stock_all, df_market_subset, on=['time', 'exchange'], how='inner')
        
        # Calculate alpha (target variable)
        merged_df['alpha'] = merged_df['FUT_RET_10D'] - merged_df['FUT_RET_10D_market']
        
        # Sort by time to ensure chronological split and handle NaNs consistently
        merged_df = merged_df.sort_values(by='time').reset_index(drop=True)

        # Drop rows with NaNs in essential columns (features or target)
        # Ensure all feature_names and 'alpha' exist before attempting to use them in dropna
        cols_for_nan_check = [col for col in features_cols if col in merged_df.columns] + ['alpha']
        merged_df = merged_df.dropna(subset=cols_for_nan_check)
        
        if merged_df.empty:
            raise ValueError("DataFrame is empty after merging and NaN drop. Check data alignment, feature calculations, or NaN handling.")

        # Separate features (X) and target (y)
        X = merged_df[features_cols].values
        y_unscaled = merged_df['alpha'].values # Unscaled target

        # Train-test split (chronological)
        split_idx = int(len(X) * (1 - self.config['test_size']))
        X_train, X_test_val = X[:split_idx], X[split_idx:] # X_test_val serves as validation/test set
        y_train_unscaled, y_test_val_unscaled = y_unscaled[:split_idx], y_unscaled[split_idx:]

        # Scale features
        X_train_scaled = self.feature_scaler.fit_transform(X_train)
        X_test_val_scaled = self.feature_scaler.transform(X_test_val)
        
        # Scale target
        y_train_scaled = self.target_scaler.fit_transform(y_train_unscaled.reshape(-1, 1)).flatten()
        y_test_val_scaled = self.target_scaler.transform(y_test_val_unscaled.reshape(-1, 1)).flatten()

        print(f"Training set size: {X_train_scaled.shape}, Test/Validation set size: {X_test_val_scaled.shape}")

        # X_test_val_scaled is used as validation set for XGBoost early stopping
        # y_test_val_unscaled is used for unscaled IC calculation on this validation set
        return (X_train_scaled, X_test_val_scaled, 
                y_train_scaled, y_test_val_scaled, # Scaled targets for training and eval_set
                y_train_unscaled, y_test_val_unscaled, # Unscaled targets for final metrics
                features_cols)

    def calculate_ic(self, predictions, targets):
        """
        Calculates Information Coefficient (Pearson) and Rank Information Coefficient (Spearman).
        Handles NaN values by removing corresponding pairs.
        """
        # Ensure inputs are numpy arrays
        predictions = np.asarray(predictions)
        targets = np.asarray(targets)

        # Remove NaN values from predictions or targets
        valid_mask = ~(np.isnan(predictions) | np.isnan(targets))
        pred_clean = predictions[valid_mask]
        target_clean = targets[valid_mask]
        
        if len(pred_clean) < 10:  # Need a minimum number of samples for meaningful correlation
            print("Warning: Less than 10 valid samples for IC calculation.")
            return 0.0, 0.0
        
        # Pearson correlation (IC)
        ic = np.corrcoef(pred_clean, target_clean)[0, 1]
        if np.isnan(ic): # Handle cases where correlation might be undefined (e.g., constant series)
            ic = 0.0
        
        # Spearman rank correlation (Rank IC)
        rank_ic, _ = spearmanr(pred_clean, target_clean)
        if np.isnan(rank_ic):
            rank_ic = 0.0
        
        return ic, rank_ic

    def train_model(self, X_train_scaled, y_train_scaled, X_val_scaled, y_val_scaled, y_val_unscaled):
        """
        Trains the XGBoost model.
        X_val_scaled, y_val_scaled are used for the evaluation set in early stopping.
        y_val_unscaled is for calculating IC on the validation set after training.
        """
        # Prepare DMatrix for train and validation
        dtrain = xgb.DMatrix(X_train_scaled, label=y_train_scaled)
        dval = xgb.DMatrix(X_val_scaled, label=y_val_scaled)

        params = {
            'learning_rate': self.config['learning_rate'],
            'max_depth': self.config['max_depth'],
            'subsample': self.config['subsample'],
            'colsample_bytree': self.config['colsample_bytree'],
            'gamma': self.config['gamma'],
            'reg_alpha': self.config['reg_alpha'],
            'reg_lambda': self.config['reg_lambda'],
            'objective': self.config['objective'],
            'seed': self.config['random_state'],
            'nthread': self.config['n_jobs'],
        }

        evals = [(dtrain, 'train'), (dval, 'eval')]
        evals_result = {}

        print(f"Starting XGBoost training with early stopping rounds: {self.config['early_stopping_rounds']}...")
        self.model = xgb.train(
            params,
            dtrain,
            num_boost_round=self.config['n_estimators'],
            evals=evals,
            early_stopping_rounds=self.config['early_stopping_rounds'],
            verbose_eval=50,
            evals_result=evals_result
        )

        self.history['eval_metric_history'] = evals_result

        print("XGBoost training completed.")
        if hasattr(self.model, 'best_iteration'):
            print(f"Best iteration (according to early stopping): {self.model.best_iteration}")

        # Predict on validation set using best_iteration+1 trees
        best_ntree = self.model.best_iteration + 1 if hasattr(self.model, 'best_iteration') else self.config['n_estimators']
        val_predictions_scaled = self.model.predict(dval, iteration_range=(0, best_ntree))
        val_predictions_unscaled = self.target_scaler.inverse_transform(val_predictions_scaled.reshape(-1, 1)).flatten()
        ic, rank_ic = self.calculate_ic(val_predictions_unscaled, y_val_unscaled)
        print(f"Validation Set IC (after training): {ic:.4f}, Rank IC: {rank_ic:.4f}")

        return self.model

    def evaluate_model(self, X_test_val_scaled, y_test_val_unscaled):
        """
        Evaluates the trained XGBoost model on the test/validation set.
        Predictions are made on scaled data, then inverse-transformed for metric calculation.
        """
        if not self.model:
            raise ValueError("Model has not been trained yet. Call train_model first.")

        dtest = xgb.DMatrix(X_test_val_scaled)
        best_ntree = self.model.best_iteration + 1 if hasattr(self.model, 'best_iteration') else self.config['n_estimators']
        predictions_scaled = self.model.predict(dtest, iteration_range=(0, best_ntree))
        predictions_unscaled = self.target_scaler.inverse_transform(predictions_scaled.reshape(-1, 1)).flatten()

        mse = mean_squared_error(y_test_val_unscaled, predictions_unscaled)
        mae = mean_absolute_error(y_test_val_unscaled, predictions_unscaled)
        ic, rank_ic = self.calculate_ic(predictions_unscaled, y_test_val_unscaled)

        return predictions_unscaled, {'mse': mse, 'mae': mae, 'ic': ic, 'rank_ic': rank_ic}

def train_xgboost_alpha_model_main(stock_df_paths, market_df_paths, config=None):
    """
    Main function to orchestrate the XGBoost model training and evaluation.
    """
    trainer = XGBoostAlphaTrainer(model_config=config)

    # Prepare data
    # X_test_val_scaled is the validation set for training, and also the final test set.
    # y_test_val_unscaled is its corresponding unscaled target.
    print("Preparing data...")
    (X_train_scaled, X_test_val_scaled, 
     y_train_scaled, y_test_val_scaled, # y_test_val_scaled is the scaled target for eval_set
     _, y_test_val_unscaled,  # y_train_unscaled not needed here, y_test_val_unscaled for final eval
     _) = trainer.prepare_data(stock_df_paths, market_df_paths)

    # Train model
    # Pass X_test_val_scaled, y_test_val_scaled for early stopping's eval_set
    # Pass y_test_val_unscaled for IC calculation on validation set after training
    print("\nTraining model...")
    model = trainer.train_model(X_train_scaled, y_train_scaled, 
                                X_test_val_scaled, y_test_val_scaled, 
                                y_test_val_unscaled)

    # Evaluate model on the same test/validation set
    print("\nEvaluating model...")
    predictions_unscaled, metrics = trainer.evaluate_model(X_test_val_scaled, y_test_val_unscaled)
    
    return model, trainer, predictions_unscaled, metrics

if __name__ == "__main__":
    # Define paths to your data
    # Assuming this script is in the 'models' directory, and 'data_prep' is a sibling directory
    current_script_directory = os.path.dirname(os.path.abspath(__file__))
    input_base_folder = os.path.join(current_script_directory, '..', 'data_prep')
    
    stock_features_path = os.path.join(input_base_folder, 'calculated_stock_features')
    # Corrected path for market_indices, it's inside 'stock_data' which is inside 'data_prep'
    market_indices_path = os.path.join(input_base_folder, 'stock_data', 'market_indices')

    print(f"Looking for stock features in: {stock_features_path}")
    print(f"Looking for market indices in: {market_indices_path}")

    # Gather CSV file paths for stock features and market indices
    stock_df_paths = glob.glob(os.path.join(stock_features_path, "*", "*.csv"))
    market_df_paths = glob.glob(os.path.join(market_indices_path, "*.csv"))

    # Train and evaluate the XGBoost alpha model
    model, trainer, predictions, metrics = train_xgboost_alpha_model_main(stock_df_paths, market_df_paths)

    # Print final evaluation metrics
    print("\nFinal Evaluation Metrics:")
    for metric, value in metrics.items():
        print(f"{metric.upper()}: {value:.6f}")


# Training set size: (664739, 15), Test/Validation set size: (166185, 15)

# Starting XGBoost training with early stopping rounds: 50...
# [0]     train-rmse:0.99908      eval-rmse:0.78253
# [50]    train-rmse:0.97847      eval-rmse:0.77906
# [100]   train-rmse:0.96902      eval-rmse:0.77826
# [150]   train-rmse:0.96237      eval-rmse:0.77792
# [192]   train-rmse:0.95790      eval-rmse:0.77788

# Best iteration (according to early stopping): 143
# Validation Set IC (after training): 0.1167, Rank IC: 0.0863

# Final Evaluation Metrics:
# MSE: 59.774368
# MAE: 4.836697
# IC: 0.116700
# RANK_IC: 0.086295