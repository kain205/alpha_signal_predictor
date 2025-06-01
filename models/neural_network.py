#!/usr/bin/env python
# coding: utf-8


import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from sklearn.preprocessing import RobustScaler, StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from pathlib import Path
from torch.utils.data import Dataset, DataLoader




class AlphaDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class AlphaNet(nn.Module):
    # Neural Network for Alpha Prediction
    def __init__(self, input_dim, hidden_dims = [128, 64, 32],
                 dropout_rate = 0.3, use_batch_norm = True):
        super().__init__()

        layers = []
        prev_dim = input_dim

        for i, hidden_dim in enumerate(hidden_dims):
            # Linear Layer
            layers.append(nn.Linear(prev_dim, hidden_dim))

            # Batch normalization
            if use_batch_norm:
                layers.append(nn.BatchNorm1d(hidden_dim))

            # Activation
            layers.append(nn.ReLU())

            # Drop out
            if i < len(hidden_dims) -1:
                layers.append(nn.Dropout(dropout_rate))

            prev_dim = hidden_dim

        # Output Layer
        layers.append(nn.Linear(prev_dim, 1))

        self.network = nn.Sequential(*layers)

        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            nn.init.constant_(module.bias, 0)

    def forward(self, x):
        return self.network(x).squeeze()




class AlphaTrainer:
    def __init__(self, model_config = None):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # Default
        self.config = {
            'hidden_dims': [128, 64, 32],
            'dropout_rate': 0.3,
            'use_batch_norm': True,
            'learning_rate': 0.001,
            'batch_size': 512,
            'epochs': 100,
            'patience': 15,
            'weight_decay': 1e-5,
            'scheduler_patience': 7,
            'scheduler_factor': 0.5
        }
        if model_config is not None:
            self.config.update(model_config)

        # Feature names
        self.feature_names = [
            'daily_return', 'FEAT_DebtEquity_quarterly', 'FEAT_PE_quarterly',
            'FEAT_EVEBITDA_quarterly', 'FEAT_ROE_quarterly', 'price_mom_5d',
            'price_mom_10d', 'price_mom_20d', 'vol_w_mom_5d', 'vol_w_mom_10d',
            'vol_w_mom_20d', 'volatility_10d', 'skewness_20d', 'rsi_14d',
            'zscore_mom_10d_60w'
        ]

        # Scalers
        self.feature_scaler = RobustScaler()
        self.target_scaler = StandardScaler()

        # Training history
        self.history = {'train_loss': [], 'val_loss': [], 'ic': [], 'rank_ic': []} 

    def prepare_data(self, stock_df_path, market_df_path, test_size = 0.2):
        features_cols = self.feature_names
        dfs = []
        for path in stock_df_path:
            p = Path(path).resolve()
            exchange = p.parts[-2]
            df = pd.read_csv(p)
            df['exchange'] = exchange
            dfs.append(df)
        df_stock_all = pd.concat(dfs, ignore_index= True)
        dfs = []
        for path in market_df_path:
            p = Path(path).resolve()
            name = p.stem.upper()
            exchange = "HNX" if "HNX" in name else "HOSE"
            df = pd.read_csv(p)
            df['exchange'] = exchange
            dfs.append(df)
        df_market_all = pd.concat(dfs, ignore_index= True)
        df_market_subset = df_market_all[['time', 'exchange', 'FUT_RET_10D_market']]        
        merged_df = pd.merge(df_stock_all, df_market_subset, on = ['time', 'exchange'], how = 'inner' )
        merged_df['alpha'] = merged_df['FUT_RET_10D'] - merged_df['FUT_RET_10D_market']
        merged_df = merged_df.dropna() 
        #Separate features and target
        X = merged_df[features_cols].values
        y = merged_df['alpha'].values

        #Train test split
        split_idx = int(len(X) * (1 - test_size))
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]

        #Scale features and target
        X_train_scaled = self.feature_scaler.fit_transform(X_train)
        X_test_scaled = self.feature_scaler.transform(X_test)
        y_train_scaled = self.target_scaler.fit_transform(y_train.reshape(-1, 1)).flatten()
        y_test_scaled = self.target_scaler.transform(y_test.reshape(-1, 1)).flatten()

        print(f"Training set: {X_train_scaled.shape}, Test set: {X_test_scaled.shape}")

        return (X_train_scaled, X_test_scaled, y_train_scaled, y_test_scaled,
                y_train, y_test, features_cols)

    def calculate_ic(self, predictions, targets):
        # Remove NaN values
        valid_mask = ~(np.isnan(predictions) | np.isnan(targets))
        pred_clean = predictions[valid_mask]
        target_clean = targets[valid_mask]

        if len(pred_clean) < 10:  # Need minimum samples
            return 0.0, 0.0

        # Pearson correlation (IC)
        ic = np.corrcoef(pred_clean, target_clean)[0, 1]
        if np.isnan(ic):
            ic = 0.0

        # Spearman rank correlation (Rank IC)
        from scipy.stats import spearmanr
        rank_ic, _ = spearmanr(pred_clean, target_clean)
        if np.isnan(rank_ic):
            rank_ic = 0.0

        return ic, rank_ic

    def train_model(self, X_train, X_val, y_train, y_val, y_val_unscaled):
        # Train the neural network

        # Create datasets and dataloaders
        train_dataset = AlphaDataset(X_train, y_train)
        val_dataset = AlphaDataset(X_val, y_val)
        train_loader = DataLoader(train_dataset, batch_size= self.config['batch_size'], shuffle = True)
        val_loader = DataLoader(val_dataset, batch_size = self.config['batch_size'], shuffle= False)

        # Initialize mode
        model = AlphaNet(
            input_dim = X_train.shape[1],
            hidden_dims = self.config['hidden_dims'],
            dropout_rate = self.config['dropout_rate'],
            use_batch_norm = self.config['use_batch_norm']
        ).to(self.device)

        # Loss function
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(),
                               lr = self.config['learning_rate'],
                               weight_decay = self.config['weight_decay'])

        # Scheduler
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode = 'min', patience = self.config['scheduler_patience'],
            factor = self.config['scheduler_factor']
        )

        # Early stopping
        best_val_loss = float('inf')
        patience_counter = 0
        best_model_state = 0

        for epoch in range(self.config['epochs']):
            # Training
            model.train()
            train_loss = 0.0

            for batch_X, batch_y in train_loader:
                batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)
                optimizer.zero_grad()
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()

                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm= 1.0)

                optimizer.step()
                train_loss += loss.item()

            # Validation
            model.eval()
            val_loss = 0.0
            val_predictions = []
            with torch.no_grad():
                for batch_X, batch_y in val_loader:
                    batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)
                    outputs = model(batch_X)
                    loss = criterion(outputs, batch_y)
                    val_loss += loss.item()
                    val_predictions.extend(outputs.cpu().numpy())

            train_loss /= len(train_loader)
            val_loss /= len(val_loader)

            # Unscale predictions for IC calculation
            val_pred_unscaled = self.target_scaler.inverse_transform(
                np.array(val_predictions).reshape(-1, 1)
            ).flatten()

            ic, rank_ic = self.calculate_ic(val_pred_unscaled, y_val_unscaled)

            # Update history
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['ic'].append(ic)
            self.history['rank_ic'].append(rank_ic)

            # Learning rate scheduling
            scheduler.step(val_loss)

            # Early stopping check
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                best_model_state = model.state_dict().copy()
            else:
                patience_counter += 1

            # Print progress
            if (epoch + 1) % 10 == 0:
                print(f'Epoch [{epoch+1}/{self.config["epochs"]}], '
                      f'Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}, '
                      f'IC: {ic:.4f}, Rank IC: {rank_ic:.4f}')

            # Early stopping
            if patience_counter >= self.config['patience']:
                print(f'Early stopping at epoch {epoch+1}')
                break

        # Load best model
        if best_model_state is not None:
            model.load_state_dict(best_model_state)

        return model      

    def evaluate_model(self, model, X_test,
                       y_test_scaled, y_test_unscaled):
        model.eval() 
        with torch.no_grad(): 
            X_test_tensor = torch.FloatTensor(X_test).to(self.device)
            predictions_scaled = model(X_test_tensor).cpu().numpy()

            predictions_scaled_reshaped = predictions_scaled.reshape(-1, 1)

            predictions_unscaled = self.target_scaler.inverse_transform(
                predictions_scaled_reshaped
            ).flatten()
            # Unscaled loss
            mse = mean_squared_error(y_test_unscaled, predictions_unscaled)
            mae = mean_absolute_error(y_test_unscaled, predictions_unscaled)
            ic, rank_ic = self.calculate_ic(predictions_unscaled, y_test_unscaled)

            print(f"\nEvaluation on Test set:")
            print(f"MSE: {mse:.6f}")
            print(f"MAE: {mae:.6f}")
            print(f"IC: {ic:.4f}")
            print(f"Rank IC: {rank_ic:.4f}")

        return predictions_unscaled, {'mse': mse, 'mae': mae, 'ic': ic, 'rank_ic': rank_ic}          



def train_alpha_model(stock_df_path, market_df_path, config = None):

    trainer = AlphaTrainer()

    # Prepare data
    X_train, X_test, y_train_scaled, y_test_scaled, y_train, y_test, features_cols = trainer.prepare_data(stock_df_path, market_df_path)

    # Train model
    model = trainer.train_model(X_train, X_test, y_train_scaled, y_test_scaled, y_test)

    # Evaluate model
    predictions, metrics = trainer.evaluate_model(model, X_test, y_test_scaled, y_test)

    return model, trainer, predictions, metrics



import os
import glob
current_script_directory = os.path.dirname(os.path.abspath(__file__)) if '__file__' in locals() else os.getcwd()
input_base_folder = os.path.join(current_script_directory, '..', 'data_prep')
stock_features_path = os.path.join(input_base_folder, 'calculated_stock_features')
market_indices_path = os.path.join(input_base_folder, 'stock_data\market_indices')
stock_df_path = glob.glob(os.path.join(stock_features_path, "*", "*.csv"))
market_df_path = glob.glob(os.path.join(market_indices_path, "*.csv"))




train_alpha_model(stock_df_path, market_df_path)


# Training set: (664739, 15), Test set: (166185, 15)
# 
# Epoch [10/100], Train Loss: 0.988329, Val Loss: 0.679555, IC: 0.0729, Rank IC: 0.0697
# 
# Epoch [20/100], Train Loss: 0.985254, Val Loss: 0.678091, IC: 0.0875, Rank IC: 0.0811
# 
# Epoch [30/100], Train Loss: 0.983203, Val Loss: 0.678065, IC: 0.0891, Rank IC: 0.0837
# 
# Epoch [40/100], Train Loss: 0.981161, Val Loss: 0.677731, IC: 0.0895, Rank IC: 0.0880
# 
# Epoch [50/100], Train Loss: 0.979822, Val Loss: 0.678361, IC: 0.0854, Rank IC: 0.0857
# 
# Epoch [60/100], Train Loss: 0.978949, Val Loss: 0.677962, IC: 0.0894, Rank IC: 0.0874
# 
# Epoch [70/100], Train Loss: 0.978627, Val Loss: 0.678225, IC: 0.0896, Rank IC: 0.0868
# 
# Epoch [80/100], Train Loss: 0.977818, Val Loss: 0.678385, IC: 0.0859, Rank IC: 0.0845
# 
# Early stopping at epoch 83
# 
# Evaluation on Test set:
# 
# MSE: 65.867974
# 
# MAE: 5.424329
# 
# IC: 0.0916
# 
# Rank IC: 0.0888