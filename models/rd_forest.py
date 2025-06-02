import os
import glob
from pathlib import Path

import pandas as pd
import numpy as np
from scipy.stats import spearmanr

from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import RobustScaler, StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error


class RandomForestAlphaTrainer:
    def __init__(self, model_config=None):
        # Default configuration
        self.config = {
            'n_estimators': 100,
            'max_depth': None,
            'random_state': 42,
            'n_jobs': -1,
            'test_size': 0.2,
        }
        if model_config:
            self.config.update(model_config)
        # Feature names (same as other models)
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
        self.model = None

    def prepare_data(self, stock_paths, market_paths):
        # Load stock features
        dfs = []
        for p in stock_paths:
            path = Path(p).resolve()
            df = pd.read_csv(path)
            df['exchange'] = path.parts[-2]
            if 'time' in df.columns and not pd.api.types.is_datetime64_any_dtype(df['time']):
                df['time'] = pd.to_datetime(df['time'], errors='coerce')
            dfs.append(df)
        df_stock = pd.concat(dfs, ignore_index=True)
        # Load market indices
        dfs = []
        for p in market_paths:
            path = Path(p).resolve()
            df = pd.read_csv(path)
            name = path.stem.upper()
            df['exchange'] = 'HNX' if 'HNX' in name else 'HOSE'
            if 'time' in df.columns and not pd.api.types.is_datetime64_any_dtype(df['time']):
                df['time'] = pd.to_datetime(df['time'], errors='coerce')
            dfs.append(df)
        df_mkt = pd.concat(dfs, ignore_index=True)
        # Merge and compute alpha target
        df_mkt_sub = df_mkt[['time', 'exchange', 'FUT_RET_10D_market']]
        df = pd.merge(df_stock, df_mkt_sub, on=['time', 'exchange'], how='inner')
        df['alpha'] = df['FUT_RET_10D'] - df['FUT_RET_10D_market']
        df = df.sort_values('time').reset_index(drop=True)
        # Drop rows with NaN in features or target
        cols = [c for c in self.feature_names if c in df.columns] + ['alpha']
        df = df.dropna(subset=cols)
        # Split features and targets
        X = df[self.feature_names].values
        y = df['alpha'].values
        # Chronological train-test split
        split = int(len(X) * (1 - self.config['test_size']))
        X_train, X_test = X[:split], X[split:]
        y_train, y_test = y[:split], y[split:]
        # Scale data
        X_train_scaled = self.feature_scaler.fit_transform(X_train)
        X_test_scaled = self.feature_scaler.transform(X_test)
        y_train_scaled = self.target_scaler.fit_transform(y_train.reshape(-1,1)).flatten()
        y_test_scaled = self.target_scaler.transform(y_test.reshape(-1,1)).flatten()
        print(f"Training set: {X_train_scaled.shape}, Test set: {X_test_scaled.shape}")
        return X_train_scaled, X_test_scaled, y_train_scaled, y_test_scaled, y_train, y_test

    def calculate_ic(self, preds, targets):
        preds = np.asarray(preds)
        targets = np.asarray(targets)
        mask = ~(np.isnan(preds) | np.isnan(targets))
        p, t = preds[mask], targets[mask]
        if len(p) < 10:
            return 0.0, 0.0
        ic = np.corrcoef(p, t)[0,1]
        rank_ic, _ = spearmanr(p, t)
        return (0.0 if np.isnan(ic) else ic, 0.0 if np.isnan(rank_ic) else rank_ic)

    def train_model(self, X_train, y_train_scaled, X_val, y_val_scaled, y_val_unscaled):
        # Train Random Forest
        self.model = RandomForestRegressor(
            n_estimators=self.config['n_estimators'],
            max_depth=self.config['max_depth'],
            random_state=self.config['random_state'],
            n_jobs=self.config['n_jobs']
        )
        print("Training Random Forest model...")
        self.model.fit(X_train, y_train_scaled)
        # Validation
        preds_scaled = self.model.predict(X_val)
        preds_unscaled = self.target_scaler.inverse_transform(preds_scaled.reshape(-1,1)).flatten()
        ic, rank_ic = self.calculate_ic(preds_unscaled, y_val_unscaled)
        print(f"Validation IC: {ic:.4f}, Rank IC: {rank_ic:.4f}")
        return self.model

    def evaluate_model(self, X_test, y_test_unscaled):
        if self.model is None:
            raise ValueError("Model not trained. Call train_model first.")
        preds_scaled = self.model.predict(X_test)
        preds_unscaled = self.target_scaler.inverse_transform(preds_scaled.reshape(-1,1)).flatten()
        mse = mean_squared_error(y_test_unscaled, preds_unscaled)
        mae = mean_absolute_error(y_test_unscaled, preds_unscaled)
        ic, rank_ic = self.calculate_ic(preds_unscaled, y_test_unscaled)
        print(f"\nTest set results:\nMSE: {mse:.6f}, MAE: {mae:.6f}, IC: {ic:.4f}, Rank IC: {rank_ic:.4f}")
        return preds_unscaled, {'mse': mse, 'mae': mae, 'ic': ic, 'rank_ic': rank_ic}


def train_rf_alpha_model_main(stock_paths, market_paths, config=None):
    trainer = RandomForestAlphaTrainer(model_config=config)
    X_train, X_test, y_train_scaled, y_test_scaled, y_train, y_test = \
        trainer.prepare_data(stock_paths, market_paths)
    trainer.train_model(X_train, y_train_scaled, X_test, y_test_scaled, y_test)
    # Evaluate and return predictions, true values, and metrics
    preds_unscaled, metrics = trainer.evaluate_model(X_test, y_test)
    return trainer.model, trainer, preds_unscaled, y_test, metrics


if __name__ == '__main__':
    base = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data_prep')
    stock_dir = os.path.join(base, 'calculated_stock_features')
    market_dir = os.path.join(base, 'stock_data', 'market_indices')
    stock_paths = glob.glob(os.path.join(stock_dir, '*', '*.csv'))
    market_paths = glob.glob(os.path.join(market_dir, '*.csv'))
    model, trainer, preds, y_true, metrics = train_rf_alpha_model_main(stock_paths, market_paths)
    print("\nFinal Evaluation Metrics:")
    for k,v in metrics.items():
        print(f"{k.upper()}: {v:.6f}")
    # Calculate and display PnL using predicted signals and actual returns
    signals = np.sign(preds)
    pnl = np.sum(signals * y_true)
    return_pct = np.mean(signals * y_true) * 100
    print(f"PNL: {pnl:.6f}")
    print(f"Return %: {return_pct:.4f}%")


# Training set: (664739, 15), Test set: (166185, 15)
# Training Random Forest model...
# Validation IC: 0.0902, Rank IC: 0.0915

# Final Evaluation Metrics:
# MSE: 63.239111
# MAE: 5.080403
# IC: 0.090161
# RANK_IC: 0.091477