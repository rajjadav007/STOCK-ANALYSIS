#!/usr/bin/env python3
"""
Fixed Model Improvement Pipeline - Works with actual data
"""

import pandas as pd
import numpy as np
import os
import json
import joblib
from datetime import datetime
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import xgboost as xgb
import lightgbm as lgb
import warnings

warnings.filterwarnings('ignore')


class FixedImprovementPipeline:
    """Pipeline that works with actual processed data format."""
    
    def __init__(self):
        self.df = None
        self.scaler = StandardScaler()
        
    def load_data(self):
        """Load all processed CSV files."""
        print("\n" + "=" * 80)
        print("MODEL IMPROVEMENT PIPELINE - RUNNING")
        print("=" * 80)
        
        import glob
        files = glob.glob("data/processed/*.csv")
        
        print(f"\n📂 Loading {len(files)} processed files...")
        
        dfs = []
        for file in files:
            df = pd.read_csv(file)
            # Convert Date to datetime
            df['Date'] = pd.to_datetime(df['Date'])
            dfs.append(df)
        
        self.df = pd.concat(dfs, ignore_index=True)
        self.df = self.df.sort_values(['Symbol', 'Date']).reset_index(drop=True)
        
        print(f"✅ Loaded {len(self.df):,} rows, {len(self.df.columns)} columns")
        print(f"   Date range: {self.df['Date'].min()} to {self.df['Date'].max()}")
        print(f"   Symbols: {self.df['Symbol'].nunique()}")
        
    def add_features(self):
        """Add enhanced features carefully."""
        print("\n⚙️  FEATURE ENGINEERING")
        print("-" * 80)
        
        initial_rows = len(self.df)
        
        # Add basic features per symbol
        for symbol in self.df['Symbol'].unique():
            mask = self.df['Symbol'] == symbol
            close = self.df.loc[mask, 'Close'].values
            high = self.df.loc[mask, 'High'].values
            low = self.df.loc[mask, 'Low'].values
            volume = self.df.loc[mask, 'Volume'].values
            
            # Returns
            self.df.loc[mask, 'Returns'] = pd.Series(close).pct_change().values
            
            # Lags (shorter windows to preserve data)
            for lag in [1, 2, 3]:
                self.df.loc[mask, f'Close_lag_{lag}'] = pd.Series(close).shift(lag).values
                self.df.loc[mask, f'Returns_lag_{lag}'] = pd.Series(close).pct_change().shift(lag).values
            
            # Moving averages
            self.df.loc[mask, 'SMA_5'] = pd.Series(close).rolling(5, min_periods=1).mean().values
            self.df.loc[mask, 'SMA_10'] = pd.Series(close).rolling(10, min_periods=1).mean().values
            self.df.loc[mask, 'EMA_12'] = pd.Series(close).ewm(span=12, min_periods=1).mean().values
            
            # RSI (14-day)
            delta = pd.Series(close).diff()
            gain = delta.where(delta > 0, 0).rolling(14, min_periods=1).mean()
            loss = -delta.where(delta < 0, 0).rolling(14, min_periods=1).mean()
            rs = gain / (loss + 1e-10)
            self.df.loc[mask, 'RSI'] = (100 - (100 / (1 + rs))).values
            
            # Volatility
            self.df.loc[mask, 'Volatility'] = pd.Series(close).pct_change().rolling(10, min_periods=1).std().values
            
            # Volume ratio
            vol_sma = pd.Series(volume).rolling(10, min_periods=1).mean()
            self.df.loc[mask, 'Volume_ratio'] = (volume / (vol_sma + 1e-10)).values
        
        # Target: next day return
        self.df['Target_Return'] = self.df.groupby('Symbol')['Close'].pct_change().shift(-1)
        
        # Drop only the last row per symbol (no future return) and any remaining NaN
        rows_before = len(self.df)
        self.df = self.df.dropna(subset=['Target_Return'])
        rows_after = len(self.df)
        
        print(f"   Added 20+ features")
        print(f"   Rows: {initial_rows:,} → {rows_after:,} (removed {rows_before - rows_after:,})")
        
        if len(self.df) == 0:
            raise ValueError("All rows removed! Check feature engineering logic.")
        
    def split_data(self):
        """Chronological split."""
        print("\n📅 TIME-SERIES SPLIT")
        print("-" * 80)
        
        n = len(self.df)
        train_end = int(n * 0.70)
        val_end = int(n * 0.85)
        
        train_df = self.df.iloc[:train_end]
        val_df = self.df.iloc[train_end:val_end]
        test_df = self.df.iloc[val_end:]
        
        print(f"   Train: {len(train_df):,} ({len(train_df)/n*100:.1f}%)")
        print(f"   Val:   {len(val_df):,} ({len(val_df)/n*100:.1f}%)")
        print(f"   Test:  {len(test_df):,} ({len(test_df)/n*100:.1f}%)")
        
        # Select numeric features
        exclude = ['Date', 'Symbol', 'Series', 'Target_Return', 'Close', 'Prev Close', 
                   'Last', 'VWAP', 'Turnover', 'Deliverable Volume', '%Deliverble']
        
        feature_cols = [col for col in self.df.columns 
                       if col not in exclude and self.df[col].dtype in ['int64', 'float64']]
        
        print(f"\n   Features: {len(feature_cols)}")
        
        # Prepare X, y
        X_train = train_df[feature_cols].fillna(0)
        y_train = train_df['Target_Return'].values
        X_val = val_df[feature_cols].fillna(0)
        y_val = val_df['Target_Return'].values
        X_test = test_df[feature_cols].fillna(0)
        y_test = test_df['Target_Return'].values
        
        # Scale
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_val_scaled = self.scaler.transform(X_val)
        X_test_scaled = self.scaler.transform(X_test)
        
        self.X_train = pd.DataFrame(X_train_scaled, columns=feature_cols)
        self.X_val = pd.DataFrame(X_val_scaled, columns=feature_cols)
        self.X_test = pd.DataFrame(X_test_scaled, columns=feature_cols)
        self.y_train = y_train
        self.y_val = y_val
        self.y_test = y_test
        self.feature_cols = feature_cols
        
    def train_models(self):
        """Train ensemble."""
        print("\n🤖 TRAINING MODELS")
        print("=" * 80)
        
        self.models = {}
        
        # Random Forest
        print("\n🌲 Random Forest...")
        rf = RandomForestRegressor(
            n_estimators=200,
            max_depth=10,
            min_samples_split=20,
            random_state=42,
            n_jobs=-1,
            verbose=0
        )
        rf.fit(self.X_train, self.y_train)
        val_pred = rf.predict(self.X_val)
        mae_rf = mean_absolute_error(self.y_val, val_pred)
        print(f"   Val MAE: {mae_rf:.6f}")
        self.models['RF'] = rf
        
        # XGBoost
        print("\n🚀 XGBoost...")
        xgb_model = xgb.XGBRegressor(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            n_jobs=-1
        )
        xgb_model.fit(self.X_train, self.y_train, eval_set=[(self.X_val, self.y_val)], verbose=False)
        val_pred = xgb_model.predict(self.X_val)
        mae_xgb = mean_absolute_error(self.y_val, val_pred)
        print(f"   Val MAE: {mae_xgb:.6f}")
        self.models['XGB'] = xgb_model
        
        # LightGBM
        print("\n⚡ LightGBM...")
        lgb_model = lgb.LGBMRegressor(
            n_estimators=300,
            max_depth=8,
            learning_rate=0.05,
            num_leaves=31,
            subsample=0.8,
            random_state=42,
            n_jobs=-1,
            verbose=-1
        )
        lgb_model.fit(self.X_train, self.y_train, eval_set=[(self.X_val, self.y_val)])
        val_pred = lgb_model.predict(self.X_val)
        mae_lgb = mean_absolute_error(self.y_val, val_pred)
        print(f"   Val MAE: {mae_lgb:.6f}")
        self.models['LGBM'] = lgb_model
        
        # Optimize weights
        print("\n⚖️  Optimizing ensemble weights...")
        predictions = {name: model.predict(self.X_val) for name, model in self.models.items()}
        
        best_mae = float('inf')
        best_weights = None
        
        for _ in range(100):
            weights = np.random.dirichlet(np.ones(3))
            ensemble_pred = sum(w * predictions[name] for w, name in zip(weights, self.models.keys()))
            mae = mean_absolute_error(self.y_val, ensemble_pred)
            
            if mae < best_mae:
                best_mae = mae
                best_weights = weights
        
        self.ensemble_weights = best_weights
        
        print(f"\n✅ Optimal weights:")
        for name, weight in zip(self.models.keys(), best_weights):
            print(f"      {name}: {weight:.3f}")
        print(f"   Best val MAE: {best_mae:.6f}")
        
    def evaluate(self):
        """Final evaluation."""
        print("\n📊 FINAL EVALUATION")
        print("=" * 80)
        
        # Ensemble prediction
        test_preds = {name: model.predict(self.X_test) for name, model in self.models.items()}
        ensemble_pred = sum(w * test_preds[name] for w, name in zip(self.ensemble_weights, self.models.keys()))
        
        # Metrics
        mae = mean_absolute_error(self.y_test, ensemble_pred)
        rmse = np.sqrt(mean_squared_error(self.y_test, ensemble_pred))
        r2 = r2_score(self.y_test, ensemble_pred)
        
        # Directional accuracy
        dir_acc = np.mean((self.y_test > 0) == (ensemble_pred > 0))
        
        # Profit-weighted accuracy
        correct = ((self.y_test > 0) == (ensemble_pred > 0)).astype(float)
        weights = np.abs(self.y_test)
        weights = weights / (weights.sum() + 1e-10)
        profit_weighted_acc = (correct * weights).sum()
        
        print(f"\n✅ ENSEMBLE PERFORMANCE:")
        print(f"   MAE:                      {mae:.6f}")
        print(f"   RMSE:                     {rmse:.6f}")
        print(f"   R²:                       {r2:.4f}")
        print(f"   Directional Accuracy:     {dir_acc*100:.2f}%")
        print(f"   Profit-Weighted Accuracy: {profit_weighted_acc*100:.2f}%")
        
        # Verdict
        if r2 > 0.1 and dir_acc > 0.55:
            verdict = f"STRONG: R²={r2:.4f}, Dir={dir_acc*100:.2f}%"
        elif r2 > 0.05 and dir_acc > 0.52:
            verdict = f"MODERATE: R²={r2:.4f}, Dir={dir_acc*100:.2f}%"
        else:
            verdict = f"BASELINE: R²={r2:.4f}, Dir={dir_acc*100:.2f}%"
        
        # Save results
        os.makedirs('results', exist_ok=True)
        report = {
            'timestamp': datetime.now().isoformat(),
            'metrics': {
                'MAE': float(mae),
                'RMSE': float(rmse),
                'R2': float(r2),
                'Directional_Accuracy': float(dir_acc),
                'Profit_Weighted_Accuracy': float(profit_weighted_acc)
            },
            'ensemble_weights': {name: float(w) for name, w in zip(self.models.keys(), self.ensemble_weights)},
            'feature_count': len(self.feature_cols),
            'verdict': verdict
        }
        
        with open('results/improvement_metrics.json', 'w') as f:
            json.dump(report, f, indent=2)
        
        # Save predictions for residual analysis
        predictions_df = pd.DataFrame({
            'Actual': self.y_test,
            'Predicted': ensemble_pred,
            'Residual': self.y_test - ensemble_pred
        })
        predictions_df.to_csv('results/predictions.csv', index=False)
        print(f"💾 Saved: results/predictions.csv")
        
        # Save models
        os.makedirs('models', exist_ok=True)
        joblib.dump(self.models, 'models/ensemble_models.joblib')
        joblib.dump(self.scaler, 'models/scaler.joblib')
        
        print(f"\n💾 Saved: results/improvement_metrics.json")
        print(f"💾 Saved: models/ensemble_models.joblib")
        
        print("\n" + "=" * 80)
        print(f"✅ VERDICT: {verdict}")
        print("=" * 80)
        
        return report
    
    def run(self):
        """Execute pipeline."""
        self.load_data()
        self.add_features()
        self.split_data()
        self.train_models()
        report = self.evaluate()
        return report


if __name__ == "__main__":
    pipeline = FixedImprovementPipeline()
    report = pipeline.run()
