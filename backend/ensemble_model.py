#!/usr/bin/env python3
"""
Ensemble Model Strategy
========================
Weighted ensemble combining Random Forest, XGBoost, and LightGBM.
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
import xgboost as xgb
import lightgbm as lgb
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, accuracy_score
import warnings

warnings.filterwarnings('ignore')


class EnsembleModel:
    """Weighted ensemble of multiple ML models."""
    
    def __init__(self, task='regression'):
        self.task = task
        self.models = {}
        self.weights = None
        self.best_weights = None
        
    def train_random_forest(self, X_train, y_train, X_val, y_val):
        """Train optimized Random Forest."""
        print("\n🌲 Training Random Forest...")
        
        if self.task == 'regression':
            model = RandomForestRegressor(
                n_estimators=300,
                max_depth=15,
                min_samples_split=10,
                min_samples_leaf=4,
                max_features='sqrt',
                max_samples=0.8,
                random_state=42,
                n_jobs=-1,
                verbose=0
            )
        else:
            model = RandomForestClassifier(
                n_estimators=300,
                max_depth=15,
                min_samples_split=10,
                min_samples_leaf=4,
                max_features='sqrt',
                max_samples=0.8,
                random_state=42,
                n_jobs=-1,
                verbose=0
            )
        
        model.fit(X_train, y_train)
        val_pred = model.predict(X_val)
        
        if self.task == 'regression':
            mae = mean_absolute_error(y_val, val_pred)
            print(f"   Validation MAE: {mae:.4f}")
        else:
            acc = accuracy_score(y_val, val_pred)
            print(f"   Validation Accuracy: {acc:.4f}")
        
        self.models['RandomForest'] = model
        return model
    
    def train_xgboost(self, X_train, y_train, X_val, y_val):
        """Train XGBoost model."""
        print("\n🚀 Training XGBoost...")
        
        if self.task == 'regression':
            model = xgb.XGBRegressor(
                n_estimators=500,
                max_depth=8,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                min_child_weight=3,
                gamma=0.1,
                random_state=42,
                n_jobs=-1,
                eval_metric='mae'
            )
        else:
            model = xgb.XGBClassifier(
                n_estimators=500,
                max_depth=8,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                min_child_weight=3,
                gamma=0.1,
                random_state=42,
                n_jobs=-1,
                eval_metric='logloss'
            )
        
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            verbose=False
        )
        
        val_pred = model.predict(X_val)
        
        if self.task == 'regression':
            mae = mean_absolute_error(y_val, val_pred)
            print(f"   Validation MAE: {mae:.4f}")
        else:
            acc = accuracy_score(y_val, val_pred)
            print(f"   Validation Accuracy: {acc:.4f}")
        
        self.models['XGBoost'] = model
        return model
    
    def train_lightgbm(self, X_train, y_train, X_val, y_val):
        """Train LightGBM model."""
        print("\n⚡ Training LightGBM...")
        
        if self.task == 'regression':
            model = lgb.LGBMRegressor(
                n_estimators=500,
                max_depth=10,
                learning_rate=0.05,
                num_leaves=63,
                subsample=0.8,
                colsample_bytree=0.8,
                min_child_samples=20,
                random_state=42,
                n_jobs=-1,
                verbose=-1
            )
        else:
            model = lgb.LGBMClassifier(
                n_estimators=500,
                max_depth=10,
                learning_rate=0.05,
                num_leaves=63,
                subsample=0.8,
                colsample_bytree=0.8,
                min_child_samples=20,
                random_state=42,
                n_jobs=-1,
                verbose=-1
            )
        
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            verbose=False
        )
        
        val_pred = model.predict(X_val)
        
        if self.task == 'regression':
            mae = mean_absolute_error(y_val, val_pred)
            print(f"   Validation MAE: {mae:.4f}")
        else:
            acc = accuracy_score(y_val, val_pred)
            print(f"   Validation Accuracy: {acc:.4f}")
        
        self.models['LightGBM'] = model
        return model
    
    def optimize_weights(self, X_val, y_val, n_trials=100):
        """Optimize ensemble weights using grid search."""
        print("\n⚖️  Optimizing ensemble weights...")
        
        if len(self.models) == 0:
            raise ValueError("No models trained yet")
        
        # Get predictions from each model
        predictions = {}
        for name, model in self.models.items():
            predictions[name] = model.predict(X_val)
        
        # Grid search for optimal weights
        best_score = float('inf') if self.task == 'regression' else -float('inf')
        best_weights = None
        
        # Generate weight combinations
        for i in range(n_trials):
            # Random weights that sum to 1
            weights = np.random.dirichlet(np.ones(len(self.models)))
            
            # Weighted ensemble prediction
            ensemble_pred = np.zeros(len(y_val))
            for idx, name in enumerate(self.models.keys()):
                ensemble_pred += weights[idx] * predictions[name]
            
            # Evaluate
            if self.task == 'regression':
                score = mean_absolute_error(y_val, ensemble_pred)
                if score < best_score:
                    best_score = score
                    best_weights = weights
            else:
                # Round to nearest class for classification
                ensemble_pred_class = np.round(ensemble_pred).astype(int)
                score = accuracy_score(y_val, ensemble_pred_class)
                if score > best_score:
                    best_score = score
                    best_weights = weights
        
        self.best_weights = best_weights
        
        print(f"   Best validation score: {best_score:.4f}")
        print(f"   Optimal weights:")
        for idx, name in enumerate(self.models.keys()):
            print(f"      {name}: {best_weights[idx]:.3f}")
        
        return best_weights
    
    def predict(self, X):
        """Make ensemble prediction using optimized weights."""
        if self.best_weights is None:
            # Equal weights if not optimized
            self.best_weights = np.ones(len(self.models)) / len(self.models)
        
        ensemble_pred = np.zeros(len(X))
        for idx, (name, model) in enumerate(self.models.items()):
            pred = model.predict(X)
            ensemble_pred += self.best_weights[idx] * pred
        
        if self.task == 'classification':
            ensemble_pred = np.round(ensemble_pred).astype(int)
        
        return ensemble_pred
    
    def train_all(self, X_train, y_train, X_val, y_val):
        """Train all models and optimize weights."""
        print("\n🤖 ENSEMBLE MODEL TRAINING")
        print("=" * 60)
        
        self.train_random_forest(X_train, y_train, X_val, y_val)
        self.train_xgboost(X_train, y_train, X_val, y_val)
        self.train_lightgbm(X_train, y_train, X_val, y_val)
        
        self.optimize_weights(X_val, y_val, n_trials=100)
        
        print("\n✅ Ensemble training complete")
        return self


if __name__ == "__main__":
    # Test ensemble model
    print("🧪 Testing Ensemble Model")
    print("=" * 60)
    
    # Generate synthetic data
    np.random.seed(42)
    n_samples = 1000
    n_features = 20
    
    X = np.random.randn(n_samples, n_features)
    y = X[:, 0] * 2 + X[:, 1] * 0.5 + np.random.randn(n_samples) * 0.1
    
    # Split
    split = int(0.7 * n_samples)
    val_split = int(0.85 * n_samples)
    
    X_train, y_train = X[:split], y[:split]
    X_val, y_val = X[split:val_split], y[split:val_split]
    X_test, y_test = X[val_split:], y[val_split:]
    
    # Train ensemble
    ensemble = EnsembleModel(task='regression')
    ensemble.train_all(X_train, y_train, X_val, y_val)
    
    # Test prediction
    y_pred = ensemble.predict(X_test)
    test_mae = mean_absolute_error(y_test, y_pred)
    print(f"\n📊 Test MAE: {test_mae:.4f}")
    
    print("\n✅ Ensemble model test complete")
