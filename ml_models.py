#!/usr/bin/env python3
"""
Comprehensive ML Models Module
================================

Implements three types of models for stock prediction:
1. Random Forest (baseline tree-based)  
2. XGBoost (advanced gradient boosting)
3. LSTM (deep learning for time-series)

Supports both:
- Regression (price prediction)
- Classification (BUY/SELL/HOLD)

Author: Stock Prediction System
Date: December 2025
"""

import pandas as pd
import numpy as np
import joblib
import os
import json
from pathlib import Path
import warnings

# Scikit-learn
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (mean_squared_error, mean_absolute_error, r2_score,
                             accuracy_score, precision_score, recall_score, f1_score,
                             classification_report, confusion_matrix)
from sklearn.preprocessing import LabelEncoder, StandardScaler

# XGBoost
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    print("⚠️  XGBoost not installed. Install with: pip install xgboost")

# TensorFlow/Keras for LSTM
try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout
    from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False
    print("⚠️  TensorFlow not installed. Install with: pip install tensorflow")

warnings.filterwarnings('ignore')


class StockPredictionModels:
    """
    Comprehensive ML model suite for stock prediction.
    
    Supports:
    - Regression models (price prediction)
    - Classification models (trading signals)
    - Model persistence and loading
    - Performance evaluation
    """
    
    def __init__(self, models_dir="models"):
        """
        Initialize models.
        
        Args:
            models_dir (str): Directory to save/load models
        """
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(parents=True, exist_ok=True)
        
        self.regression_models = {}
        self.classification_models = {}
        
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        
    # ==================== REGRESSION MODELS ====================
    
    def train_regression_models(self, X_train, y_train, X_val, y_val):
        """
        Train all regression models.
        
        Args:
            X_train, y_train: Training data
            X_val, y_val: Validation data
        """
        print("\n" + "=" * 80)
        print("TRAINING REGRESSION MODELS (Price Prediction)")
        print("=" * 80)
        
        results = {}
        
        # 1. Baseline: Linear Regression
        print("\n1️⃣  LINEAR REGRESSION (Baseline)")
        print("-" * 60)
        lr_model = LinearRegression()
        lr_model.fit(X_train, y_train)
        self.regression_models['Linear_Regression'] = lr_model
        results['Linear_Regression'] = self._evaluate_regression(lr_model, X_val, y_val)
        
        # 2. Random Forest Regressor
        print("\n2️⃣  RANDOM FOREST REGRESSOR")
        print("-" * 60)
        rf_model = RandomForestRegressor(
            n_estimators=100,
            max_depth=15,
            min_samples_split=10,
            min_samples_leaf=4,
            max_features='sqrt',
            random_state=42,
            n_jobs=-1,
            verbose=0
        )
        rf_model.fit(X_train, y_train)
        self.regression_models['Random_Forest'] = rf_model
        results['Random_Forest'] = self._evaluate_regression(rf_model, X_val, y_val)
        
        # 3. XGBoost Regressor
        if XGBOOST_AVAILABLE:
            print("\n3️⃣  XGBOOST REGRESSOR")
            print("-" * 60)
            xgb_model = xgb.XGBRegressor(
                n_estimators=100,
                max_depth=8,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                n_jobs=-1,
                verbosity=0
            )
            xgb_model.fit(X_train, y_train,
                         eval_set=[(X_val, y_val)],
                         verbose=False)
            self.regression_models['XGBoost'] = xgb_model
            results['XGBoost'] = self._evaluate_regression(xgb_model, X_val, y_val)
        
        # 4. LSTM Regressor
        if TENSORFLOW_AVAILABLE:
            print("\n4️⃣  LSTM REGRESSOR (Deep Learning)")
            print("-" * 60)
            lstm_model = self._build_lstm_regressor(X_train.shape[1])
            
            # Scale features for LSTM
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_val_scaled = self.scaler.transform(X_val)
            
            # Reshape for LSTM [samples, time_steps, features]
            X_train_lstm = X_train_scaled.reshape((X_train_scaled.shape[0], 1, X_train_scaled.shape[1]))
            X_val_lstm = X_val_scaled.reshape((X_val_scaled.shape[0], 1, X_val_scaled.shape[1]))
            
            # Early stopping
            early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
            
            # Train
            lstm_model.fit(
                X_train_lstm, y_train,
                validation_data=(X_val_lstm, y_val),
                epochs=50,
                batch_size=64,
                verbose=0,
                callbacks=[early_stop]
            )
            
            self.regression_models['LSTM'] = lstm_model
            results['LSTM'] = self._evaluate_regression_lstm(lstm_model, X_val_scaled, y_val)
        
        # Display results table
        self._display_regression_results(results)
        
        return results
    
    def _build_lstm_regressor(self, input_dim):
        """Build LSTM model for regression."""
        model = Sequential([
            LSTM(64, activation='relu', return_sequences=True, input_shape=(1, input_dim)),
            Dropout(0.2),
            LSTM(32, activation='relu'),
            Dropout(0.2),
            Dense(16, activation='relu'),
            Dense(1)  # Output: single price value
        ])
        model.compile(optimizer='adam', loss='mse', metrics=['mae'])
        return model
    
    def _evaluate_regression(self, model, X_val, y_val):
        """Evaluate regression model."""
        y_pred = model.predict(X_val)
        
        rmse = np.sqrt(mean_squared_error(y_val, y_pred))
        mae = mean_absolute_error(y_val, y_pred)
        r2 = r2_score(y_val, y_pred)
        mape = np.mean(np.abs((y_val - y_pred) / y_val)) * 100
        
        # Directional accuracy (did we predict up/down correctly?)
        actual_direction = np.sign(y_val.values[1:] - y_val.values[:-1])
        pred_direction = np.sign(y_pred[1:] - y_pred[:-1])
        dir_accuracy = (actual_direction == pred_direction).mean() * 100
        
        print(f"   RMSE: {rmse:.4f}")
        print(f"   MAE: {mae:.4f}")
        print(f"   R²: {r2:.4f}")
        print(f"   MAPE: {mape:.2f}%")
        print(f"   Directional Accuracy: {dir_accuracy:.2f}%")
        
        return {
            'RMSE': rmse,
            'MAE': mae,
            'R2': r2,
            'MAPE': mape,
            'Dir_Accuracy': dir_accuracy
        }
    
    def _evaluate_regression_lstm(self, model, X_val_scaled, y_val):
        """Evaluate LSTM regression model."""
        X_val_lstm = X_val_scaled.reshape((X_val_scaled.shape[0], 1, X_val_scaled.shape[1]))
        y_pred = model.predict(X_val_lstm, verbose=0).flatten()
        
        rmse = np.sqrt(mean_squared_error(y_val, y_pred))
        mae = mean_absolute_error(y_val, y_pred)
        r2 = r2_score(y_val, y_pred)
        mape = np.mean(np.abs((y_val - y_pred) / y_val)) * 100
        
        actual_direction = np.sign(y_val.values[1:] - y_val.values[:-1])
        pred_direction = np.sign(y_pred[1:] - y_pred[:-1])
        dir_accuracy = (actual_direction == pred_direction).mean() * 100
        
        print(f"   RMSE: {rmse:.4f}")
        print(f"   MAE: {mae:.4f}")
        print(f"   R²: {r2:.4f}")
        print(f"   MAPE: {mape:.2f}%")
        print(f"   Directional Accuracy: {dir_accuracy:.2f}%")
        
        return {
            'RMSE': rmse,
            'MAE': mae,
            'R2': r2,
            'MAPE': mape,
            'Dir_Accuracy': dir_accuracy
        }
    
    # ==================== CLASSIFICATION MODELS ====================
    
    def train_classification_models(self, X_train, y_train, X_val, y_val):
        """
        Train all classification models.
        
        Args:
            X_train, y_train: Training data
            X_val, y_val: Validation data
        """
        print("\n" + "=" * 80)
        print("TRAINING CLASSIFICATION MODELS (BUY/SELL/HOLD)")
        print("=" * 80)
        
        # Encode labels
        y_train_encoded = self.label_encoder.fit_transform(y_train)
        y_val_encoded = self.label_encoder.transform(y_val)
        
        results = {}
        
        # 1. Baseline: Logistic Regression
        print("\n1️⃣  LOGISTIC REGRESSION (Baseline)")
        print("-" * 60)
        lr_model = LogisticRegression(
            max_iter=1000,
            multi_class='multinomial',
            random_state=42
        )
        lr_model.fit(X_train, y_train)
        self.classification_models['Logistic_Regression'] = lr_model
        results['Logistic_Regression'] = self._evaluate_classification(lr_model, X_val, y_val)
        
        # 2. Random Forest Classifier
        print("\n2️⃣  RANDOM FOREST CLASSIFIER")
        print("-" * 60)
        rf_model = RandomForestClassifier(
            n_estimators=100,
            max_depth=15,
            min_samples_split=10,
            min_samples_leaf=4,
            max_features='sqrt',
            class_weight='balanced',
            random_state=42,
            n_jobs=-1,
            verbose=0
        )
        rf_model.fit(X_train, y_train)
        self.classification_models['Random_Forest'] = rf_model
        results['Random_Forest'] = self._evaluate_classification(rf_model, X_val, y_val)
        
        # 3. XGBoost Classifier
        if XGBOOST_AVAILABLE:
            print("\n3️⃣  XGBOOST CLASSIFIER")
            print("-" * 60)
            xgb_model = xgb.XGBClassifier(
                n_estimators=100,
                max_depth=8,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                n_jobs=-1,
                verbosity=0
            )
            xgb_model.fit(X_train, y_train_encoded,
                         eval_set=[(X_val, y_val_encoded)],
                         verbose=False)
            self.classification_models['XGBoost'] = xgb_model
            results['XGBoost'] = self._evaluate_classification_xgb(xgb_model, X_val, y_val_encoded)
        
        # 4. LSTM Classifier
        if TENSORFLOW_AVAILABLE:
            print("\n4️⃣  LSTM CLASSIFIER (Deep Learning)")
            print("-" * 60)
            lstm_model = self._build_lstm_classifier(X_train.shape[1], len(self.label_encoder.classes_))
            
            # Scale features
            X_train_scaled = self.scaler.transform(X_train) if hasattr(self.scaler, 'mean_') else self.scaler.fit_transform(X_train)
            X_val_scaled = self.scaler.transform(X_val)
            
            # One-hot encode labels
            y_train_onehot = tf.keras.utils.to_categorical(y_train_encoded, num_classes=len(self.label_encoder.classes_))
            y_val_onehot = tf.keras.utils.to_categorical(y_val_encoded, num_classes=len(self.label_encoder.classes_))
            
            # Reshape for LSTM
            X_train_lstm = X_train_scaled.reshape((X_train_scaled.shape[0], 1, X_train_scaled.shape[1]))
            X_val_lstm = X_val_scaled.reshape((X_val_scaled.shape[0], 1, X_val_scaled.shape[1]))
            
            # Early stopping
            early_stop = EarlyStopping(monitor='val_accuracy', patience=10, restore_best_weights=True)
            
            # Train
            lstm_model.fit(
                X_train_lstm, y_train_onehot,
                validation_data=(X_val_lstm, y_val_onehot),
                epochs=50,
                batch_size=64,
                verbose=0,
                callbacks=[early_stop]
            )
            
            self.classification_models['LSTM'] = lstm_model
            results['LSTM'] = self._evaluate_classification_lstm(lstm_model, X_val_scaled, y_val_encoded)
        
        # Display results table
        self._display_classification_results(results)
        
        return results
    
    def _build_lstm_classifier(self, input_dim, num_classes):
        """Build LSTM model for classification."""
        model = Sequential([
            LSTM(64, activation='relu', return_sequences=True, input_shape=(1, input_dim)),
            Dropout(0.3),
            LSTM(32, activation='relu'),
            Dropout(0.3),
            Dense(16, activation='relu'),
            Dense(num_classes, activation='softmax')  # Multi-class output
        ])
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        return model
    
    def _evaluate_classification(self, model, X_val, y_val):
        """Evaluate classification model."""
        y_pred = model.predict(X_val)
        
        accuracy = accuracy_score(y_val, y_pred)
        precision = precision_score(y_val, y_pred, average='weighted', zero_division=0)
        recall = recall_score(y_val, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(y_val, y_pred, average='weighted', zero_division=0)
        
        print(f"   Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
        print(f"   Precision: {precision:.4f}")
        print(f"   Recall: {recall:.4f}")
        print(f"   F1-Score: {f1:.4f}")
        
        return {
            'Accuracy': accuracy,
            'Precision': precision,
            'Recall': recall,
            'F1_Score': f1
        }
    
    def _evaluate_classification_xgb(self, model, X_val, y_val_encoded):
        """Evaluate XGBoost classification model."""
        y_pred_encoded = model.predict(X_val)
        y_pred = self.label_encoder.inverse_transform(y_pred_encoded)
        y_val_decoded = self.label_encoder.inverse_transform(y_val_encoded)
        
        accuracy = accuracy_score(y_val_decoded, y_pred)
        precision = precision_score(y_val_decoded, y_pred, average='weighted', zero_division=0)
        recall = recall_score(y_val_decoded, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(y_val_decoded, y_pred, average='weighted', zero_division=0)
        
        print(f"   Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
        print(f"   Precision: {precision:.4f}")
        print(f"   Recall: {recall:.4f}")
        print(f"   F1-Score: {f1:.4f}")
        
        return {
            'Accuracy': accuracy,
            'Precision': precision,
            'Recall': recall,
            'F1_Score': f1
        }
    
    def _evaluate_classification_lstm(self, model, X_val_scaled, y_val_encoded):
        """Evaluate LSTM classification model."""
        X_val_lstm = X_val_scaled.reshape((X_val_scaled.shape[0], 1, X_val_scaled.shape[1]))
        y_pred_prob = model.predict(X_val_lstm, verbose=0)
        y_pred_encoded = np.argmax(y_pred_prob, axis=1)
        
        y_pred = self.label_encoder.inverse_transform(y_pred_encoded)
        y_val_decoded = self.label_encoder.inverse_transform(y_val_encoded)
        
        accuracy = accuracy_score(y_val_decoded, y_pred)
        precision = precision_score(y_val_decoded, y_pred, average='weighted', zero_division=0)
        recall = recall_score(y_val_decoded, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(y_val_decoded, y_pred, average='weighted', zero_division=0)
        
        print(f"   Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
        print(f"   Precision: {precision:.4f}")
        print(f"   Recall: {recall:.4f}")
        print(f"   F1-Score: {f1:.4f}")
        
        return {
            'Accuracy': accuracy,
            'Precision': precision,
            'Recall': recall,
            'F1_Score': f1
        }
    
    def _display_regression_results(self, results):
        """Display regression results in table format."""
        print("\n" + "=" * 80)
        print("REGRESSION RESULTS SUMMARY")
        print("=" * 80)
        print(f"{'Model':<25} {'RMSE':<12} {'MAE':<12} {'R²':<10} {'MAPE':<10} {'Dir_Acc':<10}")
        print("-" * 80)
        for model_name, metrics in results.items():
            print(f"{model_name:<25} "
                  f"{metrics['RMSE']:<12.4f} "
                  f"{metrics['MAE']:<12.4f} "
                  f"{metrics['R2']:<10.4f} "
                  f"{metrics['MAPE']:<10.2f} "
                  f"{metrics['Dir_Accuracy']:<10.2f}")
    
    def _display_classification_results(self, results):
        """Display classification results in table format."""
        print("\n" + "=" * 80)
        print("CLASSIFICATION RESULTS SUMMARY")
        print("=" * 80)
        print(f"{'Model':<25} {'Accuracy':<12} {'Precision':<12} {'Recall':<12} {'F1-Score':<12}")
        print("-" * 80)
        for model_name, metrics in results.items():
            print(f"{model_name:<25} "
                  f"{metrics['Accuracy']:<12.4f} "
                  f"{metrics['Precision']:<12.4f} "
                  f"{metrics['Recall']:<12.4f} "
                  f"{metrics['F1_Score']:<12.4f}")
    
    def save_models(self):
        """Save all trained models."""
        print(f"\n💾 SAVING MODELS")
        print("-" * 60)
        
        # Save regression models
        for name, model in self.regression_models.items():
            if name == 'LSTM':
                model.save(self.models_dir / f"regression_{name.lower()}.h5")
            else:
                joblib.dump(model, self.models_dir / f"regression_{name.lower()}.joblib")
            print(f"✅ Saved regression model: {name}")
        
        # Save classification models
        for name, model in self.classification_models.items():
            if name == 'LSTM':
                model.save(self.models_dir / f"classification_{name.lower()}.h5")
            else:
                joblib.dump(model, self.models_dir / f"classification_{name.lower()}.joblib")
            print(f"✅ Saved classification model: {name}")
        
        # Save scalers and encoders
        joblib.dump(self.scaler, self.models_dir / "scaler.joblib")
        joblib.dump(self.label_encoder, self.models_dir / "label_encoder.joblib")
        print(f"✅ Saved scalers and encoders")


def main():
    """Test model training."""
    print("🧪 TESTING ML MODELS")
    print("=" * 80)
    
    # Load data with targets
    train_data = pd.read_csv('data/processed/train_data.csv')
    val_data = pd.read_csv('data/processed/val_data.csv')
    
    # Prepare feature-target split
    from target_generator import TargetGenerator
    
    # Add targets to train/val data
    generator_train = TargetGenerator(train_data)
    train_with_targets = generator_train.create_all_targets(horizons=[5])
    X_train, y_train_reg, y_train_class = generator_train.get_feature_target_split(horizon=5)
    
    generator_val = TargetGenerator(val_data)
    val_with_targets = generator_val.create_all_targets(horizons=[5])
    X_val, y_val_reg, y_val_class = generator_val.get_feature_target_split(horizon=5)
    
    print(f"\nTraining set: {len(X_train):,} samples")
    print(f"Validation set: {len(X_val):,} samples")
    
    # Create model suite
    models = StockPredictionModels()
    
    # Train regression models
    reg_results = models.train_regression_models(X_train, y_train_reg, X_val, y_val_reg)
    
    # Train classification models
    class_results = models.train_classification_models(X_train, y_train_class, X_val, y_val_class)
    
    # Save models
    models.save_models()
    
    print("\n✅ MODEL TRAINING COMPLETE!")


if __name__ == "__main__":
    main()
