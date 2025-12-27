#!/usr/bin/env python3
"""
Directional Accuracy Metrics
=============================
Classification head for up/down prediction with confidence thresholds.
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import xgboost as xgb
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import warnings

warnings.filterwarnings('ignore')


class DirectionalPredictor:
    """Binary classifier for price direction with confidence scoring."""
    
    def __init__(self, confidence_threshold=0.6):
        self.confidence_threshold = confidence_threshold
        self.direction_model = None
        self.magnitude_model = None
        
    def create_direction_labels(self, y):
        """Convert returns to binary up/down labels."""
        if isinstance(y, pd.Series):
            y = y.values
        return (y > 0).astype(int)
    
    def train_direction_classifier(self, X_train, y_train, X_val, y_val):
        """Train binary classifier for direction."""
        print("\n🎯 DIRECTIONAL CLASSIFIER TRAINING")
        print("-" * 60)
        
        # Convert to binary labels
        y_train_dir = self.create_direction_labels(y_train)
        y_val_dir = self.create_direction_labels(y_val)
        
        # Train XGBoost classifier
        print("📊 Training XGBoost direction classifier...")
        self.direction_model = xgb.XGBClassifier(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            n_jobs=-1,
            eval_metric='logloss'
        )
        
        self.direction_model.fit(
            X_train, y_train_dir,
            eval_set=[(X_val, y_val_dir)],
            verbose=False
        )
        
        # Evaluate
        y_train_pred = self.direction_model.predict(X_train)
        y_val_pred = self.direction_model.predict(X_val)
        
        train_acc = accuracy_score(y_train_dir, y_train_pred)
        val_acc = accuracy_score(y_val_dir, y_val_pred)
        
        print(f"   Train accuracy: {train_acc:.4f} ({train_acc*100:.2f}%)")
        print(f"   Val accuracy:   {val_acc:.4f} ({val_acc*100:.2f}%)")
        
        return val_acc
    
    def predict_with_confidence(self, X):
        """Predict direction with confidence scores."""
        if self.direction_model is None:
            raise ValueError("Direction model not trained")
            
        # Get probability predictions
        proba = self.direction_model.predict_proba(X)
        max_proba = np.max(proba, axis=1)
        predictions = self.direction_model.predict(X)
        
        # Apply confidence threshold
        confident_mask = max_proba >= self.confidence_threshold
        
        return predictions, max_proba, confident_mask
    
    def calculate_directional_accuracy(self, y_true, y_pred):
        """Calculate directional accuracy metric."""
        if isinstance(y_true, pd.Series):
            y_true = y_true.values
        if isinstance(y_pred, pd.Series):
            y_pred = y_pred.values
            
        # Convert to direction (up/down)
        true_direction = (y_true > 0).astype(int)
        pred_direction = (y_pred > 0).astype(int)
        
        accuracy = accuracy_score(true_direction, pred_direction)
        return accuracy
    
    def calculate_profit_weighted_accuracy(self, y_true, y_pred):
        """Calculate accuracy weighted by return magnitude."""
        if isinstance(y_true, pd.Series):
            y_true = y_true.values
        if isinstance(y_pred, pd.Series):
            y_pred = y_pred.values
            
        # Direction correctness
        correct = ((y_true > 0) == (y_pred > 0)).astype(float)
        
        # Weight by absolute true return
        weights = np.abs(y_true)
        weights = weights / (weights.sum() + 1e-10)
        
        weighted_acc = (correct * weights).sum()
        return weighted_acc
    
    def evaluate_directional_performance(self, X_test, y_test):
        """Comprehensive directional performance evaluation."""
        print("\n📊 DIRECTIONAL PERFORMANCE EVALUATION")
        print("-" * 60)
        
        predictions, confidence, confident_mask = self.predict_with_confidence(X_test)
        y_test_dir = self.create_direction_labels(y_test)
        
        # Overall accuracy
        overall_acc = accuracy_score(y_test_dir, predictions)
        print(f"📈 Overall directional accuracy: {overall_acc:.4f} ({overall_acc*100:.2f}%)")
        
        # High-confidence accuracy
        if confident_mask.sum() > 0:
            high_conf_acc = accuracy_score(y_test_dir[confident_mask], predictions[confident_mask])
            print(f"📈 High-confidence accuracy (≥{self.confidence_threshold}): {high_conf_acc:.4f} ({high_conf_acc*100:.2f}%)")
            print(f"   High-confidence predictions: {confident_mask.sum():,} ({confident_mask.sum()/len(predictions)*100:.1f}%)")
        
        # Confusion matrix
        print("\n📊 Confusion Matrix:")
        cm = confusion_matrix(y_test_dir, predictions)
        print(f"              Predicted")
        print(f"              DOWN    UP")
        print(f"Actual DOWN   {cm[0,0]:5d}  {cm[0,1]:5d}")
        print(f"       UP     {cm[1,0]:5d}  {cm[1,1]:5d}")
        
        return overall_acc, confident_mask


if __name__ == "__main__":
    # Test directional predictor
    print("🧪 Testing Directional Predictor")
    print("=" * 60)
    
    # Generate synthetic data
    np.random.seed(42)
    n_samples = 1000
    n_features = 20
    
    X = np.random.randn(n_samples, n_features)
    y = np.random.randn(n_samples) * 0.02  # Returns
    
    # Split
    split = int(0.7 * n_samples)
    val_split = int(0.85 * n_samples)
    
    X_train, y_train = X[:split], y[:split]
    X_val, y_val = X[split:val_split], y[split:val_split]
    X_test, y_test = X[val_split:], y[val_split:]
    
    # Train and evaluate
    predictor = DirectionalPredictor(confidence_threshold=0.6)
    predictor.train_direction_classifier(X_train, y_train, X_val, y_val)
    predictor.evaluate_directional_performance(X_test, y_test)
    
    print("\n✅ Directional predictor test complete")
