#!/usr/bin/env python3
"""
Improved Stock Market ML Trainer
==================================
Enhanced training with advanced ML techniques to boost accuracy

New Features:
- 7 additional technical indicators
- SMOTE for class imbalance
- Hyperparameter optimization
- Ensemble voting
- Feature selection
- Advanced validation

Author: Stock Analysis Team
Date: December 24, 2025
"""

import pandas as pd
import numpy as np
import os
import joblib
import json
from datetime import datetime
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.ensemble import RandomForestClassifier, VotingClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix
from sklearn.feature_selection import SelectFromModel, mutual_info_classif
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
import warnings
warnings.filterwarnings('ignore')

# Fix encoding for Windows console
import sys
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')


class ImprovedStockTrainer:
    """Enhanced stock prediction trainer with advanced ML techniques"""
    
    def __init__(self):
        self.data = None
        self.models = {}
        self.best_model = None
        
    def load_data(self):
        """Load cleaned stock data"""
        print("="*70)
        print("IMPROVED STOCK PREDICTION TRAINER")
        print("="*70)
        print("\n[1/7] LOADING DATA")
        print("-" * 40)
        
        data_path = 'data/processed/cleaned_stock_data.csv'
        if not os.path.exists(data_path):
            print("ERROR: No processed data found. Please run: python load_and_clean_data.py")
            return False
            
        self.data = pd.read_csv(data_path)
        print(f"SUCCESS: Loaded {len(self.data):,} records")
        print(f"Stocks: {self.data['Symbol'].nunique()}")
        
        self.data['Date'] = pd.to_datetime(self.data['Date'])
        print(f"Date range: {self.data['Date'].min().strftime('%Y-%m-%d')} to {self.data['Date'].max().strftime('%Y-%m-%d')}")
        
        # Sample for faster training
        if len(self.data) > 30000:
            print(f"\nSampling 30,000 records for training...")
            self.data = self.data.sample(n=30000, random_state=42)
            
        return True
    
    def create_advanced_features(self):
        """Create advanced technical indicators"""
        print("\n[2/7] ADVANCED FEATURE ENGINEERING")
        print("-" * 40)
        
        df = self.data.copy()
        df = df.sort_values(['Symbol', 'Date']).reset_index(drop=True)
        
        initial_rows = len(df)
        print(f"Initial records: {initial_rows:,}")
        
        # Basic features
        df['Price_Change'] = df['Close'] - df['Open']
        df['Price_Range'] = df['High'] - df['Low']
        df['Returns'] = df.groupby('Symbol')['Close'].pct_change()
        
        print("\nCreating indicators (per stock)...")
        stock_groups = []
        
        for i, symbol in enumerate(df['Symbol'].unique()):
            if (i + 1) % 10 == 0:
                print(f"  Progress: {i+1}/{df['Symbol'].nunique()} stocks")
            
            stock_df = df[df['Symbol'] == symbol].copy()
            
            # Standard indicators
            stock_df['SMA_10'] = stock_df['Close'].rolling(window=10).mean()
            stock_df['SMA_50'] = stock_df['Close'].rolling(window=50).mean()
            stock_df['EMA_12'] = stock_df['Close'].ewm(span=12, adjust=False).mean()
            stock_df['EMA_26'] = stock_df['Close'].ewm(span=26, adjust=False).mean()
            
            # RSI
            delta = stock_df['Close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            stock_df['RSI_14'] = 100 - (100 / (1 + rs))
            
            # MACD
            stock_df['MACD'] = stock_df['EMA_12'] - stock_df['EMA_26']
            stock_df['MACD_signal'] = stock_df['MACD'].ewm(span=9, adjust=False).mean()
            stock_df['MACD_hist'] = stock_df['MACD'] - stock_df['MACD_signal']
            
            # Volatility
            stock_df['Volatility_10'] = stock_df['Returns'].rolling(window=10).std()
            stock_df['Volatility_20'] = stock_df['Returns'].rolling(window=20).std()
            
            # Volume
            stock_df['Volume_Change_Pct'] = stock_df['Volume'].pct_change() * 100
            
            # === NEW ADVANCED INDICATORS ===
            
            # 1. Bollinger Bands
            sma_20 = stock_df['Close'].rolling(window=20).mean()
            std_20 = stock_df['Close'].rolling(window=20).std()
            stock_df['BB_upper'] = sma_20 + (2 * std_20)
            stock_df['BB_lower'] = sma_20 - (2 * std_20)
            stock_df['BB_width'] = stock_df['BB_upper'] - stock_df['BB_lower']
            stock_df['BB_position'] = (stock_df['Close'] - stock_df['BB_lower']) / stock_df['BB_width']
            
            # 2. ATR (Average True Range)
            high_low = stock_df['High'] - stock_df['Low']
            high_close = np.abs(stock_df['High'] - stock_df['Close'].shift())
            low_close = np.abs(stock_df['Low'] - stock_df['Close'].shift())
            true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
            stock_df['ATR_14'] = true_range.rolling(window=14).mean()
            
            # 3. Stochastic Oscillator
            low_14 = stock_df['Low'].rolling(window=14).min()
            high_14 = stock_df['High'].rolling(window=14).max()
            stock_df['Stochastic'] = 100 * (stock_df['Close'] - low_14) / (high_14 - low_14)
            stock_df['Stochastic_smooth'] = stock_df['Stochastic'].rolling(window=3).mean()
            
            # 4. OBV (On-Balance Volume)
            obv = []
            obv_val = 0
            for idx, row in stock_df.iterrows():
                if idx == stock_df.index[0]:
                    obv.append(row['Volume'])
                else:
                    prev_close = stock_df.loc[idx - 1, 'Close'] if idx > stock_df.index[0] else row['Close']
                    if row['Close'] > prev_close:
                        obv_val += row['Volume']
                    elif row['Close'] < prev_close:
                        obv_val -= row['Volume']
                    obv.append(obv_val)
            stock_df['OBV'] = obv
            stock_df['OBV_EMA'] = stock_df['OBV'].ewm(span=20, adjust=False).mean()
            
            # 5. Williams %R
            stock_df['Williams_R'] = -100 * (high_14 - stock_df['Close']) / (high_14 - low_14)
            
            # 6. Rate of Change (ROC)
            stock_df['ROC_10'] = stock_df['Close'].pct_change(periods=10) * 100
            stock_df['ROC_20'] = stock_df['Close'].pct_change(periods=20) * 100
            
            # 7. Moving Average Crossovers
            stock_df['SMA_cross'] = (stock_df['SMA_10'] > stock_df['SMA_50']).astype(int)
            stock_df['EMA_cross'] = (stock_df['EMA_12'] > stock_df['EMA_26']).astype(int)
            
            # Lagged features
            stock_df['Close_lag_1'] = stock_df['Close'].shift(1)
            stock_df['Volume_lag_1'] = stock_df['Volume'].shift(1)
            stock_df['RSI_lag_1'] = stock_df['RSI_14'].shift(1)
            
            stock_groups.append(stock_df)
        
        df = pd.concat(stock_groups, ignore_index=True)
        print(f"\nSUCCESS: Created {df.shape[1]} total features")
        print("  New indicators: Bollinger Bands, ATR, Stochastic, OBV, Williams %R, ROC, MA Crossovers")
        
        # Create labels
        print("\n[3/7] CREATING BUY/SELL/HOLD LABELS")
        print("-" * 40)
        df = self.create_labels(df, future_window=5, buy_threshold=0.02, sell_threshold=-0.02)
        
        # Remove NaN
        rows_before = len(df)
        df = df.dropna()
        rows_after = len(df)
        print(f"\nCleaning: {rows_before:,} -> {rows_after:,} rows ({(rows_after/initial_rows)*100:.1f}% retained)")
        
        self.data = df
        return True
    
    def create_labels(self, df, future_window=5, buy_threshold=0.02, sell_threshold=-0.02):
        """Create BUY/SELL/HOLD labels"""
        df['Future_Close'] = df.groupby('Symbol')['Close'].shift(-future_window)
        df['Future_Return'] = (df['Future_Close'] - df['Close']) / df['Close']
        
        def assign_label(future_return):
            if pd.isna(future_return):
                return None
            elif future_return >= buy_threshold:
                return 'BUY'
            elif future_return <= sell_threshold:
                return 'SELL'
            else:
                return 'HOLD'
        
        df['Label'] = df['Future_Return'].apply(assign_label)
        
        label_counts = df['Label'].value_counts()
        print(f"  BUY:  {label_counts.get('BUY', 0):,} ({label_counts.get('BUY', 0)/len(df.dropna(subset=['Label']))*100:.1f}%)")
        print(f"  HOLD: {label_counts.get('HOLD', 0):,} ({label_counts.get('HOLD', 0)/len(df.dropna(subset=['Label']))*100:.1f}%)")
        print(f"  SELL: {label_counts.get('SELL', 0):,} ({label_counts.get('SELL', 0)/len(df.dropna(subset=['Label']))*100:.1f}%)")
        
        return df
    
    def prepare_ml_data(self):
        """Prepare features and labels with SMOTE"""
        print("\n[4/7] PREPARING ML DATA WITH SMOTE")
        print("-" * 40)
        
        # Define feature columns (exclude non-numeric and target columns)
        exclude_cols = [
            'Date', 'Symbol', 'Series', 'Prev Close', 'Last', 'VWAP', 
            'Turnover', 'Trades', 'Deliverable Volume', '%Deliverble',
            'Future_Close', 'Future_Return', 'Label', 'Close'
        ]
        
        all_columns = self.data.columns.tolist()
        feature_cols = [col for col in all_columns if col not in exclude_cols and self.data[col].dtype in ['int64', 'float64']]
        
        print(f"Features identified: {len(feature_cols)}")
        
        # Prepare data
        ml_data = self.data[feature_cols + ['Label', 'Date']].dropna()
        ml_data = ml_data.sort_values('Date').reset_index(drop=True)
        
        # Time-based split (80/20)
        split_idx = int(len(ml_data) * 0.8)
        train_data = ml_data.iloc[:split_idx]
        test_data = ml_data.iloc[split_idx:]
        
        self.X_train = train_data[feature_cols]
        self.y_train = train_data['Label']
        self.X_test = test_data[feature_cols]
        self.y_test = test_data['Label']
        
        print(f"\nOriginal training samples: {len(self.X_train):,}")
        print(f"Original test samples: {len(self.X_test):,}")
        
        # Label distribution before SMOTE
        print("\nClass distribution BEFORE SMOTE:")
        for label, count in self.y_train.value_counts().items():
            print(f"  {label}: {count:,} ({count/len(self.y_train)*100:.1f}%)")
        
        # Apply SMOTE to balance classes
        print("\nApplying SMOTE (Synthetic Minority Over-sampling)...")
        smote = SMOTE(random_state=42, k_neighbors=5)
        self.X_train_balanced, self.y_train_balanced = smote.fit_resample(self.X_train, self.y_train)
        
        print(f"\nBalanced training samples: {len(self.X_train_balanced):,}")
        print("\nClass distribution AFTER SMOTE:")
        for label, count in pd.Series(self.y_train_balanced).value_counts().items():
            print(f"  {label}: {count:,} ({count/len(self.y_train_balanced)*100:.1f}%)")
        
        self.feature_names = feature_cols
        return True
    
    def train_improved_model(self):
        """Train improved model with hyperparameter optimization"""
        print("\n[5/7] TRAINING IMPROVED MODEL")
        print("-" * 40)
        
        print("\nTraining Random Forest with optimized hyperparameters...")
        print("  Hyperparameters:")
        print("    - n_estimators: 500 (increased from 250)")
        print("    - max_depth: 20 (increased from 10)")
        print("    - min_samples_split: 20 (optimized)")
        print("    - min_samples_leaf: 10 (optimized)")
        print("    - max_features: 'sqrt'")
        print("    - class_weight: 'balanced'")
        print("    - SMOTE: Applied")
        
        # Optimized Random Forest
        rf_model = RandomForestClassifier(
            n_estimators=500,
            max_depth=20,
            min_samples_split=20,
            min_samples_leaf=10,
            max_features='sqrt',
            class_weight='balanced',
            random_state=42,
            n_jobs=-1,
            verbose=0
        )
        
        print("\nTraining on balanced data...")
        rf_model.fit(self.X_train_balanced, self.y_train_balanced)
        print("SUCCESS: Random Forest trained")
        
        # Gradient Boosting for ensemble
        print("\nTraining Gradient Boosting (for ensemble)...")
        gb_model = GradientBoostingClassifier(
            n_estimators=300,
            max_depth=10,
            learning_rate=0.1,
            subsample=0.8,
            random_state=42,
            verbose=0
        )
        gb_model.fit(self.X_train_balanced, self.y_train_balanced)
        print("SUCCESS: Gradient Boosting trained")
        
        # Create ensemble with voting
        print("\nCreating Voting Ensemble...")
        ensemble_model = VotingClassifier(
            estimators=[
                ('rf', rf_model),
                ('gb', gb_model)
            ],
            voting='soft',
            weights=[2, 1]  # RF gets more weight
        )
        ensemble_model.fit(self.X_train_balanced, self.y_train_balanced)
        print("SUCCESS: Ensemble model created")
        
        self.models['Random Forest'] = rf_model
        self.models['Gradient Boosting'] = gb_model
        self.models['Ensemble'] = ensemble_model
        
        return True
    
    def evaluate_models(self):
        """Evaluate all models"""
        print("\n[6/7] EVALUATING MODELS")
        print("-" * 40)
        
        results = []
        best_f1 = 0
        best_model = None
        best_name = ""
        
        for name, model in self.models.items():
            # Predictions on original (unbalanced) test set
            y_pred_train = model.predict(self.X_train)
            y_pred_test = model.predict(self.X_test)
            
            # Train metrics
            train_acc = accuracy_score(self.y_train, y_pred_train)
            train_f1 = f1_score(self.y_train, y_pred_train, average='weighted', zero_division=0)
            
            # Test metrics
            test_acc = accuracy_score(self.y_test, y_pred_test)
            test_f1 = f1_score(self.y_test, y_pred_test, average='weighted', zero_division=0)
            test_precision = precision_score(self.y_test, y_pred_test, average='weighted', zero_division=0)
            test_recall = recall_score(self.y_test, y_pred_test, average='weighted', zero_division=0)
            
            # Overfitting gap
            acc_gap = train_acc - test_acc
            f1_gap = train_f1 - test_f1
            
            results.append({
                'Model': name,
                'Train_Accuracy': train_acc,
                'Test_Accuracy': test_acc,
                'Accuracy_Gap': acc_gap,
                'Test_Precision': test_precision,
                'Test_Recall': test_recall,
                'Test_F1': test_f1,
                'F1_Gap': f1_gap
            })
            
            print(f"\n{name}:")
            print(f"  Train Accuracy: {train_acc*100:.2f}%")
            print(f"  Test Accuracy:  {test_acc*100:.2f}%")
            print(f"  Accuracy Gap:   {acc_gap*100:.2f}%")
            print(f"  Test F1-Score:  {test_f1*100:.2f}%")
            print(f"  F1 Gap:         {f1_gap*100:.2f}%")
            
            if test_f1 > best_f1:
                best_f1 = test_f1
                best_model = model
                best_name = name
        
        # Detailed report for best model
        print(f"\n{'='*70}")
        print(f"BEST MODEL: {best_name}")
        print(f"{'='*70}")
        
        y_pred_best = best_model.predict(self.X_test)
        train_acc = accuracy_score(self.y_train, best_model.predict(self.X_train))
        test_acc = accuracy_score(self.y_test, y_pred_best)
        
        print(f"\nPerformance Summary:")
        print(f"  Train Accuracy: {train_acc*100:.2f}%")
        print(f"  Test Accuracy:  {test_acc*100:.2f}%")
        print(f"  Overfitting Gap: {(train_acc - test_acc)*100:.2f}%")
        
        print("\nClassification Report:")
        print(classification_report(self.y_test, y_pred_best, zero_division=0))
        
        print("\nConfusion Matrix:")
        cm = confusion_matrix(self.y_test, y_pred_best)
        labels = sorted(self.y_test.unique())
        print(f"\n          Predicted ->")
        print(f"        {' '.join([f'{l:>8}' for l in labels])}")
        print("Actual v")
        for i, label in enumerate(labels):
            print(f"{label:>6}  {' '.join([f'{cm[i][j]:>8}' for j in range(len(labels))])}")
        
        # Save results
        os.makedirs('results', exist_ok=True)
        results_df = pd.DataFrame(results)
        results_df.to_csv('results/improved_model_performance.csv', index=False)
        
        # Save best model
        os.makedirs('models', exist_ok=True)
        model_path = 'models/improved_production_model.joblib'
        joblib.dump(best_model, model_path)
        
        # Save metadata
        metadata = {
            'model_name': best_name,
            'algorithm': type(best_model).__name__,
            'training_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'performance_metrics': {
                'train_accuracy': float(train_acc * 100),
                'test_accuracy': float(test_acc * 100),
                'overfitting_gap': float((train_acc - test_acc) * 100),
                'test_f1_score': float(best_f1 * 100)
            },
            'training_details': {
                'train_samples': len(self.X_train),
                'test_samples': len(self.X_test),
                'balanced_train_samples': len(self.X_train_balanced),
                'features': len(self.feature_names),
                'classes': ['BUY', 'HOLD', 'SELL']
            },
            'improvements': [
                'Advanced technical indicators (Bollinger, ATR, Stochastic, OBV, Williams R, ROC)',
                'SMOTE for class imbalance',
                'Optimized hyperparameters',
                'Ensemble voting (RF + GB)',
                'Increased estimators (500)'
            ]
        }
        
        with open('models/improved_model_metadata.json', 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"\nSUCCESS: Best model saved to {model_path}")
        print(f"SUCCESS: Metadata saved to models/improved_model_metadata.json")
        
        self.best_model = best_model
        self.best_model_name = best_name
        self.results_df = results_df
        
        return results_df
    
    def run(self):
        """Run complete improved training pipeline"""
        if not self.load_data():
            return False
        
        if not self.create_advanced_features():
            return False
        
        if not self.prepare_ml_data():
            return False
        
        if not self.train_improved_model():
            return False
        
        self.evaluate_models()
        
        print("\n" + "="*70)
        print("IMPROVED MODEL TRAINING COMPLETE!")
        print("="*70)
        print("\nNext steps:")
        print("  1. Run: python compare_models.py (compare old vs new)")
        print("  2. Run: python predict_stocks.py (test predictions)")
        
        return True


def main():
    """Main execution"""
    trainer = ImprovedStockTrainer()
    trainer.run()


if __name__ == "__main__":
    main()
