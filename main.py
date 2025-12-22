#!/usr/bin/env python3
"""
Stock Market ML Analysis
Main script to run the complete analysis pipeline
"""

import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix
import joblib
import warnings
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.gridspec import GridSpec

warnings.filterwarnings('ignore')
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=DeprecationWarning)

class StockMarketAnalyzer:
    """Complete Stock Market ML Analysis System"""
    
    def __init__(self):
        self.data = None
        self.models = {}
        
    def load_data(self):
        """Load cleaned stock data"""
        print("📂 LOADING DATA")
        print("-" * 40)
        
        data_path = 'data/processed/cleaned_stock_data.csv'
        if not os.path.exists(data_path):
            print("❌ No processed data found. Please run data cleaning first.")
            print("   Run: python load_and_clean_data.py")
            return False
            
        self.data = pd.read_csv(data_path)
        print(f"✅ Loaded {len(self.data):,} records")
        print(f"📊 Stocks: {self.data['Symbol'].nunique()}")
        
        # Convert Date column and show range
        self.data['Date'] = pd.to_datetime(self.data['Date'])
        print(f"📅 Date range: {self.data['Date'].min().strftime('%Y-%m-%d')} to {self.data['Date'].max().strftime('%Y-%m-%d')}")
        
        # Sample data for faster training
        if len(self.data) > 20000:
            print(f"📦 Sampling 20,000 records for training...")
            self.data = self.data.sample(n=20000, random_state=42)
            
        return True
    
    def create_labels(self, df, future_window=5, buy_threshold=0.02, sell_threshold=-0.02):
        """
        Create BUY/SELL/HOLD labels using future price movement
        
        Parameters:
        -----------
        future_window : int, default=5
            Number of days to look ahead
        buy_threshold : float, default=0.02
            % gain threshold for BUY label (2%)
        sell_threshold : float, default=-0.02
            % loss threshold for SELL label (-2%)
        """
        print(f"🏷️  Creating BUY/SELL/HOLD labels (window={future_window} days, thresholds={buy_threshold*100:.1f}%/{sell_threshold*100:.1f}%)...")
        
        # Calculate future return for each stock separately
        df['Future_Close'] = df.groupby('Symbol')['Close'].shift(-future_window)
        df['Future_Return'] = (df['Future_Close'] - df['Close']) / df['Close']
        
        # Create labels based on thresholds
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
        
        # Show label distribution
        label_counts = df['Label'].value_counts()
        print(f"   BUY:  {label_counts.get('BUY', 0):,} ({label_counts.get('BUY', 0)/len(df.dropna(subset=['Label']))*100:.1f}%)")
        print(f"   HOLD: {label_counts.get('HOLD', 0):,} ({label_counts.get('HOLD', 0)/len(df.dropna(subset=['Label']))*100:.1f}%)")
        print(f"   SELL: {label_counts.get('SELL', 0):,} ({label_counts.get('SELL', 0)/len(df.dropna(subset=['Label']))*100:.1f}%)")
        
        return df
    
    def create_features(self):
        """Create ML features with comprehensive technical indicators"""
        print("\n⚙️ FEATURE ENGINEERING")
        print("-" * 40)
        
        df = self.data.copy()
        
        # Ensure proper date sorting (Date already converted in load_data)
        df = df.sort_values(['Symbol', 'Date']).reset_index(drop=True)
        
        # Track rows before indicator creation
        initial_rows = len(df)
        print(f"📊 Initial records: {initial_rows:,}")
        
        # Basic price features
        df['Price_Change'] = df['Close'] - df['Open']
        df['Price_Range'] = df['High'] - df['Low']
        df['Returns'] = df.groupby('Symbol')['Close'].pct_change()
        
        # Apply technical indicators using pandas and numpy for each stock symbol
        print("🔧 Creating technical indicators (SMA, EMA, RSI, MACD, Volatility, Volume change)...")
        
        # Process each stock separately to ensure correct calculations
        stock_groups = []
        for symbol in df['Symbol'].unique():
            stock_df = df[df['Symbol'] == symbol].copy()
            
            # SMA - Simple Moving Averages (short & long)
            stock_df['SMA_10'] = stock_df['Close'].rolling(window=10).mean()
            stock_df['SMA_50'] = stock_df['Close'].rolling(window=50).mean()
            
            # EMA - Exponential Moving Averages
            stock_df['EMA_12'] = stock_df['Close'].ewm(span=12, adjust=False).mean()
            stock_df['EMA_26'] = stock_df['Close'].ewm(span=26, adjust=False).mean()
            
            # RSI - Relative Strength Index (14-day)
            delta = stock_df['Close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            stock_df['RSI_14'] = 100 - (100 / (1 + rs))
            
            # MACD - Moving Average Convergence Divergence
            stock_df['MACD'] = stock_df['EMA_12'] - stock_df['EMA_26']
            stock_df['MACD_signal'] = stock_df['MACD'].ewm(span=9, adjust=False).mean()
            stock_df['MACD_hist'] = stock_df['MACD'] - stock_df['MACD_signal']
            
            # Volatility - Standard deviation of returns
            stock_df['Volatility_10'] = stock_df['Returns'].rolling(window=10).std()
            stock_df['Volatility_20'] = stock_df['Returns'].rolling(window=20).std()
            
            # Volume change percentage
            stock_df['Volume_Change_Pct'] = stock_df['Volume'].pct_change() * 100
            
            stock_groups.append(stock_df)
        
        # Combine all stocks back together
        df = pd.concat(stock_groups, ignore_index=True)
        print("✅ Technical indicators created successfully")
        
        # Lagged features
        df['Close_lag_1'] = df.groupby('Symbol')['Close'].shift(1)
        df['Volume_lag_1'] = df.groupby('Symbol')['Volume'].shift(1)
        
        # Time features
        df['Year'] = df['Date'].dt.year
        df['Month'] = df['Date'].dt.month
        df['DayOfWeek'] = df['Date'].dt.dayofweek
        df['Quarter'] = df['Date'].dt.quarter
        
        # CREATE BUY/SELL/HOLD LABELS before removing NaN
        print("\n🎯 SUPERVISED LEARNING LABELS")
        print("-" * 40)
        df = self.create_labels(df, future_window=5, buy_threshold=0.02, sell_threshold=-0.02)
        
        # Remove rows with NaN values created by technical indicators AND labels
        print("\n🧹 Removing NaN rows created by indicators and labels...")
        rows_before_nan_removal = len(df)
        df = df.dropna()
        rows_after_nan_removal = len(df)
        nan_rows_removed = rows_before_nan_removal - rows_after_nan_removal
        
        print(f"✅ Features created: {df.shape[1]} columns")
        print(f"📊 Records after NaN removal: {rows_after_nan_removal:,}")
        print(f"🗑️  NaN rows removed: {nan_rows_removed:,} ({(nan_rows_removed/initial_rows)*100:.2f}%)")
        print(f"💾 Data retention: {(rows_after_nan_removal/initial_rows)*100:.2f}%")
        
        self.data = df
        return True
    
    def prepare_ml_data(self):
        """Prepare data for machine learning with proper feature/label separation"""
        print("\n🎯 PREPARING ML DATA")
        print("-" * 40)
        
        # 1. Define columns to EXCLUDE (non-ML columns)
        exclude_cols = [
            'Date',           # Not numeric, temporal identifier
            'Symbol',         # Categorical identifier
            'Series',         # Categorical, not useful for prediction
            'Prev Close',     # Redundant (we have Close_lag_1)
            'Last',           # Redundant with Close
            'VWAP',           # Volume Weighted Average Price (can be noisy)
            'Turnover',       # Derived from Volume * Price
            'Trades',         # Less relevant for price prediction
            'Deliverable Volume',  # Subset of Volume
            '%Deliverble',    # Percentage metric, less relevant
            'Future_Close',   # Target leakage! Future price we're predicting
            'Future_Return',  # Target leakage! Future return we're predicting
            'Label',          # This is our TARGET (y), not a feature
            'Close'           # Also a target for regression (optional)
        ]
        
        # 2. Get all numeric columns from dataset
        print("📋 Identifying features...")
        all_columns = self.data.columns.tolist()
        
        # 3. Select only numeric columns, exclude non-ML columns
        feature_cols = []
        for col in all_columns:
            # Skip excluded columns
            if col in exclude_cols:
                continue
            # Include only numeric columns
            if self.data[col].dtype in ['int64', 'float64']:
                feature_cols.append(col)
        
        print(f"   Total columns in dataset: {len(all_columns)}")
        print(f"   Excluded columns: {len(exclude_cols)}")
        print(f"   Feature columns: {len(feature_cols)}")
        
        # 4. Ensure all required features exist
        available_features = [col for col in feature_cols if col in self.data.columns]
        
        # 5. Display feature list
        print(f"\n📊 FEATURE LIST ({len(available_features)} features):")
        print("-" * 40)
        
        # Group features by category
        price_features = [f for f in available_features if f in ['Open', 'High', 'Low', 'Price_Change', 'Price_Range', 'Close_lag_1']]
        volume_features = [f for f in available_features if 'Volume' in f or f == 'Trades']
        technical_features = [f for f in available_features if any(x in f for x in ['SMA', 'EMA', 'RSI', 'MACD', 'Volatility'])]
        time_features = [f for f in available_features if f in ['Year', 'Month', 'DayOfWeek', 'Quarter']]
        other_features = [f for f in available_features if f not in price_features + volume_features + technical_features + time_features]
        
        if price_features:
            print(f"  Price Features ({len(price_features)}):")
            for f in price_features:
                print(f"    • {f}")
        
        if volume_features:
            print(f"  Volume Features ({len(volume_features)}):")
            for f in volume_features:
                print(f"    • {f}")
        
        if technical_features:
            print(f"  Technical Indicators ({len(technical_features)}):")
            for f in technical_features:
                print(f"    • {f}")
        
        if time_features:
            print(f"  Time Features ({len(time_features)}):")
            for f in time_features:
                print(f"    • {f}")
        
        if other_features:
            print(f"  Other Features ({len(other_features)}):")
            for f in other_features:
                print(f"    • {f}")
        
        # 6. Prepare X (features) and y (labels)
        print(f"\n🔄 SEPARATING FEATURES AND LABELS")
        print("-" * 40)
        
        if 'Label' in self.data.columns:
            # Supervised learning with BUY/SELL/HOLD labels
            # Include Date for time-based splitting
            ml_data = self.data[available_features + ['Label', 'Date']].dropna()
            
            X = ml_data[available_features]
            y = ml_data['Label']  # Categorical target for classification
            
            print(f"✅ Target: Label (BUY/SELL/HOLD - Classification)")
            print(f"📊 Label distribution:")
            label_dist = y.value_counts()
            for label, count in label_dist.items():
                print(f"   {label}: {count:,} ({count/len(y)*100:.1f}%)")
        else:
            # Fallback: Regression on Close price
            # Include Date for time-based splitting
            ml_data = self.data[available_features + ['Close', 'Date']].dropna()
            
            X = ml_data[available_features]
            y = ml_data['Close']  # Continuous target for regression
            
            print(f"✅ Target: Close (Price - Regression)")
        
        # 7. Verify no data leakage
        print(f"\n🔒 DATA LEAKAGE CHECK:")
        print("-" * 40)
        leakage_cols = ['Future_Close', 'Future_Return', 'Close']
        found_leakage = [col for col in leakage_cols if col in X.columns]
        
        if found_leakage:
            print(f"⚠️  WARNING: Found potential leakage columns in features:")
            for col in found_leakage:
                print(f"   • {col}")
        else:
            print(f"✅ No data leakage detected - all future columns excluded")
        
        # 8. TIME-BASED TRAIN-TEST SPLIT (No Shuffling!)
        print(f"\n📦 TIME-BASED TRAIN-TEST SPLIT:")
        print("-" * 40)
        print(f"⚠️  Using TIME-BASED split (chronological order preserved)")
        print(f"   Train on PAST data → Test on FUTURE data")
        print(f"   NO shuffling to prevent data leakage!")
        
        # Ensure data is sorted by date (already done, but verify)
        ml_data_sorted = ml_data.sort_values('Date').reset_index(drop=True)
        
        # Calculate split index (80% train, 20% test)
        split_idx = int(len(ml_data_sorted) * 0.8)
        
        # Split chronologically
        train_data = ml_data_sorted.iloc[:split_idx]
        test_data = ml_data_sorted.iloc[split_idx:]
        
        # Extract features and labels
        self.X_train = train_data[available_features]
        self.y_train = train_data['Label'] if 'Label' in ml_data_sorted.columns else train_data['Close']
        self.X_test = test_data[available_features]
        self.y_test = test_data['Label'] if 'Label' in ml_data_sorted.columns else test_data['Close']
        
        # Get date ranges
        train_start = train_data['Date'].min()
        train_end = train_data['Date'].max()
        test_start = test_data['Date'].min()
        test_end = test_data['Date'].max()
        
        print(f"\n📅 TRAIN PERIOD:")
        print(f"   Start:  {train_start.strftime('%Y-%m-%d')}")
        print(f"   End:    {train_end.strftime('%Y-%m-%d')}")
        print(f"   Days:   {(train_end - train_start).days}")
        print(f"   Samples: {len(self.X_train):,} ({len(self.X_train)/len(ml_data_sorted)*100:.1f}%)")
        
        print(f"\n📅 TEST PERIOD:")
        print(f"   Start:  {test_start.strftime('%Y-%m-%d')}")
        print(f"   End:    {test_end.strftime('%Y-%m-%d')}")
        print(f"   Days:   {(test_end - test_start).days}")
        print(f"   Samples: {len(self.X_test):,} ({len(self.X_test)/len(ml_data_sorted)*100:.1f}%)")
        
        # Verify no temporal leakage
        if train_end >= test_start:
            print(f"\n⚠️  WARNING: Temporal overlap detected!")
            print(f"   Last train date: {train_end.strftime('%Y-%m-%d')}")
            print(f"   First test date: {test_start.strftime('%Y-%m-%d')}")
        else:
            print(f"\n✅ No temporal overlap - clean time-based split")
            gap_days = (test_start - train_end).days
            print(f"   Gap between train and test: {gap_days} day(s)")
        
        print(f"\n📊 SUMMARY:")
        print(f"   Total samples: {len(ml_data_sorted):,}")
        print(f"   Features: {X.shape[1]}")
        print(f"   Target: {y.name if hasattr(y, 'name') else 'Label'}")
        
        # Label distribution in train/test
        if 'Label' in ml_data_sorted.columns:
            print(f"\n🏷️  LABEL DISTRIBUTION:")
            print(f"   Train:")
            for label, count in self.y_train.value_counts().items():
                print(f"     {label}: {count:,} ({count/len(self.y_train)*100:.1f}%)")
            print(f"   Test:")
            for label, count in self.y_test.value_counts().items():
                print(f"     {label}: {count:,} ({count/len(self.y_test)*100:.1f}%)")
        
        # 9. Store feature names and metadata
        self.feature_names = available_features
        self.ml_data = ml_data_sorted
        self.X = X
        self.y = y
        self.train_data = train_data
        self.test_data = test_data
        
        return True
    
    def train_models(self):
        """Train ML models (classification or regression based on target)"""
        print("\n🤖 TRAINING MODELS")
        print("-" * 40)
        
        # Check if we're doing classification or regression
        is_classification = hasattr(self, 'y') and self.y.dtype == 'object'
        
        if is_classification:
            print("📊 Task: Classification (BUY/SELL/HOLD prediction)")
            
            # Logistic Regression (multi-class)
            print("📈 Training Logistic Regression...")
            lr_model = LogisticRegression(max_iter=1000, random_state=42, multi_class='multinomial')
            lr_model.fit(self.X_train, self.y_train)
            self.models['Logistic Regression'] = lr_model
            print("✅ Logistic Regression trained")
            
            # Random Forest Classifier
            print("🌲 Training Random Forest Classifier...")
            rf_model = RandomForestClassifier(
                n_estimators=100, max_depth=10, random_state=42, n_jobs=2
            )
            rf_model.fit(self.X_train, self.y_train)
            self.models['Random Forest'] = rf_model
            print("✅ Random Forest Classifier trained")
            
        else:
            print("📊 Task: Regression (Price prediction)")
            
            # Linear Regression
            print("📈 Training Linear Regression...")
            lr_model = LinearRegression()
            lr_model.fit(self.X_train, self.y_train)
            self.models['Linear Regression'] = lr_model
            print("✅ Linear Regression trained")
            
            # Random Forest Regressor
            print("🌲 Training Random Forest Regressor...")
            rf_model = RandomForestRegressor(
                n_estimators=20, max_depth=8, random_state=42, n_jobs=2
            )
            rf_model.fit(self.X_train, self.y_train)
            self.models['Random Forest'] = rf_model
            print("✅ Random Forest Regressor trained")
        
        print(f"✅ Trained {len(self.models)} models")
        self.is_classification = is_classification
        
    def evaluate_models(self):
        """Evaluate model performance (classification or regression)"""
        print("\n📊 EVALUATING MODELS")
        print("-" * 40)
        
        results = []
        best_score = -float('inf')
        best_model = None
        best_name = ""
        
        if self.is_classification:
            # Classification metrics
            for name, model in self.models.items():
                # Make predictions
                y_pred = model.predict(self.X_test)
                
                # Calculate metrics
                accuracy = accuracy_score(self.y_test, y_pred)
                precision = precision_score(self.y_test, y_pred, average='weighted', zero_division=0)
                recall = recall_score(self.y_test, y_pred, average='weighted', zero_division=0)
                f1 = f1_score(self.y_test, y_pred, average='weighted', zero_division=0)
                
                results.append({
                    'Model': name,
                    'Accuracy': accuracy,
                    'Precision': precision,
                    'Recall': recall,
                    'F1-Score': f1
                })
                
                print(f"  {name}:")
                print(f"    Accuracy:  {accuracy:.4f} ({accuracy*100:.2f}%)")
                print(f"    Precision: {precision:.4f}")
                print(f"    Recall:    {recall:.4f}")
                print(f"    F1-Score:  {f1:.4f}")
                
                # Track best model (by F1-score)
                if f1 > best_score:
                    best_score = f1
                    best_model = model
                    best_name = name
            
            # Print detailed classification report for best model
            print(f"\n📋 DETAILED CLASSIFICATION REPORT (Best Model: {best_name})")
            print("-" * 60)
            y_pred_best = best_model.predict(self.X_test)
            print(classification_report(self.y_test, y_pred_best, zero_division=0))
            
            # Confusion matrix
            print(f"\n📊 CONFUSION MATRIX:")
            cm = confusion_matrix(self.y_test, y_pred_best)
            labels = sorted(self.y_test.unique())
            print(f"\n          Predicted →")
            print(f"        {' '.join([f'{l:>8}' for l in labels])}")
            print("Actual ↓")
            for i, label in enumerate(labels):
                print(f"{label:>6}  {' '.join([f'{cm[i][j]:>8}' for j in range(len(labels))])}")
            
            metric_name = "F1-Score"
        else:
            # Regression metrics
            for name, model in self.models.items():
                # Make predictions
                y_pred = model.predict(self.X_test)
                
                # Calculate metrics
                rmse = np.sqrt(mean_squared_error(self.y_test, y_pred))
                mae = mean_absolute_error(self.y_test, y_pred)
                r2 = r2_score(self.y_test, y_pred)
                
                results.append({
                    'Model': name,
                    'RMSE': rmse,
                    'MAE': mae,
                    'R²': r2
                })
                
                print(f"  {name}:")
                print(f"    RMSE: {rmse:.2f}")
                print(f"    MAE: {mae:.2f}")
                print(f"    R²: {r2:.4f}")
                
                # Track best model (by R²)
                if r2 > best_score:
                    best_score = r2
                    best_model = model
                    best_name = name
            
            metric_name = "R²"
        
        # Save results
        os.makedirs('results', exist_ok=True)
        results_df = pd.DataFrame(results)
        results_df.to_csv('results/model_performance.csv', index=False)
        
        # Save best model
        os.makedirs('models', exist_ok=True)
        joblib.dump(best_model, 'models/best_model.joblib')
        
        print(f"\n🏆 BEST MODEL: {best_name} ({metric_name} = {best_score:.4f})")
        print("💾 Results saved to results/model_performance.csv")
        print("💾 Best model saved to models/best_model.joblib")
        
        # Store predictions for visualization
        self.best_model = best_model
        self.best_model_name = best_name
        self.y_pred = best_model.predict(self.X_test)
        
        return results_df
    
    def visualize_results(self):
        """Create comprehensive visualizations of results"""
        print("\n📊 CREATING VISUALIZATIONS")
        print("-" * 40)
        
        # Create results directory for plots
        plots_dir = 'results/plots'
        os.makedirs(plots_dir, exist_ok=True)
        
        # 1. Actual vs Predicted Prices
        self.plot_predictions()
        
        # 2. Technical Indicators Visualization
        self.plot_technical_indicators()
        
        # 3. Model Performance Comparison
        self.plot_model_comparison()
        
        # 4. Feature Importance (for Random Forest)
        self.plot_feature_importance()
        
        print(f"✅ All visualizations saved to {plots_dir}/")
        print("📈 Open the PNG files to view the graphs")
    
    def plot_predictions(self):
        """Plot actual vs predicted prices"""
        plt.figure(figsize=(14, 6))
        
        # Get sample of test data for cleaner visualization
        sample_size = min(200, len(self.y_test))
        indices = np.arange(sample_size)
        
        plt.plot(indices, self.y_test.iloc[:sample_size].values, 
                label='Actual Price', color='blue', linewidth=2, alpha=0.7)
        plt.plot(indices, self.y_pred[:sample_size], 
                label='Predicted Price', color='red', linewidth=2, alpha=0.7, linestyle='--')
        
        plt.xlabel('Sample Index', fontsize=12)
        plt.ylabel('Stock Price', fontsize=12)
        plt.title(f'Stock Price Prediction: Actual vs Predicted\n({self.best_model_name})', 
                 fontsize=14, fontweight='bold')
        plt.legend(fontsize=10)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        plt.savefig('results/plots/actual_vs_predicted.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("  ✅ Actual vs Predicted plot saved")
    
    def plot_technical_indicators(self):
        """Plot technical indicators for a sample stock"""
        # Get data with technical indicators
        sample_data = self.data.head(200).copy()
        
        if len(sample_data) == 0:
            print("  ⚠️  No data available for technical indicators plot")
            return
        
        # Create figure with subplots
        fig = plt.figure(figsize=(16, 12))
        gs = GridSpec(4, 2, figure=fig, hspace=0.3, wspace=0.3)
        
        # 1. Price with SMA
        ax1 = fig.add_subplot(gs[0, :])
        ax1.plot(sample_data.index, sample_data['Close'], label='Close Price', color='black', linewidth=2)
        ax1.plot(sample_data.index, sample_data['SMA_10'], label='SMA 10', color='blue', linewidth=1.5)
        ax1.plot(sample_data.index, sample_data['SMA_50'], label='SMA 50', color='red', linewidth=1.5)
        ax1.set_title('Stock Price with Simple Moving Averages (SMA)', fontweight='bold', fontsize=12)
        ax1.set_ylabel('Price')
        ax1.legend(loc='best')
        ax1.grid(True, alpha=0.3)
        
        # 2. Price with EMA
        ax2 = fig.add_subplot(gs[1, :])
        ax2.plot(sample_data.index, sample_data['Close'], label='Close Price', color='black', linewidth=2)
        ax2.plot(sample_data.index, sample_data['EMA_12'], label='EMA 12', color='green', linewidth=1.5)
        ax2.plot(sample_data.index, sample_data['EMA_26'], label='EMA 26', color='orange', linewidth=1.5)
        ax2.set_title('Stock Price with Exponential Moving Averages (EMA)', fontweight='bold', fontsize=12)
        ax2.set_ylabel('Price')
        ax2.legend(loc='best')
        ax2.grid(True, alpha=0.3)
        
        # 3. RSI
        ax3 = fig.add_subplot(gs[2, 0])
        ax3.plot(sample_data.index, sample_data['RSI_14'], label='RSI 14', color='purple', linewidth=2)
        ax3.axhline(y=70, color='r', linestyle='--', label='Overbought (70)', linewidth=1)
        ax3.axhline(y=30, color='g', linestyle='--', label='Oversold (30)', linewidth=1)
        ax3.set_title('Relative Strength Index (RSI)', fontweight='bold', fontsize=12)
        ax3.set_ylabel('RSI')
        ax3.set_ylim(0, 100)
        ax3.legend(loc='best', fontsize=8)
        ax3.grid(True, alpha=0.3)
        
        # 4. MACD
        ax4 = fig.add_subplot(gs[2, 1])
        ax4.plot(sample_data.index, sample_data['MACD'], label='MACD', color='blue', linewidth=2)
        ax4.plot(sample_data.index, sample_data['MACD_signal'], label='Signal', color='red', linewidth=1.5)
        ax4.bar(sample_data.index, sample_data['MACD_hist'], label='Histogram', color='gray', alpha=0.3)
        ax4.set_title('MACD (Moving Average Convergence Divergence)', fontweight='bold', fontsize=12)
        ax4.set_ylabel('MACD')
        ax4.legend(loc='best', fontsize=8)
        ax4.grid(True, alpha=0.3)
        
        # 5. Volatility
        ax5 = fig.add_subplot(gs[3, 0])
        ax5.plot(sample_data.index, sample_data['Volatility_10'], label='Volatility 10', color='red', linewidth=2)
        ax5.plot(sample_data.index, sample_data['Volatility_20'], label='Volatility 20', color='orange', linewidth=1.5)
        ax5.set_title('Price Volatility (Standard Deviation)', fontweight='bold', fontsize=12)
        ax5.set_ylabel('Volatility')
        ax5.legend(loc='best', fontsize=8)
        ax5.grid(True, alpha=0.3)
        
        # 6. Volume
        ax6 = fig.add_subplot(gs[3, 1])
        ax6.bar(sample_data.index, sample_data['Volume'], color='steelblue', alpha=0.6)
        ax6.set_title('Trading Volume', fontweight='bold', fontsize=12)
        ax6.set_ylabel('Volume')
        ax6.grid(True, alpha=0.3)
        
        plt.suptitle('Technical Indicators Analysis', fontsize=16, fontweight='bold', y=0.995)
        plt.savefig('results/plots/technical_indicators.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("  ✅ Technical indicators plot saved")
    
    def plot_model_comparison(self):
        """Plot model performance comparison (classification or regression)"""
        # Read results
        results_df = pd.read_csv('results/model_performance.csv')
        
        if self.is_classification:
            # Classification metrics
            fig, axes = plt.subplots(1, 4, figsize=(18, 5))
            
            # Accuracy
            axes[0].bar(results_df['Model'], results_df['Accuracy']*100, color=['#3498db', '#e74c3c'])
            axes[0].set_title('Accuracy\n(Higher is Better)', fontweight='bold')
            axes[0].set_ylabel('Accuracy (%)')
            axes[0].tick_params(axis='x', rotation=15)
            axes[0].grid(True, alpha=0.3, axis='y')
            axes[0].set_ylim([0, 100])
            
            # Precision
            axes[1].bar(results_df['Model'], results_df['Precision']*100, color=['#2ecc71', '#f39c12'])
            axes[1].set_title('Precision\n(Higher is Better)', fontweight='bold')
            axes[1].set_ylabel('Precision (%)')
            axes[1].tick_params(axis='x', rotation=15)
            axes[1].grid(True, alpha=0.3, axis='y')
            axes[1].set_ylim([0, 100])
            
            # Recall
            axes[2].bar(results_df['Model'], results_df['Recall']*100, color=['#9b59b6', '#1abc9c'])
            axes[2].set_title('Recall\n(Higher is Better)', fontweight='bold')
            axes[2].set_ylabel('Recall (%)')
            axes[2].tick_params(axis='x', rotation=15)
            axes[2].grid(True, alpha=0.3, axis='y')
            axes[2].set_ylim([0, 100])
            
            # F1-Score
            axes[3].bar(results_df['Model'], results_df['F1-Score']*100, color=['#e67e22', '#34495e'])
            axes[3].set_title('F1-Score\n(Higher is Better)', fontweight='bold')
            axes[3].set_ylabel('F1-Score (%)')
            axes[3].tick_params(axis='x', rotation=15)
            axes[3].grid(True, alpha=0.3, axis='y')
            axes[3].set_ylim([0, 100])
            
            plt.suptitle('Classification Model Performance Comparison', fontsize=16, fontweight='bold')
        else:
            # Regression metrics
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))
            
            # RMSE comparison
            axes[0].bar(results_df['Model'], results_df['RMSE'], color=['#3498db', '#e74c3c'])
            axes[0].set_title('RMSE Comparison\n(Lower is Better)', fontweight='bold')
            axes[0].set_ylabel('RMSE')
            axes[0].tick_params(axis='x', rotation=15)
            axes[0].grid(True, alpha=0.3, axis='y')
            
            # MAE comparison
            axes[1].bar(results_df['Model'], results_df['MAE'], color=['#2ecc71', '#f39c12'])
            axes[1].set_title('MAE Comparison\n(Lower is Better)', fontweight='bold')
            axes[1].set_ylabel('MAE')
            axes[1].tick_params(axis='x', rotation=15)
            axes[1].grid(True, alpha=0.3, axis='y')
            
            # R² comparison
            axes[2].bar(results_df['Model'], results_df['R²'], color=['#9b59b6', '#1abc9c'])
            axes[2].set_title('R² Score Comparison\n(Higher is Better)', fontweight='bold')
            axes[2].set_ylabel('R² Score')
            axes[2].tick_params(axis='x', rotation=15)
            axes[2].grid(True, alpha=0.3, axis='y')
            axes[2].set_ylim([0, 1.1])
            
            plt.suptitle('Regression Model Performance Comparison', fontsize=16, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig('results/plots/model_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("  ✅ Model comparison plot saved")
    
    def plot_feature_importance(self):
        """Plot feature importance for Random Forest model"""
        if 'Random Forest' not in self.models:
            print("  ⚠️  Random Forest model not available for feature importance")
            return
        
        rf_model = self.models['Random Forest']
        
        # Get feature importances
        importances = rf_model.feature_importances_
        feature_names = self.feature_names
        
        # Sort by importance
        indices = np.argsort(importances)[::-1][:15]  # Top 15 features
        
        plt.figure(figsize=(12, 8))
        plt.barh(range(len(indices)), importances[indices], color='steelblue', alpha=0.8)
        plt.yticks(range(len(indices)), [feature_names[i] for i in indices])
        plt.xlabel('Feature Importance Score', fontsize=12)
        plt.title('Top 15 Most Important Features\n(Random Forest Model)', 
                 fontsize=14, fontweight='bold')
        plt.gca().invert_yaxis()
        plt.grid(True, alpha=0.3, axis='x')
        plt.tight_layout()
        
        plt.savefig('results/plots/feature_importance.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("  ✅ Feature importance plot saved")
    
    def save_labeled_dataset(self):
        """Save the dataset with BUY/SELL/HOLD labels"""
        if self.data is not None and 'Label' in self.data.columns:
            output_path = 'data/processed/labeled_stock_data.csv'
            os.makedirs('data/processed', exist_ok=True)
            self.data.to_csv(output_path, index=False)
            print(f"\n💾 Labeled dataset saved to: {output_path}")
            print(f"   Total records: {len(self.data):,}")
            print(f"   Columns: {', '.join(self.data.columns.tolist()[:10])}...")
            return output_path
        return None
    
    def run_complete_analysis(self):
        """Run the complete ML analysis"""
        print("🚀 STOCK MARKET ML ANALYSIS")
        print("=" * 50)
        
        # Execute pipeline
        if not self.load_data():
            return False
            
        if not self.create_features():
            return False
        
        # Save labeled dataset
        self.save_labeled_dataset()
            
        if not self.prepare_ml_data():
            return False
            
        self.train_models()
        self.evaluate_models()
        self.visualize_results()
        
        print("\n✅ ANALYSIS COMPLETED SUCCESSFULLY!")
        print("📊 Check results/ folder for performance metrics")
        print("📁 Check data/processed/labeled_stock_data.csv for labeled data")
        print("🤖 Check models/ folder for trained model")
        print("📈 Check results/plots/ folder for visualizations")
        
        return True

if __name__ == "__main__":
    analyzer = StockMarketAnalyzer()
    analyzer.run_complete_analysis()
