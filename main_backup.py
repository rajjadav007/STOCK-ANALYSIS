#!/usr/bin/env python3
"""
Stock Market ML Analysis
Main script to run the complete analysis pipeline
"""

import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib
import warnings

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
        
        # Apply technical indicators using pandas-ta for each stock symbol
        print("🔧 Creating technical indicators with pandas-ta...")
        
        if ta is not None:
            # Process each stock separately to ensure correct calculations
            stock_groups = []
            for symbol in df['Symbol'].unique():
                stock_df = df[df['Symbol'] == symbol].copy()
                
                # SMA - Simple Moving Averages (short & long)
                stock_df['SMA_10'] = ta.sma(stock_df['Close'], length=10)
                stock_df['SMA_50'] = ta.sma(stock_df['Close'], length=50)
                
                # EMA - Exponential Moving Average
                stock_df['EMA_12'] = ta.ema(stock_df['Close'], length=12)
                stock_df['EMA_26'] = ta.ema(stock_df['Close'], length=26)
                
                # RSI - Relative Strength Index
                stock_df['RSI_14'] = ta.rsi(stock_df['Close'], length=14)
                
                # MACD - Moving Average Convergence Divergence
                macd_result = ta.macd(stock_df['Close'], fast=12, slow=26, signal=9)
                if macd_result is not None:
                    stock_df['MACD'] = macd_result['MACD_12_26_9']
                    stock_df['MACD_signal'] = macd_result['MACDs_12_26_9']
                    stock_df['MA (including new technical indicators)
        feature_cols = [
            'Open', 'High', 'Low', 'Volume', 'Price_Change', 'Price_Range',
            'Returns', 'SMA_10', 'SMA_50', 'EMA_12', 'EMA_26', 'RSI_14',
            'MACD', 'MACD_signal', 'MACD_hist', 'Volatility_10', 'Volatility_20',
            'Volume_Change_Pct'Returns'].rolling(window=10).std()
                stock_df['Volatility_20'] = stock_df['Returns'].rolling(window=20).std()
                
                # Volume change percentage
                stock_df['Volume_Change_Pct'] = stock_df['Volume'].pct_change() * 100
                
                stock_groups.append(stock_df)
            
            # Combine all stocks back together
            df = pd.concat(stock_groups, ignore_index=True)
            print("✅ Technical indicators created with pandas-ta")
        else:
            # Fallback: Create basic indicators without pandas-ta
            print("⚠️  Using fallback indicators (pandas-ta not available)")
            df['SMA_10'] = df.groupby('Symbol')['Close'].transform(lambda x: x.rolling(10, min_periods=1).mean())
            df['SMA_50'] = df.groupby('Symbol')['Close'].transform(lambda x: x.rolling(50, min_periods=1).mean())
            df['Volatility_10'] = df.groupby('Symbol')['Returns'].transform(lambda x: x.rolling(10, min_periods=1).std())
            df['Volume_Change_Pct'] = df.groupby('Symbol')['Volume'].transform(lambda x: x.pct_change() * 100)
        
        # Lagged features
        df['Close_lag_1'] = df.groupby('Symbol')['Close'].shift(1)
        df['Volume_lag_1'] = df.groupby('Symbol')['Volume'].shift(1)
        
        # Time features
        df['Year'] = df['Date'].dt.year
        df['Month'] = df['Date'].dt.month
        df['DayOfWeek'] = df['Date'].dt.dayofweek
        df['Quarter'] = df['Date'].dt.quarter
        
        # Remove rows with NaN values created by technical indicators
        print("🧹 Removing NaN rows created by indicators...")
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
        """Prepare data for machine learning"""
        print("\n🎯 PREPARING ML DATA")
        print("-" * 40)
        
        # Select features for ML
        feature_cols = [
            'Open', 'High', 'Low', 'Volume', 'Price_Change', 'Price_Range',
            'Returns', 'SMA_5', 'SMA_10', 'Volatility', 'Close_lag_1', 'Volume_lag_1',
            'Year', 'Month', 'DayOfWeek', 'Quarter'
        ]
        
        # Ensure all features exist and remove NaN
        available_features = [col for col in feature_cols if col in self.data.columns]
        ml_data = self.data[available_features + ['Close']].dropna()
        
        # Features and target
        X = ml_data[available_features]
        y = ml_data['Close']
        
        # Train-test split
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        print(f"✅ Train set: {len(self.X_train):,} samples")
        print(f"✅ Test set: {len(self.X_test):,} samples")
        print(f"📊 Features: {len(available_features)}")
        
        self.feature_names = available_features
        return True
    
    def train_models(self):
        """Train ML models"""
        print("\n🤖 TRAINING MODELS")
        print("-" * 40)
        
        # Linear Regression
        print("📈 Training Linear Regression...")
        lr_model = LinearRegression()
        lr_model.fit(self.X_train, self.y_train)
        self.models['Linear Regression'] = lr_model
        print("✅ Linear Regression trained")
        
        # Random Forest
        print("🌲 Training Random Forest...")
        rf_model = RandomForestRegressor(
            n_estimators=20, max_depth=8, random_state=42, n_jobs=2
        )
        rf_model.fit(self.X_train, self.y_train)
        self.models['Random Forest'] = rf_model
        print("✅ Random Forest trained")
        
        print(f"✅ Trained {len(self.models)} models")
        
    def evaluate_models(self):
        """Evaluate model performance"""
        print("\n📊 EVALUATING MODELS")
        print("-" * 40)
        
        results = []
        best_r2 = -float('inf')
        best_model = None
        best_name = ""
        
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
            
            # Track best model
            if r2 > best_r2:
                best_r2 = r2
                best_model = model
                best_name = name
        
        # Save results
        os.makedirs('results', exist_ok=True)
        results_df = pd.DataFrame(results)
        results_df.to_csv('results/model_performance.csv', index=False)
        
        # Save best model
        os.makedirs('models', exist_ok=True)
        joblib.dump(best_model, 'models/best_model.joblib')
        
        print(f"\n🏆 BEST MODEL: {best_name} (R² = {best_r2:.4f})")
        print("💾 Results saved to results/model_performance.csv")
        print("💾 Best model saved to models/best_model.joblib")
        
        return results_df
    
    def run_complete_analysis(self):
        """Run the complete ML analysis"""
        print("🚀 STOCK MARKET ML ANALYSIS")
        print("=" * 50)
        
        # Execute pipeline
        if not self.load_data():
            return False
            
        if not self.create_features():
            return False
            
        if not self.prepare_ml_data():
            return False
            
        self.train_models()
        self.evaluate_models()
        
        print("\n✅ ANALYSIS COMPLETED SUCCESSFULLY!")
        print("📊 Check results/ folder for performance metrics")
        print("🤖 Check models/ folder for trained model")
        
        return True

if __name__ == "__main__":
    analyzer = StockMarketAnalyzer()
    analyzer.run_complete_analysis()