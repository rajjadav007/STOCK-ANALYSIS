#!/usr/bin/env python3
"""
Stock Market ML Pipeline - Data Preparation Module
===================================================

This module implements time-series data preparation with ZERO data leakage.
All operations respect temporal ordering and use only past information.

Key Features:
- Time-series train/validation/test split (70/15/15)
- Rolling window feature engineering
- Proper handling of missing values
- Multi-stock support with symbol tracking
- Data leakage prevention safeguards

Author: Stock Prediction System
Date: December 2025
"""

import pandas as pd
import numpy as np
import os
from pathlib import Path
from datetime import datetime
import warnings

warnings.filterwarnings('ignore')


class StockDataPipeline:
    """
    Complete data preparation pipeline for stock prediction system.
    
    This class ensures:
    1. NO DATA LEAKAGE - strict chronological ordering
    2. Proper time-series splitting
    3. Rolling window feature engineering
    4. Multi-stock data handling
    """
    
    def __init__(self, data_dir="data/raw", processed_dir="data/processed"):
        """
        Initialize the data pipeline.
        
        Args:
            data_dir (str): Directory containing raw CSV files
            processed_dir (str): Directory to save processed data
        """
        self.data_dir = Path(data_dir)
        self.processed_dir = Path(processed_dir)
        self.processed_dir.mkdir(parents=True, exist_ok=True)
        
        self.data = None
        self.train_data = None
        self.val_data = None
        self.test_data = None
        
        self.feature_columns = []
        
    def load_all_stocks(self, limit_stocks=None):
        """
        Load all stock CSV files and combine into single dataset.
        
        Args:
            limit_stocks (int): Optional limit on number of stocks to load
            
        Returns:
            pd.DataFrame: Combined stock data
        """
        print("📂 LOADING STOCK DATA")
        print("=" * 60)
        
        # Discover all CSV files
        csv_files = sorted(list(self.data_dir.glob("*.csv")))
        
        # Remove NIFTY50_all.csv if present (it's an index, not a stock)
        csv_files = [f for f in csv_files if 'NIFTY50' not in f.name]
        
        if limit_stocks:
            csv_files = csv_files[:limit_stocks]
            
        print(f"✅ Found {len(csv_files)} stock files")
        
        # Load and combine all stocks
        all_stocks = []
        
        for i, csv_file in enumerate(csv_files, 1):
            symbol = csv_file.stem
            
            try:
                # Load CSV
                df = pd.read_csv(csv_file)
                
                # Add symbol column
                df['Symbol'] = symbol
                
                # Parse date
                df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
                
                # Remove rows with invalid dates
                df = df.dropna(subset=['Date'])
                
                # Sort chronologically
                df = df.sort_values('Date').reset_index(drop=True)
                
                all_stocks.append(df)
                
                if i % 10 == 0:
                    print(f"   Loaded {i}/{len(csv_files)} stocks...")
                    
            except Exception as e:
                print(f"   ⚠️  Error loading {symbol}: {str(e)}")
                continue
        
        # Combine all stocks
        self.data = pd.concat(all_stocks, ignore_index=True)
        
        # Final sort by Symbol and Date
        self.data = self.data.sort_values(['Symbol', 'Date']).reset_index(drop=True)
        
        print(f"\n✅ Combined Dataset Created:")
        print(f"   Total records: {len(self.data):,}")
        print(f"   Stocks: {self.data['Symbol'].nunique()}")
        print(f"   Date range: {self.data['Date'].min().date()} to {self.data['Date'].max().date()}")
        print(f"   Days: {(self.data['Date'].max() - self.data['Date'].min()).days}")
        
        return self.data
    
    def handle_missing_values(self):
        """
        Handle missing values using forward fill (no future data usage).
        """
        print("\n🧹 HANDLING MISSING VALUES")
        print("-" * 60)
        
        missing_before = self.data.isnull().sum().sum()
        print(f"Missing values before: {missing_before:,}")
        
        # Columns that should be numeric
        numeric_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
        
        # Forward fill prices within each stock (uses only past data)
        for col in numeric_cols:
            if col in self.data.columns:
                self.data[col] = self.data.groupby('Symbol')[col].fillna(method='ffill')
                
                # Backward fill only at the beginning (unavoidable)
                self.data[col] = self.data.groupby('Symbol')[col].fillna(method='bfill')
        
        # Drop remaining NaN rows
        rows_before = len(self.data)
        self.data = self.data.dropna(subset=numeric_cols)
        rows_after = len(self.data)
        
        print(f"Missing values after: {self.data.isnull().sum().sum():,}")
        print(f"Rows removed: {rows_before - rows_after:,}")
        print(f"✅ Data retention: {rows_after/rows_before*100:.2f}%")
        
    def create_features(self):
        """
        Create comprehensive features using ONLY rolling windows (no data leakage).
        """
        print("\n⚙️ FEATURE ENGINEERING (Rolling Windows Only)")
        print("-" * 60)
        
        df = self.data.copy()
        
        # Process each stock separately to ensure correct calculations
        stock_groups = []
        
        for symbol in df['Symbol'].unique():
            stock_df = df[df['Symbol'] == symbol].copy()
            
            # ===== PRICE FEATURES =====
            stock_df['Price_Change'] = stock_df['Close'] - stock_df['Open']
            stock_df['Price_Range'] = stock_df['High'] - stock_df['Low']
            stock_df['Returns'] = stock_df['Close'].pct_change()
            
            # ===== TREND INDICATORS =====
            # Simple Moving Averages
            stock_df['SMA_5'] = stock_df['Close'].rolling(window=5, min_periods=1).mean()
            stock_df['SMA_10'] = stock_df['Close'].rolling(window=10, min_periods=1).mean()
            stock_df['SMA_20'] = stock_df['Close'].rolling(window=20, min_periods=1).mean()
            stock_df['SMA_50'] = stock_df['Close'].rolling(window=50, min_periods=1).mean()
            stock_df['SMA_200'] = stock_df['Close'].rolling(window=200, min_periods=1).mean()
            
            # Exponential Moving Averages
            stock_df['EMA_12'] = stock_df['Close'].ewm(span=12, adjust=False).mean()
            stock_df['EMA_26'] = stock_df['Close'].ewm(span=26, adjust=False).mean()
            
            # MACD
            stock_df['MACD'] = stock_df['EMA_12'] - stock_df['EMA_26']
            stock_df['MACD_signal'] = stock_df['MACD'].ewm(span=9, adjust=False).mean()
            stock_df['MACD_hist'] = stock_df['MACD'] - stock_df['MACD_signal']
            
            # ===== MOMENTUM INDICATORS =====
            # RSI (Relative Strength Index)
            delta = stock_df['Close'].diff()
            gain = delta.where(delta > 0, 0).rolling(window=14, min_periods=1).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14, min_periods=1).mean()
            rs = gain / loss.replace(0, 1e-10)  # Avoid division by zero
            stock_df['RSI_14'] = 100 - (100 / (1 + rs))
            
            # Rate of Change
            stock_df['ROC_10'] = stock_df['Close'].pct_change(periods=10) * 100
            
            # ===== VOLATILITY INDICATORS =====
            # Bollinger Bands
            stock_df['BB_middle'] = stock_df['Close'].rolling(window=20, min_periods=1).mean()
            bb_std = stock_df['Close'].rolling(window=20, min_periods=1).std()
            stock_df['BB_upper'] = stock_df['BB_middle'] + (2 * bb_std)
            stock_df['BB_lower'] = stock_df['BB_middle'] - (2 * bb_std)
            stock_df['BB_width'] = stock_df['BB_upper'] - stock_df['BB_lower']
            
            # Historical Volatility
            stock_df['Volatility_10'] = stock_df['Returns'].rolling(window=10, min_periods=1).std()
            stock_df['Volatility_20'] = stock_df['Returns'].rolling(window=20, min_periods=1).std()
            stock_df['Volatility_50'] = stock_df['Returns'].rolling(window=50, min_periods=1).std()
            
            # Average True Range (ATR)
            high_low = stock_df['High'] - stock_df['Low']
            high_close = np.abs(stock_df['High'] - stock_df['Close'].shift(1))
            low_close = np.abs(stock_df['Low'] - stock_df['Close'].shift(1))
            true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
            stock_df['ATR_14'] = true_range.rolling(window=14, min_periods=1).mean()
            
            # ===== VOLUME INDICATORS =====
            stock_df['Volume_SMA_20'] = stock_df['Volume'].rolling(window=20, min_periods=1).mean()
            stock_df['Volume_Change'] = stock_df['Volume'].pct_change()
            stock_df['Volume_Ratio'] = stock_df['Volume'] / stock_df['Volume_SMA_20'].replace(0, 1)
            
            # On-Balance Volume (OBV)
            obv = [0]
            for i in range(1, len(stock_df)):
                if stock_df['Close'].iloc[i] > stock_df['Close'].iloc[i-1]:
                    obv.append(obv[-1] + stock_df['Volume'].iloc[i])
                elif stock_df['Close'].iloc[i] < stock_df['Close'].iloc[i-1]:
                    obv.append(obv[-1] - stock_df['Volume'].iloc[i])
                else:
                    obv.append(obv[-1])
            stock_df['OBV'] = obv
            
            # ===== LAGGED FEATURES (Uses .shift() - no data leakage) =====
            for lag in [1, 3, 5, 10]:
                stock_df[f'Close_lag_{lag}'] = stock_df['Close'].shift(lag)
                stock_df[f'Volume_lag_{lag}'] = stock_df['Volume'].shift(lag)
                stock_df[f'Returns_lag_{lag}'] = stock_df['Returns'].shift(lag)
            
            # ===== TIME FEATURES =====
            stock_df['Year'] = stock_df['Date'].dt.year
            stock_df['Month'] = stock_df['Date'].dt.month
            stock_df['DayOfWeek'] = stock_df['Date'].dt.dayofweek
            stock_df['Quarter'] = stock_df['Date'].dt.quarter
            stock_df['DayOfMonth'] = stock_df['Date'].dt.day
            
            stock_groups.append(stock_df)
        
        # Combine all stocks
        self.data = pd.concat(stock_groups, ignore_index=True)
        
        print(f"✅ Features created: {self.data.shape[1]} total columns")
        print(f"   Records: {len(self.data):,}")
        
        # Remove NaN rows created by indicators
        rows_before = len(self.data)
        self.data = self.data.dropna()
        rows_after = len(self.data)
        
        print(f"   Rows after NaN removal: {rows_after:,}")
        print(f"   Retention: {rows_after/rows_before*100:.2f}%")
        
    def create_time_series_split(self, train_ratio=0.70, val_ratio=0.15, test_ratio=0.15):
        """
        Create time-series train/validation/test split.
        
        CRITICAL: This uses chronological ordering, no shuffling!
        
        Args:
            train_ratio (float): Proportion for training (default 70%)
            val_ratio (float): Proportion for validation (default 15%)
            test_ratio (float): Proportion for test (default 15%)
        """
        print("\n📦 TIME-SERIES DATA SPLIT (70/15/15)")
        print("-" * 60)
        print("⚠️  Using CHRONOLOGICAL split (no shuffling)")
        
        # Ensure data is sorted by date
        self.data = self.data.sort_values(['Symbol', 'Date']).reset_index(drop=True)
        
        # Calculate split indices for each stock
        all_train = []
        all_val = []
        all_test = []
        
        for symbol in self.data['Symbol'].unique():
            stock_df = self.data[self.data['Symbol'] == symbol].copy()
            
            n = len(stock_df)
            train_end = int(n * train_ratio)
            val_end = int(n * (train_ratio + val_ratio))
            
            train_df = stock_df.iloc[:train_end]
            val_df = stock_df.iloc[train_end:val_end]
            test_df = stock_df.iloc[val_end:]
            
            all_train.append(train_df)
            all_val.append(val_df)
            all_test.append(test_df)
        
        self.train_data = pd.concat(all_train, ignore_index=True)
        self.val_data = pd.concat(all_val, ignore_index=True)
        self.test_data = pd.concat(all_test, ignore_index=True)
        
        print(f"\n✅ SPLIT COMPLETE:")
        print(f"   Train:      {len(self.train_data):,} ({len(self.train_data)/len(self.data)*100:.1f}%)")
        print(f"   Validation: {len(self.val_data):,} ({len(self.val_data)/len(self.data)*100:.1f}%)")
        print(f"   Test:       {len(self.test_data):,} ({len(self.test_data)/len(self.data)*100:.1f}%)")
        
        print(f"\n📅 DATE RANGES:")
        print(f"   Train:      {self.train_data['Date'].min().date()} to {self.train_data['Date'].max().date()}")
        print(f"   Validation: {self.val_data['Date'].min().date()} to {self.val_data['Date'].max().date()}")
        print(f"   Test:       {self.test_data['Date'].min().date()} to {self.test_data['Date'].max().date()}")
        
        # Verify no temporal overlap
        train_max = self.train_data['Date'].max()
        val_min = self.val_data['Date'].min()
        val_max = self.val_data['Date'].max()
        test_min = self.test_data['Date'].min()
        
        if train_max < val_min and val_max < test_min:
            print(f"\n✅ NO TEMPORAL OVERLAP - Clean time-series split")
        else:
            print(f"\n⚠️  WARNING: Temporal overlap detected!")
            
    def save_processed_data(self):
        """Save processed datasets to disk."""
        print(f"\n💾 SAVING PROCESSED DATA")
        print("-" * 60)
        
        # Save full dataset
        full_path = self.processed_dir / 'full_dataset.csv'
        self.data.to_csv(full_path, index=False)
        print(f"✅ Full dataset: {full_path}")
        
        # Save splits
        train_path = self.processed_dir / 'train_data.csv'
        val_path = self.processed_dir / 'val_data.csv'
        test_path = self.processed_dir / 'test_data.csv'
        
        self.train_data.to_csv(train_path, index=False)
        self.val_data.to_csv(val_path, index=False)
        self.test_data.to_csv(test_path, index=False)
        
        print(f"✅ Train data: {train_path}")
        print(f"✅ Val data: {val_path}")
        print(f"✅ Test data: {test_path}")
        
    def run_pipeline(self, limit_stocks=None):
        """
        Run the complete data preparation pipeline.
        
        Args:
            limit_stocks (int): Optional limit on number of stocks
        """
        print("=" * 80)
        print("STOCK MARKET ML PIPELINE - DATA PREPARATION")
        print("=" * 80)
        
        # Step 1: Load data
        self.load_all_stocks(limit_stocks=limit_stocks)
        
        # Step 2: Handle missing values
        self.handle_missing_values()
        
        # Step 3: Create features
        self.create_features()
        
        # Step 4: Create time-series split
        self.create_time_series_split()
        
        # Step 5: Save processed data
        self.save_processed_data()
        
        print("\n" + "=" * 80)
        print("✅ DATA PREPARATION COMPLETE")
        print("=" * 80)
        print(f"📊 Summary:")
        print(f"   Total records: {len(self.data):,}")
        print(f"   Features: {self.data.shape[1]}")
        print(f"   Stocks: {self.data['Symbol'].nunique()}")
        print(f"   Train/Val/Test: {len(self.train_data)}/{len(self.val_data)}/{len(self.test_data)}")
        
        return self.data, self.train_data, self.val_data, self.test_data


def main():
    """Run the data preparation pipeline."""
    # Create pipeline
    pipeline = StockDataPipeline()
    
    # Run pipeline (limit to 10 stocks for faster testing)
    # Remove limit_stocks=10 to process all 52 stocks
    data, train, val, test = pipeline.run_pipeline(limit_stocks=10)
    
    print(f"\n🚀 READY FOR MODEL TRAINING!")
    print(f"   Next step: Create target variables and train models")


if __name__ == "__main__":
    main()
