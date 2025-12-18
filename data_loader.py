#!/usr/bin/env python3
"""
Professional Stock Market Data Loading Pipeline
===============================================

This module handles the initial data loading and exploration phase for stock market analysis.
It focuses on understanding the data structure before any machine learning operations.

Key Requirements for Financial Time Series:
1. Proper date parsing and sorting (chronological order is critical)
2. Data quality assessment (missing values can skew financial models)
3. Understanding data structure (columns, types, ranges)
4. Validation of financial data integrity

Author: Stock Market ML Analysis Project
Date: December 2025
"""

import pandas as pd
import numpy as np
import os
import glob
from pathlib import Path
import warnings

# Suppress warnings for cleaner output during data exploration
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)

class StockDataLoader:
    """
    Professional Stock Market Data Loading System
    
    This class handles the critical first step of any ML pipeline - understanding your data.
    For financial time series, proper data loading is crucial because:
    - Stock prices are time-dependent (order matters)
    - Missing values can indicate market holidays or data issues
    - Data types must be correct for calculations
    """
    
    def __init__(self, data_directory="data/raw"):
        """
        Initialize the data loader
        
        Args:
            data_directory (str): Path to directory containing CSV files
        
        Why we need this:
        - Centralized data path management
        - Easy switching between different data sources
        - Consistent file handling across the project
        """
        self.data_dir = Path(data_directory)
        self.loaded_data = {}
        self.data_summary = {}
        
    def discover_csv_files(self):
        """
        Discover all CSV files in the data directory
        
        Returns:
            list: List of CSV file paths
            
        Why this step is important:
        - Dynamic file discovery (works with any number of stocks)
        - Error handling for missing directories
        - Scalable approach for large datasets
        """
        print("📁 DISCOVERING DATA FILES")
        print("=" * 50)
        
        if not self.data_dir.exists():
            raise FileNotFoundError(f"Data directory not found: {self.data_dir}")
            
        csv_files = list(self.data_dir.glob("*.csv"))
        
        print(f"✅ Found {len(csv_files)} CSV files")
        print(f"📂 Directory: {self.data_dir.absolute()}")
        
        # Show first few files as preview
        for i, file in enumerate(csv_files[:5]):
            print(f"   📄 {file.name}")
        
        if len(csv_files) > 5:
            print(f"   ... and {len(csv_files) - 5} more files")
            
        return csv_files
    
    def load_single_stock(self, file_path, symbol_name=None):
        """
        Load a single stock CSV file with proper data types and validation
        
        Args:
            file_path (Path): Path to the CSV file
            symbol_name (str): Optional symbol name (extracted from filename if not provided)
            
        Returns:
            pd.DataFrame: Loaded and validated stock data
            
        Why each parameter matters:
        - file_path: Obviously needed to load the file
        - symbol_name: For tracking which stock the data belongs to
        - Return DataFrame: Standard pandas format for further analysis
        """
        if symbol_name is None:
            symbol_name = file_path.stem  # Extract filename without extension
            
        print(f"\n📊 LOADING: {symbol_name}")
        print("-" * 30)
        
        try:
            # STEP 1: Load CSV with pandas
            # Why pandas? 
            # - Industry standard for data manipulation
            # - Built-in CSV parsing with date handling
            # - Memory efficient for large datasets
            # - Extensive data cleaning capabilities
            df = pd.read_csv(file_path)
            print(f"✅ Raw data loaded: {len(df)} rows, {len(df.columns)} columns")
            
            # STEP 2: Parse Date column correctly
            # Why this is CRITICAL for time series:
            # - Stock prices are time-dependent
            # - Sorting by date ensures chronological order
            # - Enables time-based calculations (returns, moving averages)
            # - Required for train/test splits in time series ML
            if 'Date' in df.columns:
                df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
                print(f"✅ Date column parsed: {df['Date'].dtype}")
                
                # Check for invalid dates
                invalid_dates = df['Date'].isna().sum()
                if invalid_dates > 0:
                    print(f"⚠️  Found {invalid_dates} invalid dates - will be removed")
                    df = df.dropna(subset=['Date'])
            else:
                print("❌ No 'Date' column found!")
                return None
                
            # STEP 3: Sort by date (ESSENTIAL for time series)
            # Why sorting matters:
            # - ML models expect chronological order
            # - Technical indicators (SMA, EMA) require proper sequence
            # - Prevents look-ahead bias in backtesting
            # - Ensures consistent data ordering across all stocks
            df = df.sort_values('Date').reset_index(drop=True)
            print(f"✅ Data sorted chronologically")
            print(f"📅 Date range: {df['Date'].min().date()} to {df['Date'].max().date()}")
            
            # STEP 4: Add symbol column for multi-stock analysis
            # Why add symbol:
            # - Enables combining multiple stocks in one dataset
            # - Required for portfolio analysis
            # - Helps track data source in merged datasets
            if 'Symbol' not in df.columns:
                df['Symbol'] = symbol_name
                print(f"✅ Added Symbol column: {symbol_name}")
            
            return df
            
        except Exception as e:
            print(f"❌ Error loading {file_path}: {str(e)}")
            return None
    
    def analyze_data_quality(self, df, symbol_name):
        """
        Perform comprehensive data quality analysis
        
        Args:
            df (pd.DataFrame): Stock data to analyze
            symbol_name (str): Stock symbol for reporting
            
        Why data quality analysis is essential:
        - Missing values can break ML models
        - Outliers can skew predictions
        - Data types affect calculation accuracy
        - Understanding ranges helps with feature engineering
        """
        print(f"\n🔍 DATA QUALITY ANALYSIS: {symbol_name}")
        print("=" * 40)
        
        # Basic information
        print("📋 BASIC INFORMATION:")
        print(f"   Rows: {len(df):,}")
        print(f"   Columns: {len(df.columns)}")
        print(f"   Memory usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
        
        # Column information
        print("\n📊 COLUMNS:")
        for col in df.columns:
            dtype = df[col].dtype
            non_null = df[col].count()
            null_pct = (len(df) - non_null) / len(df) * 100
            print(f"   {col:15} | {str(dtype):10} | Non-null: {non_null:,} ({100-null_pct:.1f}%)")
        
        # Missing values analysis
        print("\n❓ MISSING VALUES:")
        missing = df.isnull().sum()
        total_missing = missing.sum()
        
        if total_missing > 0:
            print(f"   Total missing values: {total_missing:,}")
            for col in missing[missing > 0].index:
                pct = (missing[col] / len(df)) * 100
                print(f"   {col}: {missing[col]:,} ({pct:.2f}%)")
        else:
            print("   ✅ No missing values found!")
        
        # Numerical columns analysis
        numerical_cols = df.select_dtypes(include=[np.number]).columns
        if len(numerical_cols) > 0:
            print(f"\n📈 NUMERICAL DATA SUMMARY:")
            print(df[numerical_cols].describe())
        
        # Date range analysis
        if 'Date' in df.columns:
            print(f"\n📅 DATE ANALYSIS:")
            print(f"   Start date: {df['Date'].min()}")
            print(f"   End date: {df['Date'].max()}")
            print(f"   Trading days: {len(df):,}")
            
            # Check for data gaps (weekends/holidays are normal)
            date_diff = (df['Date'].max() - df['Date'].min()).days
            expected_trading_days = date_diff * 5/7  # Rough estimate excluding weekends
            coverage = len(df) / expected_trading_days * 100
            print(f"   Estimated coverage: {coverage:.1f}% (excluding weekends/holidays)")
        
        return {
            'rows': len(df),
            'columns': len(df.columns),
            'missing_values': total_missing,
            'memory_mb': df.memory_usage(deep=True).sum() / 1024**2,
            'date_range_days': (df['Date'].max() - df['Date'].min()).days if 'Date' in df.columns else 0
        }
    
    def load_sample_stock(self, stock_symbol="RELIANCE"):
        """
        Load and analyze a single stock for demonstration
        
        Args:
            stock_symbol (str): Stock symbol to load
            
        Why start with one stock:
        - Easier to understand data structure
        - Faster processing for initial exploration
        - Can identify issues before processing all files
        - Good for prototyping data pipeline
        """
        print("🚀 LOADING SAMPLE STOCK FOR ANALYSIS")
        print("=" * 50)
        
        # Find the stock file
        stock_file = self.data_dir / f"{stock_symbol}.csv"
        
        if not stock_file.exists():
            print(f"❌ File not found: {stock_file}")
            available_files = [f.stem for f in self.data_dir.glob("*.csv")][:10]
            print(f"Available stocks: {', '.join(available_files)}")
            return None
        
        # Load the stock data
        df = self.load_single_stock(stock_file, stock_symbol)
        
        if df is not None:
            # Analyze data quality
            self.data_summary[stock_symbol] = self.analyze_data_quality(df, stock_symbol)
            
            # Show first few rows
            print(f"\n👀 FIRST 5 ROWS:")
            print(df.head())
            
            # Store for further analysis
            self.loaded_data[stock_symbol] = df
            
            return df
        
        return None
    
    def display_summary_statistics(self):
        """
        Display summary statistics for loaded data
        
        Why summary statistics matter:
        - Quick overview of data distribution
        - Identify potential outliers
        - Understand price ranges for normalization
        - Validate data makes financial sense
        """
        if not self.loaded_data:
            print("❌ No data loaded yet. Run load_sample_stock() first.")
            return
            
        print("\n📊 SUMMARY STATISTICS")
        print("=" * 50)
        
        for symbol, df in self.loaded_data.items():
            print(f"\n💼 {symbol}:")
            
            # Price statistics
            if 'Close' in df.columns:
                close_price = df['Close']
                print(f"   Close Price - Min: ₹{close_price.min():.2f}, Max: ₹{close_price.max():.2f}")
                print(f"   Close Price - Mean: ₹{close_price.mean():.2f}, Std: ₹{close_price.std():.2f}")
            
            # Volume statistics
            if 'Volume' in df.columns:
                volume = df['Volume']
                print(f"   Volume - Mean: {volume.mean():,.0f}, Max: {volume.max():,.0f}")
            
            # Data completeness
            completeness = (1 - df.isnull().sum().sum() / (len(df) * len(df.columns))) * 100
            print(f"   Data Completeness: {completeness:.2f}%")


def main():
    """
    Main function to demonstrate the data loading pipeline
    
    This function shows the complete workflow:
    1. Initialize data loader
    2. Discover available files
    3. Load sample data
    4. Analyze data quality
    5. Display summary statistics
    
    Why this workflow:
    - Step-by-step validation of each stage
    - Easy to identify where issues occur
    - Modular approach for debugging
    - Clear progress indicators for user
    """
    print("🏗️  STOCK MARKET DATA LOADING PIPELINE")
    print("=" * 60)
    print("Phase 1: Data Discovery and Loading")
    print("This phase focuses on understanding your data BEFORE any ML operations")
    print("=" * 60)
    
    # Initialize the data loader
    loader = StockDataLoader()
    
    # Discover available files
    try:
        csv_files = loader.discover_csv_files()
    except FileNotFoundError as e:
        print(f"❌ {e}")
        return
    
    # Load sample stock for analysis
    sample_df = loader.load_sample_stock("RELIANCE")
    
    if sample_df is not None:
        # Display summary statistics
        loader.display_summary_statistics()
        
        print(f"\n🎉 SUCCESS! Data loading pipeline completed.")
        print(f"📝 Next steps:")
        print(f"   1. Review the data quality metrics above")
        print(f"   2. Check for any missing values or outliers")
        print(f"   3. Verify date ranges make sense")
        print(f"   4. Proceed to feature engineering phase")
        
        return loader, sample_df
    else:
        print("❌ Failed to load sample data. Please check your data files.")
        return None, None


if __name__ == "__main__":
    # Run the data loading pipeline
    loader, data = main()