#!/usr/bin/env python3
"""
Professional Time-Series Data Cleaning for Stock Market Analysis
================================================================

This module implements industry-standard data cleaning specifically designed for 
financial time-series data. Proper cleaning is CRITICAL before any ML operations.

WHY DATA CLEANING IS MANDATORY BEFORE ML:
==========================================
1. Missing Values: Can break ML algorithms (NaN propagation)
2. Duplicates: Create data leakage and overfitting
3. Inconsistent Types: Cause calculation errors
4. Outliers: Skew model parameters and predictions
5. Date Issues: Break time-series sequencing

WHAT HAPPENS IF YOU SKIP THIS STEP:
===================================
❌ Model Training Failures: NaN values crash most ML algorithms
❌ Incorrect Predictions: Garbage in = garbage out
❌ Data Leakage: Duplicate rows cause overfitting
❌ Performance Issues: Inconsistent data types slow processing
❌ Time-Series Errors: Wrong chronological order breaks patterns

PANDAS vs NUMPY OPERATIONS:
===========================
- PANDAS: Data manipulation, missing value handling, date operations
- NUMPY: Mathematical operations, array computations, statistical functions
- Why PANDAS: Better for structured data with mixed types and labels
- Why NUMPY: Faster for pure numerical computations

Author: Stock Market ML Analysis Project
Date: December 2025
"""

import pandas as pd
import numpy as np
import warnings
from pathlib import Path
from datetime import datetime, timedelta

# Import our data loader
from data_loader import StockDataLoader

warnings.filterwarnings('ignore')

class StockDataCleaner:
    """
    Enterprise-Grade Data Cleaning System for Financial Time Series
    
    This class implements systematic data cleaning that transforms raw stock data
    into ML-ready format while preserving financial data integrity.
    """
    
    def __init__(self):
        self.cleaning_stats = {}
        self.original_shape = None
        self.final_shape = None
        
    def validate_required_columns(self, df, symbol):
        """
        Validate that essential columns exist for stock data
        
        Args:
            df (pd.DataFrame): Stock data to validate
            symbol (str): Stock symbol for reporting
            
        Returns:
            bool: True if all required columns present
            
        WHY THIS MATTERS:
        - Stock ML models need specific price columns (OHLC)
        - Missing essential columns means incomplete analysis
        - Early detection prevents errors downstream
        
        PANDAS OPERATION: df.columns (pandas attribute for column names)
        """
        print(f"\n🔍 VALIDATING REQUIRED COLUMNS: {symbol}")
        print("-" * 40)
        
        # Essential columns for stock analysis
        required_columns = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']
        missing_columns = []
        
        for col in required_columns:
            if col not in df.columns:
                missing_columns.append(col)
                
        if missing_columns:
            print(f"❌ Missing essential columns: {missing_columns}")
            print(f"📋 Available columns: {list(df.columns)}")
            return False
        else:
            print(f"✅ All essential columns present: {required_columns}")
            return True
    
    def handle_missing_values_strategically(self, df, symbol):
        """
        Handle missing values using financial data best practices
        
        Args:
            df (pd.DataFrame): Stock data with potential missing values
            symbol (str): Stock symbol for reporting
            
        Returns:
            pd.DataFrame: Data with missing values handled
            
        WHY STRATEGIC HANDLING MATTERS:
        - Different columns require different approaches
        - Price data: Cannot be interpolated carelessly (affects returns)
        - Volume data: Can use forward fill or median
        - Meta data: Can often be dropped or filled with constants
        
        PANDAS vs NUMPY:
        - PANDAS: .fillna(), .interpolate(), .dropna() (handles mixed types)
        - NUMPY: np.nan, np.isnan() (pure numerical operations)
        """
        print(f"\n🧹 STRATEGIC MISSING VALUE HANDLING: {symbol}")
        print("-" * 50)
        
        print(f"📊 Initial Missing Value Analysis:")
        missing_before = df.isnull().sum()
        total_missing = missing_before.sum()
        print(f"   Total missing values: {total_missing:,}")
        
        if total_missing == 0:
            print("   ✅ No missing values detected!")
            return df
        
        # Create a copy to avoid modifying original
        # PANDAS OPERATION: .copy() ensures we don't modify the original DataFrame
        df_cleaned = df.copy()
        
        # Strategy 1: Handle price columns (CRITICAL for ML)
        price_columns = ['Open', 'High', 'Low', 'Close']
        
        for col in price_columns:
            if col in df_cleaned.columns:
                missing_count = df_cleaned[col].isnull().sum()
                if missing_count > 0:
                    print(f"\n   🎯 Handling {col} ({missing_count} missing):")
                    
                    # For price data, forward fill is usually best
                    # PANDAS OPERATION: .fillna(method='ffill') - forward fill missing values
                    df_cleaned[col] = df_cleaned[col].fillna(method='ffill')
                    
                    # If still missing (at the beginning), backward fill
                    # PANDAS OPERATION: .fillna(method='bfill') - backward fill remaining NaN
                    df_cleaned[col] = df_cleaned[col].fillna(method='bfill')
                    
                    remaining = df_cleaned[col].isnull().sum()
                    print(f"      ✅ Forward/backward filled: {missing_count - remaining} values")
                    
                    if remaining > 0:
                        print(f"      ⚠️  Still missing: {remaining} values (will be dropped)")
        
        # Strategy 2: Handle Volume (can use median for missing values)
        if 'Volume' in df_cleaned.columns:
            volume_missing = df_cleaned['Volume'].isnull().sum()
            if volume_missing > 0:
                print(f"\n   📈 Handling Volume ({volume_missing} missing):")
                
                # NUMPY OPERATION: np.median() for statistical calculation
                # PANDAS OPERATION: .fillna() for filling missing values
                volume_median = np.median(df_cleaned['Volume'].dropna())
                df_cleaned['Volume'] = df_cleaned['Volume'].fillna(volume_median)
                
                print(f"      ✅ Filled with median volume: {volume_median:,.0f}")
        
        # Strategy 3: Handle non-critical columns
        non_critical_cols = ['Trades', 'Deliverable Volume', '%Deliverble', 'Turnover']
        
        for col in non_critical_cols:
            if col in df_cleaned.columns:
                missing_count = df_cleaned[col].isnull().sum()
                if missing_count > 0:
                    missing_pct = (missing_count / len(df_cleaned)) * 100
                    
                    if missing_pct > 50:
                        # Drop columns with >50% missing data
                        # PANDAS OPERATION: .drop() removes columns
                        df_cleaned = df_cleaned.drop(columns=[col])
                        print(f"   🗑️  Dropped {col} ({missing_pct:.1f}% missing)")
                    else:
                        # Forward fill for moderate missing data
                        # PANDAS OPERATION: .fillna(method='ffill')
                        df_cleaned[col] = df_cleaned[col].fillna(method='ffill')
                        remaining = df_cleaned[col].isnull().sum()
                        print(f"   📝 Forward filled {col}: {missing_count - remaining} values")
        
        # Final check and cleanup
        remaining_missing = df_cleaned.isnull().sum().sum()
        
        if remaining_missing > 0:
            print(f"\n   🧽 Final cleanup: Dropping {remaining_missing} rows with remaining NaN")
            # PANDAS OPERATION: .dropna() removes rows with any NaN values
            df_cleaned = df_cleaned.dropna()
        
        print(f"\n✅ Missing value handling complete:")
        print(f"   Before: {total_missing:,} missing values")
        print(f"   After: {df_cleaned.isnull().sum().sum():,} missing values")
        print(f"   Rows kept: {len(df_cleaned):,}/{len(df):,} ({len(df_cleaned)/len(df)*100:.1f}%)")
        
        return df_cleaned
    
    def remove_duplicate_rows(self, df, symbol):
        """
        Remove duplicate rows while preserving data integrity
        
        Args:
            df (pd.DataFrame): Stock data with potential duplicates
            symbol (str): Stock symbol for reporting
            
        Returns:
            pd.DataFrame: Data with duplicates removed
            
        WHY DUPLICATE REMOVAL IS CRITICAL:
        - Duplicates create data leakage in train/test splits
        - Cause overfitting (model sees same data multiple times)
        - Skew statistical measures and technical indicators
        - Violate time-series assumptions (unique timestamps)
        
        PANDAS OPERATION: .duplicated(), .drop_duplicates()
        - Much more efficient than manual numpy loops
        - Handles mixed data types automatically
        """
        print(f"\n🔍 DUPLICATE DETECTION & REMOVAL: {symbol}")
        print("-" * 50)
        
        # Check for exact duplicates (all columns identical)
        # PANDAS OPERATION: .duplicated() returns boolean mask of duplicate rows
        exact_duplicates = df.duplicated()
        exact_count = exact_duplicates.sum()
        
        print(f"📊 Duplicate Analysis:")
        print(f"   Exact duplicates (all columns): {exact_count}")
        
        if exact_count > 0:
            print(f"   📋 Sample duplicate rows:")
            # Show some duplicate entries
            duplicate_rows = df[exact_duplicates].head(3)
            for _, row in duplicate_rows.iterrows():
                print(f"      Date: {row['Date']}, Close: ₹{row['Close']:.2f}")
            
            # Remove exact duplicates
            # PANDAS OPERATION: .drop_duplicates() removes duplicate rows
            df_no_duplicates = df.drop_duplicates()
            print(f"   ✅ Removed {exact_count} exact duplicate rows")
        else:
            df_no_duplicates = df.copy()
            print(f"   ✅ No exact duplicates found")
        
        # Check for date-based duplicates (multiple entries for same date)
        if 'Date' in df.columns:
            # PANDAS OPERATION: .duplicated(subset=['Date']) checks specific column duplicates
            date_duplicates = df_no_duplicates.duplicated(subset=['Date'])
            date_dup_count = date_duplicates.sum()
            
            print(f"   Date duplicates (same trading day): {date_dup_count}")
            
            if date_dup_count > 0:
                print(f"   📅 Sample duplicate dates:")
                dup_dates = df_no_duplicates[date_duplicates]['Date'].head(3)
                for date in dup_dates:
                    print(f"      {date.strftime('%Y-%m-%d')}")
                
                # For date duplicates, keep the last entry (most recent data)
                # PANDAS OPERATION: .drop_duplicates(subset=['Date'], keep='last')
                df_no_duplicates = df_no_duplicates.drop_duplicates(subset=['Date'], keep='last')
                print(f"   ✅ Removed {date_dup_count} date duplicates (kept latest)")
        
        total_removed = len(df) - len(df_no_duplicates)
        
        if total_removed > 0:
            print(f"\n🎯 Duplicate Removal Summary:")
            print(f"   Original rows: {len(df):,}")
            print(f"   Final rows: {len(df_no_duplicates):,}")
            print(f"   Removed: {total_removed:,} ({total_removed/len(df)*100:.2f}%)")
        else:
            print(f"\n✅ No duplicates found - data integrity confirmed")
        
        return df_no_duplicates
    
    def validate_data_types_and_ranges(self, df, symbol):
        """
        Validate and fix data types and ranges for financial data
        
        Args:
            df (pd.DataFrame): Stock data to validate
            symbol (str): Stock symbol for reporting
            
        Returns:
            pd.DataFrame: Data with corrected types and validated ranges
            
        WHY DATA TYPE VALIDATION MATTERS:
        - Wrong types cause calculation errors in ML algorithms
        - Mixed types slow down pandas operations significantly
        - Incorrect ranges indicate data corruption
        - Financial data has logical constraints (prices > 0, etc.)
        
        PANDAS vs NUMPY:
        - PANDAS: .astype(), pd.to_datetime() for type conversion
        - NUMPY: dtype validation, range checking with np.where()
        """
        print(f"\n🎯 DATA TYPE & RANGE VALIDATION: {symbol}")
        print("-" * 50)
        
        df_validated = df.copy()
        
        # Validate Date column
        if 'Date' in df_validated.columns:
            # PANDAS OPERATION: pd.to_datetime() converts to proper datetime type
            if df_validated['Date'].dtype != 'datetime64[ns]':
                print("   📅 Converting Date column to datetime...")
                df_validated['Date'] = pd.to_datetime(df_validated['Date'])
                print("   ✅ Date column converted to datetime64[ns]")
            else:
                print("   ✅ Date column already in correct format")
        
        # Validate numeric columns
        numeric_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
        
        for col in numeric_columns:
            if col in df_validated.columns:
                print(f"\n   🔢 Validating {col}:")
                
                # Check data type
                current_dtype = df_validated[col].dtype
                print(f"      Current type: {current_dtype}")
                
                # Convert to numeric if needed
                # PANDAS OPERATION: pd.to_numeric() converts to numeric type safely
                if not pd.api.types.is_numeric_dtype(df_validated[col]):
                    df_validated[col] = pd.to_numeric(df_validated[col], errors='coerce')
                    print(f"      ✅ Converted to numeric type")
                
                # Validate ranges (financial logic)
                if col in ['Open', 'High', 'Low', 'Close']:
                    # Prices should be positive
                    # NUMPY OPERATION: comparison operations return boolean arrays
                    negative_prices = (df_validated[col] <= 0).sum()
                    if negative_prices > 0:
                        print(f"      ⚠️  Found {negative_prices} non-positive prices")
                        # Replace with NaN for later handling
                        # PANDAS + NUMPY: .loc[] with numpy boolean indexing
                        df_validated.loc[df_validated[col] <= 0, col] = np.nan
                        print(f"      ✅ Marked non-positive prices as NaN")
                    
                    # Check for extremely high prices (likely data errors)
                    # NUMPY OPERATION: np.percentile() for statistical thresholds
                    price_99th = np.percentile(df_validated[col].dropna(), 99)
                    extreme_threshold = price_99th * 10  # 10x the 99th percentile
                    
                    extreme_prices = (df_validated[col] > extreme_threshold).sum()
                    if extreme_prices > 0:
                        print(f"      ⚠️  Found {extreme_prices} extremely high prices (>{extreme_threshold:.0f})")
                        print(f"      💡 These might be data errors - consider investigation")
                
                elif col == 'Volume':
                    # Volume should be non-negative
                    negative_volume = (df_validated[col] < 0).sum()
                    if negative_volume > 0:
                        print(f"      ⚠️  Found {negative_volume} negative volume values")
                        df_validated.loc[df_validated[col] < 0, col] = np.nan
                        print(f"      ✅ Marked negative volumes as NaN")
                
                # Report final stats
                valid_values = df_validated[col].notna().sum()
                print(f"      📊 Valid values: {valid_values:,}/{len(df_validated):,}")
        
        # Validate price relationships (High >= Low, etc.)
        print(f"\n   🔍 Validating Price Relationships:")
        
        if all(col in df_validated.columns for col in ['Open', 'High', 'Low', 'Close']):
            # PANDAS OPERATION: Boolean indexing with multiple conditions
            invalid_high_low = (df_validated['High'] < df_validated['Low']).sum()
            invalid_ranges = (
                (df_validated['Open'] > df_validated['High']) |
                (df_validated['Open'] < df_validated['Low']) |
                (df_validated['Close'] > df_validated['High']) |
                (df_validated['Close'] < df_validated['Low'])
            ).sum()
            
            print(f"      High < Low violations: {invalid_high_low}")
            print(f"      Open/Close outside High-Low range: {invalid_ranges}")
            
            if invalid_high_low == 0 and invalid_ranges == 0:
                print(f"      ✅ All price relationships are logically consistent")
            else:
                print(f"      ⚠️  Found price relationship violations - investigate data source")
        
        return df_validated
    
    def ensure_chronological_order(self, df, symbol):
        """
        Ensure data is in proper chronological order for time-series analysis
        
        Args:
            df (pd.DataFrame): Stock data to sort
            symbol (str): Stock symbol for reporting
            
        Returns:
            pd.DataFrame: Chronologically sorted data
            
        WHY CHRONOLOGICAL ORDER IS CRITICAL:
        - Time-series ML models assume sequential order
        - Technical indicators require proper sequence
        - Backtesting needs historical progression
        - Feature engineering depends on correct time flow
        
        PANDAS OPERATION: .sort_values() for sorting by column
        """
        print(f"\n📅 ENSURING CHRONOLOGICAL ORDER: {symbol}")
        print("-" * 50)
        
        if 'Date' not in df.columns:
            print("   ❌ No Date column found - cannot sort chronologically")
            return df
        
        # Check if already sorted
        # PANDAS OPERATION: .is_monotonic_increasing checks if values are in ascending order
        is_sorted = df['Date'].is_monotonic_increasing
        
        if is_sorted:
            print("   ✅ Data already in chronological order")
            return df.reset_index(drop=True)
        else:
            print("   🔄 Sorting data chronologically...")
            
            # Sort by date
            # PANDAS OPERATION: .sort_values() sorts DataFrame by specified column
            df_sorted = df.sort_values('Date').reset_index(drop=True)
            
            print(f"   ✅ Data sorted by date:")
            print(f"      Start: {df_sorted['Date'].iloc[0].strftime('%Y-%m-%d')}")
            print(f"      End: {df_sorted['Date'].iloc[-1].strftime('%Y-%m-%d')}")
            print(f"      Total days: {len(df_sorted):,}")
            
            return df_sorted
    
    def generate_cleaning_summary(self, original_df, cleaned_df, symbol):
        """
        Generate comprehensive cleaning summary with before/after statistics
        
        Args:
            original_df (pd.DataFrame): Original data before cleaning
            cleaned_df (pd.DataFrame): Data after cleaning
            symbol (str): Stock symbol for reporting
        """
        print(f"\n📋 COMPREHENSIVE CLEANING SUMMARY: {symbol}")
        print("=" * 60)
        
        # Size comparison
        print(f"🔢 DATA SIZE CHANGES:")
        print(f"   Original: {len(original_df):,} rows × {len(original_df.columns)} columns")
        print(f"   Cleaned: {len(cleaned_df):,} rows × {len(cleaned_df.columns)} columns")
        
        rows_removed = len(original_df) - len(cleaned_df)
        cols_removed = len(original_df.columns) - len(cleaned_df.columns)
        
        print(f"   Rows removed: {rows_removed:,} ({rows_removed/len(original_df)*100:.2f}%)")
        print(f"   Columns removed: {cols_removed}")
        
        # Missing values comparison
        print(f"\n❓ MISSING VALUES:")
        original_missing = original_df.isnull().sum().sum()
        cleaned_missing = cleaned_df.isnull().sum().sum()
        
        print(f"   Before cleaning: {original_missing:,}")
        print(f"   After cleaning: {cleaned_missing:,}")
        print(f"   Reduction: {original_missing - cleaned_missing:,}")
        
        # Data quality metrics
        print(f"\n🎯 DATA QUALITY METRICS:")
        
        # Completeness
        completeness = ((len(cleaned_df) * len(cleaned_df.columns) - cleaned_missing) / 
                       (len(cleaned_df) * len(cleaned_df.columns))) * 100
        print(f"   Completeness: {completeness:.2f}%")
        
        # Date range
        if 'Date' in cleaned_df.columns:
            date_range_days = (cleaned_df['Date'].max() - cleaned_df['Date'].min()).days
            print(f"   Date coverage: {date_range_days:,} calendar days")
            print(f"   Trading days: {len(cleaned_df):,}")
        
        # Price data integrity
        if 'Close' in cleaned_df.columns:
            price_range = cleaned_df['Close'].max() - cleaned_df['Close'].min()
            print(f"   Price range: ₹{cleaned_df['Close'].min():.2f} - ₹{cleaned_df['Close'].max():.2f}")
        
        print(f"\n✅ CLEANING STATUS: COMPLETE")
        print(f"📝 Next step: Feature Engineering")
    
    def clean_stock_data(self, symbol="RELIANCE"):
        """
        Execute complete data cleaning pipeline
        
        Args:
            symbol (str): Stock symbol to clean
            
        Returns:
            pd.DataFrame: Cleaned data ready for feature engineering
        """
        print("🧹 COMPREHENSIVE STOCK DATA CLEANING PIPELINE")
        print("=" * 80)
        print("This pipeline ensures your data is ML-ready through systematic cleaning")
        print("=" * 80)
        
        # Load the data
        loader = StockDataLoader()
        original_df = loader.load_sample_stock(symbol)
        
        if original_df is None:
            print(f"❌ Failed to load data for {symbol}")
            return None
        
        # Store original shape for comparison
        self.original_shape = original_df.shape
        
        # Step 1: Validate required columns
        if not self.validate_required_columns(original_df, symbol):
            print("❌ Cannot proceed - missing essential columns")
            return None
        
        # Step 2: Handle missing values strategically
        df_step1 = self.handle_missing_values_strategically(original_df, symbol)
        
        # Step 3: Remove duplicates
        df_step2 = self.remove_duplicate_rows(df_step1, symbol)
        
        # Step 4: Validate data types and ranges
        df_step3 = self.validate_data_types_and_ranges(df_step2, symbol)
        
        # Step 5: Ensure chronological order
        df_cleaned = self.ensure_chronological_order(df_step3, symbol)
        
        # Store final shape
        self.final_shape = df_cleaned.shape
        
        # Generate summary
        self.generate_cleaning_summary(original_df, df_cleaned, symbol)
        
        # Save cleaned data
        output_path = f'data/processed/cleaned_{symbol.lower()}_data.csv'
        df_cleaned.to_csv(output_path, index=False)
        print(f"\n💾 Cleaned data saved to: {output_path}")
        
        return df_cleaned


def main():
    """
    Demonstrate the complete data cleaning pipeline
    """
    print("🏗️  STOCK DATA CLEANING DEMONSTRATION")
    print("=" * 80)
    print("WHY THIS MATTERS: Clean data = Reliable ML models")
    print("WHAT WE'LL DO: Transform raw data into ML-ready format")
    print("=" * 80)
    
    # Create cleaner instance
    cleaner = StockDataCleaner()
    
    # Clean the sample stock data
    cleaned_data = cleaner.clean_stock_data("RELIANCE")
    
    if cleaned_data is not None:
        print(f"\n" + "="*80)
        print("🎉 SUCCESS! DATA CLEANING COMPLETE")
        print("="*80)
        print("✅ Missing values handled strategically")
        print("✅ Duplicates removed")
        print("✅ Data types validated")
        print("✅ Chronological order ensured")
        print("✅ Ready for feature engineering!")
        
        print(f"\n📊 FINAL DATASET PREVIEW:")
        print(cleaned_data.head())
        
        print(f"\n🚀 NEXT STEPS:")
        print(f"   1. ✅ Data loading complete")
        print(f"   2. ✅ Data cleaning complete")
        print(f"   3. ➡️  Feature engineering (technical indicators)")
        print(f"   4. ➡️  Train/test split")
        print(f"   5. ➡️  Model training")
        
        return cleaned_data
    else:
        print("❌ Data cleaning failed. Please check your data source.")
        return None

if __name__ == "__main__":
    cleaned_data = main()