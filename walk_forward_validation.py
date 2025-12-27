#!/usr/bin/env python3
"""
Walk-Forward Time-Series Validation
====================================
Chronological train/validation/test split for time-series data.
"""

import pandas as pd
import numpy as np
from datetime import datetime


class WalkForwardValidator:
    """Time-series cross-validation with strict chronological ordering."""
    
    def __init__(self, train_ratio=0.70, val_ratio=0.15, test_ratio=0.15):
        assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, "Ratios must sum to 1.0"
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        
    def split_by_date(self, df, date_col='Date'):
        """Split data chronologically."""
        print("\n📅 WALK-FORWARD VALIDATION SPLIT")
        print("-" * 60)
        
        df = df.sort_values(date_col).reset_index(drop=True)
        n = len(df)
        
        if n == 0:
            raise ValueError("DataFrame is empty, cannot split")
        
        train_end_idx = int(n * self.train_ratio)
        val_end_idx = int(n * (self.train_ratio + self.val_ratio))
        
        train_df = df.iloc[:train_end_idx].copy()
        val_df = df.iloc[train_end_idx:val_end_idx].copy()
        test_df = df.iloc[val_end_idx:].copy()
        
        if len(train_df) == 0 or len(val_df) == 0 or len(test_df) == 0:
            raise ValueError(f"Split resulted in empty dataset: train={len(train_df)}, val={len(val_df)}, test={len(test_df)}")
        
        print(f"📊 Total samples: {n:,}")
        print(f"   Train: {len(train_df):,} ({len(train_df)/n*100:.1f}%) | {train_df[date_col].min()} → {train_df[date_col].max()}")
        print(f"   Val:   {len(val_df):,} ({len(val_df)/n*100:.1f}%) | {val_df[date_col].min()} → {val_df[date_col].max()}")
        print(f"   Test:  {len(test_df):,} ({len(test_df)/n*100:.1f}%) | {test_df[date_col].min()} → {test_df[date_col].max()}")
        
        # Verify chronological order
        train_max = train_df[date_col].max()
        val_min = val_df[date_col].min()
        val_max = val_df[date_col].max()
        test_min = test_df[date_col].min()
        
        assert train_max < val_min, f"Data leakage: train max {train_max} >= val min {val_min}"
        assert val_max < test_min, f"Data leakage: val max {val_max} >= test min {test_min}"
        
        print("✅ Chronological integrity verified (no data leakage)")
        
        return train_df, val_df, test_df
    
    def prepare_features_target(self, train_df, val_df, test_df, feature_cols, target_col):
        """Separate features and target for each split."""
        X_train = train_df[feature_cols]
        y_train = train_df[target_col]
        
        X_val = val_df[feature_cols]
        y_val = val_df[target_col]
        
        X_test = test_df[feature_cols]
        y_test = test_df[target_col]
        
        print(f"\n📋 Feature/Target Separation:")
        print(f"   Features: {len(feature_cols)}")
        print(f"   Target: {target_col}")
        print(f"   X_train: {X_train.shape}, y_train: {y_train.shape}")
        print(f"   X_val:   {X_val.shape}, y_val:   {y_val.shape}")
        print(f"   X_test:  {X_test.shape}, y_test:  {y_test.shape}")
        
        return X_train, y_train, X_val, y_val, X_test, y_test
    
    def rolling_window_split(self, df, date_col='Date', window_size=252, step_size=63):
        """
        Generate multiple train/test splits using rolling window.
        
        Args:
            df: DataFrame with time-series data
            date_col: Date column name
            window_size: Training window size (e.g., 252 = 1 year of trading days)
            step_size: Step size between windows (e.g., 63 = 3 months)
        
        Yields:
            (train_df, test_df) tuples
        """
        df = df.sort_values(date_col).reset_index(drop=True)
        n = len(df)
        
        splits = []
        start_idx = 0
        
        while start_idx + window_size < n:
            train_end_idx = start_idx + window_size
            test_end_idx = min(train_end_idx + step_size, n)
            
            train_df = df.iloc[start_idx:train_end_idx]
            test_df = df.iloc[train_end_idx:test_end_idx]
            
            splits.append((train_df, test_df))
            start_idx += step_size
            
        print(f"\n🔄 Generated {len(splits)} rolling window splits")
        print(f"   Window size: {window_size} samples")
        print(f"   Step size: {step_size} samples")
        
        return splits


if __name__ == "__main__":
    # Test validation splitting
    print("🧪 Testing Walk-Forward Validation")
    print("=" * 60)
    
    import glob
    data_files = glob.glob("data/processed/*.csv")
    
    if data_files:
        df = pd.read_csv(data_files[0], parse_dates=['Date'])
        print(f"✅ Loaded {len(df):,} rows")
        
        validator = WalkForwardValidator(train_ratio=0.70, val_ratio=0.15, test_ratio=0.15)
        train_df, val_df, test_df = validator.split_by_date(df, date_col='Date')
        
        print("\n✅ Walk-forward validation test complete")
    else:
        print("❌ No processed data files found")
