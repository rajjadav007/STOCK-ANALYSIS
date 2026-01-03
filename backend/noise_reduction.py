#!/usr/bin/env python3
"""
Noise Reduction and Outlier Handling
=====================================
Smoothing and outlier detection while preserving trend structure.
"""

import pandas as pd
import numpy as np
from scipy.signal import savgol_filter
from sklearn.ensemble import IsolationForest
import warnings

warnings.filterwarnings('ignore')


class NoiseReducer:
    """Noise reduction and outlier handling for time-series data."""
    
    def __init__(self, contamination=0.01):
        self.contamination = contamination
        self.outlier_detector = None
        
    def apply_savgol_smoothing(self, df, cols_to_smooth, window=11, poly_order=3, symbol_col='Symbol'):
        """Apply Savitzky-Golay filter for smoothing."""
        df = df.copy()
        
        for col in cols_to_smooth:
            if col not in df.columns:
                continue
                
            for symbol in df[symbol_col].unique():
                mask = df[symbol_col] == symbol
                values = df.loc[mask, col].values
                
                if len(values) > window:
                    smoothed = savgol_filter(values, window_length=window, polyorder=poly_order)
                    df.loc[mask, f'{col}_smooth'] = smoothed
                    
        return df
    
    def detect_outliers_isolation_forest(self, df, feature_cols):
        """Detect outliers using Isolation Forest."""
        print(f"\n🔍 Detecting outliers (contamination: {self.contamination})...")
        
        X = df[feature_cols].copy()
        X = X.replace([np.inf, -np.inf], np.nan)
        X = X.fillna(X.median())
        
        self.outlier_detector = IsolationForest(
            contamination=self.contamination,
            random_state=42,
            n_jobs=-1
        )
        
        outlier_labels = self.outlier_detector.fit_predict(X)
        n_outliers = (outlier_labels == -1).sum()
        
        print(f"   Outliers detected: {n_outliers:,} ({n_outliers/len(df)*100:.2f}%)")
        
        return outlier_labels
    
    def winsorize(self, df, cols_to_clip, lower_percentile=1, upper_percentile=99, symbol_col='Symbol'):
        """Winsorize features at specified percentiles."""
        df = df.copy()
        
        for col in cols_to_clip:
            if col not in df.columns:
                continue
                
            for symbol in df[symbol_col].unique():
                mask = df[symbol_col] == symbol
                values = df.loc[mask, col]
                
                lower = np.percentile(values.dropna(), lower_percentile)
                upper = np.percentile(values.dropna(), upper_percentile)
                
                df.loc[mask, col] = values.clip(lower=lower, upper=upper)
                
        return df
    
    def rolling_median_smoothing(self, df, cols_to_smooth, window=5, symbol_col='Symbol'):
        """Apply rolling median for trend preservation."""
        df = df.copy()
        
        for col in cols_to_smooth:
            if col not in df.columns:
                continue
                
            for symbol in df[symbol_col].unique():
                mask = df[symbol_col] == symbol
                df.loc[mask, f'{col}_median'] = df.loc[mask, col].rolling(window=window, center=True).median()
                
        return df
    
    def apply_noise_reduction(self, df, price_cols=['Close', 'Open', 'High', 'Low'], 
                             feature_cols=None, remove_outliers=False):
        """Apply complete noise reduction pipeline."""
        print("\n🧹 NOISE REDUCTION")
        print("-" * 60)
        
        df = df.copy()
        initial_rows = len(df)
        
        # Winsorize price columns
        print("📊 Winsorizing price features (1st-99th percentile)...")
        df = self.winsorize(df, price_cols, lower_percentile=1, upper_percentile=99)
        
        # Apply rolling median smoothing
        print("📊 Applying rolling median smoothing...")
        df = self.rolling_median_smoothing(df, price_cols, window=5)
        
        # Detect and optionally remove outliers
        if feature_cols and remove_outliers:
            outlier_labels = self.detect_outliers_isolation_forest(df, feature_cols)
            df = df[outlier_labels != -1].copy()
            print(f"   Removed {initial_rows - len(df):,} outlier rows")
        
        print(f"✅ Noise reduction complete: {len(df):,} rows retained")
        
        return df


if __name__ == "__main__":
    # Test noise reduction
    print("🧪 Testing Noise Reduction")
    print("=" * 60)
    
    import glob
    data_files = glob.glob("data/processed/*.csv")
    
    if data_files:
        df = pd.read_csv(data_files[0], parse_dates=['Date'])
        print(f"✅ Loaded {len(df):,} rows")
        
        reducer = NoiseReducer(contamination=0.01)
        df_clean = reducer.apply_noise_reduction(
            df, 
            price_cols=['Close', 'Open', 'High', 'Low'],
            remove_outliers=False
        )
        
        print(f"\n✅ Final shape: {df_clean.shape}")
    else:
        print("❌ No processed data files found")
