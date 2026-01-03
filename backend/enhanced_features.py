#!/usr/bin/env python3
"""
Enhanced Feature Engineering for Stock Prediction
==================================================
Advanced technical indicators and lag features with multicollinearity removal.
"""

import pandas as pd
import numpy as np
from sklearn.feature_selection import VarianceThreshold
from statsmodels.stats.outliers_influence import variance_inflation_factor
import warnings

warnings.filterwarnings('ignore')


class EnhancedFeatureEngineer:
    """Advanced feature engineering with VIF-based multicollinearity removal."""
    
    def __init__(self, vif_threshold=10.0):
        self.vif_threshold = vif_threshold
        self.selected_features = None
        
    def add_lag_features(self, df, symbol_col='Symbol'):
        """Add comprehensive lag features."""
        df = df.copy()
        
        for symbol in df[symbol_col].unique():
            mask = df[symbol_col] == symbol
            
            # Price lags
            for lag in [1, 2, 3, 5, 10]:
                df.loc[mask, f'Close_lag_{lag}'] = df.loc[mask, 'Close'].shift(lag)
                df.loc[mask, f'Returns_lag_{lag}'] = df.loc[mask, 'Returns'].shift(lag)
            
            # Volume lags
            for lag in [1, 3, 5]:
                df.loc[mask, f'Volume_lag_{lag}'] = df.loc[mask, 'Volume'].shift(lag)
                
        return df
    
    def add_momentum_indicators(self, df, symbol_col='Symbol'):
        """Add momentum-based technical indicators."""
        df = df.copy()
        
        for symbol in df[symbol_col].unique():
            mask = df[symbol_col] == symbol
            close = df.loc[mask, 'Close']
            high = df.loc[mask, 'High']
            low = df.loc[mask, 'Low']
            
            # Rate of Change (ROC)
            df.loc[mask, 'ROC_10'] = close.pct_change(periods=10) * 100
            df.loc[mask, 'ROC_20'] = close.pct_change(periods=20) * 100
            
            # Stochastic Oscillator
            low_14 = low.rolling(window=14).min()
            high_14 = high.rolling(window=14).max()
            df.loc[mask, 'Stochastic_K'] = 100 * (close - low_14) / (high_14 - low_14 + 1e-10)
            df.loc[mask, 'Stochastic_D'] = df.loc[mask, 'Stochastic_K'].rolling(window=3).mean()
            
            # Williams %R
            df.loc[mask, 'Williams_R'] = -100 * (high_14 - close) / (high_14 - low_14 + 1e-10)
            
            # Commodity Channel Index (CCI)
            typical_price = (high + low + close) / 3
            sma_tp = typical_price.rolling(window=20).mean()
            mad_tp = typical_price.rolling(window=20).apply(lambda x: np.abs(x - x.mean()).mean())
            df.loc[mask, 'CCI_20'] = (typical_price - sma_tp) / (0.015 * mad_tp + 1e-10)
            
        return df
    
    def add_trend_indicators(self, df, symbol_col='Symbol'):
        """Add trend-based technical indicators."""
        df = df.copy()
        
        for symbol in df[symbol_col].unique():
            mask = df[symbol_col] == symbol
            close = df.loc[mask, 'Close']
            high = df.loc[mask, 'High']
            low = df.loc[mask, 'Low']
            
            # Average Directional Index (ADX)
            high_diff = high.diff()
            low_diff = -low.diff()
            
            plus_dm = np.where((high_diff > low_diff) & (high_diff > 0), high_diff, 0)
            minus_dm = np.where((low_diff > high_diff) & (low_diff > 0), low_diff, 0)
            
            tr = pd.concat([
                high - low,
                (high - close.shift(1)).abs(),
                (low - close.shift(1)).abs()
            ], axis=1).max(axis=1)
            
            atr_14 = pd.Series(tr).rolling(window=14).mean()
            plus_di = 100 * pd.Series(plus_dm).rolling(window=14).mean() / (atr_14 + 1e-10)
            minus_di = 100 * pd.Series(minus_dm).rolling(window=14).mean() / (atr_14 + 1e-10)
            
            dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
            df.loc[mask, 'ADX_14'] = dx.rolling(window=14).mean()
            
            # Bollinger Bands
            sma_20 = close.rolling(window=20).mean()
            std_20 = close.rolling(window=20).std()
            df.loc[mask, 'BB_upper'] = sma_20 + (2 * std_20)
            df.loc[mask, 'BB_lower'] = sma_20 - (2 * std_20)
            df.loc[mask, 'BB_width'] = (df.loc[mask, 'BB_upper'] - df.loc[mask, 'BB_lower']) / sma_20
            df.loc[mask, 'BB_position'] = (close - df.loc[mask, 'BB_lower']) / (df.loc[mask, 'BB_upper'] - df.loc[mask, 'BB_lower'] + 1e-10)
            
        return df
    
    def add_volatility_indicators(self, df, symbol_col='Symbol'):
        """Add volatility-based technical indicators."""
        df = df.copy()
        
        for symbol in df[symbol_col].unique():
            mask = df[symbol_col] == symbol
            close = df.loc[mask, 'Close']
            high = df.loc[mask, 'High']
            low = df.loc[mask, 'Low']
            
            # Average True Range (ATR)
            tr = pd.concat([
                high - low,
                (high - close.shift(1)).abs(),
                (low - close.shift(1)).abs()
            ], axis=1).max(axis=1)
            
            df.loc[mask, 'ATR_14'] = tr.rolling(window=14).mean()
            df.loc[mask, 'ATR_20'] = tr.rolling(window=20).mean()
            
            # Keltner Channels
            ema_20 = close.ewm(span=20, adjust=False).mean()
            df.loc[mask, 'Keltner_upper'] = ema_20 + (2 * df.loc[mask, 'ATR_20'])
            df.loc[mask, 'Keltner_lower'] = ema_20 - (2 * df.loc[mask, 'ATR_20'])
            df.loc[mask, 'Keltner_width'] = (df.loc[mask, 'Keltner_upper'] - df.loc[mask, 'Keltner_lower']) / ema_20
            
        return df
    
    def add_volume_indicators(self, df, symbol_col='Symbol'):
        """Add volume-based technical indicators."""
        df = df.copy()
        
        for symbol in df[symbol_col].unique():
            mask = df[symbol_col] == symbol
            close = df.loc[mask, 'Close']
            volume = df.loc[mask, 'Volume']
            
            # On-Balance Volume (OBV)
            obv = (np.sign(close.diff()) * volume).fillna(0).cumsum()
            df.loc[mask, 'OBV'] = obv
            df.loc[mask, 'OBV_EMA'] = obv.ewm(span=20, adjust=False).mean()
            
            # Volume SMA ratios
            vol_sma_20 = volume.rolling(window=20).mean()
            df.loc[mask, 'Volume_SMA_20'] = vol_sma_20
            df.loc[mask, 'Volume_ratio'] = volume / (vol_sma_20 + 1e-10)
            
            # Volume Rate of Change
            df.loc[mask, 'Volume_ROC'] = volume.pct_change(periods=10) * 100
            
        return df
    
    def calculate_vif(self, X):
        """Calculate Variance Inflation Factor for features."""
        vif_data = pd.DataFrame()
        vif_data["Feature"] = X.columns
        vif_data["VIF"] = [variance_inflation_factor(X.values, i) for i in range(len(X.columns))]
        return vif_data.sort_values('VIF', ascending=False)
    
    def remove_multicollinearity(self, df, feature_cols, target_col=None):
        """Remove highly correlated features using correlation matrix."""
        print(f"\n🔍 Removing multicollinearity (correlation threshold: 0.95)...")
        
        # Exclude non-numeric and target columns
        exclude_cols = ['Date', 'Symbol', 'Series', target_col] if target_col else ['Date', 'Symbol', 'Series']
        X = df[[col for col in feature_cols if col not in exclude_cols and col in df.columns]].copy()
        
        # Remove constant features
        X = X.loc[:, X.std() > 0]
        
        # Handle inf and NaN
        X = X.replace([np.inf, -np.inf], np.nan)
        X = X.fillna(X.median())
        
        initial_features = len(X.columns)
        
        if initial_features == 0:
            print("⚠️  No features to process after cleaning")
            return []
        
        # Calculate correlation matrix
        corr_matrix = X.corr().abs()
        
        # Find pairs of highly correlated features
        upper_tri = corr_matrix.where(
            np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
        )
        
        # Remove features with correlation > 0.95
        to_drop = [column for column in upper_tri.columns if any(upper_tri[column] > 0.95)]
        
        if to_drop:
            print(f"   Removing {len(to_drop)} highly correlated features (>0.95):")
            for feat in to_drop[:10]:  # Show first 10
                print(f"      {feat}")
            if len(to_drop) > 10:
                print(f"      ... and {len(to_drop) - 10} more")
            X = X.drop(columns=to_drop)
        
        final_features = len(X.columns)
        self.selected_features = X.columns.tolist()
        
        print(f"✅ Features: {initial_features} → {final_features} (removed {initial_features - final_features})")
        return X.columns.tolist()
    
    def fit_transform(self, df, symbol_col='Symbol', target_col='Label'):
        """Apply all feature engineering steps."""
        print("\n⚙️ ENHANCED FEATURE ENGINEERING")
        print("-" * 60)
        
        df = df.copy()
        
        # Add all feature types
        print("📊 Adding lag features...")
        df = self.add_lag_features(df, symbol_col)
        
        print("📊 Adding momentum indicators...")
        df = self.add_momentum_indicators(df, symbol_col)
        
        print("📊 Adding trend indicators...")
        df = self.add_trend_indicators(df, symbol_col)
        
        print("📊 Adding volatility indicators...")
        df = self.add_volatility_indicators(df, symbol_col)
        
        print("📊 Adding volume indicators...")
        df = self.add_volume_indicators(df, symbol_col)
        
        # Remove NaN created by indicators
        initial_rows = len(df)
        df = df.dropna()
        final_rows = len(df)
        print(f"✅ Rows after NaN removal: {final_rows:,} (removed {initial_rows - final_rows:,})")
        
        # Remove multicollinearity
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        self.selected_features = self.remove_multicollinearity(df, numeric_cols, target_col)
        
        return df, self.selected_features


if __name__ == "__main__":
    # Test with existing data
    print("🧪 Testing Enhanced Feature Engineering")
    print("=" * 60)
    
    # Load processed data
    import glob
    data_files = glob.glob("data/processed/*.csv")
    
    if data_files:
        df = pd.read_csv(data_files[0], parse_dates=['Date'])
        print(f"✅ Loaded {len(df):,} rows from {data_files[0]}")
        
        # Check if features already exist
        if 'Returns' not in df.columns:
            df['Returns'] = df.groupby('Symbol')['Close'].pct_change()
        
        # Apply enhanced features
        engineer = EnhancedFeatureEngineer(vif_threshold=10.0)
        df_enhanced, selected_features = engineer.fit_transform(df)
        
        print(f"\n📋 Selected Features ({len(selected_features)}):")
        for i, feat in enumerate(selected_features, 1):
            print(f"  {i:2d}. {feat}")
            
        print(f"\n✅ Enhanced data shape: {df_enhanced.shape}")
    else:
        print("❌ No processed data files found")
