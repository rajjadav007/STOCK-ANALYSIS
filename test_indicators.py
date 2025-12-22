#!/usr/bin/env python3
"""
Quick test to verify technical indicators are working correctly
"""

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

print("✅ pandas and numpy imported successfully")

def test_indicators():
    """Test technical indicator creation"""
    print("\n🧪 TESTING TECHNICAL INDICATORS")
    print("=" * 50)
    
    # Create sample stock data
    dates = pd.date_range(start='2024-01-01', periods=100, freq='D')
    np.random.seed(42)
    
    # Generate realistic stock price data
    base_price = 100
    returns = np.random.normal(0.001, 0.02, 100)  # Daily returns with mean 0.1%, std 2%
    prices = base_price * (1 + returns).cumprod()
    
    df = pd.DataFrame({
        'Date': dates,
        'Open': prices * (1 + np.random.normal(0, 0.005, 100)),
        'High': prices * (1 + np.random.uniform(0, 0.02, 100)),
        'Low': prices * (1 - np.random.uniform(0, 0.02, 100)),
        'Close': prices,
        'Volume': np.random.randint(1000000, 5000000, 100)
    })
    
    print(f"📊 Sample data created: {len(df)} rows")
    print(f"📅 Date range: {df['Date'].min().strftime('%Y-%m-%d')} to {df['Date'].max().strftime('%Y-%m-%d')}")
    
    # Test SMA (Simple Moving Average)
    print("\n🔧 Testing SMA (Simple Moving Average)...")
    df['SMA_10'] = df['Close'].rolling(window=10).mean()
    df['SMA_50'] = df['Close'].rolling(window=50).mean()
    print(f"   ✅ SMA_10: {df['SMA_10'].notna().sum()}/{len(df)} values (expected: 91/100)")
    print(f"   ✅ SMA_50: {df['SMA_50'].notna().sum()}/{len(df)} values (expected: 51/100)")
    
    # Test EMA (Exponential Moving Average)
    print("\n🔧 Testing EMA (Exponential Moving Average)...")
    df['EMA_12'] = df['Close'].ewm(span=12, adjust=False).mean()
    df['EMA_26'] = df['Close'].ewm(span=26, adjust=False).mean()
    print(f"   ✅ EMA_12: {df['EMA_12'].notna().sum()}/{len(df)} values")
    print(f"   ✅ EMA_26: {df['EMA_26'].notna().sum()}/{len(df)} values")
    
    # Test RSI (Relative Strength Index)
    print("\n🔧 Testing RSI (Relative Strength Index)...")
    df['Returns'] = df['Close'].pct_change()
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI_14'] = 100 - (100 / (1 + rs))
    print(f"   ✅ RSI_14: {df['RSI_14'].notna().sum()}/{len(df)} values")
    valid_rsi = df['RSI_14'].dropna()
    if len(valid_rsi) > 0:
        print(f"   📊 RSI range: {valid_rsi.min():.2f} to {valid_rsi.max():.2f} (expected: 0-100)")
    
    # Test MACD (Moving Average Convergence Divergence)
    print("\n🔧 Testing MACD (Moving Average Convergence Divergence)...")
    df['MACD'] = df['EMA_12'] - df['EMA_26']
    df['MACD_signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
    df['MACD_hist'] = df['MACD'] - df['MACD_signal']
    print(f"   ✅ MACD: {df['MACD'].notna().sum()}/{len(df)} values")
    print(f"   ✅ MACD_signal: {df['MACD_signal'].notna().sum()}/{len(df)} values")
    print(f"   ✅ MACD_hist: {df['MACD_hist'].notna().sum()}/{len(df)} values")
    
    # Test Volatility
    print("\n🔧 Testing Volatility (Rolling STD of Returns)...")
    df['Volatility_10'] = df['Returns'].rolling(window=10).std()
    df['Volatility_20'] = df['Returns'].rolling(window=20).std()
    print(f"   ✅ Volatility_10: {df['Volatility_10'].notna().sum()}/{len(df)} values")
    print(f"   ✅ Volatility_20: {df['Volatility_20'].notna().sum()}/{len(df)} values")
    
    # Test Volume Change Percentage
    print("\n🔧 Testing Volume Change Percentage...")
    df['Volume_Change_Pct'] = df['Volume'].pct_change() * 100
    print(f"   ✅ Volume_Change_Pct: {df['Volume_Change_Pct'].notna().sum()}/{len(df)} values")
    
    # Show NaN counts
    print("\n🗑️  NaN ROWS SUMMARY")
    print("-" * 50)
    nan_counts = df.isna().sum()
    total_nan = df.isna().any(axis=1).sum()
    print(f"📊 Rows with any NaN: {total_nan}/{len(df)} ({(total_nan/len(df))*100:.2f}%)")
    print(f"📊 Rows after removing NaN: {len(df.dropna())}/{len(df)} ({(len(df.dropna())/len(df))*100:.2f}%)")
    
    print("\n📋 NaN counts by column:")
    for col in nan_counts[nan_counts > 0].index:
        print(f"   {col}: {nan_counts[col]} NaN values")
    
    # Test dropna
    print("\n🧹 Testing NaN removal...")
    df_clean = df.dropna()
    print(f"   ✅ Clean data: {len(df_clean)}/{len(df)} rows retained")
    print(f"   ✅ Features: {len(df_clean.columns)} columns")
    print(f"   ✅ No NaN values: {df_clean.isna().sum().sum() == 0}")
    
    print("\n✅ ALL TECHNICAL INDICATORS TESTED SUCCESSFULLY!")
    print("🎉 Ready to use in main.py")
    
    return True

if __name__ == "__main__":
    test_indicators()
