#!/usr/bin/env python3
"""
How to Use Your Trained Models
================================
This script shows you how to load and use the trained ensemble models.
"""

import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler

# ============================================================================
# 1. LOAD THE TRAINED MODELS
# ============================================================================

print("=" * 80)
print("LOADING TRAINED MODELS")
print("=" * 80)

# Load ensemble models (RF, XGBoost, LightGBM)
models = joblib.load('models/ensemble_models.joblib')
scaler = joblib.load('models/scaler.joblib')

print(f"\n✅ Loaded {len(models)} models:")
for name in models.keys():
    print(f"   • {name}")

# Load results
import json
with open('results/improvement_metrics.json', 'r') as f:
    results = json.load(f)

print(f"\n✅ PERFORMANCE METRICS:")
print(f"   MAE:                      {results['metrics']['MAE']:.6f}")
print(f"   RMSE:                     {results['metrics']['RMSE']:.6f}")
print(f"   R²:                       {results['metrics']['R2']:.4f}")
print(f"   Directional Accuracy:     {results['metrics']['Directional_Accuracy']*100:.2f}%")
print(f"   Profit-Weighted Accuracy: {results['metrics']['Profit_Weighted_Accuracy']*100:.2f}%")

print(f"\n✅ ENSEMBLE WEIGHTS:")
for name, weight in results['ensemble_weights'].items():
    print(f"   {name}: {weight:.3f} ({weight*100:.1f}%)")

print(f"\n✅ VERDICT: {results['verdict']}")


# ============================================================================
# 2. HOW TO MAKE PREDICTIONS ON NEW DATA
# ============================================================================

print("\n" + "=" * 80)
print("EXAMPLE: MAKING PREDICTIONS")
print("=" * 80)

# Load some test data
import glob
files = glob.glob("data/processed/*.csv")
df = pd.read_csv(files[0])
df['Date'] = pd.to_datetime(df['Date'])

print(f"\n📊 Loaded data: {len(df):,} rows")

# Take last 100 rows as example
df_sample = df.tail(100).copy()

# You need to create the SAME features that were used during training
# (This is a simplified version - in production, use the same feature engineering)

for symbol in df_sample['Symbol'].unique():
    mask = df_sample['Symbol'] == symbol
    close = df_sample.loc[mask, 'Close'].values
    
    # Add features (same as training)
    df_sample.loc[mask, 'Returns'] = pd.Series(close).pct_change().values
    
    for lag in [1, 2, 3]:
        df_sample.loc[mask, f'Close_lag_{lag}'] = pd.Series(close).shift(lag).values
        df_sample.loc[mask, f'Returns_lag_{lag}'] = pd.Series(close).pct_change().shift(lag).values
    
    df_sample.loc[mask, 'SMA_5'] = pd.Series(close).rolling(5, min_periods=1).mean().values
    df_sample.loc[mask, 'SMA_10'] = pd.Series(close).rolling(10, min_periods=1).mean().values
    # ... (add all other features)

df_sample = df_sample.dropna()

# Select features (must match training features)
exclude = ['Date', 'Symbol', 'Series', 'Close', 'Prev Close', 'Last', 'VWAP', 'Turnover', 'Deliverable Volume', '%Deliverble']
feature_cols = [col for col in df_sample.columns if col not in exclude and df_sample[col].dtype in ['int64', 'float64']]

X_new = df_sample[feature_cols].fillna(0)

# Scale features
X_new_scaled = scaler.transform(X_new)

# Make predictions with each model
predictions = {}
for name, model in models.items():
    predictions[name] = model.predict(X_new_scaled)

# Ensemble prediction (using optimized weights)
weights = results['ensemble_weights']
ensemble_pred = sum(weights[name] * predictions[name] for name in models.keys())

print(f"\n✅ Made predictions for {len(ensemble_pred)} samples")
print(f"\n📈 Last 10 predictions (expected returns):")
for i, pred in enumerate(ensemble_pred[-10:], 1):
    direction = "📈 UP" if pred > 0 else "📉 DOWN"
    print(f"   {i:2d}. {pred:+.4f} ({pred*100:+.2f}%) {direction}")


# ============================================================================
# 3. HOW TO GENERATE TRADING SIGNALS
# ============================================================================

print("\n" + "=" * 80)
print("EXAMPLE: TRADING SIGNALS")
print("=" * 80)

# Generate signals based on predictions
THRESHOLD = 0.005  # 0.5% minimum expected return

signals = []
for pred in ensemble_pred:
    if pred > THRESHOLD:
        signals.append("BUY")
    elif pred < -THRESHOLD:
        signals.append("SELL")
    else:
        signals.append("HOLD")

print(f"\n📊 Trading Signals for last 10 predictions:")
for i, (pred, signal) in enumerate(zip(ensemble_pred[-10:], signals[-10:]), 1):
    emoji = "🟢" if signal == "BUY" else "🔴" if signal == "SELL" else "⚪"
    print(f"   {i:2d}. {emoji} {signal:4s} | Expected return: {pred*100:+.2f}%")

buy_count = signals.count("BUY")
sell_count = signals.count("SELL")
hold_count = signals.count("HOLD")

print(f"\n✅ Signal Distribution:")
print(f"   🟢 BUY:  {buy_count:3d} ({buy_count/len(signals)*100:.1f}%)")
print(f"   🔴 SELL: {sell_count:3d} ({sell_count/len(signals)*100:.1f}%)")
print(f"   ⚪ HOLD: {hold_count:3d} ({hold_count/len(signals)*100:.1f}%)")


# ============================================================================
# 4. SUMMARY
# ============================================================================

print("\n" + "=" * 80)
print("📁 YOUR FILES")
print("=" * 80)
print("""
✅ Models saved in:
   • models/ensemble_models.joblib  - All 3 trained models
   • models/scaler.joblib           - Feature scaler

✅ Results saved in:
   • results/improvement_metrics.json - Full performance report

✅ To use in production:
   1. Load models: joblib.load('models/ensemble_models.joblib')
   2. Load scaler: joblib.load('models/scaler.joblib')
   3. Engineer same features on new data
   4. Scale features: scaler.transform(X_new)
   5. Get predictions: model.predict(X_scaled)
   6. Apply ensemble weights for final prediction
""")

print("=" * 80)
print("✅ DONE - Models ready to use!")
print("=" * 80)
