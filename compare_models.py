#!/usr/bin/env python3
"""
Model Comparison Script
=======================
Compare old vs new model performance with visual proof

Shows:
- Accuracy improvement
- Side-by-side metrics
- Confusion matrices
- Per-class improvements
"""

import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
import json
import os
import sys

# Fix encoding for Windows
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')


def load_model_and_metadata(model_path, metadata_path):
    """Load model and its metadata"""
    model = joblib.load(model_path)
    
    try:
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
    except:
        metadata = None
    
    return model, metadata


def compare_models():
    """Compare old vs new model performance"""
    print("="*70)
    print("MODEL COMPARISON: OLD vs NEW")
    print("="*70)
    
    # Load models
    print("\n[1/4] Loading models...")
    
    old_model_path = 'models/final_production_model.joblib'
    old_metadata_path = 'models/final_model_metadata.json'
    new_model_path = 'models/improved_production_model.joblib'
    new_metadata_path = 'models/improved_model_metadata.json'
    
    if not os.path.exists(new_model_path):
        print(f"\nERROR: New model not found at {new_model_path}")
        print("Please run: python improved_trainer.py")
        return
    
    old_model, old_metadata = load_model_and_metadata(old_model_path, old_metadata_path)
    new_model, new_metadata = load_model_and_metadata(new_model_path, new_metadata_path)
    
    print(f"  Old Model: {old_model_path}")
    print(f"  New Model: {new_model_path}")
    
    # Load test data
    print("\n[2/4] Loading test data...")
    data_path = 'data/processed/cleaned_stock_data.csv'
    
    if not os.path.exists(data_path):
        print(f"\nERROR: Data not found at {data_path}")
        return
    
    df = pd.read_csv(data_path)
    df['Date'] = pd.to_datetime(df['Date'])
    
    # Sample if needed
    if len(df) > 30000:
        df = df.sample(n=30000, random_state=42)
    
    # Prepare features (same as training)
    print("  Preparing features...")
    df = prepare_features(df)
    
    # Prepare ML data
    exclude_cols = [
        'Date', 'Symbol', 'Series', 'Prev Close', 'Last', 'VWAP', 
        'Turnover', 'Trades', 'Deliverable Volume', '%Deliverble',
        'Future_Close', 'Future_Return', 'Label', 'Close'
    ]
    
    feature_cols = [col for col in df.columns if col not in exclude_cols and df[col].dtype in ['int64', 'float64']]
    
    ml_data = df[feature_cols + ['Label', 'Date']].dropna()
    ml_data = ml_data.sort_values('Date').reset_index(drop=True)
    
    # Time-based split
    split_idx = int(len(ml_data) * 0.8)
    test_data = ml_data.iloc[split_idx:]
    
    X_test = test_data[feature_cols]
    y_test = test_data['Label']
    
    print(f"  Test samples: {len(X_test):,}")
    
    # Make predictions
    print("\n[3/4] Comparing predictions...")
    
    # Get only common features for old model
    old_features = [f for f in feature_cols if f in X_test.columns]
    
    # For old model, use only original features (no new advanced ones)
    basic_features = [
        'Open', 'High', 'Low', 'Volume', 'Price_Change', 'Price_Range',
        'Returns', 'SMA_10', 'SMA_50', 'EMA_12', 'EMA_26', 'RSI_14',
        'MACD', 'MACD_signal', 'MACD_hist', 'Volatility_10', 'Volatility_20',
        'Volume_Change_Pct', 'Close_lag_1', 'Volume_lag_1'
    ]
    
    X_test_old = X_test[[f for f in basic_features if f in X_test.columns]]
    X_test_new = X_test[feature_cols]
    
    y_pred_old = old_model.predict(X_test_old)
    y_pred_new = new_model.predict(X_test_new)
    
    # Calculate metrics
    old_acc = accuracy_score(y_test, y_pred_old)
    new_acc = accuracy_score(y_test, y_pred_new)
    
    old_f1 = f1_score(y_test, y_pred_old, average='weighted', zero_division=0)
    new_f1 = f1_score(y_test, y_pred_new, average='weighted', zero_division=0)
    
    improvement_acc = ((new_acc - old_acc) / old_acc) * 100
    improvement_f1 = ((new_f1 - old_f1) / old_f1) * 100
    
    # Display comparison
    print("\n" + "="*70)
    print("PERFORMANCE COMPARISON")
    print("="*70)
    
    print(f"\n{'Metric':<20} {'Old Model':<15} {'New Model':<15} {'Improvement':<15}")
    print("-" * 70)
    print(f"{'Accuracy':<20} {old_acc*100:<14.2f}% {new_acc*100:<14.2f}% {improvement_acc:>+13.2f}%")
    print(f"{'F1-Score':<20} {old_f1*100:<14.2f}% {new_f1*100:<14.2f}% {improvement_f1:>+13.2f}%")
    
    accuracy_gain = new_acc - old_acc
    print(f"\nAbsolute Accuracy Gain: {accuracy_gain*100:+.2f} percentage points")
    
    # Confusion matrices
    print("\n" + "="*70)
    print("OLD MODEL - Confusion Matrix")
    print("="*70)
    cm_old = confusion_matrix(y_test, y_pred_old)
    labels = sorted(y_test.unique())
    print(f"\n          Predicted ->")
    print(f"        {' '.join([f'{l:>8}' for l in labels])}")
    print("Actual v")
    for i, label in enumerate(labels):
        print(f"{label:>6}  {' '.join([f'{cm_old[i][j]:>8}' for j in range(len(labels))])}")
    
    print("\n" + "="*70)
    print("NEW MODEL - Confusion Matrix")
    print("="*70)
    cm_new = confusion_matrix(y_test, y_pred_new)
    print(f"\n          Predicted ->")
    print(f"        {' '.join([f'{l:>8}' for l in labels])}")
    print("Actual v")
    for i, label in enumerate(labels):
        print(f"{label:>6}  {' '.join([f'{cm_new[i][j]:>8}' for j in range(len(labels))])}")
    
    # Per-class comparison
    print("\n" + "="*70)
    print("PER-CLASS PERFORMANCE")
    print("="*70)
    
    old_report = classification_report(y_test, y_pred_old, output_dict=True, zero_division=0)
    new_report = classification_report(y_test, y_pred_new, output_dict=True, zero_division=0)
    
    print(f"\n{'Class':<10} {'Metric':<12} {'Old Model':<12} {'New Model':<12} {'Change':<12}")
    print("-" * 70)
    
    for label in labels:
        if label in old_report and label in new_report:
            # Precision
            old_prec = old_report[label]['precision'] * 100
            new_prec = new_report[label]['precision'] * 100
            change_prec = new_prec - old_prec
            print(f"{label:<10} {'Precision':<12} {old_prec:<11.2f}% {new_prec:<11.2f}% {change_prec:>+10.2f}%")
            
            # Recall
            old_rec = old_report[label]['recall'] * 100
            new_rec = new_report[label]['recall'] * 100
            change_rec = new_rec - old_rec
            print(f"{'':<10} {'Recall':<12} {old_rec:<11.2f}% {new_rec:<11.2f}% {change_rec:>+10.2f}%")
            
            # F1
            old_f1_class = old_report[label]['f1-score'] * 100
            new_f1_class = new_report[label]['f1-score'] * 100
            change_f1 = new_f1_class - old_f1_class
            print(f"{'':<10} {'F1-Score':<12} {old_f1_class:<11.2f}% {new_f1_class:<11.2f}% {change_f1:>+10.2f}%")
            print()
    
    # Create visualizations
    print("[4/4] Creating comparison charts...")
    create_comparison_charts(old_acc, new_acc, old_f1, new_f1, cm_old, cm_new, labels)
    
    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    
    if new_acc > old_acc:
        print(f"\nSUCCESS! New model is BETTER")
        print(f"  Accuracy improved by {accuracy_gain*100:.2f} percentage points")
        print(f"  Relative improvement: {improvement_acc:.2f}%")
    else:
        print(f"\nWARNING: New model did not improve accuracy")
        print(f"  Accuracy changed by {accuracy_gain*100:.2f} percentage points")
    
    print(f"\nNew model features:")
    if new_metadata and 'improvements' in new_metadata:
        for improvement in new_metadata['improvements']:
            print(f"  - {improvement}")
    
    print(f"\nResults saved to:")
    print(f"  - results/model_comparison.png")
    print(f"  - results/improved_model_performance.csv")


def prepare_features(df):
    """Prepare all features including advanced ones"""
    df = df.sort_values(['Symbol', 'Date']).reset_index(drop=True)
    
    # Basic features
    df['Price_Change'] = df['Close'] - df['Open']
    df['Price_Range'] = df['High'] - df['Low']
    df['Returns'] = df.groupby('Symbol')['Close'].pct_change()
    
    stock_groups = []
    for symbol in df['Symbol'].unique():
        stock_df = df[df['Symbol'] == symbol].copy()
        
        # Standard indicators
        stock_df['SMA_10'] = stock_df['Close'].rolling(window=10).mean()
        stock_df['SMA_50'] = stock_df['Close'].rolling(window=50).mean()
        stock_df['EMA_12'] = stock_df['Close'].ewm(span=12, adjust=False).mean()
        stock_df['EMA_26'] = stock_df['Close'].ewm(span=26, adjust=False).mean()
        
        # RSI
        delta = stock_df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        stock_df['RSI_14'] = 100 - (100 / (1 + rs))
        
        # MACD
        stock_df['MACD'] = stock_df['EMA_12'] - stock_df['EMA_26']
        stock_df['MACD_signal'] = stock_df['MACD'].ewm(span=9, adjust=False).mean()
        stock_df['MACD_hist'] = stock_df['MACD'] - stock_df['MACD_signal']
        
        # Volatility
        stock_df['Volatility_10'] = stock_df['Returns'].rolling(window=10).std()
        stock_df['Volatility_20'] = stock_df['Returns'].rolling(window=20).std()
        stock_df['Volume_Change_Pct'] = stock_df['Volume'].pct_change() * 100
        
        # Advanced indicators (for new model)
        sma_20 = stock_df['Close'].rolling(window=20).mean()
        std_20 = stock_df['Close'].rolling(window=20).std()
        stock_df['BB_upper'] = sma_20 + (2 * std_20)
        stock_df['BB_lower'] = sma_20 - (2 * std_20)
        stock_df['BB_width'] = stock_df['BB_upper'] - stock_df['BB_lower']
        stock_df['BB_position'] = (stock_df['Close'] - stock_df['BB_lower']) / stock_df['BB_width']
        
        high_low = stock_df['High'] - stock_df['Low']
        high_close = np.abs(stock_df['High'] - stock_df['Close'].shift())
        low_close = np.abs(stock_df['Low'] - stock_df['Close'].shift())
        true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        stock_df['ATR_14'] = true_range.rolling(window=14).mean()
        
        low_14 = stock_df['Low'].rolling(window=14).min()
        high_14 = stock_df['High'].rolling(window=14).max()
        stock_df['Stochastic'] = 100 * (stock_df['Close'] - low_14) / (high_14 - low_14)
        stock_df['Stochastic_smooth'] = stock_df['Stochastic'].rolling(window=3).mean()
        
        obv = []
        obv_val = 0
        for idx, row in stock_df.iterrows():
            if idx == stock_df.index[0]:
                obv.append(row['Volume'])
            else:
                prev_close = stock_df.loc[idx - 1, 'Close'] if idx > stock_df.index[0] else row['Close']
                if row['Close'] > prev_close:
                    obv_val += row['Volume']
                elif row['Close'] < prev_close:
                    obv_val -= row['Volume']
                obv.append(obv_val)
        stock_df['OBV'] = obv
        stock_df['OBV_EMA'] = stock_df['OBV'].ewm(span=20, adjust=False).mean()
        
        stock_df['Williams_R'] = -100 * (high_14 - stock_df['Close']) / (high_14 - low_14)
        stock_df['ROC_10'] = stock_df['Close'].pct_change(periods=10) * 100
        stock_df['ROC_20'] = stock_df['Close'].pct_change(periods=20) * 100
        stock_df['SMA_cross'] = (stock_df['SMA_10'] > stock_df['SMA_50']).astype(int)
        stock_df['EMA_cross'] = (stock_df['EMA_12'] > stock_df['EMA_26']).astype(int)
        
        # Lagged features
        stock_df['Close_lag_1'] = stock_df['Close'].shift(1)
        stock_df['Volume_lag_1'] = stock_df['Volume'].shift(1)
        stock_df['RSI_lag_1'] = stock_df['RSI_14'].shift(1)
        
        stock_groups.append(stock_df)
    
    df = pd.concat(stock_groups, ignore_index=True)
    
    # Create labels
    df['Future_Close'] = df.groupby('Symbol')['Close'].shift(-5)
    df['Future_Return'] = (df['Future_Close'] - df['Close']) / df['Close']
    
    def assign_label(future_return):
        if pd.isna(future_return):
            return None
        elif future_return >= 0.02:
            return 'BUY'
        elif future_return <= -0.02:
            return 'SELL'
        else:
            return 'HOLD'
    
    df['Label'] = df['Future_Return'].apply(assign_label)
    
    return df


def create_comparison_charts(old_acc, new_acc, old_f1, new_f1, cm_old, cm_new, labels):
    """Create visual comparison charts"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 1. Accuracy Comparison
    ax1 = axes[0, 0]
    models = ['Old Model', 'New Model']
    accuracies = [old_acc * 100, new_acc * 100]
    colors = ['#e74c3c', '#2ecc71']
    bars = ax1.bar(models, accuracies, color=colors, alpha=0.7, edgecolor='black')
    ax1.set_ylabel('Accuracy (%)', fontweight='bold')
    ax1.set_title('Accuracy Comparison', fontweight='bold', fontsize=12)
    ax1.set_ylim([0, 100])
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Add value labels
    for bar, acc in zip(bars, accuracies):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{acc:.2f}%', ha='center', va='bottom', fontweight='bold')
    
    # Add improvement arrow
    improvement = new_acc - old_acc
    ax1.annotate(f'+{improvement*100:.2f}%',
                xy=(1, new_acc*100), xytext=(0.5, (old_acc + new_acc)*50),
                arrowprops=dict(arrowstyle='->', color='green', lw=2),
                fontsize=11, fontweight='bold', color='green')
    
    # 2. F1-Score Comparison
    ax2 = axes[0, 1]
    f1_scores = [old_f1 * 100, new_f1 * 100]
    bars = ax2.bar(models, f1_scores, color=colors, alpha=0.7, edgecolor='black')
    ax2.set_ylabel('F1-Score (%)', fontweight='bold')
    ax2.set_title('F1-Score Comparison', fontweight='bold', fontsize=12)
    ax2.set_ylim([0, 100])
    ax2.grid(True, alpha=0.3, axis='y')
    
    for bar, f1 in zip(bars, f1_scores):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{f1:.2f}%', ha='center', va='bottom', fontweight='bold')
    
    # 3. Old Model Confusion Matrix
    ax3 = axes[1, 0]
    im1 = ax3.imshow(cm_old, cmap='Reds', alpha=0.6)
    ax3.set_xticks(np.arange(len(labels)))
    ax3.set_yticks(np.arange(len(labels)))
    ax3.set_xticklabels(labels)
    ax3.set_yticklabels(labels)
    ax3.set_xlabel('Predicted', fontweight='bold')
    ax3.set_ylabel('Actual', fontweight='bold')
    ax3.set_title('Old Model - Confusion Matrix', fontweight='bold', fontsize=12)
    
    for i in range(len(labels)):
        for j in range(len(labels)):
            ax3.text(j, i, str(cm_old[i, j]), ha='center', va='center', fontweight='bold')
    
    # 4. New Model Confusion Matrix
    ax4 = axes[1, 1]
    im2 = ax4.imshow(cm_new, cmap='Greens', alpha=0.6)
    ax4.set_xticks(np.arange(len(labels)))
    ax4.set_yticks(np.arange(len(labels)))
    ax4.set_xticklabels(labels)
    ax4.set_yticklabels(labels)
    ax4.set_xlabel('Predicted', fontweight='bold')
    ax4.set_ylabel('Actual', fontweight='bold')
    ax4.set_title('New Model - Confusion Matrix', fontweight='bold', fontsize=12)
    
    for i in range(len(labels)):
        for j in range(len(labels)):
            ax4.text(j, i, str(cm_new[i, j]), ha='center', va='center', fontweight='bold')
    
    plt.suptitle('Model Comparison: Old vs Improved', fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()
    
    os.makedirs('results', exist_ok=True)
    plt.savefig('results/model_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("  SUCCESS: Comparison chart saved to results/model_comparison.png")


if __name__ == "__main__":
    compare_models()
