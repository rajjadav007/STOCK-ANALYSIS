#!/usr/bin/env python3
"""
Candlestick Chart Visualization with ML Predictions
Shows actual stock prices as candlesticks with BUY/SELL/HOLD signals
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Rectangle
import joblib
import os
import warnings
warnings.filterwarnings('ignore')

def plot_candlestick_with_predictions(df, predictions, start_idx=0, num_candles=100):
    """
    Create candlestick chart with prediction signals
    
    Parameters:
    -----------
    df : DataFrame with OHLC data and Date
    predictions : Array of predictions (BUY/SELL/HOLD)
    start_idx : Starting index for visualization
    num_candles : Number of candles to display
    """
    # Select subset of data
    end_idx = min(start_idx + num_candles, len(df))
    df_subset = df.iloc[start_idx:end_idx].copy()
    pred_subset = predictions[start_idx:end_idx]
    
    # Create figure
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(20, 12), 
                                     gridspec_kw={'height_ratios': [3, 1]},
                                     sharex=True)
    
    # Color mapping
    colors = []
    for i in range(len(df_subset)):
        if df_subset.iloc[i]['Close'] >= df_subset.iloc[i]['Open']:
            colors.append('green')  # Bullish candle
        else:
            colors.append('red')    # Bearish candle
    
    # Plot candlesticks
    width = 0.6
    for i in range(len(df_subset)):
        row = df_subset.iloc[i]
        
        # High-Low line (wick)
        ax1.plot([i, i], [row['Low'], row['High']], 
                color='black', linewidth=1, zorder=1)
        
        # Open-Close rectangle (body)
        height = abs(row['Close'] - row['Open'])
        bottom = min(row['Open'], row['Close'])
        
        rect = Rectangle((i - width/2, bottom), width, height,
                         facecolor=colors[i], edgecolor='black',
                         linewidth=1, alpha=0.8, zorder=2)
        ax1.add_patch(rect)
    
    # Overlay prediction signals
    buy_signals = []
    hold_signals = []
    sell_signals = []
    
    for i in range(len(pred_subset)):
        if pred_subset[i] == 'BUY':
            buy_signals.append((i, df_subset.iloc[i]['Low'] * 0.995))
        elif pred_subset[i] == 'HOLD':
            hold_signals.append((i, df_subset.iloc[i]['Low'] * 0.995))
        elif pred_subset[i] == 'SELL':
            sell_signals.append((i, df_subset.iloc[i]['High'] * 1.005))
    
    # Plot signals
    if buy_signals:
        buy_x, buy_y = zip(*buy_signals)
        ax1.scatter(buy_x, buy_y, marker='^', s=150, color='lime', 
                   edgecolors='darkgreen', linewidth=1.5, 
                   label='BUY Signal', zorder=3, alpha=0.9)
    
    if hold_signals:
        hold_x, hold_y = zip(*hold_signals)
        ax1.scatter(hold_x, hold_y, marker='o', s=100, color='yellow', 
                   edgecolors='orange', linewidth=1.5, 
                   label='HOLD Signal', zorder=3, alpha=0.7)
    
    if sell_signals:
        sell_x, sell_y = zip(*sell_signals)
        ax1.scatter(sell_x, sell_y, marker='v', s=150, color='red', 
                   edgecolors='darkred', linewidth=1.5, 
                   label='SELL Signal', zorder=3, alpha=0.9)
    
    # Formatting for main chart
    ax1.set_ylabel('Price (₹)', fontsize=14, fontweight='bold')
    ax1.set_title('Stock Price Candlestick Chart with ML Predictions\n(Random Forest Classifier)', 
                 fontsize=16, fontweight='bold', pad=20)
    ax1.grid(True, alpha=0.3, linestyle='--')
    ax1.legend(loc='upper left', fontsize=11, framealpha=0.9)
    
    # Format y-axis
    y_min = df_subset['Low'].min() * 0.99
    y_max = df_subset['High'].max() * 1.01
    ax1.set_ylim(y_min, y_max)
    
    # Add moving averages if available
    if 'SMA_10' in df_subset.columns:
        ax1.plot(range(len(df_subset)), df_subset['SMA_10'], 
                label='SMA 10', color='blue', linewidth=2, alpha=0.7, zorder=1)
    if 'SMA_50' in df_subset.columns:
        ax1.plot(range(len(df_subset)), df_subset['SMA_50'], 
                label='SMA 50', color='orange', linewidth=2, alpha=0.7, zorder=1)
    
    # Volume bar chart
    volume_colors = ['green' if c == 'green' else 'red' for c in colors]
    ax2.bar(range(len(df_subset)), df_subset['Volume'], 
           color=volume_colors, alpha=0.6, width=0.8)
    ax2.set_ylabel('Volume', fontsize=12, fontweight='bold')
    ax2.set_xlabel('Trading Days', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3, linestyle='--', axis='y')
    
    # X-axis labels with dates
    date_labels = df_subset['Date'].dt.strftime('%Y-%m-%d').tolist()
    tick_positions = range(0, len(df_subset), max(1, len(df_subset)//10))
    ax2.set_xticks(tick_positions)
    ax2.set_xticklabels([date_labels[i] for i in tick_positions], 
                        rotation=45, ha='right')
    
    plt.tight_layout()
    return fig

def main():
    print("📊 CANDLESTICK CHART WITH ML PREDICTIONS")
    print("=" * 60)
    
    # Load the trained model
    print("\n🤖 Loading Random Forest model...")
    if not os.path.exists('models/best_model.joblib'):
        print("❌ Model not found. Please train the model first (run main.py)")
        return
    
    model = joblib.load('models/best_model.joblib')
    print(f"✅ Model loaded: {type(model).__name__}")
    print(f"   Trees: {model.n_estimators}")
    print(f"   Classes: {list(model.classes_)}")
    
    # Load the labeled dataset
    print("\n📂 Loading labeled stock data...")
    if not os.path.exists('data/processed/labeled_stock_data.csv'):
        print("❌ Labeled data not found. Please run main.py first.")
        return
    
    data = pd.read_csv('data/processed/labeled_stock_data.csv')
    data['Date'] = pd.to_datetime(data['Date'])
    print(f"✅ Data loaded: {len(data):,} records")
    print(f"   Date range: {data['Date'].min().strftime('%Y-%m-%d')} to {data['Date'].max().strftime('%Y-%m-%d')}")
    
    # Get test set predictions
    print("\n🎯 Preparing predictions for visualization...")
    
    # Define features (same as in main.py)
    # Use exact feature names from the trained model
    expected_features = list(model.feature_names_in_)
    
    # Verify all features exist in the data
    missing_features = [f for f in expected_features if f not in data.columns]
    if missing_features:
        print(f"❌ Missing features: {missing_features}")
        return
    
    feature_cols = expected_features
    
    # Sort by date
    data_sorted = data.sort_values('Date').reset_index(drop=True)
    
    # Use same split as training (80/20)
    split_idx = int(len(data_sorted) * 0.8)
    
    # Get test data
    test_data = data_sorted.iloc[split_idx:].copy().reset_index(drop=True)
    
    # Make predictions on test set
    X_test = test_data[feature_cols]
    y_true = test_data['Label']
    y_pred = model.predict(X_test)
    
    print(f"✅ Generated {len(y_pred):,} predictions")
    print(f"\n📊 Prediction distribution:")
    unique, counts = np.unique(y_pred, return_counts=True)
    for label, count in zip(unique, counts):
        print(f"   {label}: {count:,} ({count/len(y_pred)*100:.1f}%)")
    
    # Create output directory
    os.makedirs('results/candlestick_charts', exist_ok=True)
    
    # Generate multiple charts for different time periods
    print("\n📈 Generating candlestick charts...")
    
    num_charts = 5
    samples_per_chart = 100
    step = len(test_data) // (num_charts + 1)
    
    for i in range(num_charts):
        start_idx = i * step
        
        print(f"\n  Chart {i+1}/{num_charts}:")
        print(f"    Period: {test_data.iloc[start_idx]['Date'].strftime('%Y-%m-%d')} to "
              f"{test_data.iloc[min(start_idx + samples_per_chart, len(test_data)-1)]['Date'].strftime('%Y-%m-%d')}")
        
        # Create chart
        fig = plot_candlestick_with_predictions(
            test_data, y_pred, 
            start_idx=start_idx, 
            num_candles=samples_per_chart
        )
        
        # Save
        filename = f'results/candlestick_charts/candlestick_chart_{i+1}.png'
        fig.savefig(filename, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close(fig)
        
        print(f"    ✅ Saved: {filename}")
    
    # Create a comprehensive view with actual vs predicted
    print("\n📊 Creating accuracy comparison chart...")
    
    fig, ax = plt.subplots(figsize=(16, 8))
    
    # Sample for cleaner visualization
    sample_size = min(200, len(test_data))
    sample_indices = np.linspace(0, len(test_data)-1, sample_size, dtype=int)
    
    dates = test_data.iloc[sample_indices]['Date']
    actual = y_true.iloc[sample_indices]
    predicted = y_pred[sample_indices]
    
    # Convert labels to numeric for plotting
    label_map = {'BUY': 1, 'HOLD': 0, 'SELL': -1}
    actual_numeric = [label_map[label] for label in actual]
    predicted_numeric = [label_map[label] for label in predicted]
    
    # Plot
    ax.plot(dates, actual_numeric, 'o-', label='Actual Label', 
           color='blue', linewidth=2, markersize=4, alpha=0.7)
    ax.plot(dates, predicted_numeric, 's--', label='Predicted Label', 
           color='red', linewidth=2, markersize=4, alpha=0.7)
    
    # Formatting
    ax.set_yticks([-1, 0, 1])
    ax.set_yticklabels(['SELL', 'HOLD', 'BUY'])
    ax.set_xlabel('Date', fontsize=12, fontweight='bold')
    ax.set_ylabel('Signal', fontsize=12, fontweight='bold')
    ax.set_title('Actual vs Predicted Trading Signals\n(Random Forest Classifier)', 
                fontsize=14, fontweight='bold')
    ax.legend(loc='upper left', fontsize=11)
    ax.grid(True, alpha=0.3, linestyle='--')
    
    # Rotate x-axis labels
    plt.xticks(rotation=45, ha='right')
    
    plt.tight_layout()
    filename = 'results/candlestick_charts/actual_vs_predicted_signals.png'
    fig.savefig(filename, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"  ✅ Saved: {filename}")
    
    # Calculate and display accuracy
    print("\n✅ VISUALIZATION COMPLETE!")
    print("=" * 60)
    print(f"\n📁 Charts saved to: results/candlestick_charts/")
    print(f"   • {num_charts} candlestick charts with ML predictions")
    print(f"   • 1 actual vs predicted comparison chart")
    
    # Calculate accuracy
    correct = sum(1 for a, p in zip(y_true, y_pred) if a == p)
    accuracy = correct / len(y_true)
    print(f"\n🎯 Overall Test Accuracy: {accuracy*100:.2f}%")
    print(f"   Correct predictions: {correct:,}/{len(y_true):,}")
    
    print("\n💡 TIP: Open the PNG files to view the candlestick charts!")
    print("    Each chart shows 100 trading days with BUY/SELL/HOLD signals")

if __name__ == "__main__":
    main()
