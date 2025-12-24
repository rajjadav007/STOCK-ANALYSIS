#!/usr/bin/env python3
"""
Stock Prediction Candlestick Visualizer
========================================
Shows stock predictions on a candlestick chart with BUY/SELL/HOLD signals

Features:
- Interactive candlestick chart
- BUY signals marked in GREEN
- SELL signals marked in RED
- HOLD signals marked in YELLOW
- Shows last prediction
- Volume bars
- Moving averages

Author: Stock Analysis Team
Date: December 24, 2025
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.lines import Line2D
import joblib
import os
import sys
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Fix encoding
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')


def load_stock_data(stock_symbol, num_days=100):
    """Load stock data"""
    print("="*70)
    print(f"CANDLESTICK CHART: {stock_symbol}")
    print("="*70)
    
    print(f"\n[1/5] Loading {stock_symbol} data...")
    
    file_path = f"data/raw/{stock_symbol}.csv"
    
    if not os.path.exists(file_path):
        print(f"\nERROR: Stock file not found: {file_path}")
        print("\nAvailable stocks:")
        if os.path.exists('data/raw'):
            stocks = [f.replace('.csv', '') for f in os.listdir('data/raw') if f.endswith('.csv')]
            for i, stock in enumerate(stocks[:20], 1):
                print(f"  {i}. {stock}")
        return None
    
    df = pd.read_csv(file_path)
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values('Date')
    
    # Get last N days + buffer for indicators
    df = df.tail(num_days + 60).reset_index(drop=True)
    
    print(f"  Loaded {len(df)} days")
    print(f"  Range: {df['Date'].min().strftime('%Y-%m-%d')} to {df['Date'].max().strftime('%Y-%m-%d')}")
    print(f"  Latest price: Rs.{df.iloc[-1]['Close']:.2f}")
    
    return df


def calculate_indicators(df):
    """Calculate technical indicators"""
    print("\n[2/5] Calculating indicators...")
    
    stock_df = df.copy()
    
    # Basic features
    stock_df['Price_Change'] = stock_df['Close'] - stock_df['Open']
    stock_df['Price_Range'] = stock_df['High'] - stock_df['Low']
    stock_df['Returns'] = stock_df['Close'].pct_change()
    
    # Moving Averages
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
    
    # Advanced indicators
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
            prev_close = stock_df.loc[idx - 1, 'Close']
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
    
    # Remove NaN
    stock_df = stock_df.dropna().reset_index(drop=True)
    
    print(f"  Calculated {stock_df.shape[1]} features")
    print(f"  Valid data points: {len(stock_df)}")
    
    return stock_df


def make_predictions(df):
    """Make BUY/SELL/HOLD predictions"""
    print("\n[3/5] Making predictions...")
    
    # Load model (use old one with basic features)
    model_path = 'models/final_production_model.joblib'
    
    model = joblib.load(model_path)
    print(f"  Model: {type(model).__name__}")
    
    # Basic features only (20 features that old model was trained on)
    feature_cols = [
        'Open', 'High', 'Low', 'Volume', 'Price_Change', 'Price_Range',
        'Returns', 'SMA_10', 'SMA_50', 'EMA_12', 'EMA_26', 'RSI_14',
        'MACD', 'MACD_signal', 'MACD_hist', 'Volatility_10', 'Volatility_20',
        'Volume_Change_Pct', 'Close_lag_1', 'Volume_lag_1'
    ]
    
    # Use only available features
    available_features = [f for f in feature_cols if f in df.columns]
    print(f"  Using {len(available_features)} features")
    X = df[available_features]
    
    # Make predictions
    predictions = model.predict(X)
    probabilities = model.predict_proba(X)
    
    df['Prediction'] = predictions
    df['BUY_Prob'] = probabilities[:, 0] * 100
    df['HOLD_Prob'] = probabilities[:, 1] * 100
    df['SELL_Prob'] = probabilities[:, 2] * 100
    df['Confidence'] = probabilities.max(axis=1) * 100
    
    print(f"  Predictions made: {len(df)}")
    
    # Count signals
    pred_counts = df['Prediction'].value_counts()
    print(f"\n  Signals:")
    print(f"    BUY:  {pred_counts.get('BUY', 0)} ({pred_counts.get('BUY', 0)/len(df)*100:.1f}%)")
    print(f"    HOLD: {pred_counts.get('HOLD', 0)} ({pred_counts.get('HOLD', 0)/len(df)*100:.1f}%)")
    print(f"    SELL: {pred_counts.get('SELL', 0)} ({pred_counts.get('SELL', 0)/len(df)*100:.1f}%)")
    
    return df


def plot_candlestick_chart(df, stock_symbol):
    """Create candlestick chart with predictions"""
    print("\n[4/5] Creating candlestick chart...")
    
    # Create figure
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 10), 
                                    gridspec_kw={'height_ratios': [3, 1]})
    
    # Prepare data
    data = df.copy()
    data['Date_num'] = np.arange(len(data))
    
    # Plot candlesticks
    for idx, row in data.iterrows():
        # Determine color
        if row['Close'] >= row['Open']:
            color = '#26a69a'  # Green
            body_color = '#26a69a'
        else:
            color = '#ef5350'  # Red
            body_color = '#ef5350'
        
        # Draw high-low line
        ax1.plot([row['Date_num'], row['Date_num']], 
                [row['Low'], row['High']], 
                color=color, linewidth=0.8, alpha=0.8)
        
        # Draw open-close body
        height = abs(row['Close'] - row['Open'])
        bottom = min(row['Open'], row['Close'])
        
        rect = Rectangle((row['Date_num']-0.3, bottom), 0.6, height,
                        facecolor=body_color, edgecolor=color, 
                        linewidth=1, alpha=0.9)
        ax1.add_patch(rect)
    
    # Plot moving averages
    ax1.plot(data['Date_num'], data['SMA_10'], 
            label='SMA 10', color='blue', linewidth=1, alpha=0.7)
    ax1.plot(data['Date_num'], data['SMA_50'], 
            label='SMA 50', color='orange', linewidth=1, alpha=0.7)
    
    # Mark BUY/SELL/HOLD signals
    buy_signals = data[data['Prediction'] == 'BUY']
    sell_signals = data[data['Prediction'] == 'SELL']
    hold_signals = data[data['Prediction'] == 'HOLD']
    
    # Plot signals
    if len(buy_signals) > 0:
        ax1.scatter(buy_signals['Date_num'], buy_signals['Low'] * 0.995, 
                   marker='^', color='green', s=100, alpha=0.8, 
                   label='BUY Signal', zorder=5)
    
    if len(sell_signals) > 0:
        ax1.scatter(sell_signals['Date_num'], sell_signals['High'] * 1.005, 
                   marker='v', color='red', s=100, alpha=0.8, 
                   label='SELL Signal', zorder=5)
    
    if len(hold_signals) > 0:
        ax1.scatter(hold_signals['Date_num'], hold_signals['Close'], 
                   marker='o', color='yellow', s=30, alpha=0.5, 
                   label='HOLD Signal', zorder=4, edgecolors='orange')
    
    # Highlight last prediction
    last_row = data.iloc[-1]
    last_color = {'BUY': 'green', 'SELL': 'red', 'HOLD': 'orange'}[last_row['Prediction']]
    
    ax1.scatter(last_row['Date_num'], last_row['Close'], 
               marker='*', color=last_color, s=500, alpha=0.9, 
               zorder=6, edgecolors='black', linewidths=2)
    
    # Annotate last prediction
    ax1.annotate(f"Last Predicted\n{last_row['Prediction']}\nConfidence: {last_row['Confidence']:.1f}%",
                xy=(last_row['Date_num'], last_row['Close']),
                xytext=(last_row['Date_num'] - len(data)*0.15, last_row['Close']),
                fontsize=11, fontweight='bold', color=last_color,
                bbox=dict(boxstyle='round,pad=0.5', facecolor='white', 
                         edgecolor=last_color, linewidth=2),
                arrowprops=dict(arrowstyle='->', color=last_color, lw=2))
    
    # Format price axis
    ax1.set_ylabel('Price (Rs.)', fontsize=12, fontweight='bold')
    ax1.set_title(f'{stock_symbol} - Stock Price with BUY/SELL/HOLD Predictions', 
                 fontsize=14, fontweight='bold', pad=20)
    ax1.legend(loc='upper left', fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    # Format x-axis with dates
    tick_spacing = max(1, len(data) // 10)
    tick_positions = data['Date_num'][::tick_spacing]
    tick_labels = data['Date'].dt.strftime('%Y-%m-%d')[::tick_spacing]
    ax1.set_xticks(tick_positions)
    ax1.set_xticklabels(tick_labels, rotation=45, ha='right')
    
    # Volume chart
    volume_colors = ['#26a69a' if data.loc[i, 'Close'] >= data.loc[i, 'Open'] 
                    else '#ef5350' for i in data.index]
    
    ax2.bar(data['Date_num'], data['Volume'], color=volume_colors, alpha=0.6)
    ax2.set_ylabel('Volume', fontsize=12, fontweight='bold')
    ax2.set_xlabel('Date', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='y')
    
    # Format volume x-axis
    ax2.set_xticks(tick_positions)
    ax2.set_xticklabels(tick_labels, rotation=45, ha='right')
    
    plt.tight_layout()
    
    # Save chart
    os.makedirs('results', exist_ok=True)
    filename = f'results/candlestick_{stock_symbol}_{datetime.now().strftime("%Y%m%d_%H%M%S")}.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"  Chart saved: {filename}")
    
    plt.show()
    
    return filename


def show_prediction_summary(df, stock_symbol):
    """Show prediction summary"""
    print("\n[5/5] Prediction Summary")
    print("="*70)
    
    latest = df.iloc[-1]
    
    print(f"\n{stock_symbol} - Latest Prediction:")
    print(f"  Date: {latest['Date'].strftime('%Y-%m-%d')}")
    print(f"  Close Price: Rs.{latest['Close']:.2f}")
    print(f"\n  PREDICTION: {latest['Prediction']}")
    print(f"  Confidence: {latest['Confidence']:.2f}%")
    print(f"\n  Probabilities:")
    print(f"    BUY:  {latest['BUY_Prob']:.2f}%  {'█' * int(latest['BUY_Prob']/5)}")
    print(f"    HOLD: {latest['HOLD_Prob']:.2f}%  {'█' * int(latest['HOLD_Prob']/5)}")
    print(f"    SELL: {latest['SELL_Prob']:.2f}%  {'█' * int(latest['SELL_Prob']/5)}")
    
    print(f"\n  Key Indicators:")
    print(f"    RSI-14: {latest['RSI_14']:.2f}", end='')
    if latest['RSI_14'] > 70:
        print(" (Overbought)")
    elif latest['RSI_14'] < 30:
        print(" (Oversold)")
    else:
        print(" (Neutral)")
    
    print(f"    MACD: {latest['MACD']:.2f}", end='')
    if latest['MACD'] > latest['MACD_signal']:
        print(" (Bullish)")
    else:
        print(" (Bearish)")
    
    print(f"    SMA-10: Rs.{latest['SMA_10']:.2f}")
    print(f"    SMA-50: Rs.{latest['SMA_50']:.2f}")
    
    # Action recommendation
    print(f"\n  RECOMMENDATION:")
    if latest['Prediction'] == 'BUY' and latest['Confidence'] > 60:
        print(f"    Strong BUY - Consider opening a long position")
        print(f"    Target: Rs.{latest['Close'] * 1.05:.2f} (+5%)")
        print(f"    Stop Loss: Rs.{latest['Close'] * 0.98:.2f} (-2%)")
    elif latest['Prediction'] == 'SELL' and latest['Confidence'] > 60:
        print(f"    Strong SELL - Consider closing positions")
        print(f"    Avoid buying at current level")
    elif latest['Prediction'] == 'HOLD':
        print(f"    HOLD - Maintain current positions")
        print(f"    Wait for clearer signal")
    else:
        print(f"    Low confidence - Monitor the market")
    
    print(f"\n{'='*70}")


def main():
    """Main execution"""
    # Available stocks
    print("\nAvailable stocks in data/raw/:")
    stocks = []
    if os.path.exists('data/raw'):
        stocks = sorted([f.replace('.csv', '') for f in os.listdir('data/raw') if f.endswith('.csv')])
        for i, stock in enumerate(stocks[:20], 1):
            print(f"  {i}. {stock}")
    
    # Select stock
    print(f"\n" + "="*70)
    stock_symbol = input("Enter stock symbol (e.g., RELIANCE, TCS, INFY): ").strip().upper()
    
    if not stock_symbol:
        stock_symbol = 'RELIANCE'  # Default
        print(f"Using default: {stock_symbol}")
    
    # Load data
    df = load_stock_data(stock_symbol, num_days=60)
    if df is None:
        return
    
    # Calculate indicators
    df = calculate_indicators(df)
    
    # Make predictions
    df = make_predictions(df)
    
    # Show summary
    show_prediction_summary(df, stock_symbol)
    
    # Plot chart
    plot_candlestick_chart(df, stock_symbol)
    
    print(f"\nDONE! Candlestick chart created successfully!")


if __name__ == "__main__":
    main()
