"""
Stock Price Prediction System
==============================
Uses trained ML model to predict BUY/SELL/HOLD signals for stocks.

Tasks:
1. Load saved model
2. Load latest stock data
3. Apply SAME feature engineering
4. Predict BUY / SELL / HOLD
5. Apply probability threshold logic

Author: Stock Analysis Team
Date: December 23, 2025
"""

import joblib
import pandas as pd
import numpy as np
import json
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')


class StockPredictor:
    """Production-ready stock prediction system."""
    
    def __init__(self, model_path='models/final_production_model.joblib'):
        """Initialize predictor with trained model."""
        print("="*70)
        print("STOCK PRICE PREDICTION SYSTEM")
        print("="*70)
        
        # Task 1: Load saved model
        print("\n[TASK 1] Loading saved model...")
        self.model = joblib.load(model_path)
        print(f"✅ Model loaded: {model_path}")
        print(f"   Algorithm: {type(self.model).__name__}")
        print(f"   Classes: {self.model.classes_}")
        
        # Load metadata
        try:
            with open('models/final_model_metadata.json', 'r') as f:
                self.metadata = json.load(f)
                print(f"✅ Metadata loaded")
                print(f"   Test Accuracy: {self.metadata['performance_metrics']['test_accuracy']:.2f}%")
                print(f"   Overfitting Gap: {self.metadata['performance_metrics']['overfitting_gap']:.2f}%")
        except:
            self.metadata = None
            print("⚠️  Metadata not found")
    
    def load_stock_data(self, stock_symbol='RELIANCE', num_days=100):
        """
        Task 2: Load latest stock data.
        
        Parameters:
        - stock_symbol: Stock ticker (e.g., 'RELIANCE', 'TCS')
        - num_days: Number of recent days to load
        """
        print(f"\n[TASK 2] Loading latest stock data...")
        
        file_path = f"data/raw/{stock_symbol}.csv"
        
        try:
            df = pd.read_csv(file_path)
            df['Date'] = pd.to_datetime(df['Date'])
            df = df.sort_values('Date')
            
            # Get latest N days (need extra for indicators)
            df = df.tail(num_days + 60).reset_index(drop=True)
            
            print(f"✅ Loaded {len(df)} days of {stock_symbol} data")
            print(f"   Date Range: {df['Date'].min().strftime('%Y-%m-%d')} to {df['Date'].max().strftime('%Y-%m-%d')}")
            print(f"   Latest Close: ₹{df.iloc[-1]['Close']:.2f}")
            
            return df, stock_symbol
            
        except FileNotFoundError:
            print(f"❌ Error: Stock data not found - {file_path}")
            print(f"   Available stocks in data/raw/:")
            import os
            if os.path.exists('data/raw'):
                stocks = [f.replace('.csv', '') for f in os.listdir('data/raw') if f.endswith('.csv')]
                print(f"   {', '.join(stocks[:10])}...")
            return None, None
    
    def apply_feature_engineering(self, df):
        """
        Task 3: Apply SAME feature engineering as training.
        
        Calculates all 20 technical indicators:
        - Price features (7): Open, High, Low, Volume, Price_Change, Price_Range, Returns
        - Moving Averages (4): SMA_10, SMA_50, EMA_12, EMA_26
        - Momentum (1): RSI_14
        - Trend (3): MACD, MACD_signal, MACD_hist
        - Volatility (2): Volatility_10, Volatility_20
        - Volume (1): Volume_Change_Pct
        - Lagged (2): Close_lag_1, Volume_lag_1
        """
        print(f"\n[TASK 3] Applying feature engineering...")
        
        data = df.copy()
        
        # 1. Price-based features
        data['Price_Change'] = data['Close'].diff()
        data['Price_Range'] = data['High'] - data['Low']
        data['Returns'] = data['Close'].pct_change() * 100
        
        # 2. Moving Averages
        data['SMA_10'] = data['Close'].rolling(window=10).mean()
        data['SMA_50'] = data['Close'].rolling(window=50).mean()
        
        # 3. Exponential Moving Averages
        data['EMA_12'] = data['Close'].ewm(span=12, adjust=False).mean()
        data['EMA_26'] = data['Close'].ewm(span=26, adjust=False).mean()
        
        # 4. RSI (Relative Strength Index)
        delta = data['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        data['RSI_14'] = 100 - (100 / (1 + rs))
        
        # 5. MACD (Moving Average Convergence Divergence)
        data['MACD'] = data['EMA_12'] - data['EMA_26']
        data['MACD_signal'] = data['MACD'].ewm(span=9, adjust=False).mean()
        data['MACD_hist'] = data['MACD'] - data['MACD_signal']
        
        # 6. Volatility
        data['Volatility_10'] = data['Returns'].rolling(window=10).std()
        data['Volatility_20'] = data['Returns'].rolling(window=20).std()
        
        # 7. Volume features
        data['Volume_Change_Pct'] = data['Volume'].pct_change() * 100
        
        # 8. Lagged features
        data['Close_lag_1'] = data['Close'].shift(1)
        data['Volume_lag_1'] = data['Volume'].shift(1)
        
        # Remove NaN rows
        initial_rows = len(data)
        data = data.dropna().reset_index(drop=True)
        final_rows = len(data)
        
        print(f"✅ Calculated 20 technical indicators")
        print(f"   Rows: {initial_rows} → {final_rows} (removed {initial_rows - final_rows} NaN rows)")
        print(f"   Features: Price, SMA, EMA, RSI, MACD, Volatility, Volume")
        
        return data
    
    def predict(self, df):
        """
        Task 4: Predict BUY / SELL / HOLD signals.
        
        Returns DataFrame with predictions and probabilities.
        """
        print(f"\n[TASK 4] Making predictions...")
        
        # Feature columns (same order as training)
        feature_columns = [
            'Open', 'High', 'Low', 'Volume', 'Price_Change', 'Price_Range',
            'Returns', 'SMA_10', 'SMA_50', 'EMA_12', 'EMA_26', 'RSI_14',
            'MACD', 'MACD_signal', 'MACD_hist', 'Volatility_10', 'Volatility_20',
            'Volume_Change_Pct', 'Close_lag_1', 'Volume_lag_1'
        ]
        
        # Prepare features
        X = df[feature_columns]
        
        # Make predictions
        predictions = self.model.predict(X)
        probabilities = self.model.predict_proba(X)
        
        # Add to dataframe
        df['Prediction'] = predictions
        df['BUY_Prob'] = probabilities[:, 0] * 100
        df['HOLD_Prob'] = probabilities[:, 1] * 100
        df['SELL_Prob'] = probabilities[:, 2] * 100
        df['Confidence'] = probabilities.max(axis=1) * 100
        
        print(f"✅ Predictions complete for {len(df)} days")
        
        # Show prediction distribution
        pred_counts = df['Prediction'].value_counts()
        print(f"\n   Prediction Distribution:")
        for signal in ['BUY', 'HOLD', 'SELL']:
            count = pred_counts.get(signal, 0)
            pct = (count / len(df)) * 100 if len(df) > 0 else 0
            print(f"     {signal:<6}: {count:>4} ({pct:>5.1f}%)")
        
        return df
    
    def apply_threshold_logic(self, df, 
                             min_confidence=35.0,
                             buy_threshold=40.0,
                             sell_threshold=40.0):
        """
        Task 5: Apply probability threshold logic.
        
        Parameters:
        - min_confidence: Minimum confidence to act (default: 35%)
        - buy_threshold: Minimum BUY probability (default: 40%)
        - sell_threshold: Minimum SELL probability (default: 40%)
        
        Returns:
        - Filtered signals with actionable trades only
        """
        print(f"\n[TASK 5] Applying probability threshold logic...")
        print(f"   Min Confidence: {min_confidence:.1f}%")
        print(f"   BUY Threshold: {buy_threshold:.1f}%")
        print(f"   SELL Threshold: {sell_threshold:.1f}%")
        
        # Create filtered signal column
        df['Action'] = 'NO_ACTION'
        
        # Apply thresholds
        buy_mask = (df['Prediction'] == 'BUY') & (df['Confidence'] >= min_confidence) & (df['BUY_Prob'] >= buy_threshold)
        sell_mask = (df['Prediction'] == 'SELL') & (df['Confidence'] >= min_confidence) & (df['SELL_Prob'] >= sell_threshold)
        hold_mask = (df['Prediction'] == 'HOLD') & (df['Confidence'] >= min_confidence)
        
        df.loc[buy_mask, 'Action'] = 'BUY'
        df.loc[sell_mask, 'Action'] = 'SELL'
        df.loc[hold_mask, 'Action'] = 'HOLD'
        
        # Count actionable signals
        action_counts = df['Action'].value_counts()
        
        print(f"\n✅ Threshold logic applied")
        print(f"\n   Actionable Signals:")
        for action in ['BUY', 'SELL', 'HOLD', 'NO_ACTION']:
            count = action_counts.get(action, 0)
            pct = (count / len(df)) * 100 if len(df) > 0 else 0
            symbol = "🟢" if action == 'BUY' else "🔴" if action == 'SELL' else "🟡" if action == 'HOLD' else "⚪"
            print(f"     {symbol} {action:<10}: {count:>4} ({pct:>5.1f}%)")
        
        return df
    
    def get_latest_signal(self, df):
        """Get the most recent trading signal."""
        latest = df.iloc[-1]
        
        print(f"\n{'='*70}")
        print("LATEST TRADING SIGNAL")
        print(f"{'='*70}")
        
        print(f"\n📅 Date: {latest['Date'].strftime('%Y-%m-%d')}")
        print(f"💰 Close Price: ₹{latest['Close']:.2f}")
        
        print(f"\n🎯 MODEL PREDICTION:")
        print(f"   Signal: {latest['Prediction']}")
        print(f"   Confidence: {latest['Confidence']:.2f}%")
        
        print(f"\n📊 PROBABILITIES:")
        print(f"   BUY:  {latest['BUY_Prob']:>6.2f}%  {'█' * int(latest['BUY_Prob']/5)}")
        print(f"   HOLD: {latest['HOLD_Prob']:>6.2f}%  {'█' * int(latest['HOLD_Prob']/5)}")
        print(f"   SELL: {latest['SELL_Prob']:>6.2f}%  {'█' * int(latest['SELL_Prob']/5)}")
        
        print(f"\n✅ ACTIONABLE SIGNAL: {latest['Action']}")
        
        # Interpretation
        print(f"\n💡 INTERPRETATION:")
        
        if latest['Action'] == 'BUY':
            print(f"   🟢 STRONG BUY SIGNAL")
            print(f"   • Consider opening a long position")
            print(f"   • Set stop-loss at ₹{latest['Close'] * 0.98:.2f} (-2%)")
            print(f"   • Target profit at ₹{latest['Close'] * 1.05:.2f} (+5%)")
            
        elif latest['Action'] == 'SELL':
            print(f"   🔴 STRONG SELL SIGNAL")
            print(f"   • Consider closing long positions")
            print(f"   • Avoid buying at current level")
            print(f"   • Wait for price stabilization")
            
        elif latest['Action'] == 'HOLD':
            print(f"   🟡 HOLD POSITION")
            print(f"   • Maintain current positions")
            print(f"   • Monitor price action")
            print(f"   • Wait for clearer signal")
            
        else:  # NO_ACTION
            print(f"   ⚪ NO ACTION")
            print(f"   • Confidence too low ({latest['Confidence']:.2f}%)")
            print(f"   • Wait for higher conviction signal")
            print(f"   • Monitor market conditions")
        
        # Technical indicators
        print(f"\n📈 KEY INDICATORS:")
        print(f"   RSI-14: {latest['RSI_14']:.2f} {'(Overbought)' if latest['RSI_14'] > 70 else '(Oversold)' if latest['RSI_14'] < 30 else '(Neutral)'}")
        print(f"   MACD: {latest['MACD']:.2f} {'(Bullish)' if latest['MACD'] > latest['MACD_signal'] else '(Bearish)'}")
        print(f"   SMA-10: ₹{latest['SMA_10']:.2f} | SMA-50: ₹{latest['SMA_50']:.2f}")
        print(f"   Volatility (20d): {latest['Volatility_20']:.2f}%")
        
        return latest
    
    def show_recent_signals(self, df, num_days=10):
        """Show recent trading signals."""
        print(f"\n{'='*70}")
        print(f"RECENT {num_days}-DAY SIGNALS")
        print(f"{'='*70}")
        
        recent = df.tail(num_days)
        
        print(f"\n{'Date':<12} {'Close':<10} {'Prediction':<8} {'Confidence':<12} {'Action':<12}")
        print("-" * 70)
        
        for _, row in recent.iterrows():
            action_symbol = "🟢" if row['Action'] == 'BUY' else "🔴" if row['Action'] == 'SELL' else "🟡" if row['Action'] == 'HOLD' else "⚪"
            print(f"{row['Date'].strftime('%Y-%m-%d'):<12} "
                  f"₹{row['Close']:<9.2f} "
                  f"{row['Prediction']:<8} "
                  f"{row['Confidence']:<11.2f}% "
                  f"{action_symbol} {row['Action']:<10}")
    
    def save_predictions(self, df, stock_symbol):
        """Save predictions to CSV."""
        filename = f"results/predictions_{stock_symbol}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        
        # Select relevant columns
        output_columns = [
            'Date', 'Open', 'High', 'Low', 'Close', 'Volume',
            'RSI_14', 'MACD', 'SMA_10', 'SMA_50',
            'Prediction', 'BUY_Prob', 'HOLD_Prob', 'SELL_Prob', 
            'Confidence', 'Action'
        ]
        
        df[output_columns].to_csv(filename, index=False)
        print(f"\n💾 Predictions saved: {filename}")
        
        return filename
    
    def run(self, stock_symbol='RELIANCE', 
            num_days=100,
            min_confidence=35.0,
            buy_threshold=40.0,
            sell_threshold=40.0,
            show_recent=10,
            save_results=True):
        """
        Run complete prediction pipeline.
        
        Parameters:
        - stock_symbol: Stock to predict (default: 'RELIANCE')
        - num_days: Number of recent days to analyze (default: 100)
        - min_confidence: Minimum confidence threshold (default: 35%)
        - buy_threshold: BUY probability threshold (default: 40%)
        - sell_threshold: SELL probability threshold (default: 40%)
        - show_recent: Number of recent signals to display (default: 10)
        - save_results: Save predictions to CSV (default: True)
        """
        
        # Task 2: Load stock data
        df, symbol = self.load_stock_data(stock_symbol, num_days)
        if df is None:
            return None
        
        # Task 3: Feature engineering
        df = self.apply_feature_engineering(df)
        
        # Task 4: Make predictions
        df = self.predict(df)
        
        # Task 5: Apply thresholds
        df = self.apply_threshold_logic(df, min_confidence, buy_threshold, sell_threshold)
        
        # Get latest signal
        latest = self.get_latest_signal(df)
        
        # Show recent signals
        if show_recent > 0:
            self.show_recent_signals(df, show_recent)
        
        # Save results
        if save_results:
            self.save_predictions(df, symbol)
        
        print(f"\n{'='*70}")
        print("PREDICTION COMPLETE!")
        print(f"{'='*70}")
        
        print(f"\n⚠️  IMPORTANT REMINDERS:")
        print(f"   • Model accuracy: 35.72% (stock prediction is difficult)")
        print(f"   • Current confidence: {latest['Confidence']:.2f}%")
        print(f"   • Always use stop-loss orders")
        print(f"   • This is a tool, not financial advice")
        print(f"   • Do your own research before trading")
        
        return df


def main():
    """Main execution function with examples."""
    
    # Initialize predictor
    predictor = StockPredictor()
    
    print("\n" + "="*70)
    print("EXAMPLE 1: RELIANCE Stock Prediction")
    print("="*70)
    
    # Run prediction on RELIANCE
    df_reliance = predictor.run(
        stock_symbol='RELIANCE',
        num_days=100,
        min_confidence=30.0,      # Lowered to get some signals
        buy_threshold=35.0,       # Lowered to get some BUY signals
        sell_threshold=35.0,      # Lowered to get some SELL signals
        show_recent=10,
        save_results=True
    )
    
    # Example 2: Different stock with stricter thresholds
    print("\n\n" + "="*70)
    print("EXAMPLE 2: TCS Stock Prediction (Stricter Thresholds)")
    print("="*70)
    
    df_tcs = predictor.run(
        stock_symbol='TCS',
        num_days=50,
        min_confidence=40.0,      # Higher confidence required
        buy_threshold=45.0,       # Stricter BUY threshold
        sell_threshold=45.0,      # Stricter SELL threshold
        show_recent=5,
        save_results=True
    )
    
    print("\n\n" + "="*70)
    print("ALL PREDICTIONS COMPLETE!")
    print("="*70)
    
    print("\n📊 Available stocks for prediction:")
    import os
    if os.path.exists('data/raw'):
        stocks = sorted([f.replace('.csv', '') for f in os.listdir('data/raw') if f.endswith('.csv')])
        for i in range(0, len(stocks), 10):
            print(f"   {', '.join(stocks[i:i+10])}")


if __name__ == "__main__":
    main()
