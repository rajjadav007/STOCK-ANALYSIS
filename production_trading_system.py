"""
Production Trading System with Backtesting
===========================================
Complete system for:
1. Loading actual stock data
2. Calculating 20 technical indicators
3. Making predictions with production model
4. Implementing risk management
5. Backtesting trading strategies

Author: Stock Analysis Team
Date: December 23, 2025
"""

import joblib
import pandas as pd
import numpy as np
import json
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class ProductionTradingSystem:
    """Complete production trading system with backtesting."""
    
    def __init__(self, model_path='models/final_production_model.joblib'):
        """Initialize the trading system."""
        self.model = joblib.load(model_path)
        self.metadata = self._load_metadata()
        print("✅ Production model loaded successfully")
        
    def _load_metadata(self):
        """Load model metadata."""
        with open('models/final_model_metadata.json', 'r') as f:
            return json.load(f)
    
    def load_stock_data(self, stock_symbol='RELIANCE', data_path='data/raw'):
        """Load actual stock data from CSV file."""
        print(f"\n{'='*70}")
        print(f"STEP 1: LOADING STOCK DATA - {stock_symbol}")
        print(f"{'='*70}")
        
        file_path = f"{data_path}/{stock_symbol}.csv"
        
        try:
            df = pd.read_csv(file_path)
            df['Date'] = pd.to_datetime(df['Date'])
            df = df.sort_values('Date').reset_index(drop=True)
            
            print(f"✅ Loaded {len(df):,} records")
            print(f"📅 Date Range: {df['Date'].min()} to {df['Date'].max()}")
            print(f"💰 Price Range: ₹{df['Close'].min():.2f} to ₹{df['Close'].max():.2f}")
            
            return df
            
        except FileNotFoundError:
            print(f"❌ Error: File not found - {file_path}")
            return None
    
    def calculate_technical_indicators(self, df):
        """Calculate all 20 required technical indicators."""
        print(f"\n{'='*70}")
        print("STEP 2: CALCULATING 20 TECHNICAL INDICATORS")
        print(f"{'='*70}")
        
        data = df.copy()
        
        # Price-based features
        data['Price_Change'] = data['Close'].diff()
        data['Price_Range'] = data['High'] - data['Low']
        data['Returns'] = data['Close'].pct_change() * 100
        
        # Moving Averages
        data['SMA_10'] = data['Close'].rolling(window=10).mean()
        data['SMA_50'] = data['Close'].rolling(window=50).mean()
        
        # Exponential Moving Averages
        data['EMA_12'] = data['Close'].ewm(span=12, adjust=False).mean()
        data['EMA_26'] = data['Close'].ewm(span=26, adjust=False).mean()
        
        # RSI (Relative Strength Index)
        delta = data['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        data['RSI_14'] = 100 - (100 / (1 + rs))
        
        # MACD (Moving Average Convergence Divergence)
        data['MACD'] = data['EMA_12'] - data['EMA_26']
        data['MACD_signal'] = data['MACD'].ewm(span=9, adjust=False).mean()
        data['MACD_hist'] = data['MACD'] - data['MACD_signal']
        
        # Volatility
        data['Volatility_10'] = data['Returns'].rolling(window=10).std()
        data['Volatility_20'] = data['Returns'].rolling(window=20).std()
        
        # Volume features
        data['Volume_Change_Pct'] = data['Volume'].pct_change() * 100
        
        # Lagged features
        data['Close_lag_1'] = data['Close'].shift(1)
        data['Volume_lag_1'] = data['Volume'].shift(1)
        
        # Remove rows with NaN values
        initial_rows = len(data)
        data = data.dropna().reset_index(drop=True)
        final_rows = len(data)
        
        print(f"✅ All 20 indicators calculated")
        print(f"📊 Records: {initial_rows:,} → {final_rows:,} (after removing NaN)")
        print(f"📈 Indicators: SMA, EMA, RSI, MACD, Volatility, Volume")
        
        return data
    
    def prepare_features(self, df):
        """Prepare feature matrix for model prediction."""
        feature_columns = [
            'Open', 'High', 'Low', 'Volume', 'Price_Change', 'Price_Range',
            'Returns', 'SMA_10', 'SMA_50', 'EMA_12', 'EMA_26', 'RSI_14',
            'MACD', 'MACD_signal', 'MACD_hist', 'Volatility_10', 'Volatility_20',
            'Volume_Change_Pct', 'Close_lag_1', 'Volume_lag_1'
        ]
        
        return df[feature_columns]
    
    def make_predictions(self, df):
        """Make predictions using the production model."""
        print(f"\n{'='*70}")
        print("STEP 3: MAKING PREDICTIONS WITH PRODUCTION MODEL")
        print(f"{'='*70}")
        
        # Prepare features
        features = self.prepare_features(df)
        
        # Make predictions
        predictions = self.model.predict(features)
        probabilities = self.model.predict_proba(features)
        
        # Add to dataframe
        df['Prediction'] = predictions
        df['BUY_Prob'] = probabilities[:, 0]
        df['HOLD_Prob'] = probabilities[:, 1]
        df['SELL_Prob'] = probabilities[:, 2]
        df['Confidence'] = probabilities.max(axis=1)
        
        print(f"✅ Predictions complete for {len(df):,} records")
        print(f"\n📊 PREDICTION DISTRIBUTION:")
        pred_dist = df['Prediction'].value_counts()
        for signal, count in pred_dist.items():
            pct = (count / len(df)) * 100
            print(f"  {signal:>6}: {count:>6,} ({pct:>5.2f}%)")
        
        print(f"\n📈 AVERAGE CONFIDENCE:")
        for signal in ['BUY', 'HOLD', 'SELL']:
            if signal in pred_dist.index:
                avg_conf = df[df['Prediction'] == signal]['Confidence'].mean() * 100
                print(f"  {signal:>6}: {avg_conf:>5.2f}%")
        
        return df
    
    def implement_risk_management(self, initial_capital=100000, 
                                  stop_loss_pct=2.0, 
                                  take_profit_pct=5.0,
                                  position_size_pct=10.0,
                                  min_confidence=0.4):
        """
        Implement risk management rules.
        
        Parameters:
        - initial_capital: Starting capital in rupees
        - stop_loss_pct: Stop loss percentage (2.0 = 2%)
        - take_profit_pct: Take profit percentage (5.0 = 5%)
        - position_size_pct: Position size as % of capital (10.0 = 10%)
        - min_confidence: Minimum prediction confidence (0.4 = 40%)
        """
        print(f"\n{'='*70}")
        print("STEP 4: IMPLEMENTING RISK MANAGEMENT")
        print(f"{'='*70}")
        
        self.risk_params = {
            'initial_capital': initial_capital,
            'stop_loss_pct': stop_loss_pct,
            'take_profit_pct': take_profit_pct,
            'position_size_pct': position_size_pct,
            'min_confidence': min_confidence
        }
        
        print(f"\n💰 RISK PARAMETERS:")
        print(f"  Initial Capital:    ₹{initial_capital:,.0f}")
        print(f"  Stop Loss:          {stop_loss_pct}%")
        print(f"  Take Profit:        {take_profit_pct}%")
        print(f"  Position Size:      {position_size_pct}% of capital")
        print(f"  Min Confidence:     {min_confidence*100:.0f}%")
        
        # Calculate position sizing
        position_size = (initial_capital * position_size_pct) / 100
        print(f"\n📊 POSITION SIZING:")
        print(f"  Max Position Value: ₹{position_size:,.0f}")
        print(f"  Max Risk per Trade: ₹{position_size * stop_loss_pct / 100:,.0f}")
        
        return self.risk_params
    
    def backtest_strategy(self, df, risk_params):
        """
        Backtest the trading strategy with risk management.
        
        Returns portfolio performance metrics and trade history.
        """
        print(f"\n{'='*70}")
        print("STEP 5: BACKTESTING TRADING STRATEGY")
        print(f"{'='*70}")
        
        capital = risk_params['initial_capital']
        position_size_pct = risk_params['position_size_pct']
        stop_loss_pct = risk_params['stop_loss_pct']
        take_profit_pct = risk_params['take_profit_pct']
        min_confidence = risk_params['min_confidence']
        
        # Initialize tracking variables
        position = None  # Current position: None, 'LONG'
        entry_price = 0
        entry_date = None
        shares = 0
        trades = []
        portfolio_values = []
        
        print(f"\n🔄 Running backtest on {len(df):,} days...")
        
        for i in range(len(df)):
            row = df.iloc[i]
            current_price = row['Close']
            current_date = row['Date']
            prediction = row['Prediction']
            confidence = row['Confidence']
            
            # Track portfolio value
            if position == 'LONG':
                portfolio_value = capital + (shares * current_price)
            else:
                portfolio_value = capital
            
            portfolio_values.append({
                'Date': current_date,
                'Portfolio_Value': portfolio_value,
                'Position': position if position else 'NONE'
            })
            
            # Check if we're in a position
            if position == 'LONG':
                # Calculate P&L percentage
                pnl_pct = ((current_price - entry_price) / entry_price) * 100
                
                # Check stop loss
                if pnl_pct <= -stop_loss_pct:
                    # Stop loss hit
                    exit_value = shares * current_price
                    pnl = exit_value - (shares * entry_price)
                    capital += pnl
                    
                    trades.append({
                        'Entry_Date': entry_date,
                        'Entry_Price': entry_price,
                        'Exit_Date': current_date,
                        'Exit_Price': current_price,
                        'Shares': shares,
                        'PnL': pnl,
                        'PnL_Pct': pnl_pct,
                        'Exit_Reason': 'STOP_LOSS',
                        'Days_Held': (current_date - entry_date).days
                    })
                    
                    position = None
                    shares = 0
                
                # Check take profit
                elif pnl_pct >= take_profit_pct:
                    # Take profit hit
                    exit_value = shares * current_price
                    pnl = exit_value - (shares * entry_price)
                    capital += pnl
                    
                    trades.append({
                        'Entry_Date': entry_date,
                        'Entry_Price': entry_price,
                        'Exit_Date': current_date,
                        'Exit_Price': current_price,
                        'Shares': shares,
                        'PnL': pnl,
                        'PnL_Pct': pnl_pct,
                        'Exit_Reason': 'TAKE_PROFIT',
                        'Days_Held': (current_date - entry_date).days
                    })
                    
                    position = None
                    shares = 0
                
                # Check SELL signal
                elif prediction == 'SELL' and confidence >= min_confidence:
                    # Exit on SELL signal
                    exit_value = shares * current_price
                    pnl = exit_value - (shares * entry_price)
                    capital += pnl
                    
                    trades.append({
                        'Entry_Date': entry_date,
                        'Entry_Price': entry_price,
                        'Exit_Date': current_date,
                        'Exit_Price': current_price,
                        'Shares': shares,
                        'PnL': pnl,
                        'PnL_Pct': pnl_pct,
                        'Exit_Reason': 'SELL_SIGNAL',
                        'Days_Held': (current_date - entry_date).days
                    })
                    
                    position = None
                    shares = 0
            
            # Check for entry signal (only if not in position)
            elif position is None:
                if prediction == 'BUY' and confidence >= min_confidence:
                    # Enter LONG position
                    position_value = (capital * position_size_pct) / 100
                    shares = int(position_value / current_price)
                    
                    if shares > 0:
                        position = 'LONG'
                        entry_price = current_price
                        entry_date = current_date
        
        # Close any open position at the end
        if position == 'LONG':
            final_price = df.iloc[-1]['Close']
            final_date = df.iloc[-1]['Date']
            exit_value = shares * final_price
            pnl = exit_value - (shares * entry_price)
            pnl_pct = ((final_price - entry_price) / entry_price) * 100
            capital += pnl
            
            trades.append({
                'Entry_Date': entry_date,
                'Entry_Price': entry_price,
                'Exit_Date': final_date,
                'Exit_Price': final_price,
                'Shares': shares,
                'PnL': pnl,
                'PnL_Pct': pnl_pct,
                'Exit_Reason': 'END_OF_PERIOD',
                'Days_Held': (final_date - entry_date).days
            })
        
        # Convert to DataFrames
        trades_df = pd.DataFrame(trades)
        portfolio_df = pd.DataFrame(portfolio_values)
        
        # Calculate performance metrics
        metrics = self._calculate_performance_metrics(
            trades_df, portfolio_df, risk_params['initial_capital']
        )
        
        return trades_df, portfolio_df, metrics
    
    def _calculate_performance_metrics(self, trades_df, portfolio_df, initial_capital):
        """Calculate comprehensive performance metrics."""
        
        if len(trades_df) == 0:
            print("\n⚠️  NO TRADES EXECUTED")
            return None
        
        final_capital = portfolio_df.iloc[-1]['Portfolio_Value']
        total_return = ((final_capital - initial_capital) / initial_capital) * 100
        
        winning_trades = trades_df[trades_df['PnL'] > 0]
        losing_trades = trades_df[trades_df['PnL'] < 0]
        
        win_rate = (len(winning_trades) / len(trades_df)) * 100 if len(trades_df) > 0 else 0
        
        avg_win = winning_trades['PnL'].mean() if len(winning_trades) > 0 else 0
        avg_loss = losing_trades['PnL'].mean() if len(losing_trades) > 0 else 0
        
        profit_factor = (winning_trades['PnL'].sum() / abs(losing_trades['PnL'].sum())) if len(losing_trades) > 0 else float('inf')
        
        # Calculate maximum drawdown
        portfolio_df['Peak'] = portfolio_df['Portfolio_Value'].cummax()
        portfolio_df['Drawdown'] = ((portfolio_df['Portfolio_Value'] - portfolio_df['Peak']) / portfolio_df['Peak']) * 100
        max_drawdown = portfolio_df['Drawdown'].min()
        
        # Calculate Sharpe Ratio (simplified - using daily returns)
        portfolio_df['Returns'] = portfolio_df['Portfolio_Value'].pct_change()
        sharpe_ratio = (portfolio_df['Returns'].mean() / portfolio_df['Returns'].std()) * np.sqrt(252) if portfolio_df['Returns'].std() > 0 else 0
        
        metrics = {
            'initial_capital': initial_capital,
            'final_capital': final_capital,
            'total_return': total_return,
            'total_trades': len(trades_df),
            'winning_trades': len(winning_trades),
            'losing_trades': len(losing_trades),
            'win_rate': win_rate,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'profit_factor': profit_factor,
            'max_drawdown': max_drawdown,
            'sharpe_ratio': sharpe_ratio,
            'avg_days_held': trades_df['Days_Held'].mean()
        }
        
        return metrics
    
    def print_backtest_results(self, trades_df, portfolio_df, metrics):
        """Print comprehensive backtest results."""
        
        if metrics is None:
            return
        
        print(f"\n{'='*70}")
        print("BACKTEST RESULTS")
        print(f"{'='*70}")
        
        print(f"\n💰 CAPITAL:")
        print(f"  Initial Capital:    ₹{metrics['initial_capital']:>12,.0f}")
        print(f"  Final Capital:      ₹{metrics['final_capital']:>12,.0f}")
        print(f"  Total Return:       {metrics['total_return']:>12.2f}%")
        
        # Determine if profitable
        if metrics['total_return'] > 0:
            status = "✅ PROFITABLE"
        elif metrics['total_return'] == 0:
            status = "⚪ BREAKEVEN"
        else:
            status = "❌ LOSS"
        print(f"  Status:             {status}")
        
        print(f"\n📊 TRADE STATISTICS:")
        print(f"  Total Trades:       {metrics['total_trades']:>12,}")
        print(f"  Winning Trades:     {metrics['winning_trades']:>12,}")
        print(f"  Losing Trades:      {metrics['losing_trades']:>12,}")
        print(f"  Win Rate:           {metrics['win_rate']:>12.2f}%")
        
        print(f"\n💵 PROFIT/LOSS:")
        print(f"  Average Win:        ₹{metrics['avg_win']:>12,.0f}")
        print(f"  Average Loss:       ₹{metrics['avg_loss']:>12,.0f}")
        print(f"  Profit Factor:      {metrics['profit_factor']:>12.2f}")
        
        print(f"\n📉 RISK METRICS:")
        print(f"  Max Drawdown:       {metrics['max_drawdown']:>12.2f}%")
        print(f"  Sharpe Ratio:       {metrics['sharpe_ratio']:>12.2f}")
        print(f"  Avg Days Held:      {metrics['avg_days_held']:>12.1f}")
        
        # Exit reason breakdown
        print(f"\n🚪 EXIT REASONS:")
        exit_reasons = trades_df['Exit_Reason'].value_counts()
        for reason, count in exit_reasons.items():
            pct = (count / len(trades_df)) * 100
            print(f"  {reason:<15} {count:>5,} ({pct:>5.2f}%)")
        
        # Recent trades
        print(f"\n📋 RECENT TRADES (Last 5):")
        recent_trades = trades_df.tail(5)
        for idx, trade in recent_trades.iterrows():
            pnl_symbol = "+" if trade['PnL'] > 0 else ""
            print(f"  {trade['Entry_Date'].strftime('%Y-%m-%d')} → {trade['Exit_Date'].strftime('%Y-%m-%d')} | "
                  f"₹{trade['Entry_Price']:.2f} → ₹{trade['Exit_Price']:.2f} | "
                  f"{pnl_symbol}₹{trade['PnL']:.0f} ({pnl_symbol}{trade['PnL_Pct']:.2f}%) | "
                  f"{trade['Exit_Reason']}")
    
    def visualize_backtest(self, trades_df, portfolio_df, stock_symbol):
        """Create comprehensive backtest visualizations."""
        
        if len(trades_df) == 0:
            print("\n⚠️  No trades to visualize")
            return
        
        print(f"\n{'='*70}")
        print("GENERATING BACKTEST VISUALIZATIONS")
        print(f"{'='*70}")
        
        fig = plt.figure(figsize=(16, 12))
        
        # 1. Portfolio Value Over Time
        ax1 = plt.subplot(3, 2, 1)
        ax1.plot(portfolio_df['Date'], portfolio_df['Portfolio_Value'], 
                linewidth=2, color='#2E86AB')
        ax1.axhline(y=self.risk_params['initial_capital'], 
                   color='red', linestyle='--', alpha=0.5, label='Initial Capital')
        ax1.set_title('Portfolio Value Over Time', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Date')
        ax1.set_ylabel('Portfolio Value (₹)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'₹{x/1000:.0f}K'))
        
        # 2. Drawdown Chart
        ax2 = plt.subplot(3, 2, 2)
        ax2.fill_between(portfolio_df['Date'], portfolio_df['Drawdown'], 0, 
                         color='red', alpha=0.3)
        ax2.plot(portfolio_df['Date'], portfolio_df['Drawdown'], color='red', linewidth=2)
        ax2.set_title('Drawdown Over Time', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Date')
        ax2.set_ylabel('Drawdown (%)')
        ax2.grid(True, alpha=0.3)
        
        # 3. Trade P&L Distribution
        ax3 = plt.subplot(3, 2, 3)
        colors = ['green' if x > 0 else 'red' for x in trades_df['PnL']]
        ax3.bar(range(len(trades_df)), trades_df['PnL'], color=colors, alpha=0.6)
        ax3.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        ax3.set_title('Trade-by-Trade P&L', fontsize=14, fontweight='bold')
        ax3.set_xlabel('Trade Number')
        ax3.set_ylabel('P&L (₹)')
        ax3.grid(True, alpha=0.3)
        
        # 4. Cumulative P&L
        ax4 = plt.subplot(3, 2, 4)
        trades_df['Cumulative_PnL'] = trades_df['PnL'].cumsum()
        ax4.plot(range(len(trades_df)), trades_df['Cumulative_PnL'], 
                linewidth=2, color='#A23B72', marker='o', markersize=4)
        ax4.axhline(y=0, color='black', linestyle='--', linewidth=0.5)
        ax4.set_title('Cumulative P&L', fontsize=14, fontweight='bold')
        ax4.set_xlabel('Trade Number')
        ax4.set_ylabel('Cumulative P&L (₹)')
        ax4.grid(True, alpha=0.3)
        
        # 5. Exit Reason Breakdown
        ax5 = plt.subplot(3, 2, 5)
        exit_counts = trades_df['Exit_Reason'].value_counts()
        colors_pie = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D']
        ax5.pie(exit_counts.values, labels=exit_counts.index, autopct='%1.1f%%',
               colors=colors_pie, startangle=90)
        ax5.set_title('Exit Reason Distribution', fontsize=14, fontweight='bold')
        
        # 6. Win/Loss Statistics
        ax6 = plt.subplot(3, 2, 6)
        win_loss_data = [
            len(trades_df[trades_df['PnL'] > 0]),
            len(trades_df[trades_df['PnL'] < 0])
        ]
        ax6.bar(['Winning Trades', 'Losing Trades'], win_loss_data, 
               color=['green', 'red'], alpha=0.6)
        ax6.set_title('Win/Loss Distribution', fontsize=14, fontweight='bold')
        ax6.set_ylabel('Number of Trades')
        ax6.grid(True, alpha=0.3, axis='y')
        
        # Add totals on bars
        for i, v in enumerate(win_loss_data):
            ax6.text(i, v + 1, str(v), ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        
        # Save figure
        filename = f'results/backtest_{stock_symbol}_{datetime.now().strftime("%Y%m%d_%H%M%S")}.png'
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"✅ Backtest visualization saved: {filename}")
        
        plt.show()
    
    def run_complete_system(self, stock_symbol='RELIANCE', 
                           initial_capital=100000,
                           stop_loss_pct=2.0,
                           take_profit_pct=5.0,
                           position_size_pct=10.0,
                           min_confidence=0.4):
        """
        Run the complete production trading system.
        
        Parameters:
        - stock_symbol: Stock to analyze (default: RELIANCE)
        - initial_capital: Starting capital (default: ₹100,000)
        - stop_loss_pct: Stop loss % (default: 2%)
        - take_profit_pct: Take profit % (default: 5%)
        - position_size_pct: Position size % (default: 10%)
        - min_confidence: Minimum confidence (default: 40%)
        """
        
        print("\n")
        print("╔" + "="*68 + "╗")
        print("║" + " "*12 + "PRODUCTION TRADING SYSTEM WITH BACKTEST" + " "*17 + "║")
        print("╚" + "="*68 + "╝")
        print("\n")
        
        # Step 1: Load stock data
        df = self.load_stock_data(stock_symbol)
        if df is None:
            return None, None, None
        
        # Step 2: Calculate technical indicators
        df = self.calculate_technical_indicators(df)
        
        # Step 3: Make predictions
        df = self.make_predictions(df)
        
        # Step 4: Implement risk management
        risk_params = self.implement_risk_management(
            initial_capital=initial_capital,
            stop_loss_pct=stop_loss_pct,
            take_profit_pct=take_profit_pct,
            position_size_pct=position_size_pct,
            min_confidence=min_confidence
        )
        
        # Step 5: Backtest strategy
        trades_df, portfolio_df, metrics = self.backtest_strategy(df, risk_params)
        
        # Print results
        self.print_backtest_results(trades_df, portfolio_df, metrics)
        
        # Visualize
        if metrics is not None:
            self.visualize_backtest(trades_df, portfolio_df, stock_symbol)
        
        print("\n" + "="*70)
        print("✅ COMPLETE SYSTEM EXECUTION FINISHED")
        print("="*70)
        
        return trades_df, portfolio_df, metrics


def main():
    """Main execution function."""
    
    # Initialize system
    system = ProductionTradingSystem()
    
    # Run complete system on RELIANCE
    print("\n📊 Running backtest on RELIANCE stock...")
    trades, portfolio, metrics = system.run_complete_system(
        stock_symbol='RELIANCE',
        initial_capital=100000,    # ₹1 Lakh
        stop_loss_pct=2.0,         # 2% stop loss
        take_profit_pct=5.0,       # 5% take profit
        position_size_pct=10.0,    # 10% position size
        min_confidence=0.4         # 40% minimum confidence
    )
    
    # Save results
    if trades is not None and len(trades) > 0:
        trades.to_csv('results/backtest_trades.csv', index=False)
        portfolio.to_csv('results/backtest_portfolio.csv', index=False)
        print(f"\n💾 Results saved:")
        print(f"  • results/backtest_trades.csv")
        print(f"  • results/backtest_portfolio.csv")
    
    print("\n" + "="*70)
    print("🎉 PRODUCTION TRADING SYSTEM DEMO COMPLETE!")
    print("="*70)
    print("\n⚠️  IMPORTANT REMINDERS:")
    print("  • This is for educational purposes only")
    print("  • Past performance does not guarantee future results")
    print("  • Always do your own research before trading")
    print("  • Never invest more than you can afford to lose")
    print("  • Use proper risk management in live trading")
    print("\n")


if __name__ == "__main__":
    main()
