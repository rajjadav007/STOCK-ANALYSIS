#!/usr/bin/env python3
"""
Formula Visualization - Show all stock analysis formulas with graphs
Demonstrates how each formula works visually
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import warnings
warnings.filterwarnings('ignore')

class FormulaVisualizer:
    """Visualize all stock market formulas"""
    
    def __init__(self):
        self.sample_days = 100
        self.create_sample_data()
        
    def create_sample_data(self):
        """Create realistic sample stock data"""
        print("📊 Creating sample stock data...")
        np.random.seed(42)
        
        # Generate realistic price data
        base_price = 100
        trend = 0.001  # Slight upward trend
        volatility = 0.02
        
        returns = np.random.normal(trend, volatility, self.sample_days)
        prices = base_price * np.cumprod(1 + returns)
        
        dates = pd.date_range(start='2024-01-01', periods=self.sample_days, freq='D')
        
        self.df = pd.DataFrame({
            'Date': dates,
            'Close': prices,
            'Open': prices * (1 + np.random.normal(0, 0.005, self.sample_days)),
            'High': prices * (1 + np.random.uniform(0, 0.02, self.sample_days)),
            'Low': prices * (1 - np.random.uniform(0, 0.02, self.sample_days)),
            'Volume': np.random.randint(1000000, 5000000, self.sample_days)
        })
        
        print(f"✅ Created {len(self.df)} days of sample data")
    
    def calculate_all_indicators(self):
        """Calculate all technical indicators"""
        print("🔧 Calculating all indicators...")
        
        df = self.df.copy()
        
        # Returns
        df['Returns'] = df['Close'].pct_change()
        
        # SMA
        df['SMA_10'] = df['Close'].rolling(window=10).mean()
        df['SMA_50'] = df['Close'].rolling(window=50).mean()
        
        # EMA
        df['EMA_12'] = df['Close'].ewm(span=12, adjust=False).mean()
        df['EMA_26'] = df['Close'].ewm(span=26, adjust=False).mean()
        
        # RSI
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['RSI_14'] = 100 - (100 / (1 + rs))
        
        # MACD
        df['MACD'] = df['EMA_12'] - df['EMA_26']
        df['MACD_signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
        df['MACD_hist'] = df['MACD'] - df['MACD_signal']
        
        # Volatility
        df['Volatility_10'] = df['Returns'].rolling(window=10).std()
        df['Volatility_20'] = df['Returns'].rolling(window=20).std()
        
        self.df = df
        print("✅ All indicators calculated")
    
    def visualize_formula_sma(self):
        """Visualize SMA formula and interpretation"""
        fig, axes = plt.subplots(2, 2, figsize=(16, 10))
        fig.suptitle('Simple Moving Average (SMA) Formula & Interpretation', 
                     fontsize=16, fontweight='bold')
        
        # 1. Price with SMA
        ax = axes[0, 0]
        ax.plot(self.df.index, self.df['Close'], label='Close Price', 
                color='black', linewidth=2, alpha=0.7)
        ax.plot(self.df.index, self.df['SMA_10'], label='SMA 10', 
                color='blue', linewidth=2)
        ax.plot(self.df.index, self.df['SMA_50'], label='SMA 50', 
                color='red', linewidth=2)
        ax.set_title('SMA = (P₁ + P₂ + ... + Pₙ) / n', fontweight='bold')
        ax.set_ylabel('Price ($)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Add annotations
        # Golden Cross
        golden_cross = None
        for i in range(50, len(self.df)-1):
            if (self.df['SMA_10'].iloc[i-1] <= self.df['SMA_50'].iloc[i-1] and 
                self.df['SMA_10'].iloc[i] > self.df['SMA_50'].iloc[i]):
                golden_cross = i
                break
        
        if golden_cross:
            ax.annotate('Golden Cross\n(BUY Signal)', 
                       xy=(golden_cross, self.df['SMA_10'].iloc[golden_cross]),
                       xytext=(golden_cross+10, self.df['SMA_10'].iloc[golden_cross]+5),
                       arrowprops=dict(arrowstyle='->', color='green', lw=2),
                       fontsize=10, fontweight='bold', color='green')
        
        # 2. Formula explanation
        ax = axes[0, 1]
        ax.axis('off')
        formula_text = """
SMA FORMULA BREAKDOWN:

📐 Formula:
   SMA(n) = Sum of prices / Number of days
   
📊 Example (10-day SMA):
   Day 1-10 prices: [100, 102, 101, 103, 104, 
                     105, 103, 102, 101, 100]
   
   SMA₁₀ = (100+102+101+...+100) / 10
        = 1021 / 10
        = 102.1
        
🎯 Trading Signals:
   ✅ Price > SMA  → BULLISH (BUY)
   ❌ Price < SMA  → BEARISH (SELL)
   
   ✅ SMA₁₀ > SMA₅₀ → GOLDEN CROSS (Strong BUY)
   ❌ SMA₁₀ < SMA₅₀ → DEATH CROSS (Strong SELL)
   
💡 Interpretation:
   • Smooths out price volatility
   • Shows average price over period
   • Identifies trend direction
   • Support/resistance levels
        """
        ax.text(0.1, 0.5, formula_text, fontsize=11, family='monospace',
                verticalalignment='center', bbox=dict(boxstyle='round', 
                facecolor='wheat', alpha=0.3))
        
        # 3. Buy/Sell signals
        ax = axes[1, 0]
        buy_signals = (self.df['Close'] > self.df['SMA_10']) & \
                     (self.df['Close'].shift(1) <= self.df['SMA_10'].shift(1))
        sell_signals = (self.df['Close'] < self.df['SMA_10']) & \
                      (self.df['Close'].shift(1) >= self.df['SMA_10'].shift(1))
        
        ax.plot(self.df.index, self.df['Close'], label='Price', 
                color='black', linewidth=2)
        ax.plot(self.df.index, self.df['SMA_10'], label='SMA 10', 
                color='blue', linewidth=2)
        ax.scatter(self.df.index[buy_signals], self.df['Close'][buy_signals],
                  color='green', marker='^', s=200, label='BUY Signal', zorder=5)
        ax.scatter(self.df.index[sell_signals], self.df['Close'][sell_signals],
                  color='red', marker='v', s=200, label='SELL Signal', zorder=5)
        ax.set_title('SMA Trading Signals', fontweight='bold')
        ax.set_ylabel('Price ($)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 4. Distance from SMA (momentum)
        ax = axes[1, 1]
        distance = ((self.df['Close'] - self.df['SMA_10']) / self.df['SMA_10'] * 100).dropna()
        colors = ['green' if x > 0 else 'red' for x in distance]
        ax.bar(distance.index, distance, color=colors, alpha=0.6)
        ax.axhline(y=0, color='black', linestyle='-', linewidth=1)
        ax.axhline(y=5, color='green', linestyle='--', label='Strong Bullish (+5%)')
        ax.axhline(y=-5, color='red', linestyle='--', label='Strong Bearish (-5%)')
        ax.set_title('Distance from SMA (Price - SMA) / SMA × 100%', fontweight='bold')
        ax.set_ylabel('Distance (%)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('results/plots/formula_sma.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("  ✅ SMA formula visualization saved")
    
    def visualize_formula_ema(self):
        """Visualize EMA formula"""
        fig, axes = plt.subplots(2, 2, figsize=(16, 10))
        fig.suptitle('Exponential Moving Average (EMA) Formula & Interpretation', 
                     fontsize=16, fontweight='bold')
        
        # 1. EMA vs SMA comparison
        ax = axes[0, 0]
        ax.plot(self.df.index, self.df['Close'], label='Close Price', 
                color='black', linewidth=2, alpha=0.7)
        ax.plot(self.df.index, self.df['SMA_10'], label='SMA 10', 
                color='blue', linewidth=2, linestyle='--', alpha=0.7)
        ax.plot(self.df.index, self.df['EMA_12'], label='EMA 12', 
                color='green', linewidth=2)
        ax.set_title('EMA = (Price × K) + (Previous EMA × (1-K))', fontweight='bold')
        ax.set_ylabel('Price ($)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Annotation
        ax.text(0.02, 0.98, 'EMA responds faster\nto price changes', 
                transform=ax.transAxes, fontsize=10, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.5))
        
        # 2. Formula explanation
        ax = axes[0, 1]
        ax.axis('off')
        formula_text = """
EMA FORMULA BREAKDOWN:

📐 Formula:
   EMA(today) = (Price × K) + (EMA(yesterday) × (1-K))
   
   K = 2 / (Period + 1)
   
📊 Example (12-day EMA):
   K = 2 / (12 + 1) = 0.1538 (15.38%)
   
   If:
   - Today's price = $105
   - Yesterday's EMA = $100
   
   EMA(today) = (105 × 0.1538) + (100 × 0.8462)
              = 16.15 + 84.62
              = $100.77
              
🎯 Trading Signals:
   ✅ EMA₁₂ > EMA₂₆ → BULLISH (BUY)
   ❌ EMA₁₂ < EMA₂₆ → BEARISH (SELL)
   
💡 Key Points:
   • Recent price has 15% weight
   • Past prices have 85% weight
   • More responsive than SMA
   • Better for short-term trading
   • Used in MACD calculation
        """
        ax.text(0.1, 0.5, formula_text, fontsize=11, family='monospace',
                verticalalignment='center', bbox=dict(boxstyle='round', 
                facecolor='lightgreen', alpha=0.3))
        
        # 3. EMA responsiveness
        ax = axes[1, 0]
        # Create a price spike to show EMA response
        sample_range = slice(40, 70)
        ax.plot(self.df.index[sample_range], self.df['Close'].iloc[sample_range], 
                label='Price', color='black', linewidth=3, marker='o')
        ax.plot(self.df.index[sample_range], self.df['SMA_10'].iloc[sample_range], 
                label='SMA 10 (Slower)', color='blue', linewidth=2, linestyle='--')
        ax.plot(self.df.index[sample_range], self.df['EMA_12'].iloc[sample_range], 
                label='EMA 12 (Faster)', color='green', linewidth=2)
        ax.set_title('EMA Responds Faster to Price Changes', fontweight='bold')
        ax.set_ylabel('Price ($)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 4. Weight distribution
        ax = axes[1, 1]
        periods = np.arange(1, 31)
        k_values = 2 / (periods + 1)
        ax.plot(periods, k_values * 100, marker='o', linewidth=2, color='green')
        ax.axhline(y=15.38, color='red', linestyle='--', 
                  label='EMA-12: K=15.38%', linewidth=2)
        ax.axhline(y=7.41, color='blue', linestyle='--', 
                  label='EMA-26: K=7.41%', linewidth=2)
        ax.set_title('EMA Weight Factor (K) vs Period\nK = 2/(Period+1)', fontweight='bold')
        ax.set_xlabel('Period (days)')
        ax.set_ylabel('Weight of Today\'s Price (%)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('results/plots/formula_ema.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("  ✅ EMA formula visualization saved")
    
    def visualize_formula_rsi(self):
        """Visualize RSI formula"""
        fig = plt.figure(figsize=(16, 10))
        gs = GridSpec(3, 2, figure=fig, hspace=0.3, wspace=0.3)
        fig.suptitle('RSI (Relative Strength Index) Formula & Interpretation', 
                     fontsize=16, fontweight='bold')
        
        # 1. Price chart
        ax1 = fig.add_subplot(gs[0, :])
        ax1.plot(self.df.index, self.df['Close'], label='Close Price', 
                color='black', linewidth=2)
        ax1.set_title('Stock Price', fontweight='bold')
        ax1.set_ylabel('Price ($)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. RSI with zones
        ax2 = fig.add_subplot(gs[1, :])
        ax2.plot(self.df.index, self.df['RSI_14'], label='RSI 14', 
                color='purple', linewidth=2)
        ax2.axhline(y=70, color='red', linestyle='--', linewidth=2, 
                   label='Overbought (70)')
        ax2.axhline(y=30, color='green', linestyle='--', linewidth=2, 
                   label='Oversold (30)')
        ax2.fill_between(self.df.index, 70, 100, alpha=0.2, color='red', 
                        label='SELL Zone')
        ax2.fill_between(self.df.index, 0, 30, alpha=0.2, color='green', 
                        label='BUY Zone')
        ax2.set_title('RSI = 100 - (100 / (1 + RS)), where RS = Avg Gain / Avg Loss', 
                     fontweight='bold')
        ax2.set_ylabel('RSI')
        ax2.set_ylim(0, 100)
        ax2.legend(loc='upper right')
        ax2.grid(True, alpha=0.3)
        
        # Mark buy/sell signals
        oversold = self.df['RSI_14'] < 30
        overbought = self.df['RSI_14'] > 70
        ax2.scatter(self.df.index[oversold], self.df['RSI_14'][oversold],
                   color='green', marker='^', s=100, zorder=5)
        ax2.scatter(self.df.index[overbought], self.df['RSI_14'][overbought],
                   color='red', marker='v', s=100, zorder=5)
        
        # 3. Formula explanation
        ax3 = fig.add_subplot(gs[2, 0])
        ax3.axis('off')
        formula_text = """
RSI CALCULATION STEPS:

Step 1: Calculate price changes
  Δ = Close(today) - Close(yesterday)
  
Step 2: Separate gains and losses
  Gain = Δ if positive, else 0
  Loss = |Δ| if negative, else 0
  
Step 3: Calculate averages (14 days)
  Avg Gain = Sum(Gains) / 14
  Avg Loss = Sum(Losses) / 14
  
Step 4: Calculate RS
  RS = Avg Gain / Avg Loss
  
Step 5: Calculate RSI
  RSI = 100 - (100 / (1 + RS))
  
EXAMPLE:
  Avg Gain = $2.50
  Avg Loss = $1.00
  RS = 2.50 / 1.00 = 2.5
  RSI = 100 - (100/(1+2.5))
      = 100 - (100/3.5)
      = 100 - 28.57
      = 71.43 → OVERBOUGHT (SELL)
        """
        ax3.text(0.05, 0.5, formula_text, fontsize=9, family='monospace',
                verticalalignment='center', bbox=dict(boxstyle='round', 
                facecolor='lavender', alpha=0.5))
        
        # 4. Trading signals
        ax4 = fig.add_subplot(gs[2, 1])
        ax4.axis('off')
        signals_text = """
🎯 RSI TRADING SIGNALS:

RSI Value    Condition      Action
─────────    ─────────      ──────
0 - 30       OVERSOLD       🟢 STRONG BUY
                            Stock too cheap
                            Price will rise
                            
30 - 40      Weak           ↗️ Consider BUY
                            Potential entry
                            
40 - 60      NEUTRAL        ⏸️ HOLD
                            No clear signal
                            Wait for clarity
                            
60 - 70      Strong         ↘️ Consider SELL
                            Potential exit
                            
70 - 100     OVERBOUGHT     🔴 STRONG SELL
                            Stock too expensive
                            Price will drop

💡 Key Points:
  • RSI < 30: Too many sellers
  • RSI > 70: Too many buyers
  • Mean reversion expected
  • Use with other indicators
        """
        ax4.text(0.05, 0.5, signals_text, fontsize=9, family='monospace',
                verticalalignment='center', bbox=dict(boxstyle='round', 
                facecolor='lightyellow', alpha=0.5))
        
        plt.savefig('results/plots/formula_rsi.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("  ✅ RSI formula visualization saved")
    
    def visualize_formula_macd(self):
        """Visualize MACD formula"""
        fig = plt.figure(figsize=(16, 12))
        gs = GridSpec(4, 2, figure=fig, hspace=0.3, wspace=0.3)
        fig.suptitle('MACD (Moving Average Convergence Divergence) Formula', 
                     fontsize=16, fontweight='bold')
        
        # 1. Price with EMAs
        ax1 = fig.add_subplot(gs[0, :])
        ax1.plot(self.df.index, self.df['Close'], label='Close Price', 
                color='black', linewidth=2)
        ax1.plot(self.df.index, self.df['EMA_12'], label='EMA 12 (Fast)', 
                color='blue', linewidth=1.5)
        ax1.plot(self.df.index, self.df['EMA_26'], label='EMA 26 (Slow)', 
                color='red', linewidth=1.5)
        ax1.set_title('Step 1: Calculate EMAs', fontweight='bold')
        ax1.set_ylabel('Price ($)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. MACD Line
        ax2 = fig.add_subplot(gs[1, :])
        ax2.plot(self.df.index, self.df['MACD'], label='MACD = EMA12 - EMA26', 
                color='blue', linewidth=2)
        ax2.axhline(y=0, color='black', linestyle='-', linewidth=1)
        ax2.fill_between(self.df.index, 0, self.df['MACD'], 
                        where=(self.df['MACD'] > 0), alpha=0.3, color='green',
                        label='Positive (Bullish)')
        ax2.fill_between(self.df.index, 0, self.df['MACD'], 
                        where=(self.df['MACD'] < 0), alpha=0.3, color='red',
                        label='Negative (Bearish)')
        ax2.set_title('Step 2: MACD Line = EMA12 - EMA26', fontweight='bold')
        ax2.set_ylabel('MACD Value')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 3. MACD with Signal
        ax3 = fig.add_subplot(gs[2, :])
        ax3.plot(self.df.index, self.df['MACD'], label='MACD Line', 
                color='blue', linewidth=2)
        ax3.plot(self.df.index, self.df['MACD_signal'], label='Signal Line (9-EMA of MACD)', 
                color='red', linewidth=2)
        ax3.axhline(y=0, color='black', linestyle='-', linewidth=1)
        
        # Mark crossovers
        bullish_cross = (self.df['MACD'] > self.df['MACD_signal']) & \
                       (self.df['MACD'].shift(1) <= self.df['MACD_signal'].shift(1))
        bearish_cross = (self.df['MACD'] < self.df['MACD_signal']) & \
                       (self.df['MACD'].shift(1) >= self.df['MACD_signal'].shift(1))
        
        ax3.scatter(self.df.index[bullish_cross], self.df['MACD'][bullish_cross],
                   color='green', marker='^', s=200, label='BUY Signal', zorder=5)
        ax3.scatter(self.df.index[bearish_cross], self.df['MACD'][bearish_cross],
                   color='red', marker='v', s=200, label='SELL Signal', zorder=5)
        
        ax3.set_title('Step 3: Add Signal Line & Identify Crossovers', fontweight='bold')
        ax3.set_ylabel('MACD Value')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # 4. Histogram
        ax4 = fig.add_subplot(gs[3, :])
        colors = ['green' if x > 0 else 'red' for x in self.df['MACD_hist'].dropna()]
        ax4.bar(self.df.index, self.df['MACD_hist'], color=colors, alpha=0.6,
               label='Histogram = MACD - Signal')
        ax4.axhline(y=0, color='black', linestyle='-', linewidth=1)
        ax4.plot(self.df.index, self.df['MACD'], label='MACD', 
                color='blue', linewidth=1, alpha=0.5)
        ax4.plot(self.df.index, self.df['MACD_signal'], label='Signal', 
                color='red', linewidth=1, alpha=0.5)
        ax4.set_title('Step 4: Histogram = MACD - Signal (Shows Momentum)', fontweight='bold')
        ax4.set_ylabel('Value')
        ax4.set_xlabel('Days')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        plt.savefig('results/plots/formula_macd.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("  ✅ MACD formula visualization saved")
    
    def visualize_formula_volatility(self):
        """Visualize volatility formula"""
        fig, axes = plt.subplots(2, 2, figsize=(16, 10))
        fig.suptitle('Volatility (Standard Deviation of Returns) Formula', 
                     fontsize=16, fontweight='bold')
        
        # 1. Price chart
        ax = axes[0, 0]
        ax.plot(self.df.index, self.df['Close'], label='Close Price', 
                color='black', linewidth=2)
        # Highlight high/low volatility periods
        high_vol = self.df['Volatility_10'] > self.df['Volatility_10'].quantile(0.75)
        ax.fill_between(self.df.index, self.df['Close'].min(), self.df['Close'].max(),
                        where=high_vol, alpha=0.2, color='red', 
                        label='High Volatility Periods')
        ax.set_title('Stock Price with Volatility Zones', fontweight='bold')
        ax.set_ylabel('Price ($)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 2. Returns distribution
        ax = axes[0, 1]
        returns_clean = self.df['Returns'].dropna()
        ax.hist(returns_clean * 100, bins=30, edgecolor='black', alpha=0.7)
        ax.axvline(x=returns_clean.mean() * 100, color='red', linestyle='--', 
                  linewidth=2, label=f'Mean: {returns_clean.mean()*100:.2f}%')
        ax.axvline(x=returns_clean.std() * 100, color='green', linestyle='--', 
                  linewidth=2, label=f'Std Dev: {returns_clean.std()*100:.2f}%')
        ax.axvline(x=-returns_clean.std() * 100, color='green', linestyle='--', linewidth=2)
        ax.set_title('Returns Distribution\nσ = √(Σ(Return - Mean)² / n)', fontweight='bold')
        ax.set_xlabel('Daily Return (%)')
        ax.set_ylabel('Frequency')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 3. Volatility over time
        ax = axes[1, 0]
        ax.plot(self.df.index, self.df['Volatility_10'] * 100, 
               label='10-Day Volatility', color='red', linewidth=2)
        ax.plot(self.df.index, self.df['Volatility_20'] * 100, 
               label='20-Day Volatility', color='blue', linewidth=2)
        ax.axhline(y=2, color='green', linestyle='--', 
                  label='Low Risk (<2%)', linewidth=1)
        ax.axhline(y=4, color='red', linestyle='--', 
                  label='High Risk (>4%)', linewidth=1)
        ax.fill_between(self.df.index, 0, 2, alpha=0.2, color='green')
        ax.fill_between(self.df.index, 4, 10, alpha=0.2, color='red')
        ax.set_title('Volatility = Rolling Std Dev of Returns', fontweight='bold')
        ax.set_ylabel('Volatility (%)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 4. Formula explanation
        ax = axes[1, 1]
        ax.axis('off')
        formula_text = """
VOLATILITY FORMULA:

σ = √(Σ(xᵢ - μ)² / (n-1))

Where:
  σ = Volatility (std deviation)
  xᵢ = Individual return
  μ = Mean return
  n = Number of periods

CALCULATION STEPS:

1. Calculate returns:
   Return = (Price_today - Price_yesterday) 
            / Price_yesterday
            
2. Calculate mean return:
   μ = Sum(returns) / n
   
3. Calculate variance:
   Variance = Σ(Return - μ)² / (n-1)
   
4. Calculate volatility:
   σ = √Variance

EXAMPLE (10-day):
  Returns: [0.5%, -0.3%, 0.8%, 1.2%, -0.5%,
            0.3%, -0.2%, 0.6%, -0.1%, 0.4%]
  Mean = 0.27%
  Variance = 0.0003
  Volatility = √0.0003 = 1.73%

INTERPRETATION:
  < 1%:  Very Low Risk
  1-2%:  Low Risk
  2-3%:  Medium Risk
  3-5%:  High Risk
  > 5%:  Very High Risk
        """
        ax.text(0.05, 0.5, formula_text, fontsize=9, family='monospace',
                verticalalignment='center', bbox=dict(boxstyle='round', 
                facecolor='lightcoral', alpha=0.3))
        
        plt.tight_layout()
        plt.savefig('results/plots/formula_volatility.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("  ✅ Volatility formula visualization saved")
    
    def visualize_formula_returns(self):
        """Visualize returns formula"""
        fig, axes = plt.subplots(2, 2, figsize=(16, 10))
        fig.suptitle('Price Returns Formula & Interpretation', 
                     fontsize=16, fontweight='bold')
        
        # 1. Price chart
        ax = axes[0, 0]
        ax.plot(self.df.index, self.df['Close'], label='Close Price', 
                color='black', linewidth=2, marker='o', markersize=3)
        # Annotate example
        if len(self.df) > 50:
            i = 50
            ax.annotate(f'Day {i}: ${self.df["Close"].iloc[i]:.2f}', 
                       xy=(i, self.df['Close'].iloc[i]),
                       xytext=(i+10, self.df['Close'].iloc[i]+2),
                       arrowprops=dict(arrowstyle='->', color='blue', lw=2),
                       fontsize=10, fontweight='bold')
            ax.annotate(f'Day {i+1}: ${self.df["Close"].iloc[i+1]:.2f}', 
                       xy=(i+1, self.df['Close'].iloc[i+1]),
                       xytext=(i+10, self.df['Close'].iloc[i+1]-2),
                       arrowprops=dict(arrowstyle='->', color='red', lw=2),
                       fontsize=10, fontweight='bold')
        ax.set_title('Stock Price Over Time', fontweight='bold')
        ax.set_ylabel('Price ($)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 2. Daily returns
        ax = axes[0, 1]
        returns_pct = self.df['Returns'] * 100
        colors = ['green' if x > 0 else 'red' for x in returns_pct.dropna()]
        ax.bar(self.df.index, returns_pct, color=colors, alpha=0.6)
        ax.axhline(y=0, color='black', linestyle='-', linewidth=1)
        ax.set_title('Return = (Price_today - Price_yesterday) / Price_yesterday', 
                    fontweight='bold')
        ax.set_ylabel('Daily Return (%)')
        ax.grid(True, alpha=0.3)
        
        # 3. Cumulative returns
        ax = axes[1, 0]
        cumulative_returns = (1 + self.df['Returns']).cumprod() - 1
        ax.plot(self.df.index, cumulative_returns * 100, 
               label='Cumulative Returns', color='blue', linewidth=2)
        ax.axhline(y=0, color='black', linestyle='-', linewidth=1)
        ax.fill_between(self.df.index, 0, cumulative_returns * 100,
                       where=(cumulative_returns > 0), alpha=0.3, color='green',
                       label='Profit')
        ax.fill_between(self.df.index, 0, cumulative_returns * 100,
                       where=(cumulative_returns < 0), alpha=0.3, color='red',
                       label='Loss')
        ax.set_title('Cumulative Returns (Total % Gain/Loss)', fontweight='bold')
        ax.set_ylabel('Cumulative Return (%)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 4. Formula explanation
        ax = axes[1, 1]
        ax.axis('off')
        formula_text = """
RETURNS FORMULA:

Return(t) = (Price(t) - Price(t-1)) 
            / Price(t-1)

In percentage:
Return(t) = ((Price(t) - Price(t-1)) 
            / Price(t-1)) × 100%

EXAMPLE:
  Yesterday: $100
  Today: $105
  
  Return = (105 - 100) / 100
         = 5 / 100
         = 0.05
         = 5%
  
  → Stock gained 5% in one day!

TRADING SIGNALS:

Return     Meaning        Action
------     -------        ------
> +5%      Huge gain      Take profit
+2% to +5% Good gain      Hold
0% to +2%  Small gain     Hold
0%         No change      Hold
-2% to 0%  Small loss     Hold/review
-5% to -2% Moderate loss  Consider sell
< -5%      Large loss     Sell/re-evaluate

💡 Key Points:
  • Positive return = Profit
  • Negative return = Loss
  • Used to calculate volatility
  • Shows daily performance
        """
        ax.text(0.05, 0.5, formula_text, fontsize=10, family='monospace',
                verticalalignment='center', bbox=dict(boxstyle='round', 
                facecolor='lightblue', alpha=0.3))
        
        plt.tight_layout()
        plt.savefig('results/plots/formula_returns.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("  ✅ Returns formula visualization saved")
    
    def visualize_buy_sell_labels(self):
        """Visualize BUY/SELL label creation"""
        fig, axes = plt.subplots(2, 2, figsize=(16, 10))
        fig.suptitle('BUY/SELL Label Creation Formula', 
                     fontsize=16, fontweight='bold')
        
        # Create future price and labels
        df_sample = self.df.copy()
        df_sample['Future_Price'] = df_sample['Close'].shift(-1)
        df_sample['Label'] = (df_sample['Future_Price'] > df_sample['Close']).astype(int)
        df_sample = df_sample.dropna()
        
        # 1. Price with labels
        ax = axes[0, 0]
        buy_days = df_sample['Label'] == 1
        sell_days = df_sample['Label'] == 0
        
        ax.plot(df_sample.index, df_sample['Close'], label='Close Price', 
                color='black', linewidth=2)
        ax.scatter(df_sample.index[buy_days], df_sample['Close'][buy_days],
                  color='green', marker='^', s=100, label='BUY Label (Price↑tomorrow)', 
                  alpha=0.6, zorder=5)
        ax.scatter(df_sample.index[sell_days], df_sample['Close'][sell_days],
                  color='red', marker='v', s=100, label='SELL Label (Price↓tomorrow)', 
                  alpha=0.6, zorder=5)
        ax.set_title('Label = 1 if Price(t+1) > Price(t) else 0', fontweight='bold')
        ax.set_ylabel('Price ($)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 2. Label distribution
        ax = axes[0, 1]
        label_counts = df_sample['Label'].value_counts()
        colors_bar = ['red', 'green']
        ax.bar(['SELL (0)', 'BUY (1)'], label_counts, color=colors_bar, alpha=0.7,
              edgecolor='black', linewidth=2)
        ax.set_title('Label Distribution', fontweight='bold')
        ax.set_ylabel('Count')
        for i, v in enumerate(label_counts):
            ax.text(i, v + 1, str(v), ha='center', fontweight='bold', fontsize=12)
        ax.grid(True, alpha=0.3, axis='y')
        
        # 3. Example visualization
        ax = axes[1, 0]
        # Show 10 days as example
        sample_range = slice(40, 50)
        sample_df = df_sample.iloc[sample_range]
        
        x = range(len(sample_df))
        ax.plot(x, sample_df['Close'].values, marker='o', linewidth=2, 
               markersize=10, color='black', label='Today Price')
        ax.plot(x, sample_df['Future_Price'].values, marker='s', linewidth=2, 
               linestyle='--', markersize=10, color='blue', label='Tomorrow Price')
        
        # Draw arrows and labels
        for i, (idx, row) in enumerate(sample_df.iterrows()):
            if row['Label'] == 1:
                ax.annotate('', xy=(i, row['Future_Price']), xytext=(i, row['Close']),
                           arrowprops=dict(arrowstyle='->', color='green', lw=3))
                ax.text(i, row['Close']-1, 'BUY', ha='center', fontsize=8, 
                       fontweight='bold', color='green',
                       bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.7))
            else:
                ax.annotate('', xy=(i, row['Future_Price']), xytext=(i, row['Close']),
                           arrowprops=dict(arrowstyle='->', color='red', lw=3))
                ax.text(i, row['Close']+1, 'SELL', ha='center', fontsize=8, 
                       fontweight='bold', color='red',
                       bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.7))
        
        ax.set_title('Label Creation: Compare Today vs Tomorrow', fontweight='bold')
        ax.set_xlabel('Days')
        ax.set_ylabel('Price ($)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 4. Formula explanation
        ax = axes[1, 1]
        ax.axis('off')
        formula_text = """
BUY/SELL LABEL FORMULA:

IF Price(tomorrow) > Price(today) THEN
    Label = BUY (1)
ELSE
    Label = SELL (0)

PYTHON CODE:
  df['Future_Price'] = df['Close'].shift(-1)
  df['Label'] = (df['Future_Price'] > 
                 df['Close']).astype(int)

EXAMPLE:

Day  Today    Tomorrow   Change    Label
---  -----    --------   ------    -----
1    $100     $105       +$5       1 (BUY)
2    $105     $103       -$2       0 (SELL)
3    $103     $108       +$5       1 (BUY)
4    $108     $107       -$1       0 (SELL)
5    $107     $110       +$3       1 (BUY)

PURPOSE:
  • Train ML model to predict
  • If Label=1: Model predicts BUY
  • If Label=0: Model predicts SELL
  • Model learns patterns from
    indicators (SMA, RSI, MACD)

WITH THRESHOLD (0.5%):
  Label = 1 if Return > 0.5%
  Label = 0 if Return < -0.5%
  (Optional: HOLD for -0.5% to 0.5%)
        """
        ax.text(0.05, 0.5, formula_text, fontsize=9, family='monospace',
                verticalalignment='center', bbox=dict(boxstyle='round', 
                facecolor='lightyellow', alpha=0.5))
        
        plt.tight_layout()
        plt.savefig('results/plots/formula_labels.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("  ✅ BUY/SELL labels formula visualization saved")
    
    def visualize_logistic_regression(self):
        """Visualize logistic regression probability formula"""
        fig, axes = plt.subplots(2, 2, figsize=(16, 10))
        fig.suptitle('Logistic Regression Probability Formula', 
                     fontsize=16, fontweight='bold')
        
        # 1. Sigmoid function
        ax = axes[0, 0]
        z = np.linspace(-10, 10, 100)
        probability = 1 / (1 + np.exp(-z))
        
        ax.plot(z, probability, linewidth=3, color='blue')
        ax.axhline(y=0.5, color='red', linestyle='--', linewidth=2, label='Decision Boundary')
        ax.axvline(x=0, color='green', linestyle='--', linewidth=2)
        ax.fill_between(z, 0, probability, where=(z < 0), alpha=0.2, color='red', 
                        label='SELL Zone (P < 0.5)')
        ax.fill_between(z, 0, probability, where=(z > 0), alpha=0.2, color='green', 
                        label='BUY Zone (P > 0.5)')
        
        ax.set_title('Sigmoid Function: P = 1 / (1 + e^(-z))', fontweight='bold')
        ax.set_xlabel('z (Linear combination)')
        ax.set_ylabel('Probability')
        ax.set_ylim(-0.1, 1.1)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Add annotations
        ax.text(-5, 0.1, 'z < 0\nSELL', ha='center', fontsize=12, fontweight='bold',
               bbox=dict(boxstyle='round', facecolor='red', alpha=0.3))
        ax.text(5, 0.9, 'z > 0\nBUY', ha='center', fontsize=12, fontweight='bold',
               bbox=dict(boxstyle='round', facecolor='green', alpha=0.3))
        
        # 2. Probability interpretation
        ax = axes[0, 1]
        probs = [0.1, 0.3, 0.5, 0.7, 0.9]
        labels = ['Strong\nSELL\n10%', 'SELL\n30%', 'Neutral\n50%', 'BUY\n70%', 'Strong\nBUY\n90%']
        colors_probs = ['darkred', 'red', 'gray', 'green', 'darkgreen']
        
        ax.barh(range(len(probs)), probs, color=colors_probs, alpha=0.7,
               edgecolor='black', linewidth=2)
        ax.set_yticks(range(len(probs)))
        ax.set_yticklabels(labels)
        ax.axvline(x=0.5, color='black', linestyle='--', linewidth=2, label='Decision Line')
        ax.set_title('Probability Interpretation', fontweight='bold')
        ax.set_xlabel('Probability')
        ax.set_xlim(0, 1)
        ax.legend()
        ax.grid(True, alpha=0.3, axis='x')
        
        # 3. Example calculation
        ax = axes[1, 0]
        ax.axis('off')
        example_text = """
EXAMPLE CALCULATION:

Given:
  SMA_10 = 150
  RSI = 65
  MACD = 2.5
  EMA_12 = 152

Trained weights (β):
  β₀ = -2.0 (intercept)
  β₁ = 0.05 (SMA_10)
  β₂ = 0.03 (RSI)
  β₃ = 0.8 (MACD)
  β₄ = 0.02 (EMA_12)

Step 1: Calculate z
  z = β₀ + β₁×SMA + β₂×RSI + β₃×MACD + β₄×EMA
  z = -2.0 + (0.05×150) + (0.03×65) + 
      (0.8×2.5) + (0.02×152)
  z = -2.0 + 7.5 + 1.95 + 2.0 + 3.04
  z = 12.49

Step 2: Calculate probability
  P = 1 / (1 + e^(-12.49))
  P = 1 / (1 + 0.0000039)
  P = 0.999996
  P ≈ 99.99%

INTERPRETATION:
  P = 99.99% → VERY HIGH confidence BUY!
  Model is 99.99% sure price will rise.
  
ACTION:
  Strong BUY signal - Invest with confidence!
        """
        ax.text(0.05, 0.5, example_text, fontsize=10, family='monospace',
                verticalalignment='center', bbox=dict(boxstyle='round', 
                facecolor='lightgreen', alpha=0.3))
        
        # 4. Decision zones
        ax = axes[1, 1]
        ax.axis('off')
        zones_text = """
DECISION ZONES:

Probability    Confidence    Action
-----------    ----------    ------
> 0.90         Very High     🟢🟢 STRONG BUY
                             Invest heavily
                             
0.70 - 0.90    High          🟢 BUY
                             Good opportunity
                             
0.60 - 0.70    Moderate      ↗️ BUY
                             Small position
                             
0.50 - 0.60    Weak          ⏸️ HOLD
                             Wait for clarity
                             
0.40 - 0.50    Weak          ⏸️ HOLD/SELL
                             Consider exit
                             
0.30 - 0.40    Moderate      🔴 SELL
                             Exit position
                             
0.10 - 0.30    High          🔴 STRONG SELL
                             Exit immediately
                             
< 0.10         Very High     🔴🔴 SHORT
                             Short opportunity

KEY FORMULA:
  z = β₀ + Σ(βᵢ × featureᵢ)
  P = 1 / (1 + e^(-z))
  Decision = BUY if P > 0.5 else SELL
        """
        ax.text(0.05, 0.5, zones_text, fontsize=10, family='monospace',
                verticalalignment='center', bbox=dict(boxstyle='round', 
                facecolor='lightyellow', alpha=0.5))
        
        plt.tight_layout()
        plt.savefig('results/plots/formula_logistic.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("  ✅ Logistic regression formula visualization saved")
    
    def create_all_visualizations(self):
        """Create all formula visualizations"""
        print("\n📊 CREATING FORMULA VISUALIZATIONS")
        print("=" * 50)
        
        self.calculate_all_indicators()
        
        print("\nGenerating formula graphs...")
        self.visualize_formula_sma()
        self.visualize_formula_ema()
        self.visualize_formula_rsi()
        self.visualize_formula_macd()
        self.visualize_formula_volatility()
        self.visualize_formula_returns()
        self.visualize_buy_sell_labels()
        self.visualize_logistic_regression()
        
        print("\n✅ ALL FORMULA VISUALIZATIONS CREATED!")
        print(f"📁 Saved in: results/plots/")
        print(f"📊 Total graphs: 8 formula explanations")

def main():
    """Main function"""
    print("🎓 STOCK MARKET FORMULA VISUALIZER")
    print("=" * 50)
    
    visualizer = FormulaVisualizer()
    visualizer.create_all_visualizations()
    
    print("\n📚 Documentation created:")
    print("   📄 FORMULAS_EXPLAINED.md - Complete formula guide")
    print("   📊 results/plots/formula_*.png - Visual explanations")
    print("\n🎉 All formulas explained with graphs!")

if __name__ == "__main__":
    main()
