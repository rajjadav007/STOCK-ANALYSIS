"""
📚 COMPLETE FORMULA GUIDE - QUICK REFERENCE
============================================

All formulas used in stock market ML analysis, explained with graphs!

📖 DOCUMENTATION FILES:
----------------------
1. FORMULAS_EXPLAINED.md          - Complete written guide (ALL formulas)
2. FORMULA_VISUALIZATIONS.md      - Graph descriptions
3. This file                       - Quick reference

📊 FORMULA GRAPHS (8 total):
---------------------------
Located in: results/plots/

1. formula_sma.png        - Simple Moving Average
2. formula_ema.png        - Exponential Moving Average  
3. formula_rsi.png        - Relative Strength Index
4. formula_macd.png       - MACD Indicator
5. formula_volatility.png - Risk Measurement
6. formula_returns.png    - Price Returns
7. formula_labels.png     - BUY/SELL Label Creation
8. formula_logistic.png   - ML Probability Calculation

🎯 QUICK FORMULA REFERENCE:
===========================

1. SMA (Simple Moving Average):
   Formula: SMA = (P₁ + P₂ + ... + Pₙ) / n
   Shows: Trend direction
   Buy: Price > SMA, Golden Cross
   Sell: Price < SMA, Death Cross

2. EMA (Exponential Moving Average):
   Formula: EMA = (Price × K) + (Previous EMA × (1-K))
           K = 2/(Period+1)
   Shows: Recent price trend
   Buy: EMA_fast > EMA_slow
   Sell: EMA_fast < EMA_slow

3. RSI (Relative Strength Index):
   Formula: RSI = 100 - (100/(1 + RS))
           RS = Avg Gain / Avg Loss
   Shows: Momentum (0-100 scale)
   Buy: RSI < 30 (oversold)
   Sell: RSI > 70 (overbought)

4. MACD:
   Formula: MACD = EMA₁₂ - EMA₂₆
           Signal = 9-day EMA of MACD
           Histogram = MACD - Signal
   Shows: Trend changes
   Buy: MACD crosses above Signal
   Sell: MACD crosses below Signal

5. Volatility:
   Formula: σ = √(Σ(Return - Mean)² / n)
   Shows: Risk level (price fluctuation)
   High: >3% = Risky
   Low: <1% = Safe

6. Returns:
   Formula: Return = (Price_t - Price_t-1) / Price_t-1
   Shows: % price change
   Positive: Profit
   Negative: Loss

7. BUY/SELL Labels:
   Formula: Label = 1 if Price(t+1) > Price(t) else 0
   Shows: Future direction for ML training
   Label=1: Price will rise (BUY)
   Label=0: Price will fall (SELL)

8. Logistic Regression:
   Formula: P = 1 / (1 + e^(-z))
           z = β₀ + Σ(βᵢ × featureᵢ)
   Shows: Probability of BUY
   P > 0.7: BUY signal
   P < 0.3: SELL signal

📊 WHO CALCULATES WHAT:
========================

✅ Pandas Library:
   - SMA: .rolling().mean()
   - EMA: .ewm().mean()
   - Volatility: .rolling().std()
   - Returns: .pct_change()

🛠️ We Calculate:
   - RSI: Manual gain/loss calculation
   - MACD: Combine EMAs manually
   - Labels: Future price comparison

🤖 Scikit-learn:
   - Logistic Regression: Model training
   - Predictions: Probability calculation

🎯 TRADING SIGNALS SUMMARY:
============================

Indicator    BUY Signal              SELL Signal
---------    ----------              -----------
SMA          Price > SMA             Price < SMA
             Golden Cross            Death Cross
             
EMA          EMA₁₂ > EMA₂₆          EMA₁₂ < EMA₂₆
             
RSI          < 30 (oversold)        > 70 (overbought)
             
MACD         MACD > Signal          MACD < Signal
             Positive histogram     Negative histogram
             
Volatility   Low (<1%)              High (>5%)
             Safe entry             Risky, exit
             
Returns      Positive (+)           Negative (-)
             Profit                 Loss
             
Probability  P > 0.7                P < 0.3
             Confident BUY          Confident SELL

🚀 HOW TO USE:
==============

1. VIEW DOCUMENTATION:
   notepad FORMULAS_EXPLAINED.md

2. VIEW FORMULA GRAPHS:
   python view_graphs.py
   (Shows all 12 graphs including formulas)

3. CREATE NEW FORMULA GRAPHS:
   python visualize_formulas.py

4. OPEN SPECIFIC GRAPH:
   start results/plots/formula_sma.png
   start results/plots/formula_rsi.png
   ... etc

5. OPEN FOLDER:
   explorer results\plots

💡 REAL TRADING EXAMPLE:
=========================

Stock: XYZ at $100

Indicators:
  SMA_10: $98    → Price above SMA ✅ BULLISH
  SMA_50: $95    → Golden Cross ✅ STRONG BUY
  RSI: 45        → Below 50, not overbought ✅
  MACD: 1.2      → Above signal (0.8) ✅ BUY
  Volatility: 2% → Normal risk ✅
  Returns: +1.5% → Positive momentum ✅

ML Prediction:
  z = calculate from all features
  P = 1/(1 + e^(-z)) = 0.95 = 95%

DECISION: STRONG BUY! 🚀
Confidence: 95%
All indicators agree!

🎓 LEARNING ORDER:
==================

Beginner:
1. Read FORMULAS_EXPLAINED.md (SMA, EMA, Returns)
2. View formula_sma.png and formula_ema.png
3. Understand trend following

Intermediate:
4. Read RSI and MACD sections
5. View formula_rsi.png and formula_macd.png
6. Understand momentum indicators

Advanced:
7. Read Volatility and Labels
8. View formula_labels.png
9. Understand ML training

Expert:
10. Read Logistic Regression
11. View formula_logistic.png
12. Understand probability prediction

📁 FILE STRUCTURE:
==================

stock-analysis/
├── FORMULAS_EXPLAINED.md           ← Start here! (Complete guide)
├── FORMULA_VISUALIZATIONS.md       ← Graph descriptions
├── FORMULA_QUICKSTART.py           ← This file
├── visualize_formulas.py           ← Creates graphs
├── view_graphs.py                  ← Views all graphs
└── results/plots/
    ├── formula_sma.png             ← 8 formula graphs
    ├── formula_ema.png
    ├── formula_rsi.png
    ├── formula_macd.png
    ├── formula_volatility.png
    ├── formula_returns.png
    ├── formula_labels.png
    └── formula_logistic.png

✅ WHAT YOU HAVE NOW:
======================

✅ Complete written explanations (FORMULAS_EXPLAINED.md)
✅ 8 visual formula explanations (300 DPI graphs)
✅ Step-by-step calculations with examples
✅ Trading signals for each indicator
✅ Library vs manual calculations explained
✅ BUY/SELL zones clearly marked
✅ Real trading examples
✅ ML probability interpretation
✅ Quick reference (this file)

🎉 YOU'RE READY TO TRADE WITH CONFIDENCE! 📈

Run this file to see this guide:
  python FORMULA_QUICKSTART.py
"""

if __name__ == "__main__":
    print(__doc__)
    
    print("\n" + "="*60)
    print("📊 AVAILABLE COMMANDS:")
    print("="*60)
    print("1. python view_graphs.py           - View all graphs")
    print("2. python visualize_formulas.py    - Recreate formula graphs")
    print("3. python main.py                  - Run full analysis")
    print("4. notepad FORMULAS_EXPLAINED.md   - Read complete guide")
    print("5. explorer results\\plots          - Open graphs folder")
    print("="*60)
