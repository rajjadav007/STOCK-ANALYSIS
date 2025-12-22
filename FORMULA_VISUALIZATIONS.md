# 📚 Formula Visualizations - Quick Reference

## 🎯 What Was Created

I've created **comprehensive visual explanations** of all formulas used in your stock market ML project!

## 📊 Formula Graphs Created (8 total)

### 1. **SMA Formula** (`formula_sma.png`)
- Shows how SMA is calculated: (P₁ + P₂ + ... + Pₙ) / n
- 4 panels:
  - Price with SMA 10 & 50 lines
  - Formula breakdown with examples
  - Buy/Sell signals (Golden Cross, Death Cross)
  - Distance from SMA (momentum indicator)
- **Trading Signal**: Price > SMA = BUY, Price < SMA = SELL

### 2. **EMA Formula** (`formula_ema.png`)
- Shows how EMA weights recent prices more
- Formula: EMA = (Price × K) + (Previous EMA × (1-K))
- 4 panels:
  - EMA vs SMA comparison (shows EMA responds faster)
  - Formula breakdown with K factor calculation
  - Responsiveness comparison
  - Weight distribution chart
- **Trading Signal**: EMA₁₂ > EMA₂₆ = BUY

### 3. **RSI Formula** (`formula_rsi.png`)
- Shows momentum indicator (0-100 scale)
- Formula: RSI = 100 - (100 / (1 + RS))
- 3 panels:
  - Stock price chart
  - RSI with overbought (>70) and oversold (<30) zones
  - Formula calculation steps
  - Trading signals table
- **Trading Signal**: RSI < 30 = BUY, RSI > 70 = SELL

### 4. **MACD Formula** (`formula_macd.png`)
- Shows trend and momentum indicator
- Formula: MACD = EMA₁₂ - EMA₂₆
- 4 panels showing step-by-step:
  - Step 1: Calculate EMAs
  - Step 2: Calculate MACD line
  - Step 3: Add signal line and identify crossovers
  - Step 4: Histogram (momentum visualization)
- **Trading Signal**: MACD crosses above Signal = BUY

### 5. **Volatility Formula** (`formula_volatility.png`)
- Shows risk measurement (standard deviation)
- Formula: σ = √(Σ(Return - Mean)² / n)
- 4 panels:
  - Price with volatility zones
  - Returns distribution histogram
  - Volatility over time (10 & 20 day)
  - Formula explanation
- **Trading Signal**: High volatility = High risk

### 6. **Returns Formula** (`formula_returns.png`)
- Shows percentage change calculation
- Formula: Return = (Price_today - Price_yesterday) / Price_yesterday
- 4 panels:
  - Price chart with annotations
  - Daily returns bar chart
  - Cumulative returns (total gain/loss)
  - Formula examples
- **Trading Signal**: Positive returns = Profit, Negative = Loss

### 7. **BUY/SELL Labels** (`formula_labels.png`)
- Shows how training labels are created
- Formula: Label = 1 if Price(tomorrow) > Price(today) else 0
- 4 panels:
  - Price chart with labels
  - Label distribution
  - Example: today vs tomorrow comparison
  - Formula explanation with examples
- **Purpose**: Train ML model to predict future direction

### 8. **Logistic Regression** (`formula_logistic.png`)
- Shows probability calculation
- Formula: P = 1 / (1 + e^(-z)), where z = β₀ + Σ(βᵢ × featureᵢ)
- 4 panels:
  - Sigmoid function curve
  - Probability interpretation zones
  - Example calculation with real numbers
  - Decision zones table
- **Trading Signal**: P > 0.7 = BUY, P < 0.3 = SELL

## 🎨 How to View

### Method 1: View All Graphs
```bash
python view_graphs.py
```
Will show all 12 graphs (4 results + 8 formulas)

### Method 2: Open Specific Formula
```bash
# Windows
start results/plots/formula_sma.png
start results/plots/formula_rsi.png
# ... etc
```

### Method 3: Open Folder
```bash
explorer results\plots
```
Then double-click any `formula_*.png` file

## 📖 Documentation

### Complete Written Guide
**File**: `FORMULAS_EXPLAINED.md`
- Detailed explanation of each formula
- Step-by-step calculations
- Trading interpretations
- Real examples
- Library vs manual calculations

### Quick Commands
```bash
# View documentation
notepad FORMULAS_EXPLAINED.md

# Create all formula visualizations
python visualize_formulas.py

# View all graphs
python view_graphs.py
```

## 🎯 What Each Formula Shows (Summary)

| Formula | What It Shows | Calculated By | BUY Signal | SELL Signal |
|---------|---------------|---------------|------------|-------------|
| **SMA** | Trend direction | pandas `.rolling().mean()` | Price > SMA | Price < SMA |
| **EMA** | Recent trend | pandas `.ewm().mean()` | EMA fast > slow | EMA fast < slow |
| **RSI** | Momentum | **Manual** (we calculate) | RSI < 30 | RSI > 70 |
| **MACD** | Trend change | **Manual** (we calculate) | MACD > Signal | MACD < Signal |
| **Volatility** | Risk level | pandas `.rolling().std()` | Low = safe | High = risky |
| **Returns** | % change | pandas `.pct_change()` | Positive | Negative |
| **Labels** | Future direction | **Manual** (we create) | Label = 1 | Label = 0 |
| **Logistic** | Buy probability | sklearn `LogisticRegression` | P > 0.7 | P < 0.3 |

## 🔍 Key Insights from Graphs

### SMA (Simple Moving Average)
- **Golden Cross**: SMA₁₀ crosses above SMA₅₀ = STRONG BUY
- **Death Cross**: SMA₁₀ crosses below SMA₅₀ = STRONG SELL
- **Support/Resistance**: Price bounces off SMA lines

### RSI (Relative Strength Index)
- **Overbought (>70)**: Too many buyers, price will drop = SELL
- **Oversold (<30)**: Too many sellers, price will rise = BUY
- **Mean reversion**: Extreme RSI values return to 50

### MACD (Moving Average Convergence Divergence)
- **Bullish Crossover**: MACD crosses above signal = BUY
- **Bearish Crossover**: MACD crosses below signal = SELL
- **Histogram**: Shows momentum strength

### Volatility
- **High Volatility**: Large price swings = High risk
- **Low Volatility**: Stable prices = Low risk
- **Increasing**: Market uncertainty

### Logistic Regression
- **P > 0.9**: Very confident BUY (99%+ confidence)
- **P = 0.5**: Neutral (50-50 chance)
- **P < 0.1**: Very confident SELL

## 📚 File Structure

```
stock-analysis/
├── visualize_formulas.py          # ← Creates formula graphs
├── FORMULAS_EXPLAINED.md          # ← Complete written guide
├── FORMULA_VISUALIZATIONS.md      # ← This file
└── results/plots/
    ├── formula_sma.png            # ← SMA explanation
    ├── formula_ema.png            # ← EMA explanation
    ├── formula_rsi.png            # ← RSI explanation
    ├── formula_macd.png           # ← MACD explanation
    ├── formula_volatility.png     # ← Volatility explanation
    ├── formula_returns.png        # ← Returns explanation
    ├── formula_labels.png         # ← Labels explanation
    └── formula_logistic.png       # ← Logistic regression
```

## 🎓 Learning Path

1. **Start with**: `FORMULAS_EXPLAINED.md` (read the theory)
2. **Then view**: Formula graphs (visual understanding)
3. **Finally**: Run `main.py` to see formulas in action on real data

## 💡 Pro Tips

1. **Print these graphs** for reference while trading
2. **Compare formula graphs** with your actual stock data
3. **Use RSI + MACD together** for stronger signals
4. **Check volatility** before taking positions
5. **Trust high-probability predictions** (P > 0.8 or P < 0.2)

## 🎉 Complete Package

✅ **8 formula explanations** with visual graphs  
✅ **Complete written documentation** (FORMULAS_EXPLAINED.md)  
✅ **Trading signals** for each indicator  
✅ **Real examples** with calculations  
✅ **Buy/Sell zones** clearly marked  
✅ **Library vs manual** calculations explained  
✅ **300 DPI quality** for printing/presentations  

---

**Now you have complete understanding of every formula used in your ML project!** 📊📈🎓
