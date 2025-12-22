# 📐 Stock Market Analysis - Complete Formula Guide

## 📊 Technical Indicator Formulas

### 1. Simple Moving Average (SMA)

**Formula Type:** Statistical Average

**Formula:**
```
SMA(n) = (P₁ + P₂ + P₃ + ... + Pₙ) / n

Where:
- P = Price (usually Close price)
- n = Number of periods (e.g., 10, 50 days)
```

**Python Implementation:**
```python
# Calculated by pandas
SMA_10 = stock_df['Close'].rolling(window=10).mean()
SMA_50 = stock_df['Close'].rolling(window=50).mean()
```

**Calculated By:** pandas `.rolling().mean()` - Library calculates this

**What It SHOWS in Trading:**
- **Trend Direction**: Line going up = uptrend, down = downtrend
- **Support/Resistance**: Price often bounces off SMA lines
- **Golden Cross**: SMA_10 crosses above SMA_50 = STRONG BUY signal
- **Death Cross**: SMA_10 crosses below SMA_50 = STRONG SELL signal
- **Above SMA**: Price above SMA = Bullish (BUY zone)
- **Below SMA**: Price below SMA = Bearish (SELL zone)

**Interpretation:**
- Short SMA (10-day): Responds quickly to price changes
- Long SMA (50-day): Shows long-term trend
- Smooth out price volatility

---

### 2. Exponential Moving Average (EMA)

**Formula Type:** Weighted Average (Recent prices matter more)

**Formula:**
```
EMA(today) = (Price(today) × K) + (EMA(yesterday) × (1 - K))

Where:
K = 2 / (n + 1)  (Smoothing factor)
n = Number of periods

Example for 12-day EMA:
K = 2 / (12 + 1) = 0.1538 (15.38%)
```

**Step-by-Step:**
```
Day 1 EMA = SMA of first n periods
Day 2 EMA = (Close × 0.1538) + (Previous EMA × 0.8462)
Day 3 EMA = (Close × 0.1538) + (Previous EMA × 0.8462)
...and so on
```

**Python Implementation:**
```python
# Calculated by pandas
EMA_12 = stock_df['Close'].ewm(span=12, adjust=False).mean()
EMA_26 = stock_df['Close'].ewm(span=26, adjust=False).mean()
```

**Calculated By:** pandas `.ewm().mean()` - Library calculates this

**What It SHOWS in Trading:**
- **Faster Response**: Reacts quicker than SMA to price changes
- **Recent Price Weight**: Today's price has 15-20% impact on EMA
- **Trend Confirmation**: Price above EMA = Uptrend
- **Entry/Exit Points**: EMA crossovers signal trades
- **EMA Crossover**: EMA_12 crosses EMA_26 = Trend change signal

**Interpretation:**
- More sensitive to recent price movements than SMA
- Better for short-term trading
- Used in MACD calculation

---

### 3. Relative Strength Index (RSI)

**Formula Type:** Momentum Oscillator (0-100 scale)

**Formula:**
```
RSI = 100 - (100 / (1 + RS))

Where:
RS = Average Gain / Average Loss (over n periods, typically 14)

Step-by-step:
1. Calculate price changes: Δ = Close(today) - Close(yesterday)
2. Separate gains and losses:
   - Gain = Δ if positive, else 0
   - Loss = |Δ| if negative, else 0
3. Average Gain = Sum of gains over 14 days / 14
4. Average Loss = Sum of losses over 14 days / 14
5. RS = Average Gain / Average Loss
6. RSI = 100 - (100 / (1 + RS))
```

**Python Implementation:**
```python
# Manual calculation (we calculate this ourselves)
delta = stock_df['Close'].diff()
gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
rs = gain / loss
RSI_14 = 100 - (100 / (1 + rs))
```

**Calculated By:** **We calculate this** using pandas operations (not a single library function)

**What It SHOWS in Trading:**
- **RSI > 70**: OVERBOUGHT = Stock too expensive, likely to drop (SELL signal)
- **RSI < 30**: OVERSOLD = Stock too cheap, likely to rise (BUY signal)
- **RSI = 50**: Neutral, no clear signal
- **RSI 40-60**: Normal trading range
- **Divergence**: Price goes up but RSI goes down = Reversal coming

**Interpretation:**
```
RSI Value    Trading Signal       Action
---------    --------------       ------
0-30         Oversold            STRONG BUY
30-40        Weak                Potential BUY
40-60        Neutral             HOLD
60-70        Strong              Potential SELL
70-100       Overbought          STRONG SELL
```

**Example:**
If RSI = 75, stock is overbought, 75% of traders are buying. Soon, sellers will dominate → Price drops → SELL!

---

### 4. MACD (Moving Average Convergence Divergence)

**Formula Type:** Trend and Momentum Indicator

**Formula:**
```
MACD Line = EMA(12) - EMA(26)
Signal Line = EMA(9) of MACD Line
MACD Histogram = MACD Line - Signal Line
```

**Detailed Calculation:**
```
Step 1: Calculate fast EMA
EMA_12 = 12-day exponential moving average

Step 2: Calculate slow EMA
EMA_26 = 26-day exponential moving average

Step 3: Calculate MACD Line
MACD = EMA_12 - EMA_26

Step 4: Calculate Signal Line
Signal = 9-day EMA of MACD

Step 5: Calculate Histogram
Histogram = MACD - Signal
```

**Python Implementation:**
```python
# We calculate this ourselves using pandas
stock_df['MACD'] = stock_df['EMA_12'] - stock_df['EMA_26']
stock_df['MACD_signal'] = stock_df['MACD'].ewm(span=9, adjust=False).mean()
stock_df['MACD_hist'] = stock_df['MACD'] - stock_df['MACD_signal']
```

**Calculated By:** **We calculate this** using EMA results

**What It SHOWS in Trading:**

**MACD Line:**
- Above 0: Bullish (short-term MA above long-term MA)
- Below 0: Bearish (short-term MA below long-term MA)

**Signal Crossovers:**
- **MACD crosses above Signal**: BULLISH = BUY signal
- **MACD crosses below Signal**: BEARISH = SELL signal

**Histogram:**
- Positive (green bars): Bullish momentum
- Negative (red bars): Bearish momentum
- Growing bars: Momentum increasing
- Shrinking bars: Momentum decreasing

**Interpretation:**
```
Condition                      Trading Signal
---------                      --------------
MACD > Signal, Histogram > 0   STRONG BUY
MACD crosses above Signal      BUY (Bullish crossover)
MACD < Signal, Histogram < 0   STRONG SELL
MACD crosses below Signal      SELL (Bearish crossover)
MACD near 0                    Weak trend, HOLD
```

**Example:**
MACD = 2.5, Signal = 1.5, Histogram = 1.0
→ MACD is above signal → Bullish → BUY signal!

---

### 5. Volatility

**Formula Type:** Statistical Dispersion (Risk Measure)

**Formula:**
```
Volatility = Standard Deviation of Returns over n periods

Step-by-step:
1. Calculate daily returns:
   Return(t) = (Price(t) - Price(t-1)) / Price(t-1)

2. Calculate mean return:
   Mean = Sum of all returns / n

3. Calculate variance:
   Variance = Σ(Return - Mean)² / (n - 1)

4. Calculate standard deviation:
   Volatility = √Variance
```

**Mathematical Formula:**
```
σ = √(Σ(xᵢ - μ)² / (n-1))

Where:
σ = Volatility (standard deviation)
xᵢ = Individual return
μ = Mean return
n = Number of periods
```

**Python Implementation:**
```python
# Calculated by pandas
stock_df['Returns'] = stock_df['Close'].pct_change()
stock_df['Volatility_10'] = stock_df['Returns'].rolling(window=10).std()
stock_df['Volatility_20'] = stock_df['Returns'].rolling(window=20).std()
```

**Calculated By:** pandas `.rolling().std()` - Library calculates this

**What It SHOWS in Trading:**
- **High Volatility (>3%)**: Stock price swings wildly = HIGH RISK, HIGH REWARD
- **Low Volatility (<1%)**: Stock price stable = LOW RISK, LOW REWARD
- **Increasing Volatility**: Market uncertainty, prepare for big moves
- **Decreasing Volatility**: Market stabilizing, less risk

**Interpretation:**
```
Volatility Level    Risk Level    Trading Strategy
----------------    ----------    ----------------
< 1%                Very Low      Safe for long-term hold
1-2%                Low           Normal trading
2-3%                Medium        Use stop-loss orders
3-5%                High          Day trading, quick exits
> 5%                Very High     Extreme caution, options trading
```

**Example:**
Volatility = 0.025 (2.5%)
→ Stock typically moves ±2.5% per day
→ If price = $100, expect daily range: $97.50 - $102.50

---

### 6. Price Return

**Formula Type:** Percentage Change

**Formula:**
```
Return(t) = (Price(t) - Price(t-1)) / Price(t-1)

Or in percentage:
Return(t) = ((Price(t) - Price(t-1)) / Price(t-1)) × 100%
```

**Alternative notation:**
```
Return = (Current Price - Previous Price) / Previous Price
       = (ΔPrice) / Previous Price
```

**Python Implementation:**
```python
# Calculated by pandas
stock_df['Returns'] = stock_df['Close'].pct_change()
# This is equivalent to:
# (Close[today] - Close[yesterday]) / Close[yesterday]
```

**Calculated By:** pandas `.pct_change()` - Library calculates this

**What It SHOWS in Trading:**
- **Positive Return (+)**: Stock went up, profit
- **Negative Return (-)**: Stock went down, loss
- **Return > 2%**: Strong upward movement (bullish)
- **Return < -2%**: Strong downward movement (bearish)
- **Return near 0%**: No significant movement

**Interpretation:**
```
Return Value    Meaning           Trading Signal
------------    -------           --------------
> +5%           Huge gain         Consider taking profit (SELL)
+2% to +5%      Good gain         HOLD or partial SELL
+0% to +2%      Small gain        HOLD
0%              No change         HOLD
-2% to 0%       Small loss        HOLD or review
-5% to -2%      Moderate loss     Consider SELL or add stop-loss
< -5%           Large loss        SELL or re-evaluate position
```

**Example:**
Yesterday: $100, Today: $105
Return = (105 - 100) / 100 = 0.05 = 5%
→ Stock gained 5% in one day → Strong performance!

---

## 🎯 BUY/SELL Label Creation Formula

**Formula Type:** Classification Logic (Binary Label)

**Formula:**
```
IF Price(tomorrow) > Price(today) THEN
    Label = BUY (1)
ELSE
    Label = SELL (0)

More precisely:
Label = 1 if Close(t+1) > Close(t) else 0
```

**With Return Threshold:**
```
IF Return(tomorrow) > Threshold (e.g., 0.5%) THEN
    Label = BUY (1)
ELSE IF Return(tomorrow) < -Threshold THEN
    Label = SELL (0)
ELSE
    Label = HOLD (not used in binary classification)
```

**Python Implementation:**
```python
# Create future price
df['Future_Price'] = df.groupby('Symbol')['Close'].shift(-1)

# Create label
df['Label'] = (df['Future_Price'] > df['Close']).astype(int)

# With threshold (0.5%)
df['Future_Return'] = (df['Future_Price'] - df['Close']) / df['Close']
df['Label'] = (df['Future_Return'] > 0.005).astype(int)
```

**Calculated By:** **We create this** (conceptual, not from a library)

**What It SHOWS in Trading:**
- **Label = 1 (BUY)**: Tomorrow's price will be higher, profit opportunity
- **Label = 0 (SELL)**: Tomorrow's price will be lower, avoid or short
- **Purpose**: Train ML model to predict profitable trades

**Interpretation:**
```
Current State           Future Prediction    Action
-------------           -----------------    ------
All indicators + Label  Price will rise     BUY stock today
All indicators + Label  Price will fall     SELL/avoid stock today
```

---

## 🤖 Logistic Regression Probability Formula

**Formula Type:** Sigmoid Function (Probability Output)

**Formula:**
```
P(Y=1|X) = 1 / (1 + e^(-z))

Where:
z = β₀ + β₁X₁ + β₂X₂ + ... + βₙXₙ

Components:
- P(Y=1|X) = Probability of BUY signal given features X
- e = Euler's number (≈ 2.71828)
- z = Linear combination of features
- β₀ = Intercept (bias)
- β₁, β₂, ..., βₙ = Feature weights
- X₁, X₂, ..., Xₙ = Feature values (SMA, RSI, MACD, etc.)
```

**Detailed Calculation:**
```
Step 1: Calculate linear combination (z)
z = β₀ + (β₁ × SMA_10) + (β₂ × RSI) + (β₃ × MACD) + ...

Step 2: Apply sigmoid function
P = 1 / (1 + e^(-z))

Step 3: Interpret probability
IF P > 0.5 THEN Predict BUY (1)
ELSE Predict SELL (0)
```

**Example Calculation:**
```
Assume trained weights:
β₀ = -2.0 (intercept)
β₁ = 0.05 (SMA_10 weight)
β₂ = 0.03 (RSI weight)
β₃ = 0.8 (MACD weight)

Current values:
SMA_10 = 150
RSI = 65
MACD = 2.5

Step 1: Calculate z
z = -2.0 + (0.05 × 150) + (0.03 × 65) + (0.8 × 2.5)
z = -2.0 + 7.5 + 1.95 + 2.0
z = 9.45

Step 2: Calculate probability
P = 1 / (1 + e^(-9.45))
P = 1 / (1 + 0.000079)
P = 0.9999 ≈ 99.99%

Interpretation: 99.99% probability that price will rise → STRONG BUY!
```

**Python Implementation:**
```python
# Calculated by scikit-learn
from sklearn.linear_model import LogisticRegression

model = LogisticRegression()
model.fit(X_train, y_train)

# Get probabilities
probabilities = model.predict_proba(X_test)
# Returns: [[P(SELL), P(BUY)] for each sample]
```

**Calculated By:** scikit-learn `LogisticRegression` - Library calculates this

**What It SHOWS in Trading:**
- **P > 0.8 (80%)**: High confidence BUY signal
- **P = 0.5-0.8**: Moderate BUY signal
- **P = 0.5**: Neutral, coin flip
- **P = 0.2-0.5**: Moderate SELL signal
- **P < 0.2 (20%)**: High confidence SELL signal

**Interpretation:**
```
Probability    Confidence    Trading Action
-----------    ----------    --------------
> 0.90         Very High     STRONG BUY - Invest heavily
0.70 - 0.90    High          BUY - Good opportunity
0.60 - 0.70    Moderate      BUY - Small position
0.50 - 0.60    Weak          HOLD - Wait for clarity
0.40 - 0.50    Weak          HOLD or small SELL
0.30 - 0.40    Moderate      SELL - Exit position
0.10 - 0.30    High          STRONG SELL - Exit immediately
< 0.10         Very High     STRONG SELL - Short opportunity
```

---

## 📚 Library vs Manual Calculations

### ✅ Calculated by Libraries (pandas/sklearn):
1. **SMA**: `rolling().mean()` - pandas calculates average
2. **EMA**: `ewm().mean()` - pandas calculates weighted average
3. **Volatility**: `rolling().std()` - pandas calculates standard deviation
4. **Returns**: `pct_change()` - pandas calculates percentage change
5. **Logistic Regression**: `LogisticRegression()` - sklearn trains and predicts

### 🛠️ Calculated Manually (we implement):
1. **RSI**: We calculate gains/losses, then apply formula
2. **MACD**: We combine EMAs and apply formula
3. **BUY/SELL Labels**: We create based on future price logic

### 💡 Conceptual (ML learns patterns):
1. **Feature Weights (β)**: Learned by ML model during training
2. **Decision Boundaries**: Learned by model to separate BUY/SELL
3. **Predictions**: Model combines all indicators using learned weights

---

## 🎯 Summary: What Each Formula Shows

| Formula | Shows | BUY Signal | SELL Signal |
|---------|-------|------------|-------------|
| **SMA** | Trend direction | Price > SMA, Golden Cross | Price < SMA, Death Cross |
| **EMA** | Recent trend | Price > EMA, Fast > Slow | Price < EMA, Fast < Slow |
| **RSI** | Momentum strength | RSI < 30 (oversold) | RSI > 70 (overbought) |
| **MACD** | Trend change | MACD > Signal, Hist > 0 | MACD < Signal, Hist < 0 |
| **Volatility** | Risk level | Low volatility = safe entry | High volatility = risky |
| **Returns** | Price change % | Positive returns | Negative returns |
| **Label** | Future direction | Label = 1 (price will rise) | Label = 0 (price will fall) |
| **Probability** | Confidence | P > 0.7 (confident BUY) | P < 0.3 (confident SELL) |

---

## 🔍 Real Trading Example

**Stock: XYZ, Current Price: $100**

```
Indicator Values:
- SMA_10: $98 → Price above SMA ✅ (Bullish)
- SMA_50: $95 → SMA_10 > SMA_50 ✅ (Golden Cross)
- RSI: 45 → Neutral, but below 50 (Slight bullish)
- MACD: 1.2, Signal: 0.8 → MACD > Signal ✅ (Bullish)
- MACD_Hist: 0.4 → Positive and growing ✅ (Strong)
- Volatility: 2.1% → Normal risk ✅
- Returns: +1.5% → Positive momentum ✅

ML Model Calculation:
z = -2.0 + (0.05×98) + (0.03×45) + (0.8×1.2) + ...
z = 5.5
P(BUY) = 1 / (1 + e^(-5.5)) = 0.996 = 99.6%

DECISION: STRONG BUY! 🚀
Confidence: 99.6%
Expected: Price will rise tomorrow
```

---

**This guide covers ALL formulas used in your ML stock prediction project!** 📊
