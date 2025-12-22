# 📊 SUPERVISED LEARNING LABELS - DOCUMENTATION

## Overview
The dataset has been converted into a **supervised machine learning problem** with **BUY/SELL/HOLD labels** based on future price movements.

---

## 🎯 Labeling Strategy

### Parameters
- **Future Window**: 5 days (look ahead)
- **BUY Threshold**: +2% (future return ≥ 2%)
- **SELL Threshold**: -2% (future return ≤ -2%)
- **HOLD Range**: Between -2% and +2%

### Logic
```python
Future_Return = (Price[t+5] - Price[t]) / Price[t]

if Future_Return >= 0.02:
    Label = 'BUY'   # Price will rise by 2% or more
elif Future_Return <= -0.02:
    Label = 'SELL'  # Price will drop by 2% or more
else:
    Label = 'HOLD'  # Price will stay relatively stable
```

---

## 📈 Label Distribution

| Label | Count | Percentage |
|-------|-------|------------|
| **BUY**  | 809   | 33.01%     |
| **HOLD** | 922   | 37.62%     |
| **SELL** | 720   | 29.38%     |
| **TOTAL**| 2,451 | 100.00%    |

**Balance**: The dataset is reasonably balanced across all three classes, which is ideal for ML training.

---

## 📊 Future Return Statistics

### BUY Label (809 samples)
- **Mean Return**: +5.62%
- **Median Return**: +4.73%
- **Range**: +2.00% to +26.38%
- **Interpretation**: Stocks labeled BUY increased by an average of 5.62% in the next 5 days

### HOLD Label (922 samples)
- **Mean Return**: +0.02%
- **Median Return**: +0.03%
- **Range**: -1.99% to +2.00%
- **Interpretation**: Stocks labeled HOLD stayed relatively stable with minimal movement

### SELL Label (720 samples)
- **Mean Return**: -5.86%
- **Median Return**: -4.08%
- **Range**: -51.50% to -2.00%
- **Interpretation**: Stocks labeled SELL decreased by an average of 5.86% in the next 5 days

---

## 🎓 ML Problem Type

### Classification Problem
This is now a **multi-class classification problem**:
- **Input**: 24 technical indicator features
- **Output**: One of 3 classes (BUY, HOLD, SELL)

### Suitable Algorithms
1. **Logistic Regression** (multi-class)
2. **Random Forest Classifier**
3. **XGBoost Classifier**
4. **Neural Networks**
5. **Support Vector Machines (SVM)**

---

## 📁 Dataset Files

### Input Files
- `data/processed/cleaned_stock_data.csv` - Clean stock data (5,306 records)

### Output Files
- `data/processed/labeled_stock_data.csv` - Dataset with BUY/SELL/HOLD labels (2,451 records)
  - **Columns**: All original features + technical indicators + `Label` + `Future_Return`
  - **Date Range**: 2011-06-01 to 2021-04-23
  - **Records Lost**: 2,855 (53.81%) due to NaN values from indicators and future window

---

## 🔧 Features in Dataset

### Total Features: 38 columns

#### Price Features (6)
1. Open
2. High
3. Low
4. Close
5. Price_Change
6. Price_Range

#### Volume Features (3)
7. Volume
8. Volume_lag_1
9. Volume_Change_Pct

#### Technical Indicators (13)
10. SMA_10 (Simple Moving Average - 10 days)
11. SMA_50 (Simple Moving Average - 50 days)
12. EMA_12 (Exponential Moving Average - 12 days)
13. EMA_26 (Exponential Moving Average - 26 days)
14. RSI_14 (Relative Strength Index - 14 days)
15. MACD (Moving Average Convergence Divergence)
16. MACD_signal (MACD Signal Line)
17. MACD_hist (MACD Histogram)
18. Volatility_10 (10-day volatility)
19. Volatility_20 (20-day volatility)
20. Returns (Daily returns)
21. Close_lag_1 (Previous day close)
22. Future_Close (Close price 5 days ahead)

#### Time Features (4)
23. Year
24. Month
25. DayOfWeek
26. Quarter

#### Target Variables (2)
27. **Future_Return** (Continuous: actual % change in 5 days)
28. **Label** (Categorical: BUY/HOLD/SELL)

---

## 💡 How to Use This Dataset

### Option 1: Classification (Recommended)
Train a classifier to predict BUY/SELL/HOLD labels:
```python
from sklearn.ensemble import RandomForestClassifier

X = df[feature_columns]
y = df['Label']  # Categorical target

model = RandomForestClassifier()
model.fit(X_train, y_train)
predictions = model.predict(X_test)
```

### Option 2: Regression → Classification
Train a regressor to predict Future_Return, then convert to labels:
```python
from sklearn.ensemble import RandomForestRegressor

X = df[feature_columns]
y = df['Future_Return']  # Continuous target

model = RandomForestRegressor()
model.fit(X_train, y_train)
predicted_returns = model.predict(X_test)

# Convert to labels
predicted_labels = ['BUY' if r >= 0.02 else 'SELL' if r <= -0.02 else 'HOLD' 
                   for r in predicted_returns]
```

---

## 📊 Visualization

Run the analysis script to see comprehensive visualizations:
```bash
python view_labels.py
```

**Graphs include**:
1. Label distribution (pie chart)
2. Label counts (bar chart)
3. Future return distribution by label (box plot)
4. Overall return distribution with thresholds
5. Label distribution over time
6. Mean future return by label

**Output**: `results/plots/label_analysis.png`

---

## 🚀 Next Steps

### 1. Train Classification Models
```bash
# Update main.py to use classification instead of regression
python main.py
```

### 2. Evaluate with Classification Metrics
- **Accuracy**: Overall correctness
- **Precision**: How many predicted BUYs were correct
- **Recall**: How many actual BUYs were caught
- **F1-Score**: Harmonic mean of precision and recall
- **Confusion Matrix**: Detailed breakdown of predictions

### 3. Trading Strategy
Use the model predictions for trading decisions:
- **BUY**: If model predicts BUY → Purchase stock
- **SELL**: If model predicts SELL → Exit position
- **HOLD**: If model predicts HOLD → Maintain current position

### 4. Backtesting
Test the strategy on historical data to measure:
- Total return
- Win rate
- Maximum drawdown
- Sharpe ratio

---

## ⚙️ Customization

### Change Window Size
```python
# In main.py, modify create_labels() call:
df = self.create_labels(df, 
                       future_window=10,      # Look 10 days ahead
                       buy_threshold=0.03,    # 3% for BUY
                       sell_threshold=-0.03)  # -3% for SELL
```

### Different Thresholds
- **Conservative**: ±3% (fewer trades, higher confidence)
- **Aggressive**: ±1% (more trades, lower confidence)
- **Current**: ±2% (balanced approach)

---

## 📝 Summary

✅ **Dataset converted to supervised learning problem**
✅ **BUY/SELL/HOLD labels created using 5-day future window**
✅ **±2% thresholds for trading signals**
✅ **2,451 labeled samples ready for training**
✅ **Balanced distribution: 33% BUY, 38% HOLD, 29% SELL**
✅ **38 features including technical indicators**
✅ **Comprehensive visualizations created**

**Ready for ML training! 🎉**
