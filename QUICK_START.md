# Stock Prediction System - Quick Start Guide

## 🚀 Getting Started

### Installation

```bash
# Install dependencies
pip install -r requirements.txt
```

### Running the Complete System

```bash
# 1. Prepare data (demo with 10 stocks)
python stock_ml_pipeline.py

# 2. Train all models (RF, XGBoost, LSTM)
python ml_models.py

# 3. Create visualizations
python create_visualizations.py

# 4. Test the system
python test_system.py
```

## 📊 Making Predictions

### Using Python API

```python
import pandas as pd
import joblib

# Load your stock data
X = pd.read_csv('your_stock_features.csv')

# Load models
rf_reg = joblib.load('models/regression_random_forest.joblib')
rf_class = joblib.load('models/classification_random_forest.joblib')

# Predict price
predicted_price = rf_reg.predict(X)
print(f"Predicted Price: ₹{predicted_price[0]:.2f}")

# Predict action
action = rf_class.predict(X)
print(f"Trading Action: {action[0]}")

# Get confidence (probability)
action_prob = rf_class.predict_proba(X)
confidence = action_prob.max(axis=1) * 100
print(f"Confidence: {confidence[0]:.1f}%")
```

## 🎯 System Outputs

### 1. Price Prediction (Regression)
- **Output**: Future stock price (5 days ahead)
- **Models**: Linear Regression, Random Forest, XGBoost, LSTM
- **Metrics**: RMSE, MAE, R², MAPE, Directional Accuracy

### 2. Trading Action (Classification)
- **Output**: BUY, SELL, or HOLD recommendation
- **Models**: Logistic Regression, Random Forest, XGBoost, LSTM
- **Metrics**: Accuracy, Precision, Recall, F1-Score

### 3. Confidence Score
- **Output**: 0-100% confidence in prediction
- **Method**: Based on model probability/ensemble agreement  

## 📁 Output Files

### Models (saved to `models/`)
- `regression_random_forest.joblib` - Best price predictor
- `classification_random_forest.joblib` - Best action predictor
- `regression_xgboost.joblib` - Alternative price model
- `classification_xgboost.joblib` - Alternative action model
- `scaler.joblib` - Feature scaler
- `label_encoder.joblib` - Label encoder

### Visualizations (saved to `results/visualizations/`)
- Price prediction charts
- Error analysis plots
- Confusion matrices
- Feature importance rankings
- Trading signal overlays

### Data (saved to `data/processed/`)
- `train_data.csv` - Training set (70%)
- `val_data.csv` - Validation set (15%)
- `test_data.csv` - Test set (15%)
- `full_dataset.csv` - Complete processed data

## 🔧 Configuration

### Change Prediction Horizon

```python
# In target_generator.py
generator.create_all_targets(horizons=[1, 5, 10, 20])  # Days ahead
```

### Adjust BUY/SELL Thresholds

```python
# In target_generator.py
generator.create_classification_target(
    horizon=5,
    buy_threshold=0.03,   # 3% gain = BUY
    sell_threshold=-0.03  # 3% loss = SELL
)
```

### Use All 52 Stocks

```python
# In stock_ml_pipeline.py, modify main():
pipeline.run_pipeline(limit_stocks=None)  # Remove limit
```

## 📊 Current Dataset

- **Stocks**: 10 demo stocks (can scale to 52)
- **Records**: 24,560 total
- **Features**: 59 technical indicators
- **Split**: Train (17,190) / Val (3,680) / Test (3,690)
- **Date Range**: 2000-2021

## 🎓 Model Performance

| Task | Model | Primary Metric |
|------|-------|----------------|
| Regression | Random Forest | Best R² score |
| Regression | XGBoost | Fast training |
| Regression | LSTM | Deep learning |
| Classification | Random Forest | Best F1-score |
| Classification | XGBoost | Fast inference |
| Classification | LSTM | Sequence modeling |

## 🆘 Troubleshooting

**Issue**: TensorFlow/LSTM errors
- **Solution**: Use Random Forest or XGBoost models (fully functional)

**Issue**: Out of memory
- **Solution**: Reduce number of stocks or use sampling

**Issue**: Slow training
- **Solution**: Reduce `n_estimators` in model parameters

## 📞 Support

See the full `walkthrough.md` for detailed documentation, architecture, and advanced usage.

---

**System Status**: ✅ Fully Operational
**Last Updated**: December 2025
