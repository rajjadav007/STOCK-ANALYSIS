# 🚀 HOW TO RUN THE STOCK PREDICTION SYSTEM

## Step-by-Step Guide to Run and See Results

### 📋 Prerequisites
```bash
# Make sure you're in the project directory
cd D:\STOCK-ANALYSIS

# Verify Python is working
python --version
```

---

## 🎯 Option 1: Quick Demo (See Results Immediately)

### Run the Final Report
```bash
python generate_report.py
```

**What you'll see:**
- ✅ Model performance metrics (Accuracy, RMSE, MAE, R²)
- ✅ 10 sample predictions with actual vs predicted values
- ✅ Top 15 most important features
- ✅ Results saved to `results/final_predictions.csv`

**Expected Output:**
```
STOCK PREDICTION SYSTEM - FINAL REPORT
Test Set: 3,640 samples with 49 features

RANDOM FOREST MODEL PERFORMANCE
💰 PRICE PREDICTION (Regression):
  RMSE: ₹XX.XX
  MAE:  ₹XX.XX
  R²:   0.XXXX

📊 TRADING SIGNALS (Classification):
  Accuracy: XX.XX%
  F1-Score: 0.XXXX

🔮 Sample 1:
   Price:  Actual=₹XXX.XX, Predicted=₹XXX.XX, Error=₹X.XX ✅
   Action: Actual=BUY, Predicted=BUY, Confidence=XX.X% ✅
...
```

---

## 🎨 Option 2: View Visualizations

### Generate All Charts
```bash
python create_visualizations.py
```

**What you'll get:**
- 📊 Price prediction charts
- 📈 Confusion matrices
- 🎯 Feature importance plots
- 📉 Error analysis
- 💹 Trading signals overlay

**Results Location:** `results/visualizations/`

**Files Created:**
- `price_prediction_random_forest.png`
- `error_analysis_random_forest.png`
- `confusion_matrix_random_forest.png`
- `feature_importance_random_forest_(regression).png`
- `feature_importance_random_forest_(classification).png`
- `trading_signals_*.png`

### View the Images:
1. Open File Explorer
2. Navigate to `D:\STOCK-ANALYSIS\results\visualizations\`
3. Double-click any PNG file to view

---

## 🔧 Option 3: Run Complete System (From Scratch)

### Step 1: Prepare Data
```bash
python stock_ml_pipeline.py
```

**Output:** Creates train/val/test splits in `data/processed/`
**Time:** ~30 seconds

### Step 2: Train Models
```bash
python ml_models.py
```

**Output:** Trains all models (RF, XGBoost, LSTM)
**Time:** ~2-5 minutes
**Models saved to:** `models/` directory

### Step 3: Generate Report & Visualizations
```bash
python generate_report.py
python create_visualizations.py
```

---

## 📊 Option 4: Make Custom Predictions

### Create a test script:
```python
# test_prediction.py
import joblib
import pandas as pd

# Load model
model = joblib.load('models/regression_random_forest.joblib')
classifier = joblib.load('models/classification_random_forest.joblib')

# Load test data
test_data = pd.read_csv('data/processed/test_data.csv')

from target_generator import TargetGenerator
gen = TargetGenerator(test_data)
data_with_targets = gen.create_all_targets(horizons=[5])
X, y_price, y_action = gen.get_feature_target_split(horizon=5)

# Make prediction on first sample
price_pred = model.predict(X.head(1))
action_pred = classifier.predict(X.head(1))
confidence = classifier.predict_proba(X.head(1)).max() * 100

print(f"Predicted Price: ₹{price_pred[0]:.2f}")
print(f"Trading Action: {action_pred[0]}")
print(f"Confidence: {confidence:.1f}%")
```

### Run it:
```bash
python test_prediction.py
```

---

## 📁 Where to Find Results

### 1. Predictions Data
- **File:** `results/final_predictions.csv`
- **Open with:** Excel or any CSV viewer
- **Contains:** All predictions with actual vs predicted values

### 2. Visualizations
- **Folder:** `results/visualizations/`
- **Files:** PNG images
- **View:** Any image viewer

### 3. Models
- **Folder:** `models/`
- **Files:** .joblib and .h5 files
- **Use:** Load with joblib or Keras

### 4. Processed Data
- **Folder:** `data/processed/`
- **Files:** 
  - `train_data.csv` (17,190 samples)
  - `val_data.csv` (3,680 samples)
  - `test_data.csv` (3,690 samples)

---

## 🔍 Quick System Test

```bash
python test_system.py
```

**This will verify:**
- ✅ Data pipeline working
- ✅ Models loading correctly
- ✅ Predictions generating
- ✅ All components operational

---

## 📱 Quick Commands Cheat Sheet

```bash
# See model performance + sample predictions
python generate_report.py

# Create all charts
python create_visualizations.py

# Verify system
python test_system.py

# View results
start results\visualizations\              # Windows
explorer results\visualizations\           # Windows Explorer

# Open predictions CSV
start results\final_predictions.csv        # Opens in Excel
```

---

## 🎯 What Each File Does

| File | Purpose | Output |
|------|---------|--------|
| `stock_ml_pipeline.py` | Data preparation | Processed CSVs |
| `target_generator.py` | Create targets | Data with labels |
| `ml_models.py` | Train models | Saved models |
| `generate_report.py` | Show results | Console + CSV |
| `create_visualizations.py` | Make charts | PNG images |
| `test_system.py` | Verify system | Status check |

---

## 💡 TIP: Quick Start

**For fastest results, just run:**
```bash
python generate_report.py
```

This will show you everything working with actual predictions!

**To see charts:**
```bash
python create_visualizations.py
start results\visualizations\
```

---

## ❓ Troubleshooting

**Problem:** "File not found"
- **Solution:** Make sure you're in `D:\STOCK-ANALYSIS` directory

**Problem:** "Module not found"
- **Solution:** Run `pip install -r requirements.txt`

**Problem:** "No such file: train_data.csv"
- **Solution:** Run `python stock_ml_pipeline.py` first

**Problem:** "Models not found"
- **Solution:** Run `python ml_models.py` first

---

## 🎉 Ready to Go!

The system is already trained with models in `models/` directory.

**Just run:** `python generate_report.py` to see it in action! 🚀
