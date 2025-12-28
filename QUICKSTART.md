# 🚀 QUICK START GUIDE

**Get your stock prediction model running in 5 minutes!**

---

## ⚡ Quick Commands

### 1. Setup (First Time Only)
```powershell
# Activate virtual environment
.\.venv\Scripts\Activate.ps1

# Install dependencies
pip install -r requirements.txt
```

### 2. Train Model (Required First Run)
```powershell
python model_improvement_pipeline.py
```
⏱️ **Takes:** 5-15 minutes  
💾 **Creates:** Trained models in `models/` folder

### 3. View Results
```powershell
python visualize_results.py
```
📊 **Creates:** 5 visualization charts in `results/` folder

### 4. Make Predictions
```powershell
python production_predictor.py
```
🔮 **Shows:** Buy/Sell signals for all stocks

---

## 📋 Complete Workflow

```powershell
# ONE-LINE COMMAND (runs everything)
.\.venv\Scripts\Activate.ps1; python model_improvement_pipeline.py; python visualize_results.py; python production_predictor.py
```

---

## 🎯 What You Get

After running the commands above, you'll have:

✅ **Trained AI Models** (64% accuracy)  
✅ **Performance Charts** (5 visualizations)  
✅ **Stock Predictions** (Buy/Sell signals)  
✅ **Metrics Report** (JSON file)

---

## 📁 Check Your Results

```powershell
# Open results folder
explorer results

# View metrics
type results\improvement_metrics.json

# View predictions
type results\predictions.csv
```

---

## 🔄 Re-training Schedule

| Frequency | When | Command |
|-----------|------|---------|
| **Daily** | Active trading | `python production_predictor.py` |
| **Weekly** | Update model | `python model_improvement_pipeline.py` |
| **Monthly** | Full re-train | All 3 commands |

---

## 💡 Quick Tips

### ✅ Best Practices:
- Train model weekly for best accuracy
- Use predictions as ONE input (not sole decision)
- Always use stop-loss orders
- Start with paper trading first

### ❌ Common Mistakes:
- Not re-training regularly (accuracy degrades)
- Trading all signals (focus on high confidence)
- Ignoring market conditions (news, events)
- Over-leveraging (risk management!)

---

## 📊 Understanding Output

### Model Training Output:
```
✅ VERDICT: STRONG: R²=0.1964, Dir=64.32%
         ↑                ↑           ↑
      Status      Variance    Directional
                  Explained   Accuracy
```

### Prediction Output:
```
RELIANCE: +4.2%  ← Predicted return (percentage)
Signal: BUY      ← Trading recommendation
Confidence: HIGH ← How confident the model is
```

---

## 🆘 Quick Troubleshooting

| Error | Solution |
|-------|----------|
| `FileNotFoundError: improvement_metrics.json` | Run `python model_improvement_pipeline.py` first |
| `ModuleNotFoundError: sklearn` | Run `pip install -r requirements.txt` |
| `No data found` | Add CSV files to `data/raw/` folder |
| `Low accuracy (<55%)` | Add more data or re-train |

---

## 🎓 Next Steps

1. ✅ Read full documentation: [README.md](README.md)
2. 📊 Check visualization guide: Section "Understanding Results"
3. 🔮 Start making predictions: `production_predictor.py`
4. 📈 Monitor performance: Re-train weekly

---

## 📞 Need Help?

1. Check [README.md](README.md) for detailed guide
2. Review error messages in terminal
3. Verify data format (Date, Symbol, Open, High, Low, Close, Volume)

---

**Ready to predict stocks? Run the first command! 🚀**

```powershell
python model_improvement_pipeline.py
```
