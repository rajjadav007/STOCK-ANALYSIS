"""
QUICK START - Graph Visualization
==================================

Your stock analysis now creates beautiful graphs automatically!

GENERATED GRAPHS (4 total):
---------------------------
1. actual_vs_predicted.png     - Model accuracy visualization
2. technical_indicators.png    - All technical indicators in one view
3. model_comparison.png        - Compare different ML models
4. feature_importance.png      - Which features matter most

LOCATION:
---------
results/plots/

HOW TO VIEW:
------------
Method 1 - Interactive Viewer (Recommended):
    python view_graphs.py

Method 2 - Run Full Analysis:
    python main.py
    (Graphs auto-generated at the end)

Method 3 - Open Files Directly:
    Navigate to results/plots/ folder
    Double-click any PNG file

WHAT'S IN EACH GRAPH:
---------------------

📊 technical_indicators.png (6 panels):
   - Stock Price with SMA 10 & 50
   - Stock Price with EMA 12 & 26
   - RSI (Relative Strength Index)
   - MACD (Moving Average Convergence Divergence)
   - Volatility (10 & 20 day)
   - Trading Volume

📈 actual_vs_predicted.png:
   - Blue line: Actual stock prices
   - Red dashed line: Model predictions
   - Shows how accurate your model is

📊 model_comparison.png:
   - RMSE comparison (lower is better)
   - MAE comparison (lower is better)
   - R² Score comparison (higher is better)

📊 feature_importance.png:
   - Top 15 most important features
   - Shows which indicators drive predictions

GRAPH QUALITY:
--------------
✅ 300 DPI (publication quality)
✅ Professional color schemes
✅ Clear labels and legends
✅ Grid lines for easy reading
✅ Optimized sizes for viewing

COMMANDS:
---------
# Generate all graphs:
python main.py

# View all graphs:
python view_graphs.py

# Open specific graph (Windows):
start results/plots/technical_indicators.png
start results/plots/actual_vs_predicted.png
start results/plots/model_comparison.png
start results/plots/feature_importance.png

# Open folder:
explorer results\plots

DOCUMENTATION:
--------------
📖 Full guide: VISUALIZATION_GUIDE.md
📖 Summary: VISUALIZATION_SUMMARY.md
📖 Technical indicators: TECHNICAL_INDICATORS_SUMMARY.md

ENJOY YOUR GRAPHS! 📈📊
"""

if __name__ == "__main__":
    print(__doc__)
