#!/usr/bin/env python3
"""
Graph Viewer - Display all generated visualizations
Opens all graphs in a viewing window
"""

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os

def view_graphs():
    """Display all generated graphs"""
    print("📊 STOCK MARKET ANALYSIS - GRAPH VIEWER")
    print("=" * 50)
    
    plots_dir = 'results/plots'
    
    if not os.path.exists(plots_dir):
        print("❌ No plots directory found!")
        print("   Run main.py first to generate graphs")
        return
    
    # Get all PNG files
    graph_files = {
        'Actual vs Predicted': 'actual_vs_predicted.png',
        'Technical Indicators': 'technical_indicators.png',
        'Model Comparison': 'model_comparison.png',
        'Feature Importance': 'feature_importance.png',
        'Formula: SMA': 'formula_sma.png',
        'Formula: EMA': 'formula_ema.png',
        'Formula: RSI': 'formula_rsi.png',
        'Formula: MACD': 'formula_macd.png',
        'Formula: Volatility': 'formula_volatility.png',
        'Formula: Returns': 'formula_returns.png',
        'Formula: BUY/SELL Labels': 'formula_labels.png',
        'Formula: Logistic Regression': 'formula_logistic.png'
    }
    
    print("\n📈 Available Graphs:")
    for i, (name, filename) in enumerate(graph_files.items(), 1):
        filepath = os.path.join(plots_dir, filename)
        if os.path.exists(filepath):
            print(f"   {i}. {name}: {filename} ✅")
        else:
            print(f"   {i}. {name}: {filename} ❌ (not found)")
    
    print("\n" + "=" * 50)
    print("Opening all graphs...")
    print("Close each window to see the next graph")
    print("=" * 50 + "\n")
    
    # Display each graph
    for name, filename in graph_files.items():
        filepath = os.path.join(plots_dir, filename)
        
        if os.path.exists(filepath):
            print(f"📊 Displaying: {name}")
            
            # Read and display the image
            img = mpimg.imread(filepath)
            
            plt.figure(figsize=(16, 10))
            plt.imshow(img)
            plt.axis('off')
            plt.title(name, fontsize=16, fontweight='bold', pad=20)
            plt.tight_layout()
            
            # Show the plot
            plt.show()
        else:
            print(f"⚠️  Skipping: {name} (file not found)")
    
    print("\n✅ All graphs displayed!")
    print(f"📁 Graphs are saved in: {os.path.abspath(plots_dir)}")

if __name__ == "__main__":
    view_graphs()
