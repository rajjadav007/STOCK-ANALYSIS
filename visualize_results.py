#!/usr/bin/env python3
"""
Model Results Visualization
============================
Creates graphs and charts to visualize model performance.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
import joblib
import warnings

warnings.filterwarnings('ignore')

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (15, 10)


def load_results():
    """Load saved results and models."""
    with open('results/improvement_metrics.json') as f:
        results = json.load(f)
    
    models = joblib.load('models/ensemble_models.joblib')
    scaler = joblib.load('models/scaler.joblib')
    
    return results, models, scaler


def plot_performance_metrics(results):
    """Plot model performance metrics."""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Model Performance Metrics', fontsize=16, fontweight='bold')
    
    metrics = results['metrics']
    
    # 1. MAE and RMSE
    ax1 = axes[0, 0]
    errors = ['MAE', 'RMSE']
    values = [metrics['MAE'], metrics['RMSE']]
    colors = ['#3498db', '#e74c3c']
    bars = ax1.bar(errors, values, color=colors, alpha=0.7, edgecolor='black')
    ax1.set_ylabel('Error Value', fontsize=12)
    ax1.set_title('Prediction Errors', fontsize=14, fontweight='bold')
    ax1.grid(axis='y', alpha=0.3)
    
    # Add value labels
    for bar in bars:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.6f}',
                ha='center', va='bottom', fontsize=10)
    
    # 2. R² Score
    ax2 = axes[0, 1]
    r2 = metrics['R2']
    ax2.barh(['R² Score'], [r2], color='#2ecc71', alpha=0.7, edgecolor='black')
    ax2.set_xlim(0, 1)
    ax2.set_xlabel('Score (0-1)', fontsize=12)
    ax2.set_title('R² Score (Variance Explained)', fontsize=14, fontweight='bold')
    ax2.text(r2, 0, f'  {r2:.4f} ({r2*100:.2f}%)', va='center', fontsize=12, fontweight='bold')
    ax2.grid(axis='x', alpha=0.3)
    
    # 3. Directional Accuracy
    ax3 = axes[1, 0]
    dir_acc = metrics['Directional_Accuracy']
    profit_acc = metrics['Profit_Weighted_Accuracy']
    
    accuracies = ['Directional\nAccuracy', 'Profit-Weighted\nAccuracy']
    values = [dir_acc, profit_acc]
    colors = ['#9b59b6', '#f39c12']
    bars = ax3.bar(accuracies, values, color=colors, alpha=0.7, edgecolor='black')
    ax3.set_ylabel('Accuracy (%)', fontsize=12)
    ax3.set_ylim(0, 1)
    ax3.set_title('Prediction Accuracy', fontsize=14, fontweight='bold')
    ax3.axhline(y=0.5, color='red', linestyle='--', label='Random Guess (50%)', linewidth=2)
    ax3.legend()
    ax3.grid(axis='y', alpha=0.3)
    
    # Add percentage labels
    for bar in bars:
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height,
                f'{height*100:.2f}%',
                ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    # 4. Overall Verdict
    ax4 = axes[1, 1]
    ax4.axis('off')
    
    verdict = results['verdict']
    verdict_text = f"{verdict}\n\n"
    verdict_text += f"MAE: {metrics['MAE']:.6f}\n"
    verdict_text += f"RMSE: {metrics['RMSE']:.6f}\n"
    verdict_text += f"R²: {metrics['R2']:.4f}\n"
    verdict_text += f"Dir Acc: {metrics['Directional_Accuracy']*100:.2f}%\n"
    verdict_text += f"Profit Acc: {metrics['Profit_Weighted_Accuracy']*100:.2f}%"
    
    ax4.text(0.5, 0.5, verdict_text, 
            ha='center', va='center', fontsize=14,
            bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.5, pad=1),
            fontfamily='monospace')
    
    plt.tight_layout()
    plt.savefig('results/performance_metrics.png', dpi=300, bbox_inches='tight')
    print("✅ Saved: results/performance_metrics.png")
    plt.show()


def plot_ensemble_weights(results):
    """Plot ensemble model weights."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    fig.suptitle('Ensemble Model Weights', fontsize=16, fontweight='bold')
    
    weights = results['ensemble_weights']
    models = list(weights.keys())
    values = list(weights.values())
    
    # Bar chart
    colors = ['#e74c3c', '#3498db', '#2ecc71']
    bars = ax1.bar(models, values, color=colors, alpha=0.7, edgecolor='black')
    ax1.set_ylabel('Weight', fontsize=12)
    ax1.set_title('Model Contribution Weights', fontsize=14, fontweight='bold')
    ax1.set_ylim(0, 1)
    ax1.grid(axis='y', alpha=0.3)
    
    # Add percentage labels
    for bar in bars:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.3f}\n({height*100:.1f}%)',
                ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    # Pie chart
    ax2.pie(values, labels=models, autopct='%1.1f%%', colors=colors, 
            startangle=90, textprops={'fontsize': 12, 'fontweight': 'bold'})
    ax2.set_title('Ensemble Distribution', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('results/ensemble_weights.png', dpi=300, bbox_inches='tight')
    print("✅ Saved: results/ensemble_weights.png")
    plt.show()


def plot_accuracy_comparison(results):
    """Plot accuracy comparison with baseline."""
    fig, ax = plt.subplots(figsize=(12, 6))
    
    dir_acc = results['metrics']['Directional_Accuracy']
    profit_acc = results['metrics']['Profit_Weighted_Accuracy']
    
    categories = ['Random Guess', 'Your Model\n(Directional)', 'Your Model\n(Profit-Weighted)']
    values = [0.50, dir_acc, profit_acc]
    colors = ['#95a5a6', '#3498db', '#2ecc71']
    
    bars = ax.bar(categories, values, color=colors, alpha=0.7, edgecolor='black')
    ax.set_ylabel('Accuracy', fontsize=14)
    ax.set_ylim(0, 1)
    ax.set_title('Accuracy Comparison vs Random Baseline', fontsize=16, fontweight='bold')
    ax.axhline(y=0.5, color='red', linestyle='--', linewidth=2, alpha=0.5)
    ax.grid(axis='y', alpha=0.3)
    
    # Add percentage and improvement labels
    for i, bar in enumerate(bars):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height*100:.2f}%',
                ha='center', va='bottom', fontsize=12, fontweight='bold')
        
        if i > 0:  # Show improvement over baseline
            improvement = (height - 0.5) * 100
            ax.text(bar.get_x() + bar.get_width()/2., height/2,
                    f'+{improvement:.1f}%\nedge',
                    ha='center', va='center', fontsize=10, 
                    color='white', fontweight='bold',
                    bbox=dict(boxstyle='round', facecolor='black', alpha=0.7))
    
    plt.tight_layout()
    plt.savefig('results/accuracy_comparison.png', dpi=300, bbox_inches='tight')
    print("✅ Saved: results/accuracy_comparison.png")
    plt.show()


def plot_metrics_dashboard(results):
    """Create a comprehensive metrics dashboard."""
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
    
    fig.suptitle('Model Performance Dashboard', fontsize=18, fontweight='bold')
    
    metrics = results['metrics']
    
    # 1. MAE gauge
    ax1 = fig.add_subplot(gs[0, 0])
    mae = metrics['MAE']
    ax1.barh(['MAE'], [mae], color='#3498db', alpha=0.7)
    ax1.set_xlim(0, 0.05)
    ax1.set_title('Mean Absolute Error', fontweight='bold')
    ax1.text(mae, 0, f'  {mae:.6f}', va='center', fontsize=11)
    
    # 2. RMSE gauge
    ax2 = fig.add_subplot(gs[0, 1])
    rmse = metrics['RMSE']
    ax2.barh(['RMSE'], [rmse], color='#e74c3c', alpha=0.7)
    ax2.set_xlim(0, 0.05)
    ax2.set_title('Root Mean Square Error', fontweight='bold')
    ax2.text(rmse, 0, f'  {rmse:.6f}', va='center', fontsize=11)
    
    # 3. R² gauge
    ax3 = fig.add_subplot(gs[0, 2])
    r2 = metrics['R2']
    ax3.barh(['R²'], [r2], color='#2ecc71', alpha=0.7)
    ax3.set_xlim(0, 1)
    ax3.set_title('R² Score', fontweight='bold')
    ax3.text(r2, 0, f'  {r2:.4f}', va='center', fontsize=11)
    
    # 4. Directional Accuracy (large)
    ax4 = fig.add_subplot(gs[1, :2])
    dir_acc = metrics['Directional_Accuracy']
    sizes = [dir_acc, 1-dir_acc]
    colors_pie = ['#2ecc71', '#ecf0f1']
    explode = (0.1, 0)
    ax4.pie(sizes, labels=['Correct', 'Incorrect'], autopct='%1.1f%%',
            colors=colors_pie, explode=explode, startangle=90,
            textprops={'fontsize': 14, 'fontweight': 'bold'})
    ax4.set_title(f'Directional Accuracy: {dir_acc*100:.2f}%', 
                  fontsize=14, fontweight='bold', pad=20)
    
    # 5. Ensemble weights (pie)
    ax5 = fig.add_subplot(gs[1, 2])
    weights = results['ensemble_weights']
    ax5.pie(weights.values(), labels=weights.keys(), autopct='%1.0f%%',
            colors=['#e74c3c', '#3498db', '#2ecc71'], startangle=90)
    ax5.set_title('Ensemble Mix', fontweight='bold')
    
    # 6. Accuracy bar comparison
    ax6 = fig.add_subplot(gs[2, :])
    categories = ['Random\nGuess', 'Directional\nAccuracy', 'Profit-Weighted\nAccuracy']
    values = [0.50, dir_acc, metrics['Profit_Weighted_Accuracy']]
    colors_bar = ['#95a5a6', '#3498db', '#f39c12']
    bars = ax6.bar(categories, values, color=colors_bar, alpha=0.7, edgecolor='black')
    ax6.set_ylim(0, 1)
    ax6.set_ylabel('Accuracy', fontsize=12)
    ax6.set_title('Accuracy Metrics Comparison', fontweight='bold', fontsize=14)
    ax6.axhline(y=0.5, color='red', linestyle='--', alpha=0.5)
    ax6.grid(axis='y', alpha=0.3)
    
    for bar in bars:
        height = bar.get_height()
        ax6.text(bar.get_x() + bar.get_width()/2., height,
                f'{height*100:.1f}%',
                ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    plt.savefig('results/dashboard.png', dpi=300, bbox_inches='tight')
    print("✅ Saved: results/dashboard.png")
    plt.show()


def main():
    """Generate all visualizations."""
    print("\n" + "=" * 80)
    print("GENERATING VISUALIZATIONS")
    print("=" * 80 + "\n")
    
    # Load results
    results, models, scaler = load_results()
    
    print(f"📊 Loaded results:")
    print(f"   Verdict: {results['verdict']}")
    print(f"   Features: {results['feature_count']}")
    print(f"   Models: {len(models)}\n")
    
    # Generate plots
    print("📈 Creating visualizations...\n")
    
    plot_performance_metrics(results)
    plot_ensemble_weights(results)
    plot_accuracy_comparison(results)
    plot_metrics_dashboard(results)
    
    print("\n" + "=" * 80)
    print("✅ ALL VISUALIZATIONS CREATED!")
    print("=" * 80)
    print("\n📁 Saved files:")
    print("   • results/performance_metrics.png")
    print("   • results/ensemble_weights.png")
    print("   • results/accuracy_comparison.png")
    print("   • results/dashboard.png")
    print("\n💡 Open these files to see your results as graphs!\n")


if __name__ == "__main__":
    main()
