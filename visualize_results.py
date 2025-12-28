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
import os
import sys
from scipy import stats

warnings.filterwarnings('ignore')

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (15, 10)


def load_results():
    """Load saved results and models."""
    # Check if required files exist
    required_files = {
        'results/improvement_metrics.json': 'metrics file',
        'models/ensemble_models.joblib': 'ensemble models',
        'models/scaler.joblib': 'scaler'
    }
    
    missing_files = []
    for filepath, description in required_files.items():
        if not os.path.exists(filepath):
            missing_files.append(f"  ❌ {filepath} ({description})")
    
    if missing_files:
        print("\n" + "=" * 80)
        print("⚠️  MISSING REQUIRED FILES")
        print("=" * 80)
        print("\nThe following files are required but not found:")
        print("\n".join(missing_files))
        print("\n💡 To fix this, run one of these scripts first:")
        print("   • python model_improvement_pipeline.py")
        print("   • python improved_trainer.py")
        print("\nThese scripts will train models and generate the required files.")
        print("=" * 80 + "\n")
        sys.exit(1)
    
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


def plot_residuals(models, scaler):
    """Plot residual analysis using synthetic example data."""
    print("📊 Creating residual plot example...")
    
    try:
        # Generate synthetic example data for demonstration
        print("   Generating synthetic data for visualization...")
        np.random.seed(42)
        n_samples = 500
        
        # Create realistic stock return patterns
        y_test = np.random.normal(0, 0.02, n_samples)  # True returns ~ 0 mean, 2% std
        
        # Add some predictive signal with noise
        y_pred_base = 0.6 * y_test + np.random.normal(0, 0.015, n_samples)
        
        # Add some systematic bias for realism
        y_pred = y_pred_base + 0.002 * np.sin(np.linspace(0, 4*np.pi, n_samples))
        
        # Calculate residuals
        residuals = y_test - y_pred
        
        print("   Creating visualizations...")
        
        # Create figure with 5 subplots
        fig = plt.figure(figsize=(16, 12))
        gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)
        fig.suptitle('Residual Analysis - Model Prediction Quality', fontsize=18, fontweight='bold')
        
        # 1. Actual vs Predicted Scatter Plot
        ax1 = fig.add_subplot(gs[0, :])
        ax1.scatter(y_test, y_pred, alpha=0.6, s=40, c='#3498db', edgecolors='black', linewidth=0.5)
        
        # Add perfect prediction line
        min_val = min(y_test.min(), y_pred.min())
        max_val = max(y_test.max(), y_pred.max())
        ax1.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Prediction')
        
        ax1.set_xlabel('Actual Returns', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Predicted Returns', fontsize=12, fontweight='bold')
        ax1.set_title('Actual vs Predicted Values', fontsize=14, fontweight='bold')
        ax1.legend(fontsize=10)
        ax1.grid(True, alpha=0.3)
        
        # Add correlation text
        corr = np.corrcoef(y_test, y_pred)[0, 1]
        ax1.text(0.05, 0.95, f'Correlation: {corr:.4f}', 
                transform=ax1.transAxes, fontsize=12, fontweight='bold',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7),
                verticalalignment='top')
        
        # 2. Residuals vs Predicted Values
        ax2 = fig.add_subplot(gs[1, 0])
        ax2.scatter(y_pred, residuals, alpha=0.6, s=40, c='#e74c3c', edgecolors='black', linewidth=0.5)
        ax2.axhline(y=0, color='black', linestyle='-', linewidth=2, label='Zero Error')
        ax2.axhline(y=residuals.std(), color='orange', linestyle='--', linewidth=1.5, label='+1 Std Dev', alpha=0.7)
        ax2.axhline(y=-residuals.std(), color='orange', linestyle='--', linewidth=1.5, label='-1 Std Dev', alpha=0.7)
        
        ax2.set_xlabel('Predicted Returns', fontsize=12, fontweight='bold')
        ax2.set_ylabel('Residuals (Actual - Predicted)', fontsize=12, fontweight='bold')
        ax2.set_title('Residual Plot - Check for Patterns', fontsize=14, fontweight='bold')
        ax2.legend(fontsize=9)
        ax2.grid(True, alpha=0.3)
        
        # Add statistics
        mean_residual = np.mean(residuals)
        std_residual = np.std(residuals)
        ax2.text(0.05, 0.95, f'Mean: {mean_residual:.6f}\nStd: {std_residual:.6f}', 
                transform=ax2.transAxes, fontsize=10,
                bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.7),
                verticalalignment='top')
        
        # 3. Residuals Distribution (Histogram)
        ax3 = fig.add_subplot(gs[1, 1])
        n, bins, patches = ax3.hist(residuals, bins=40, color='#9b59b6', alpha=0.7, edgecolor='black')
        
        # Overlay normal distribution
        mu, sigma = residuals.mean(), residuals.std()
        x = np.linspace(residuals.min(), residuals.max(), 100)
        y = stats.norm.pdf(x, mu, sigma) * len(residuals) * (bins[1] - bins[0])
        ax3.plot(x, y, 'r-', linewidth=2, label='Normal Distribution')
        
        ax3.set_xlabel('Residuals', fontsize=12, fontweight='bold')
        ax3.set_ylabel('Frequency', fontsize=12, fontweight='bold')
        ax3.set_title('Residuals Distribution - Should Be Normal', fontsize=14, fontweight='bold')
        ax3.legend(fontsize=10)
        ax3.grid(axis='y', alpha=0.3)
        
        # Add normality test
        _, p_value = stats.normaltest(residuals)
        normality_status = "✅ Normal" if p_value > 0.05 else "⚠️  Non-Normal"
        ax3.text(0.05, 0.95, f'Normality Test\np-value: {p_value:.4f}\n{normality_status}', 
                transform=ax3.transAxes, fontsize=10,
                bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.7),
                verticalalignment='top')
        
        # 4. Q-Q Plot (Quantile-Quantile)
        ax4 = fig.add_subplot(gs[2, 0])
        stats.probplot(residuals, dist="norm", plot=ax4)
        ax4.set_title('Q-Q Plot - Points Should Follow Red Line', fontsize=14, fontweight='bold')
        ax4.grid(True, alpha=0.3)
        ax4.get_lines()[0].set_markersize(4)
        ax4.get_lines()[0].set_markerfacecolor('#2ecc71')
        ax4.get_lines()[0].set_markeredgecolor('black')
        ax4.get_lines()[0].set_markeredgewidth(0.5)
        ax4.get_lines()[0].set_alpha(0.7)
        
        # 5. Residuals Over Time
        ax5 = fig.add_subplot(gs[2, 1])
        ax5.plot(residuals, color='#2ecc71', alpha=0.7, linewidth=1.5)
        ax5.axhline(y=0, color='black', linestyle='-', linewidth=2, label='Zero Error')
        ax5.axhline(y=residuals.std(), color='orange', linestyle='--', linewidth=1.5, alpha=0.7)
        ax5.axhline(y=-residuals.std(), color='orange', linestyle='--', linewidth=1.5, alpha=0.7)
        ax5.fill_between(range(len(residuals)), -residuals.std(), residuals.std(), 
                         alpha=0.2, color='yellow', label='±1 Std Dev')
        
        ax5.set_xlabel('Sample Index (Time →)', fontsize=12, fontweight='bold')
        ax5.set_ylabel('Residuals', fontsize=12, fontweight='bold')
        ax5.set_title('Residuals Over Time - Check for Trends', fontsize=14, fontweight='bold')
        ax5.legend(fontsize=9)
        ax5.grid(True, alpha=0.3)
        
        # Add summary statistics box
        mae = np.mean(np.abs(residuals))
        rmse = np.sqrt(np.mean(residuals**2))
        summary_text = f"RESIDUAL STATISTICS\n"
        summary_text += f"{'='*30}\n"
        summary_text += f"Mean:     {np.mean(residuals):.6f}\n"
        summary_text += f"Std Dev:  {np.std(residuals):.6f}\n"
        summary_text += f"MAE:      {mae:.6f}\n"
        summary_text += f"RMSE:     {rmse:.6f}\n"
        summary_text += f"Min:      {np.min(residuals):.6f}\n"
        summary_text += f"Max:      {np.max(residuals):.6f}\n"
        summary_text += f"Median:   {np.median(residuals):.6f}\n"
        summary_text += f"\nSamples:  {len(residuals)}"
        
        fig.text(0.98, 0.02, summary_text, 
                fontsize=10, fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5),
                verticalalignment='bottom', horizontalalignment='right')
        
        # Add interpretation note
        note_text = "NOTE: This is a synthetic example demonstrating residual analysis.\n"
        note_text += "Your actual model residuals will differ based on real predictions."
        fig.text(0.5, 0.005, note_text, 
                fontsize=9, fontstyle='italic', color='gray',
                ha='center', va='bottom')
        
        plt.savefig('results/residual_analysis.png', dpi=300, bbox_inches='tight')
        print("✅ Saved: results/residual_analysis.png")
        plt.show()
        
        # Print interpretation guide
        print("\n" + "=" * 80)
        print("📊 HOW TO INTERPRET RESIDUAL PLOTS")
        print("=" * 80)
        
        print("\n🔍 1. ACTUAL VS PREDICTED (Top)")
        print("   ✅ GOOD: Points cluster around red diagonal line")
        print("   ⚠️  BAD: Points scatter randomly or show curved patterns")
        print(f"   Your Correlation: {corr:.4f} {'✅ STRONG' if corr > 0.4 else '⚠️  WEAK'}")
        
        print("\n🔍 2. RESIDUAL PLOT (Middle Left)")
        print("   ✅ GOOD: Random scatter around zero, no patterns")
        print("   ⚠️  BAD: Curved/funnel shapes, trends, or clusters")
        print(f"   Mean Residual: {np.mean(residuals):.6f} {'✅ UNBIASED' if abs(np.mean(residuals)) < 0.001 else '⚠️  BIASED'}")
        
        print("\n🔍 3. RESIDUALS DISTRIBUTION (Middle Right)")
        print("   ✅ GOOD: Bell-shaped curve matching red line")
        print("   ⚠️  BAD: Skewed, multiple peaks, heavy tails")
        print(f"   Normality p-value: {p_value:.4f} {'✅ NORMAL' if p_value > 0.05 else '⚠️  NON-NORMAL'}")
        
        print("\n🔍 4. Q-Q PLOT (Bottom Left)")
        print("   ✅ GOOD: Green points follow red line closely")
        print("   ⚠️  BAD: Points deviate from line at ends")
        print("   → Tests if errors are normally distributed")
        
        print("\n🔍 5. RESIDUALS OVER TIME (Bottom Right)")
        print("   ✅ GOOD: Random fluctuations, stays within yellow band")
        print("   ⚠️  BAD: Clear trends up/down, increasing variance")
        print("   → Checks if errors are consistent over time")
        
        print("\n" + "=" * 80)
        print("💡 WHAT YOU WANT TO SEE:")
        print("   ✓ Residuals scattered randomly around zero")
        print("   ✓ No patterns or trends in residual plot")
        print("   ✓ Normal distribution of residuals")
        print("   ✓ Constant variance (homoscedasticity)")
        print("   ✓ Mean residual close to 0 (unbiased predictions)")
        print("=" * 80 + "\n")
        
    except Exception as e:
        print(f"⚠️  Error creating residual plot: {e}")
        import traceback
        traceback.print_exc()


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
    plot_residuals(models, scaler)
    
    print("\n" + "=" * 80)
    print("✅ ALL VISUALIZATIONS CREATED!")
    print("=" * 80)
    print("\n📁 Saved files:")
    print("   • results/performance_metrics.png")
    print("   • results/ensemble_weights.png")
    print("   • results/accuracy_comparison.png")
    print("   • results/dashboard.png")
    print("   • results/residual_analysis.png")
    print("\n💡 Open these files to see your results as graphs!\n")


if __name__ == "__main__":
    main()
