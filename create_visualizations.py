#!/usr/bin/env python3
"""
Visualization Suite for Stock Prediction System
================================================

Creates comprehensive visualizations:
1. Price prediction charts
2. Feature importance plots
3. Confusion matrices
4. Model performance comparisons
5. Trading signal visualizations

Author: Stock Prediction System
Date: December 2025
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from pathlib import Path
import warnings

warnings.filterwarnings('ignore')

# Set style
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (14, 8)


class StockVisualizer:
    """Create visualizations for stock prediction results."""
    
    def __init__(self, output_dir='results/visualizations'):
        """Initialize visualizer."""
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def plot_price_predictions(self, y_true, y_pred, model_name='Model', stock_symbol='Stock'):
        """
        Plot actual vs predicted prices.
        
        Args:
            y_true: Actual prices
            y_pred: Predicted prices
            model_name: Name of the model
            stock_symbol: Stock symbol
        """
        plt.figure(figsize=(16, 6))
        
        # Take sample for clearer visualization
        n_samples = min(200, len(y_true))
        indices = range(n_samples)
        
        plt.plot(indices, y_true.iloc[:n_samples].values, 
                label='Actual Price', color='#2E86AB', linewidth=2.5, alpha=0.8)
        plt.plot(indices, y_pred[:n_samples], 
                label='Predicted Price', color='#A23B72', linewidth=2.5, 
                alpha=0.8, linestyle='--')
        
        plt.xlabel('Time Steps', fontsize=12, fontweight='bold')
        plt.ylabel('Stock Price (₹)', fontsize=12, fontweight='bold')
        plt.title(f'Stock Price Prediction: {stock_symbol}\n{model_name}', 
                 fontsize=14, fontweight='bold', pad=20)
        plt.legend(fontsize=11, loc='best', framealpha=0.9)
        plt.grid(True, alpha=0.3, linestyle='--')
        plt.tight_layout()
        
        filename = self.output_dir / f'price_prediction_{model_name.lower().replace(" ", "_")}.png'
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f'✅ Saved: {filename}')
        
    def plot_prediction_error(self, y_true, y_pred, model_name='Model'):
        """
        Plot prediction error distribution.
        """
        errors = y_true - y_pred
        
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        
        # Error distribution
        axes[0].hist(errors, bins=50, color='#F18F01', alpha=0.7, edgecolor='black')
        axes[0].axvline(0, color='red', linestyle='--', linewidth=2, label='Zero Error')
        axes[0].set_xlabel('Prediction Error (₹)', fontsize=12, fontweight='bold')
        axes[0].set_ylabel('Frequency', fontsize=12, fontweight='bold')
        axes[0].set_title('Error Distribution', fontsize=13, fontweight='bold')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Residual plot
        axes[1].scatter(y_pred, errors, alpha=0.5, color='#C73E1D', s=30)
        axes[1].axhline(0, color='red', linestyle='--', linewidth=2)
        axes[1].set_xlabel('Predicted Price (₹)', fontsize=12, fontweight='bold')
        axes[1].set_ylabel('Residual Error (₹)', fontsize=12, fontweight='bold')
        axes[1].set_title('Residual Plot', fontsize=13, fontweight='bold')
        axes[1].grid(True, alpha=0.3)
        
        plt.suptitle(f'Prediction Error Analysis - {model_name}', 
                    fontsize=14, fontweight='bold', y=1.02)
        plt.tight_layout()
        
        filename = self.output_dir / f'error_analysis_{model_name.lower().replace(" ", "_")}.png'
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f'✅ Saved: {filename}')
    
    def plot_confusion_matrix(self, y_true, y_pred, model_name='Model'):
        """
        Plot confusion matrix for classification.
        """
        from sklearn.metrics import confusion_matrix
        
        # Get confusion matrix
        labels = sorted(y_true.unique())
        cm = confusion_matrix(y_true, y_pred, labels=labels)
        
        # Normalize
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        
        # Raw counts
        sns.heatmap(cm, annot=True, fmt='d', cmap='YlOrRd', 
                   xticklabels=labels, yticklabels=labels, ax=axes[0],
                   cbar_kws={'label': 'Count'})
        axes[0].set_xlabel('Predicted', fontsize=12, fontweight='bold')
        axes[0].set_ylabel('Actual', fontsize=12, fontweight='bold')
        axes[0].set_title('Confusion Matrix (Counts)', fontsize=13, fontweight='bold')
        
        # Normalized
        sns.heatmap(cm_normalized, annot=True, fmt='.2%', cmap='Blues',
                   xticklabels=labels, yticklabels=labels, ax=axes[1],
                   cbar_kws={'label': 'Percentage'})
        axes[1].set_xlabel('Predicted', fontsize=12, fontweight='bold')
        axes[1].set_ylabel('Actual', fontsize=12, fontweight='bold')
        axes[1].set_title('Confusion Matrix (Normalized)', fontsize=13, fontweight='bold')
        
        plt.suptitle(f'Classification Performance - {model_name}', 
                    fontsize=14, fontweight='bold', y=1.02)
        plt.tight_layout()
        
        filename = self.output_dir / f'confusion_matrix_{model_name.lower().replace(" ", "_")}.png'
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f'✅ Saved: {filename}')
    
    def plot_feature_importance(self, model, feature_names, model_name='Random Forest', top_n=20):
        """
        Plot feature importance.
        """
        if not hasattr(model, 'feature_importances_'):
            print(f'⚠️  {model_name} does not have feature importance')
            return
        
        importances = model.feature_importances_
        indices = np.argsort(importances)[::-1][:top_n]
        
        plt.figure(figsize=(12, 8))
        
        colors = plt.cm.viridis(np.linspace(0, 1, top_n))
        plt.barh(range(top_n), importances[indices], color=colors, alpha=0.8)
        plt.yticks(range(top_n), [feature_names[i] for i in indices])
        plt.xlabel('Importance Score', fontsize=12, fontweight='bold')
        plt.title(f'Top {top_n} Most Important Features\n{model_name}', 
                 fontsize=14, fontweight='bold', pad=20)
        plt.gca().invert_yaxis()
        plt.grid(True, alpha=0.3, axis='x')
        plt.tight_layout()
        
        filename = self.output_dir / f'feature_importance_{model_name.lower().replace(" ", "_")}.png'
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f'✅ Saved: {filename}')
    
    def plot_model_comparison(self, results_df, task='regression'):
        """
        Compare multiple models.
        
        Args:
            results_df: DataFrame with model results
            task: 'regression' or 'classification'
        """
        if task == 'regression':
            metrics = ['RMSE', 'MAE', 'R2']
            fig, axes = plt.subplots(1, 3, figsize=(18, 6))
            
            for i, metric in enumerate(metrics):
                if metric in results_df.columns:
                    bars = axes[i].bar(results_df['Model'], results_df[metric], 
                                      color=['#2E86AB', '#A23B72', '#F18F01', '#C73E1D'][:len(results_df)])
                    axes[i].set_xlabel('Model', fontsize=11, fontweight='bold')
                    axes[i].set_ylabel(metric, fontsize=11, fontweight='bold')
                    axes[i].set_title(f'{metric} Comparison', fontsize=12, fontweight='bold')
                    axes[i].tick_params(axis='x', rotation=15)
                    axes[i].grid(True, alpha=0.3, axis='y')
                    
                    # Add value labels
                    for bar in bars:
                        height = bar.get_height()
                        axes[i].text(bar.get_x() + bar.get_width()/2., height,
                                   f'{height:.4f}', ha='center', va='bottom', fontsize=9)
            
            plt.suptitle('Regression Model Performance Comparison', 
                        fontsize=14, fontweight='bold')
            
        else:  # classification
            metrics = ['Accuracy', 'Precision', 'Recall', 'F1_Score']
            fig, axes = plt.subplots(2, 2, figsize=(16, 12))
            axes = axes.flatten()
            
            for i, metric in enumerate(metrics):
                if metric in results_df.columns:
                    bars = axes[i].bar(results_df['Model'], results_df[metric] * 100,
                                      color=['#2E86AB', '#A23B72', '#F18F01', '#C73E1D'][:len(results_df)])
                    axes[i].set_xlabel('Model', fontsize=11, fontweight='bold')
                    axes[i].set_ylabel(f'{metric} (%)', fontsize=11, fontweight='bold')
                    axes[i].set_title(f'{metric} Comparison', fontsize=12, fontweight='bold')
                    axes[i].tick_params(axis='x', rotation=15)
                    axes[i].grid(True, alpha=0.3, axis='y')
                    axes[i].set_ylim([0, 100])
                    
                    # Add value labels
                    for bar in bars:
                        height = bar.get_height()
                        axes[i].text(bar.get_x() + bar.get_width()/2., height,
                                   f'{height:.1f}%', ha='center', va='bottom', fontsize=9)
            
            plt.suptitle('Classification Model Performance Comparison',
                        fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        
        filename = self.output_dir / f'{task}_model_comparison.png'
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f'✅ Saved: {filename}')
    
    def plot_trading_signals(self, data, predictions, stock_symbol='Stock'):
        """
        Plot trading signals on price chart.
        
        Args:
            data: DataFrame with Date and Close price
            predictions: Array of BUY/SELL/HOLD predictions
        """
        plt.figure(figsize=(16, 8))
        
        # Take sample
        n_samples = min(200, len(data))
        sample_data = data.iloc[:n_samples].copy()
        sample_pred = predictions[:n_samples]
        
        # Plot price
        plt.plot(range(n_samples), sample_data['Close'].values, 
                color='black', linewidth=2, label='Close Price', alpha=0.7)
        
        # Mark BUY signals
        buy_mask = sample_pred == 'BUY'
        plt.scatter(np.where(buy_mask)[0], sample_data['Close'].values[buy_mask],
                   color='green', marker='^', s=200, label='BUY Signal', zorder=5, alpha=0.8)
        
        # Mark SELL signals
        sell_mask = sample_pred == 'SELL'
        plt.scatter(np.where(sell_mask)[0], sample_data['Close'].values[sell_mask],
                   color='red', marker='v', s=200, label='SELL Signal', zorder=5, alpha=0.8)
        
        plt.xlabel('Time Steps', fontsize=12, fontweight='bold')
        plt.ylabel('Stock Price (₹)', fontsize=12, fontweight='bold')
        plt.title(f'Trading Signals: {stock_symbol}', fontsize=14, fontweight='bold', pad=20)
        plt.legend(fontsize=11, loc='best', framealpha=0.9)
        plt.grid(True, alpha=0.3, linestyle='--')
        plt.tight_layout()
        
        filename = self.output_dir / f'trading_signals_{stock_symbol.lower()}.png'
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f'✅ Saved: {filename}')


def main():
    """Generate all visualizations."""
    print("=" * 80)
    print("GENERATING VISUALIZATIONS")
    print("=" * 80)
    
    # Load test data
    test_data = pd.read_csv('data/processed/test_data.csv')
    test_data['Date'] = pd.to_datetime(test_data['Date'])
    
    # Generate targets
    from target_generator import TargetGenerator
    generator = TargetGenerator(test_data)
    test_with_targets = generator.create_all_targets(horizons=[5])
    X_test, y_test_reg, y_test_class = generator.get_feature_target_split(horizon=5)
    
    print(f"\nTest set: {len(X_test):,} samples")
    
    # Load models
    models_dir = Path('models')
    rf_reg = joblib.load(models_dir / 'regression_random_forest.joblib')
    rf_class = joblib.load(models_dir / 'classification_random_forest.joblib')
    
    # Make predictions
    y_pred_price = rf_reg.predict(X_test)
    y_pred_action = rf_class.predict(X_test)
    
    # Create visualizer
    viz = StockVisualizer()
    
    print("\n📊 Creating Visualizations...")
    
    # 1. Price predictions
    viz.plot_price_predictions(y_test_reg, y_pred_price, 
                               model_name='Random Forest', 
                               stock_symbol='Multi-Stock')
    
    # 2. Error analysis
    viz.plot_prediction_error(y_test_reg, y_pred_price, model_name='Random Forest')
    
    # 3. Confusion matrix
    viz.plot_confusion_matrix(y_test_class, y_pred_action, model_name='Random Forest')
    
    # 4. Feature importance
    feature_names = X_test.columns.tolist()
    viz.plot_feature_importance(rf_reg, feature_names, 
                                model_name='Random Forest (Regression)', top_n=20)
    viz.plot_feature_importance(rf_class, feature_names,
                                model_name='Random Forest (Classification)', top_n=20)
    
    # 5. Trading signals
    test_sample = test_with_targets[test_with_targets['Symbol'] == test_with_targets['Symbol'].iloc[0]].head(200)
    X_sample, _, _ = generator.get_feature_target_split(horizon=5)
    pred_sample = rf_class.predict(X_sample[:200])
    viz.plot_trading_signals(test_sample, pred_sample, 
                            stock_symbol=test_sample['Symbol'].iloc[0])
    
    print("\n" + "=" * 80)
    print("✅ ALL VISUALIZATIONS CREATED")
    print("=" * 80)
    print(f"\n📁 Saved to: {viz.output_dir}/")
    print("\nGenerated:")
    print("   • Price prediction charts")
    print("   • Error analysis plots")
    print("   • Confusion matrices")
    print("   • Feature importance (top 20)")
    print("   • Trading signal visualization")


if __name__ == "__main__":
    main()
