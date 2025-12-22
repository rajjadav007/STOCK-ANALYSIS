#!/usr/bin/env python3
"""
View BUY/SELL/HOLD Label Analysis
Visualize label distribution and statistics
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

def analyze_labels():
    """Analyze and visualize BUY/SELL/HOLD labels"""
    
    print("📊 LABEL ANALYSIS")
    print("=" * 60)
    
    # Load labeled dataset
    data_path = 'data/processed/labeled_stock_data.csv'
    if not os.path.exists(data_path):
        print(f"❌ File not found: {data_path}")
        print("   Run: python main.py first to create labeled dataset")
        return
    
    df = pd.read_csv(data_path)
    df['Date'] = pd.to_datetime(df['Date'])
    
    print(f"\n📁 Dataset: {data_path}")
    print(f"📊 Total records: {len(df):,}")
    print(f"📅 Date range: {df['Date'].min().strftime('%Y-%m-%d')} to {df['Date'].max().strftime('%Y-%m-%d')}")
    
    # Label distribution
    print(f"\n🏷️  LABEL DISTRIBUTION:")
    print("-" * 60)
    label_counts = df['Label'].value_counts()
    total = len(df)
    
    for label in ['BUY', 'HOLD', 'SELL']:
        count = label_counts.get(label, 0)
        pct = (count / total) * 100
        print(f"   {label:5s}: {count:5,} ({pct:5.2f}%)")
    
    # Future return statistics by label
    print(f"\n📈 FUTURE RETURN STATISTICS BY LABEL:")
    print("-" * 60)
    for label in ['BUY', 'HOLD', 'SELL']:
        label_data = df[df['Label'] == label]['Future_Return']
        print(f"\n   {label}:")
        print(f"      Mean:   {label_data.mean()*100:6.2f}%")
        print(f"      Median: {label_data.median()*100:6.2f}%")
        print(f"      Min:    {label_data.min()*100:6.2f}%")
        print(f"      Max:    {label_data.max()*100:6.2f}%")
    
    # Create visualizations
    print(f"\n📊 Creating visualizations...")
    create_label_visualizations(df)
    
    print(f"\n✅ Analysis complete!")
    print(f"📈 Graphs saved to: results/plots/label_analysis.png")

def create_label_visualizations(df):
    """Create comprehensive label visualization"""
    
    fig = plt.figure(figsize=(16, 10))
    
    # 1. Label Distribution (Pie Chart)
    ax1 = plt.subplot(2, 3, 1)
    label_counts = df['Label'].value_counts()
    colors = ['#2ecc71', '#f39c12', '#e74c3c']  # Green, Orange, Red
    ax1.pie(label_counts.values, labels=label_counts.index, autopct='%1.1f%%',
            colors=colors, startangle=90, textprops={'fontsize': 10, 'fontweight': 'bold'})
    ax1.set_title('Label Distribution', fontsize=14, fontweight='bold')
    
    # 2. Label Distribution (Bar Chart)
    ax2 = plt.subplot(2, 3, 2)
    bars = ax2.bar(label_counts.index, label_counts.values, color=colors, alpha=0.7)
    ax2.set_title('Label Counts', fontsize=14, fontweight='bold')
    ax2.set_ylabel('Count', fontsize=11)
    ax2.grid(True, alpha=0.3, axis='y')
    for bar in bars:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height):,}',
                ha='center', va='bottom', fontweight='bold')
    
    # 3. Future Return Distribution by Label (Box Plot)
    ax3 = plt.subplot(2, 3, 3)
    buy_returns = df[df['Label'] == 'BUY']['Future_Return'] * 100
    hold_returns = df[df['Label'] == 'HOLD']['Future_Return'] * 100
    sell_returns = df[df['Label'] == 'SELL']['Future_Return'] * 100
    
    bp = ax3.boxplot([buy_returns, hold_returns, sell_returns],
                     labels=['BUY', 'HOLD', 'SELL'],
                     patch_artist=True,
                     showmeans=True)
    
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)
    
    ax3.set_title('Future Return Distribution by Label', fontsize=14, fontweight='bold')
    ax3.set_ylabel('Future Return (%)', fontsize=11)
    ax3.grid(True, alpha=0.3, axis='y')
    ax3.axhline(y=0, color='black', linestyle='--', linewidth=1, alpha=0.5)
    
    # 4. Future Return Histogram
    ax4 = plt.subplot(2, 3, 4)
    ax4.hist(df['Future_Return'] * 100, bins=50, color='steelblue', alpha=0.7, edgecolor='black')
    ax4.axvline(x=2, color='green', linestyle='--', linewidth=2, label='BUY threshold (+2%)')
    ax4.axvline(x=-2, color='red', linestyle='--', linewidth=2, label='SELL threshold (-2%)')
    ax4.axvline(x=0, color='black', linestyle='-', linewidth=1, alpha=0.5)
    ax4.set_title('Future Return Distribution (All Labels)', fontsize=14, fontweight='bold')
    ax4.set_xlabel('Future Return (%)', fontsize=11)
    ax4.set_ylabel('Frequency', fontsize=11)
    ax4.legend(fontsize=9)
    ax4.grid(True, alpha=0.3, axis='y')
    
    # 5. Labels Over Time
    ax5 = plt.subplot(2, 3, 5)
    df_sorted = df.sort_values('Date')
    
    # Create monthly aggregation
    df_sorted['YearMonth'] = df_sorted['Date'].dt.to_period('M')
    monthly_labels = df_sorted.groupby(['YearMonth', 'Label']).size().unstack(fill_value=0)
    
    # Plot stacked area chart
    monthly_labels.plot(kind='area', stacked=True, ax=ax5, 
                       color=colors, alpha=0.7)
    ax5.set_title('Label Distribution Over Time', fontsize=14, fontweight='bold')
    ax5.set_xlabel('Date', fontsize=11)
    ax5.set_ylabel('Count', fontsize=11)
    ax5.legend(title='Label', fontsize=9)
    ax5.grid(True, alpha=0.3, axis='y')
    
    # 6. Mean Future Return by Label
    ax6 = plt.subplot(2, 3, 6)
    mean_returns = df.groupby('Label')['Future_Return'].mean() * 100
    bars = ax6.bar(mean_returns.index, mean_returns.values, color=colors, alpha=0.7)
    ax6.set_title('Mean Future Return by Label', fontsize=14, fontweight='bold')
    ax6.set_ylabel('Mean Future Return (%)', fontsize=11)
    ax6.grid(True, alpha=0.3, axis='y')
    ax6.axhline(y=0, color='black', linestyle='-', linewidth=1)
    
    for bar in bars:
        height = bar.get_height()
        ax6.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.2f}%',
                ha='center', va='bottom' if height > 0 else 'top',
                fontweight='bold', fontsize=10)
    
    plt.suptitle('📊 BUY/SELL/HOLD Label Analysis\n' + 
                f'Window: 5 days | BUY: >2% | SELL: <-2% | HOLD: between',
                fontsize=16, fontweight='bold', y=0.995)
    
    plt.tight_layout()
    
    # Save
    os.makedirs('results/plots', exist_ok=True)
    plt.savefig('results/plots/label_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()

if __name__ == "__main__":
    analyze_labels()
    print("\n" + "=" * 60)
    print("💡 TIP: Run 'python view_labels.py' to see this analysis")
    print("=" * 60)
