#!/usr/bin/env python3
"""
Visualize Time-Based Train-Test Split
Show how data is split chronologically for proper time series validation
"""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Rectangle
import numpy as np
import os

def visualize_time_split():
    """Create visualization of time-based train-test split"""
    
    print("📊 TIME-BASED TRAIN-TEST SPLIT VISUALIZATION")
    print("=" * 70)
    
    # Load data
    data_path = 'data/processed/labeled_stock_data.csv'
    if not os.path.exists(data_path):
        print(f"❌ File not found: {data_path}")
        return
    
    df = pd.read_csv(data_path)
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values('Date').reset_index(drop=True)
    
    # Calculate split
    split_idx = int(len(df) * 0.8)
    train_df = df.iloc[:split_idx]
    test_df = df.iloc[split_idx:]
    
    # Get date ranges
    train_start = train_df['Date'].min()
    train_end = train_df['Date'].max()
    test_start = test_df['Date'].min()
    test_end = test_df['Date'].max()
    
    print(f"\n📅 TRAIN PERIOD: {train_start.strftime('%Y-%m-%d')} to {train_end.strftime('%Y-%m-%d')}")
    print(f"   Samples: {len(train_df):,} ({len(train_df)/len(df)*100:.1f}%)")
    print(f"   Days: {(train_end - train_start).days}")
    
    print(f"\n📅 TEST PERIOD: {test_start.strftime('%Y-%m-%d')} to {test_end.strftime('%Y-%m-%d')}")
    print(f"   Samples: {len(test_df):,} ({len(test_df)/len(df)*100:.1f}%)")
    print(f"   Days: {(test_end - test_start).days}")
    
    # Create figure
    fig = plt.figure(figsize=(18, 12))
    
    # ============================================================
    # PANEL 1: Timeline Visualization
    # ============================================================
    ax1 = plt.subplot(3, 2, 1)
    
    # Draw timeline
    y_pos = 0.5
    timeline_start = 0
    timeline_end = 100
    
    # Train period (80%)
    train_width = 80
    train_rect = Rectangle((timeline_start, y_pos - 0.15), train_width, 0.3,
                           facecolor='#3498db', alpha=0.6, edgecolor='#2980b9', linewidth=2)
    ax1.add_patch(train_rect)
    ax1.text(train_width/2, y_pos, 'TRAIN\n80%', ha='center', va='center',
            fontsize=14, fontweight='bold', color='white')
    
    # Test period (20%)
    test_width = 20
    test_rect = Rectangle((train_width, y_pos - 0.15), test_width, 0.3,
                          facecolor='#e74c3c', alpha=0.6, edgecolor='#c0392b', linewidth=2)
    ax1.add_patch(test_rect)
    ax1.text(train_width + test_width/2, y_pos, 'TEST\n20%', ha='center', va='center',
            fontsize=14, fontweight='bold', color='white')
    
    # Time arrow
    ax1.arrow(-5, y_pos, 110, 0, head_width=0.1, head_length=3, fc='black', ec='black')
    ax1.text(-5, y_pos + 0.3, 'PAST', ha='center', fontsize=10, fontweight='bold')
    ax1.text(105, y_pos + 0.3, 'FUTURE', ha='center', fontsize=10, fontweight='bold')
    
    # Add dates
    ax1.text(0, y_pos - 0.35, train_start.strftime('%Y-%m-%d'), ha='left', fontsize=9)
    ax1.text(train_width, y_pos - 0.35, train_end.strftime('%Y-%m-%d'), ha='center', fontsize=9)
    ax1.text(100, y_pos - 0.35, test_end.strftime('%Y-%m-%d'), ha='right', fontsize=9)
    
    ax1.set_xlim(-10, 110)
    ax1.set_ylim(0, 1)
    ax1.axis('off')
    ax1.set_title('Time-Based Split: Train on PAST → Test on FUTURE', 
                 fontsize=14, fontweight='bold', pad=20)
    
    # ============================================================
    # PANEL 2: Why No Shuffling?
    # ============================================================
    ax2 = plt.subplot(3, 2, 2)
    ax2.axis('off')
    ax2.set_xlim(0, 10)
    ax2.set_ylim(0, 10)
    
    ax2.text(5, 9.5, '⚠️  Why NO Shuffling?', ha='center', fontsize=14, fontweight='bold',
            bbox=dict(boxstyle='round', facecolor='#ffcccc', alpha=0.7))
    
    # Bad example (with shuffling)
    ax2.text(5, 8.5, '❌ BAD: Random Split (Shuffled)', ha='center', fontsize=11, 
            fontweight='bold', color='red')
    
    bad_example = [
        '• Train: 2011, 2015, 2019, 2021',
        '• Test: 2012, 2016, 2020',
        '• Problem: Training on FUTURE data!',
        '• Model learns from data it shouldn\'t see',
        '• Unrealistic high accuracy',
        '• Fails in real trading'
    ]
    
    y_pos = 7.5
    for line in bad_example:
        ax2.text(0.5, y_pos, line, fontsize=9, va='top', color='darkred')
        y_pos -= 0.6
    
    # Good example (time-based)
    ax2.text(5, 4.5, '✅ GOOD: Time-Based Split', ha='center', fontsize=11,
            fontweight='bold', color='green')
    
    good_example = [
        '• Train: 2011-2019 (past only)',
        '• Test: 2019-2021 (future only)',
        '• Realistic: Simulates real trading',
        '• Model never sees future',
        '• Honest performance metrics',
        '• Works in live trading'
    ]
    
    y_pos = 3.5
    for line in good_example:
        ax2.text(0.5, y_pos, line, fontsize=9, va='top', color='darkgreen')
        y_pos -= 0.6
    
    # ============================================================
    # PANEL 3: Price Chart with Split
    # ============================================================
    ax3 = plt.subplot(3, 1, 2)
    
    # Plot entire price history
    ax3.plot(df['Date'], df['Close'], color='gray', alpha=0.3, linewidth=1, label='All Data')
    
    # Highlight train period
    ax3.plot(train_df['Date'], train_df['Close'], color='#3498db', linewidth=2, label='Train Data')
    
    # Highlight test period
    ax3.plot(test_df['Date'], test_df['Close'], color='#e74c3c', linewidth=2, label='Test Data')
    
    # Add vertical line at split
    split_date = test_start
    ax3.axvline(x=split_date, color='black', linestyle='--', linewidth=2, alpha=0.7)
    ax3.text(split_date, ax3.get_ylim()[1] * 0.95, 'SPLIT', 
            ha='center', fontsize=10, fontweight='bold',
            bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))
    
    ax3.set_xlabel('Date', fontsize=11, fontweight='bold')
    ax3.set_ylabel('Close Price (₹)', fontsize=11, fontweight='bold')
    ax3.set_title('Stock Price Over Time: Train vs Test Periods', fontsize=13, fontweight='bold')
    ax3.legend(loc='best', fontsize=10)
    ax3.grid(True, alpha=0.3)
    
    # ============================================================
    # PANEL 4: Label Distribution Comparison
    # ============================================================
    ax4 = plt.subplot(3, 3, 7)
    
    train_labels = train_df['Label'].value_counts()
    test_labels = test_df['Label'].value_counts()
    
    labels = ['BUY', 'HOLD', 'SELL']
    train_counts = [train_labels.get(l, 0) for l in labels]
    test_counts = [test_labels.get(l, 0) for l in labels]
    
    x = np.arange(len(labels))
    width = 0.35
    
    bars1 = ax4.bar(x - width/2, train_counts, width, label='Train', 
                   color='#3498db', alpha=0.7)
    bars2 = ax4.bar(x + width/2, test_counts, width, label='Test',
                   color='#e74c3c', alpha=0.7)
    
    ax4.set_xlabel('Label', fontsize=11, fontweight='bold')
    ax4.set_ylabel('Count', fontsize=11, fontweight='bold')
    ax4.set_title('Label Distribution: Train vs Test', fontsize=12, fontweight='bold')
    ax4.set_xticks(x)
    ax4.set_xticklabels(labels)
    ax4.legend()
    ax4.grid(True, alpha=0.3, axis='y')
    
    # Add value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width()/2., height,
                    f'{int(height)}',
                    ha='center', va='bottom', fontsize=8)
    
    # ============================================================
    # PANEL 5: Sample Distribution Over Time
    # ============================================================
    ax5 = plt.subplot(3, 3, 8)
    
    # Monthly sample count
    df['YearMonth'] = df['Date'].dt.to_period('M')
    monthly_counts = df.groupby('YearMonth').size()
    
    train_monthly = train_df.groupby(train_df['Date'].dt.to_period('M')).size()
    test_monthly = test_df.groupby(test_df['Date'].dt.to_period('M')).size()
    
    # Plot
    dates = [p.to_timestamp() for p in monthly_counts.index]
    ax5.bar(dates, monthly_counts.values, color='lightgray', alpha=0.5, label='All')
    
    train_dates = [p.to_timestamp() for p in train_monthly.index]
    ax5.bar(train_dates, train_monthly.values, color='#3498db', alpha=0.7, label='Train')
    
    test_dates = [p.to_timestamp() for p in test_monthly.index]
    ax5.bar(test_dates, test_monthly.values, color='#e74c3c', alpha=0.7, label='Test')
    
    ax5.axvline(x=split_date, color='black', linestyle='--', linewidth=2)
    
    ax5.set_xlabel('Date', fontsize=11, fontweight='bold')
    ax5.set_ylabel('Samples per Month', fontsize=11, fontweight='bold')
    ax5.set_title('Sample Distribution Over Time', fontsize=12, fontweight='bold')
    ax5.legend(fontsize=9)
    ax5.grid(True, alpha=0.3, axis='y')
    
    # ============================================================
    # PANEL 6: Key Metrics Table
    # ============================================================
    ax6 = plt.subplot(3, 3, 9)
    ax6.axis('off')
    ax6.set_xlim(0, 10)
    ax6.set_ylim(0, 10)
    
    ax6.text(5, 9.5, 'Split Summary', ha='center', fontsize=13, fontweight='bold',
            bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))
    
    # Create table data
    table_data = [
        ['Metric', 'Train', 'Test'],
        ['Samples', f"{len(train_df):,}", f"{len(test_df):,}"],
        ['Percentage', f"{len(train_df)/len(df)*100:.1f}%", f"{len(test_df)/len(df)*100:.1f}%"],
        ['Start Date', train_start.strftime('%Y-%m-%d'), test_start.strftime('%Y-%m-%d')],
        ['End Date', train_end.strftime('%Y-%m-%d'), test_end.strftime('%Y-%m-%d')],
        ['Days', f"{(train_end-train_start).days}", f"{(test_end-test_start).days}"],
        ['BUY', f"{train_labels.get('BUY', 0):,}", f"{test_labels.get('BUY', 0):,}"],
        ['HOLD', f"{train_labels.get('HOLD', 0):,}", f"{test_labels.get('HOLD', 0):,}"],
        ['SELL', f"{train_labels.get('SELL', 0):,}", f"{test_labels.get('SELL', 0):,}"]
    ]
    
    y_pos = 8.5
    col_widths = [3, 3.5, 3.5]
    
    # Draw table
    for i, row in enumerate(table_data):
        x_pos = 0.5
        bg_color = 'lightgray' if i == 0 else 'white'
        font_weight = 'bold' if i == 0 else 'normal'
        
        for j, cell in enumerate(row):
            ax6.text(x_pos, y_pos, cell, fontsize=9, va='center',
                    fontweight=font_weight,
                    bbox=dict(boxstyle='round', facecolor=bg_color, alpha=0.3))
            x_pos += col_widths[j]
        
        y_pos -= 0.9
    
    # Overall title
    plt.suptitle('📊 Time-Based Train-Test Split Analysis\n' +
                'Training on Past Data | Testing on Future Data | No Shuffling',
                fontsize=16, fontweight='bold', y=0.995)
    
    plt.tight_layout()
    
    # Save
    os.makedirs('results/plots', exist_ok=True)
    plt.savefig('results/plots/time_based_split.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"\n✅ Visualization created!")
    print(f"📁 Saved to: results/plots/time_based_split.png")
    
    # Print summary
    print(f"\n" + "=" * 70)
    print("KEY POINTS:")
    print("=" * 70)
    print(f"✅ Time-based split preserves chronological order")
    print(f"✅ Train: {train_start.strftime('%Y-%m-%d')} to {train_end.strftime('%Y-%m-%d')} ({len(train_df):,} samples)")
    print(f"✅ Test: {test_start.strftime('%Y-%m-%d')} to {test_end.strftime('%Y-%m-%d')} ({len(test_df):,} samples)")
    print(f"✅ No shuffling - prevents data leakage")
    print(f"✅ Simulates real trading: train on past, predict future")
    print(f"✅ Gap: {(test_start - train_end).days} day(s) between train and test")

if __name__ == "__main__":
    visualize_time_split()
    print("\n🎯 View the comprehensive visualization:")
    print("   Start-Process results/plots/time_based_split.png")
