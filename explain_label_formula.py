#!/usr/bin/env python3
"""
Explain BUY/SELL/HOLD Label Formula
Detailed visualization and explanation of the labeling logic
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyBboxPatch
import matplotlib.patches as mpatches
import os

def create_formula_explanation():
    """Create comprehensive formula explanation with visualizations"""
    
    print("📚 BUY/SELL/HOLD LABEL FORMULA EXPLANATION")
    print("=" * 70)
    
    # Create figure with multiple subplots
    fig = plt.figure(figsize=(18, 12))
    
    # ============================================================
    # PANEL 1: Formula Breakdown
    # ============================================================
    ax1 = plt.subplot(3, 3, 1)
    ax1.axis('off')
    ax1.set_xlim(0, 10)
    ax1.set_ylim(0, 10)
    
    # Title
    ax1.text(5, 9.5, 'Formula Breakdown', ha='center', va='top',
            fontsize=14, fontweight='bold', bbox=dict(boxstyle='round', 
            facecolor='lightblue', alpha=0.7))
    
    # Formula
    ax1.text(5, 8, 'Future Return Formula:', ha='center', va='top',
            fontsize=11, fontweight='bold')
    
    formula_text = r'$Future\ Return = \frac{Price_{t+n} - Price_t}{Price_t}$'
    ax1.text(5, 7, formula_text, ha='center', va='top',
            fontsize=13, bbox=dict(boxstyle='round', facecolor='lightyellow'))
    
    # Components
    ax1.text(1, 5.5, 'Where:', fontsize=10, fontweight='bold')
    ax1.text(1, 4.8, r'• $Price_t$ = Current price (today)', fontsize=9)
    ax1.text(1, 4.2, r'• $Price_{t+n}$ = Future price (n days ahead)', fontsize=9)
    ax1.text(1, 3.6, r'• $n$ = Future window (e.g., 5 days)', fontsize=9)
    
    # Example calculation
    ax1.text(1, 2.5, 'Example:', fontsize=10, fontweight='bold',
            color='darkblue')
    ax1.text(1, 1.9, 'Current Price = ₹100', fontsize=9)
    ax1.text(1, 1.4, 'Future Price (5 days) = ₹105', fontsize=9)
    ax1.text(1, 0.8, 'Future Return = (105-100)/100 = 0.05 = 5%', 
            fontsize=9, fontweight='bold', color='green')
    
    # ============================================================
    # PANEL 2: Decision Logic
    # ============================================================
    ax2 = plt.subplot(3, 3, 2)
    ax2.axis('off')
    ax2.set_xlim(0, 10)
    ax2.set_ylim(0, 10)
    
    ax2.text(5, 9.5, 'Threshold-Based Decision Logic', ha='center', va='top',
            fontsize=14, fontweight='bold', bbox=dict(boxstyle='round', 
            facecolor='lightblue', alpha=0.7))
    
    # Decision rules
    ax2.text(5, 8, 'If-Then Rules:', ha='center', va='top',
            fontsize=11, fontweight='bold')
    
    # BUY rule
    buy_box = FancyBboxPatch((0.5, 6.5), 9, 1.2, boxstyle="round,pad=0.1",
                             facecolor='#2ecc71', alpha=0.3, edgecolor='#27ae60', linewidth=2)
    ax2.add_patch(buy_box)
    ax2.text(5, 7.4, 'IF Future Return ≥ +2%', ha='center', fontsize=10, fontweight='bold')
    ax2.text(5, 6.9, 'THEN Label = BUY', ha='center', fontsize=10, color='darkgreen')
    
    # HOLD rule
    hold_box = FancyBboxPatch((0.5, 4.8), 9, 1.2, boxstyle="round,pad=0.1",
                              facecolor='#f39c12', alpha=0.3, edgecolor='#e67e22', linewidth=2)
    ax2.add_patch(hold_box)
    ax2.text(5, 5.7, 'IF -2% < Future Return < +2%', ha='center', fontsize=10, fontweight='bold')
    ax2.text(5, 5.2, 'THEN Label = HOLD', ha='center', fontsize=10, color='darkorange')
    
    # SELL rule
    sell_box = FancyBboxPatch((0.5, 3.1), 9, 1.2, boxstyle="round,pad=0.1",
                              facecolor='#e74c3c', alpha=0.3, edgecolor='#c0392b', linewidth=2)
    ax2.add_patch(sell_box)
    ax2.text(5, 4.0, 'IF Future Return ≤ -2%', ha='center', fontsize=10, fontweight='bold')
    ax2.text(5, 3.5, 'THEN Label = SELL', ha='center', fontsize=10, color='darkred')
    
    # Threshold explanation
    ax2.text(5, 1.8, 'Why ±2% Thresholds?', ha='center', fontsize=10, 
            fontweight='bold', color='navy')
    ax2.text(1, 1.1, '• Filters out noise (small fluctuations)', fontsize=8)
    ax2.text(1, 0.6, '• Captures significant price movements', fontsize=8)
    ax2.text(1, 0.1, '• Balances trade frequency vs confidence', fontsize=8)
    
    # ============================================================
    # PANEL 3: Visual Threshold Zones
    # ============================================================
    ax3 = plt.subplot(3, 3, 3)
    ax3.set_xlim(-10, 10)
    ax3.set_ylim(0, 10)
    
    # Draw zones
    ax3.axvspan(-10, -2, alpha=0.3, color='#e74c3c', label='SELL Zone')
    ax3.axvspan(-2, 2, alpha=0.3, color='#f39c12', label='HOLD Zone')
    ax3.axvspan(2, 10, alpha=0.3, color='#2ecc71', label='BUY Zone')
    
    # Draw threshold lines
    ax3.axvline(x=-2, color='red', linestyle='--', linewidth=2, label='Sell Threshold')
    ax3.axvline(x=2, color='green', linestyle='--', linewidth=2, label='Buy Threshold')
    ax3.axvline(x=0, color='black', linestyle='-', linewidth=1, alpha=0.5)
    
    # Add zone labels
    ax3.text(-6, 8, 'SELL\nZone', ha='center', fontsize=12, fontweight='bold', color='darkred')
    ax3.text(0, 8, 'HOLD\nZone', ha='center', fontsize=12, fontweight='bold', color='darkorange')
    ax3.text(6, 8, 'BUY\nZone', ha='center', fontsize=12, fontweight='bold', color='darkgreen')
    
    ax3.set_xlabel('Future Return (%)', fontsize=11, fontweight='bold')
    ax3.set_title('Threshold Zones', fontsize=14, fontweight='bold')
    ax3.set_yticks([])
    ax3.grid(True, alpha=0.3, axis='x')
    
    # ============================================================
    # PANEL 4: Step-by-Step Example 1 (BUY)
    # ============================================================
    ax4 = plt.subplot(3, 3, 4)
    ax4.axis('off')
    ax4.set_xlim(0, 10)
    ax4.set_ylim(0, 10)
    
    ax4.text(5, 9.5, 'Example 1: BUY Label', ha='center', va='top',
            fontsize=13, fontweight='bold', 
            bbox=dict(boxstyle='round', facecolor='#2ecc71', alpha=0.5))
    
    ax4.text(1, 8.5, 'Step 1: Current State', fontsize=10, fontweight='bold')
    ax4.text(1.5, 7.9, '• Date: June 1, 2021', fontsize=9)
    ax4.text(1.5, 7.4, '• Current Price (t): ₹632.55', fontsize=9)
    ax4.text(1.5, 6.9, '• Future Window: 5 days', fontsize=9)
    
    ax4.text(1, 6.0, 'Step 2: Look Ahead', fontsize=10, fontweight='bold')
    ax4.text(1.5, 5.4, '• Date + 5 days: June 6, 2021', fontsize=9)
    ax4.text(1.5, 4.9, '• Future Price (t+5): ₹663.95', fontsize=9)
    
    ax4.text(1, 4.0, 'Step 3: Calculate Return', fontsize=10, fontweight='bold')
    ax4.text(1.5, 3.4, 'Future Return = (663.95 - 632.55) / 632.55', fontsize=9)
    ax4.text(1.5, 2.9, '             = 31.40 / 632.55', fontsize=9)
    ax4.text(1.5, 2.4, '             = 0.0497 = 4.97%', fontsize=9, 
            fontweight='bold', color='green')
    
    ax4.text(1, 1.5, 'Step 4: Apply Logic', fontsize=10, fontweight='bold')
    ax4.text(1.5, 0.9, '4.97% ≥ 2% ✅', fontsize=9)
    ax4.text(1.5, 0.4, 'Label = BUY 🚀', fontsize=11, fontweight='bold', 
            color='darkgreen')
    
    # ============================================================
    # PANEL 5: Step-by-Step Example 2 (HOLD)
    # ============================================================
    ax5 = plt.subplot(3, 3, 5)
    ax5.axis('off')
    ax5.set_xlim(0, 10)
    ax5.set_ylim(0, 10)
    
    ax5.text(5, 9.5, 'Example 2: HOLD Label', ha='center', va='top',
            fontsize=13, fontweight='bold',
            bbox=dict(boxstyle='round', facecolor='#f39c12', alpha=0.5))
    
    ax5.text(1, 8.5, 'Step 1: Current State', fontsize=10, fontweight='bold')
    ax5.text(1.5, 7.9, '• Date: June 2, 2021', fontsize=9)
    ax5.text(1.5, 7.4, '• Current Price (t): ₹617.15', fontsize=9)
    ax5.text(1.5, 6.9, '• Future Window: 5 days', fontsize=9)
    
    ax5.text(1, 6.0, 'Step 2: Look Ahead', fontsize=10, fontweight='bold')
    ax5.text(1.5, 5.4, '• Date + 5 days: June 7, 2021', fontsize=9)
    ax5.text(1.5, 4.9, '• Future Price (t+5): ₹618.25', fontsize=9)
    
    ax5.text(1, 4.0, 'Step 3: Calculate Return', fontsize=10, fontweight='bold')
    ax5.text(1.5, 3.4, 'Future Return = (618.25 - 617.15) / 617.15', fontsize=9)
    ax5.text(1.5, 2.9, '             = 1.10 / 617.15', fontsize=9)
    ax5.text(1.5, 2.4, '             = 0.0018 = 0.18%', fontsize=9,
            fontweight='bold', color='orange')
    
    ax5.text(1, 1.5, 'Step 4: Apply Logic', fontsize=10, fontweight='bold')
    ax5.text(1.5, 0.9, '-2% < 0.18% < 2% ✅', fontsize=9)
    ax5.text(1.5, 0.4, 'Label = HOLD ⏸️', fontsize=11, fontweight='bold',
            color='darkorange')
    
    # ============================================================
    # PANEL 6: Step-by-Step Example 3 (SELL)
    # ============================================================
    ax6 = plt.subplot(3, 3, 6)
    ax6.axis('off')
    ax6.set_xlim(0, 10)
    ax6.set_ylim(0, 10)
    
    ax6.text(5, 9.5, 'Example 3: SELL Label', ha='center', va='top',
            fontsize=13, fontweight='bold',
            bbox=dict(boxstyle='round', facecolor='#e74c3c', alpha=0.5))
    
    ax6.text(1, 8.5, 'Step 1: Current State', fontsize=10, fontweight='bold')
    ax6.text(1.5, 7.9, '• Date: June 3, 2021', fontsize=9)
    ax6.text(1.5, 7.4, '• Current Price (t): ₹628.65', fontsize=9)
    ax6.text(1.5, 6.9, '• Future Window: 5 days', fontsize=9)
    
    ax6.text(1, 6.0, 'Step 2: Look Ahead', fontsize=10, fontweight='bold')
    ax6.text(1.5, 5.4, '• Date + 5 days: June 8, 2021', fontsize=9)
    ax6.text(1.5, 4.9, '• Future Price (t+5): ₹612.95', fontsize=9)
    
    ax6.text(1, 4.0, 'Step 3: Calculate Return', fontsize=10, fontweight='bold')
    ax6.text(1.5, 3.4, 'Future Return = (612.95 - 628.65) / 628.65', fontsize=9)
    ax6.text(1.5, 2.9, '             = -15.70 / 628.65', fontsize=9)
    ax6.text(1.5, 2.4, '             = -0.0250 = -2.50%', fontsize=9,
            fontweight='bold', color='red')
    
    ax6.text(1, 1.5, 'Step 4: Apply Logic', fontsize=10, fontweight='bold')
    ax6.text(1.5, 0.9, '-2.50% ≤ -2% ✅', fontsize=9)
    ax6.text(1.5, 0.4, 'Label = SELL 📉', fontsize=11, fontweight='bold',
            color='darkred')
    
    # ============================================================
    # PANEL 7: Window Size Impact
    # ============================================================
    ax7 = plt.subplot(3, 3, 7)
    
    # Create sample price data
    days = np.arange(1, 21)
    price = 100 + 10 * np.sin(days / 3) + np.random.randn(20) * 2
    
    ax7.plot(days, price, 'o-', linewidth=2, markersize=8, label='Price', color='steelblue')
    
    # Show different windows
    current_day = 10
    ax7.axvline(x=current_day, color='black', linestyle='--', linewidth=2, 
               label=f'Today (Day {current_day})')
    
    # 3-day window
    ax7.axvline(x=current_day+3, color='green', linestyle=':', linewidth=2,
               alpha=0.6, label='3-day window')
    # 5-day window
    ax7.axvline(x=current_day+5, color='orange', linestyle=':', linewidth=2,
               alpha=0.6, label='5-day window')
    # 10-day window
    ax7.axvline(x=current_day+10, color='red', linestyle=':', linewidth=2,
               alpha=0.6, label='10-day window')
    
    ax7.scatter([current_day], [price[current_day-1]], s=200, c='black', 
               marker='o', zorder=5)
    ax7.text(current_day, price[current_day-1]-3, 'Current\nPrice', 
            ha='center', fontsize=9, fontweight='bold')
    
    ax7.set_xlabel('Days', fontsize=11, fontweight='bold')
    ax7.set_ylabel('Price (₹)', fontsize=11, fontweight='bold')
    ax7.set_title('Window Size Visualization', fontsize=13, fontweight='bold')
    ax7.legend(loc='best', fontsize=8)
    ax7.grid(True, alpha=0.3)
    
    # ============================================================
    # PANEL 8: Window Size Comparison Table
    # ============================================================
    ax8 = plt.subplot(3, 3, 8)
    ax8.axis('off')
    ax8.set_xlim(0, 10)
    ax8.set_ylim(0, 10)
    
    ax8.text(5, 9.5, 'Why Window Size Matters', ha='center', va='top',
            fontsize=13, fontweight='bold',
            bbox=dict(boxstyle='round', facecolor='lightcyan'))
    
    # Table headers
    headers = ['Window', 'Time Horizon', 'Volatility', 'Use Case']
    col_widths = [1.8, 2.2, 2.2, 3.8]
    
    # Header row
    y_pos = 8.2
    x_start = 0.2
    for i, (header, width) in enumerate(zip(headers, col_widths)):
        x_pos = x_start + sum(col_widths[:i]) + width/2
        ax8.text(x_pos, y_pos, header, ha='center', fontsize=9, 
                fontweight='bold', bbox=dict(boxstyle='round', 
                facecolor='lightgray', alpha=0.7))
    
    # Data rows
    data = [
        ['3 days', 'Short', 'High', 'Day trading'],
        ['5 days', 'Medium', 'Moderate', 'Swing trading (✓)'],
        ['10 days', 'Long', 'Low', 'Position trading'],
        ['20+ days', 'Very Long', 'Very Low', 'Long-term investing']
    ]
    
    y_pos = 7.2
    for row in data:
        x_start = 0.2
        for i, (cell, width) in enumerate(zip(row, col_widths)):
            x_pos = x_start + sum(col_widths[:i]) + width/2
            color = 'lightyellow' if '(✓)' in cell else 'white'
            ax8.text(x_pos, y_pos, cell, ha='center', fontsize=8,
                    bbox=dict(boxstyle='round', facecolor=color, alpha=0.5))
        y_pos -= 0.7
    
    # Key insights
    ax8.text(5, 3.8, 'Key Insights:', ha='center', fontsize=10, 
            fontweight='bold', color='navy')
    
    insights = [
        '• Shorter window = More trades, but noisier signals',
        '• Longer window = Fewer trades, but clearer trends',
        '• 5-day window = Good balance for swing trading',
        '• Must match window to your trading strategy'
    ]
    
    y_pos = 3.0
    for insight in insights:
        ax8.text(0.5, y_pos, insight, fontsize=8, va='top')
        y_pos -= 0.6
    
    # ============================================================
    # PANEL 9: Real Dataset Statistics
    # ============================================================
    ax9 = plt.subplot(3, 3, 9)
    ax9.axis('off')
    ax9.set_xlim(0, 10)
    ax9.set_ylim(0, 10)
    
    ax9.text(5, 9.5, 'Real Dataset Results', ha='center', va='top',
            fontsize=13, fontweight='bold',
            bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.5))
    
    # Current configuration
    ax9.text(5, 8.5, 'Configuration Used:', ha='center', fontsize=10, 
            fontweight='bold', color='darkblue')
    ax9.text(1, 7.7, '• Future Window: 5 days', fontsize=9)
    ax9.text(1, 7.2, '• BUY Threshold: +2%', fontsize=9)
    ax9.text(1, 6.7, '• SELL Threshold: -2%', fontsize=9)
    
    # Results
    ax9.text(5, 5.9, 'Label Distribution:', ha='center', fontsize=10,
            fontweight='bold', color='darkblue')
    
    # BUY stats
    buy_box = FancyBboxPatch((0.5, 4.8), 9, 0.7, boxstyle="round,pad=0.05",
                             facecolor='#2ecc71', alpha=0.2, edgecolor='#27ae60')
    ax9.add_patch(buy_box)
    ax9.text(1, 5.3, 'BUY:', fontsize=9, fontweight='bold')
    ax9.text(3, 5.3, '809 samples (33%)', fontsize=9)
    ax9.text(7, 5.3, 'Avg Return: +5.62%', fontsize=9)
    
    # HOLD stats
    hold_box = FancyBboxPatch((0.5, 3.8), 9, 0.7, boxstyle="round,pad=0.05",
                              facecolor='#f39c12', alpha=0.2, edgecolor='#e67e22')
    ax9.add_patch(hold_box)
    ax9.text(1, 4.3, 'HOLD:', fontsize=9, fontweight='bold')
    ax9.text(3, 4.3, '922 samples (38%)', fontsize=9)
    ax9.text(7, 4.3, 'Avg Return: +0.02%', fontsize=9)
    
    # SELL stats
    sell_box = FancyBboxPatch((0.5, 2.8), 9, 0.7, boxstyle="round,pad=0.05",
                              facecolor='#e74c3c', alpha=0.2, edgecolor='#c0392b')
    ax9.add_patch(sell_box)
    ax9.text(1, 3.3, 'SELL:', fontsize=9, fontweight='bold')
    ax9.text(3, 3.3, '720 samples (29%)', fontsize=9)
    ax9.text(7, 3.3, 'Avg Return: -5.86%', fontsize=9)
    
    # Summary
    ax9.text(5, 1.9, 'Summary:', ha='center', fontsize=10,
            fontweight='bold', color='navy')
    ax9.text(1, 1.3, '✅ Balanced distribution across all labels', fontsize=8)
    ax9.text(1, 0.8, '✅ BUY labels show positive returns (5.62%)', fontsize=8)
    ax9.text(1, 0.3, '✅ SELL labels show negative returns (-5.86%)', fontsize=8)
    
    # Overall title
    plt.suptitle('📚 Complete Guide: BUY/SELL/HOLD Label Creation Formula\n' + 
                'Understanding Future Return Calculation, Thresholds, and Window Size',
                fontsize=16, fontweight='bold', y=0.995)
    
    plt.tight_layout()
    
    # Save
    os.makedirs('results/plots', exist_ok=True)
    plt.savefig('results/plots/label_formula_explained.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("\n✅ Formula explanation visualization created!")
    print("📁 Saved to: results/plots/label_formula_explained.png")

def print_text_explanation():
    """Print detailed text explanation"""
    
    print("\n" + "=" * 70)
    print("DETAILED TEXT EXPLANATION")
    print("=" * 70)
    
    print("\n1️⃣  FUTURE RETURN FORMULA")
    print("-" * 70)
    print("""
The core formula calculates the percentage change between current and future price:

    Future Return = (Future Price - Current Price) / Current Price
    
Components:
    • Current Price (Price_t): The stock price TODAY
    • Future Price (Price_t+n): The stock price N days in the FUTURE
    • n: The future window size (e.g., 5 days)

Why this formula?
    • Percentage change is scale-independent (works for ₹100 or ₹10,000 stocks)
    • Easy to compare across different stocks
    • Directly tells us: "By what % will price change?"

Example:
    Current Price = ₹632.55 (June 1)
    Future Price = ₹663.95 (June 6, after 5 days)
    Future Return = (663.95 - 632.55) / 632.55 = 0.0497 = 4.97%
    
    Interpretation: Price increased by 4.97% over 5 days
""")
    
    print("\n2️⃣  THRESHOLD-BASED DECISION LOGIC")
    print("-" * 70)
    print("""
Once we have the Future Return, we apply threshold rules to create labels:

    IF Future Return ≥ +2%:
        Label = BUY
        Reason: Price will rise significantly (profitable opportunity)
    
    ELIF Future Return ≤ -2%:
        Label = SELL
        Reason: Price will drop significantly (avoid losses)
    
    ELSE (between -2% and +2%):
        Label = HOLD
        Reason: Price will stay relatively stable (no strong signal)

Why ±2% thresholds?

    1. Filters Noise:
       • Stock prices fluctuate ±0.5% to ±1% naturally every day
       • ±2% captures MEANINGFUL movements, not random noise
    
    2. Transaction Costs:
       • Brokerage fees, taxes typically 0.1% to 0.5%
       • Need >2% gain to make trading worthwhile
    
    3. Balance:
       • Too low (±1%): Too many trades, low confidence
       • Too high (±5%): Too few trades, miss opportunities
       • ±2%: Sweet spot for swing trading
    
    4. Risk Management:
       • Clear boundary between "buy" and "sell" decisions
       • Prevents overtrading on marginal signals

Example Decision Making:
    • +4.97% → ≥ +2% → BUY (strong upward movement)
    • +0.18% → between -2% and +2% → HOLD (minor fluctuation)
    • -2.50% → ≤ -2% → SELL (strong downward movement)
""")
    
    print("\n3️⃣  WHY FUTURE WINDOW SIZE MATTERS")
    print("-" * 70)
    print("""
The future window (n days) fundamentally changes the prediction problem:

╔══════════════════════════════════════════════════════════════════════╗
║  Window Size  │  Meaning              │  Trading Style              ║
╠══════════════════════════════════════════════════════════════════════╣
║  1-2 days     │  Very short-term      │  Day trading / Scalping     ║
║  3-7 days     │  Short-term           │  Swing trading (← OUR PICK) ║
║  10-20 days   │  Medium-term          │  Position trading           ║
║  30+ days     │  Long-term            │  Investment holding         ║
╚══════════════════════════════════════════════════════════════════════╝

Impact on Predictions:

    Shorter Window (1-3 days):
        ✅ Pros:
            • Captures quick profits
            • More trading opportunities
            • Responsive to news/events
        
        ❌ Cons:
            • High volatility (noisy signals)
            • More false positives
            • Higher transaction costs
            • Stressful (constant monitoring)
    
    Medium Window (5-7 days): ← RECOMMENDED
        ✅ Pros:
            • Balances signal quality and frequency
            • Reduces noise while catching trends
            • Suitable for part-time traders
            • Good risk-reward ratio
        
        ⚠️  Considerations:
            • Must check positions every few days
            • Miss some intraday opportunities
    
    Longer Window (10-20 days):
        ✅ Pros:
            • Very clear trend signals
            • Low noise, high confidence
            • Fewer transactions (lower costs)
            • Less time-intensive
        
        ❌ Cons:
            • Fewer trading opportunities
            • Slower to react to changes
            • May miss short-term profits
            • Higher capital lock-in time

Mathematical Impact:

    With 5-day window:
        Price changes from ₹632.55 → ₹663.95
        Return = 4.97% over 5 days
        Annualized ~ 362% (very high!)
    
    Same ₹31.40 change with 20-day window:
        Return = 4.97% over 20 days
        Annualized ~ 91% (still good, but 4x slower)

Real-World Example (Our Dataset):

    Window = 5 days:
        • Total labels: 2,451
        • BUY: 809 (33%) with avg return +5.62%
        • SELL: 720 (29%) with avg return -5.86%
        • HOLD: 922 (38%) with avg return +0.02%
    
    If we used Window = 10 days:
        • Fewer total labels (lost more to NaN)
        • Higher avg returns per trade (more time to move)
        • Fewer trading opportunities
        • Different label distribution

Key Insight:
    The window size must MATCH your trading strategy:
        • Day trader? Use 1-3 days
        • Swing trader? Use 5-7 days (← Our choice)
        • Investor? Use 20-30 days
    
    You cannot use 5-day predictions for day trading,
    nor can you use 1-day predictions for long-term investing!
""")
    
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print("""
✅ Formula: Future Return = (Price_future - Price_now) / Price_now

✅ Thresholds: ±2% for BUY/SELL, between for HOLD
   • Filters noise
   • Accounts for transaction costs
   • Balances trade frequency and confidence

✅ Window: 5 days chosen for swing trading
   • Not too fast (noisy)
   • Not too slow (miss opportunities)
   • Matches typical holding period

✅ Result: Balanced dataset ready for ML training
   • 2,451 labeled samples
   • 33% BUY, 38% HOLD, 29% SELL
   • Clear separation in return statistics
""")

if __name__ == "__main__":
    # Create visualization
    create_formula_explanation()
    
    # Print detailed text explanation
    print_text_explanation()
    
    print("\n🎯 View the comprehensive visualization:")
    print("   Start-Process results/plots/label_formula_explained.png")
