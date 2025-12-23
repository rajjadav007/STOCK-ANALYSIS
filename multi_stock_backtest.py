"""
Multi-Stock Analysis with Adjusted Parameters
==============================================
Analyzes multiple stocks with different confidence thresholds.

Author: Stock Analysis Team
Date: December 23, 2025
"""

from production_trading_system import ProductionTradingSystem
import pandas as pd

def analyze_multiple_stocks():
    """Analyze multiple stocks with adjusted parameters."""
    
    system = ProductionTradingSystem()
    
    # Test stocks
    stocks = ['RELIANCE', 'TCS', 'INFY', 'HDFCBANK', 'ICICIBANK']
    
    # Adjusted parameters for more trades
    params = {
        'initial_capital': 100000,
        'stop_loss_pct': 2.0,
        'take_profit_pct': 5.0,
        'position_size_pct': 20.0,  # Increased to 20%
        'min_confidence': 0.30      # Lowered to 30% to allow more trades
    }
    
    all_results = []
    
    print("\n" + "="*70)
    print("MULTI-STOCK BACKTEST ANALYSIS")
    print("="*70)
    print(f"\nTesting {len(stocks)} stocks with adjusted parameters:")
    print(f"  • Position Size: {params['position_size_pct']}%")
    print(f"  • Min Confidence: {params['min_confidence']*100}%")
    print(f"  • Stop Loss: {params['stop_loss_pct']}%")
    print(f"  • Take Profit: {params['take_profit_pct']}%")
    
    for stock in stocks:
        print("\n" + "="*70)
        print(f"📊 ANALYZING: {stock}")
        print("="*70)
        
        try:
            trades, portfolio, metrics = system.run_complete_system(
                stock_symbol=stock,
                **params
            )
            
            if metrics is not None:
                result = {
                    'Stock': stock,
                    'Total_Return': metrics['total_return'],
                    'Total_Trades': metrics['total_trades'],
                    'Win_Rate': metrics['win_rate'],
                    'Profit_Factor': metrics['profit_factor'],
                    'Max_Drawdown': metrics['max_drawdown'],
                    'Sharpe_Ratio': metrics['sharpe_ratio']
                }
                all_results.append(result)
            else:
                all_results.append({
                    'Stock': stock,
                    'Total_Return': 0,
                    'Total_Trades': 0,
                    'Win_Rate': 0,
                    'Profit_Factor': 0,
                    'Max_Drawdown': 0,
                    'Sharpe_Ratio': 0
                })
        
        except Exception as e:
            print(f"❌ Error analyzing {stock}: {e}")
            all_results.append({
                'Stock': stock,
                'Total_Return': 0,
                'Total_Trades': 0,
                'Win_Rate': 0,
                'Profit_Factor': 0,
                'Max_Drawdown': 0,
                'Sharpe_Ratio': 0
            })
    
    # Summary table
    print("\n" + "="*70)
    print("MULTI-STOCK SUMMARY")
    print("="*70)
    
    results_df = pd.DataFrame(all_results)
    
    print(f"\n{'Stock':<12} {'Return%':<10} {'Trades':<8} {'Win%':<8} {'PF':<8} {'MaxDD%':<10} {'Sharpe':<8}")
    print("-" * 70)
    
    for _, row in results_df.iterrows():
        print(f"{row['Stock']:<12} {row['Total_Return']:>8.2f}% {row['Total_Trades']:>7,} "
              f"{row['Win_Rate']:>7.2f}% {row['Profit_Factor']:>7.2f} "
              f"{row['Max_Drawdown']:>9.2f}% {row['Sharpe_Ratio']:>7.2f}")
    
    # Overall statistics
    print("\n" + "="*70)
    print("OVERALL STATISTICS")
    print("="*70)
    
    total_trades = results_df['Total_Trades'].sum()
    avg_return = results_df['Total_Return'].mean()
    avg_win_rate = results_df[results_df['Total_Trades'] > 0]['Win_Rate'].mean()
    
    print(f"  Total Trades Across All Stocks: {total_trades:,}")
    print(f"  Average Return per Stock:       {avg_return:.2f}%")
    print(f"  Average Win Rate:               {avg_win_rate:.2f}%")
    
    # Save results
    results_df.to_csv('results/multi_stock_backtest_results.csv', index=False)
    print(f"\n💾 Results saved: results/multi_stock_backtest_results.csv")
    
    return results_df


def test_different_confidence_levels():
    """Test different confidence thresholds on one stock."""
    
    print("\n" + "="*70)
    print("CONFIDENCE THRESHOLD SENSITIVITY ANALYSIS")
    print("="*70)
    
    system = ProductionTradingSystem()
    confidence_levels = [0.25, 0.30, 0.35, 0.40, 0.45]
    
    results = []
    
    for conf in confidence_levels:
        print(f"\n🔍 Testing with {conf*100}% minimum confidence...")
        
        trades, portfolio, metrics = system.run_complete_system(
            stock_symbol='RELIANCE',
            initial_capital=100000,
            stop_loss_pct=2.0,
            take_profit_pct=5.0,
            position_size_pct=20.0,
            min_confidence=conf
        )
        
        if metrics is not None:
            results.append({
                'Min_Confidence': f"{conf*100:.0f}%",
                'Total_Trades': metrics['total_trades'],
                'Total_Return': metrics['total_return'],
                'Win_Rate': metrics['win_rate'],
                'Profit_Factor': metrics['profit_factor']
            })
        else:
            results.append({
                'Min_Confidence': f"{conf*100:.0f}%",
                'Total_Trades': 0,
                'Total_Return': 0.0,
                'Win_Rate': 0.0,
                'Profit_Factor': 0.0
            })
    
    # Summary
    print("\n" + "="*70)
    print("CONFIDENCE THRESHOLD IMPACT")
    print("="*70)
    
    results_df = pd.DataFrame(results)
    print(results_df.to_string(index=False))
    
    return results_df


if __name__ == "__main__":
    import sys
    import io
    
    # Set UTF-8 encoding for Windows console
    if sys.platform == 'win32':
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    
    # Test 1: Multiple stocks
    print("\n>>> TEST 1: Multi-Stock Analysis")
    multi_stock_results = analyze_multiple_stocks()
    
    # Test 2: Different confidence levels
    print("\n\n>>> TEST 2: Confidence Threshold Analysis")
    confidence_results = test_different_confidence_levels()
    
    print("\n" + "="*70)
    print("ANALYSIS COMPLETE!")
    print("="*70)
