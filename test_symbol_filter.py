"""
Test script to verify trade symbol filtering
"""
import sys
sys.path.insert(0, 'D:/STOCK-ANALYSIS')

from dashboard_api import load_stock_data, format_strategy_data

# Test RELIANCE
print("\n=== Testing RELIANCE ===")
df_reliance = load_stock_data('RELIANCE')
if df_reliance is not None:
    data_reliance = format_strategy_data('RELIANCE', df_reliance)
    trades = data_reliance['tradeAnalysis']['tableData']
    print(f"Total trades: {len(trades)}")
    print(f"First 3 trades:")
    for i, trade in enumerate(trades[:3]):
        print(f"  {i+1}. Symbol: {trade['symbol']}, Entry: {trade['entry']}, P/L: {trade['profitLoss']}, Result: {trade['result']}")

# Test INFY
print("\n=== Testing INFY ===")
df_infy = load_stock_data('INFY')
if df_infy is not None:
    data_infy = format_strategy_data('INFY', df_infy)
    trades = data_infy['tradeAnalysis']['tableData']
    print(f"Total trades: {len(trades)}")
    print(f"First 3 trades:")
    for i, trade in enumerate(trades[:3]):
        print(f"  {i+1}. Symbol: {trade['symbol']}, Entry: {trade['entry']}, P/L: {trade['profitLoss']}, Result: {trade['result']}")

print("\n=== Test Complete ===")
