import requests
import json

try:
    print("Testing API endpoint...")
    r = requests.get('http://localhost:5000/api/stock/RELIANCE', timeout=5)
    data = r.json()
    
    trades = data['tradeAnalysis']['tableData'][:5]
    print(f"\n=== RELIANCE Trades from API ({len(data['tradeAnalysis']['tableData'])} total) ===")
    for i, t in enumerate(trades, 1):
        print(f"{i}. Symbol: {t['symbol']}, Entry: {t['entry']}, P/L: {t['profitLoss']}")
    
    print("\n=== Testing HDFCBANK ===")
    r2 = requests.get('http://localhost:5000/api/stock/HDFCBANK', timeout=5)
    data2 = r2.json()
    
    trades2 = data2['tradeAnalysis']['tableData'][:5]
    print(f"HDFCBANK Trades from API ({len(data2['tradeAnalysis']['tableData'])} total) ===")
    for i, t in enumerate(trades2, 1):
        print(f"{i}. Symbol: {t['symbol']}, Entry: {t['entry']}, P/L: {t['profitLoss']}")
        
except Exception as e:
    print(f"Error: {e}")
