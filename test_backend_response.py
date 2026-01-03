import requests
import time

start = time.time()
try:
    r = requests.get('http://localhost:5000/api/stock/RELIANCE', timeout=15)
    elapsed = time.time() - start
    print(f'✓ Response time: {elapsed:.2f}s')
    print(f'✓ Status: {r.status_code}')
    
    data = r.json()
    print(f'✓ Has tradeAnalysis: {"tradeAnalysis" in data}')
    print(f'✓ Trades: {len(data.get("tradeAnalysis", {}).get("tableData", []))}')
    print(f'✓ Has strategy: {"strategy" in data}')
    print(f'✓ Has summary: {"summary" in data}')
    
    if 'tradeAnalysis' in data and data['tradeAnalysis']['tableData']:
        first_trade = data['tradeAnalysis']['tableData'][0]
        print(f'✓ First trade symbol: {first_trade.get("symbol")}')
except Exception as e:
    print(f'✗ Error: {e}')
