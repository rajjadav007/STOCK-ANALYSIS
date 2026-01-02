# Quick Start Guide - Stock Analysis Dashboard

## 🚀 Running the Dashboard

### Method 1: One-Click Start (Easiest)
```bash
# Double-click this file:
start_dashboard.bat
```

### Method 2: Manual Start

**Terminal 1 - Backend API:**
```bash
python dashboard_api.py
```

**Terminal 2 - Frontend Dashboard:**
```bash
cd dashboard
npm start
```

## ✅ What Should Happen

1. Backend starts on **http://localhost:5000**
2. Frontend starts on **http://localhost:3000**
3. Browser automatically opens the dashboard
4. Dashboard loads stock data and shows:
   - ML predictions
   - Performance charts
   - Trading statistics
   - Candlestick patterns

## 🔧 If You See a Blank Page

### Check 1: Is Backend Running?
Open http://localhost:5000/api/stocks in your browser
- ✅ Should show: `{"stocks": ["RELIANCE", "TCS", ...]}`
- ❌ If error: Backend is not running

### Check 2: Check Browser Console
Press **F12** in browser, go to **Console** tab
- Look for red error messages
- Common issue: "Failed to fetch" = Backend not running

### Check 3: Verify Data Files
Ensure you have CSV files in `data/raw/` folder:
```
data/raw/RELIANCE.csv
data/raw/TCS.csv
data/raw/HDFCBANK.csv
etc...
```

## 🛠️ Solutions

### Problem: "Module not found" error
```bash
# Install Python dependencies
pip install -r requirements.txt
```

### Problem: Backend port already in use
```bash
# Find and kill process on port 5000
netstat -ano | findstr :5000
taskkill /PID <process_id> /F
```

### Problem: Frontend won't start
```bash
cd dashboard
npm install
npm start
```

### Problem: Dashboard loads but no data
1. Check that stock CSV files exist in `data/raw/`
2. Files should have: Date, Open, High, Low, Close, Volume columns
3. Restart the backend: `python dashboard_api.py`

## 📊 Expected Output

When working correctly, you'll see:
- Strategy name and backtest period
- Summary metrics (Capital, P/L, ROI, etc.)
- Stock selector dropdown
- Performance equity curve chart
- Candlestick chart with ML annotations
- Day/Month/Year analysis tabs
- Trade statistics tables

## 🔗 Useful URLs

- Dashboard: http://localhost:3000
- API Health: http://localhost:5000/api/stocks
- Specific Stock: http://localhost:5000/api/stock/RELIANCE

## 📝 Notes

- Both backend AND frontend must be running simultaneously
- Backend loads data from `data/raw/*.csv` files
- Dashboard automatically refreshes when you select different stocks
- Close terminals or use Ctrl+C to stop services
