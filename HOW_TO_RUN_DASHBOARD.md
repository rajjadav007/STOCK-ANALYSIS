# Running the Stock Analysis Dashboard

## Quick Start

### Option 1: Automated Startup (Recommended)
Simply double-click `start_dashboard.bat` - this will automatically:
1. Start the Flask backend API on port 5000
2. Start the React frontend on port 3000
3. Open the dashboard in your browser

### Option 2: Manual Startup

#### Step 1: Install Python Dependencies
```bash
pip install -r requirements.txt
```

#### Step 2: Install Node.js Dependencies
```bash
cd dashboard
npm install
cd ..
```

#### Step 3: Start Backend API (Terminal 1)
```bash
python dashboard_api.py
```

#### Step 4: Start Frontend (Terminal 2)
```bash
cd dashboard
npm start
```

The dashboard will automatically open at `http://localhost:3000`

## Troubleshooting

### Blank Page Issue
If you see a blank page:
1. Check that both backend (port 5000) and frontend (port 3000) are running
2. Open browser console (F12) to check for errors
3. Ensure you have stock data in `data/raw/` directory

### Backend Not Starting
```bash
# Install missing dependencies
pip install flask flask-cors scipy pandas numpy
```

### Frontend Not Starting
```bash
# In dashboard directory
npm install
npm start
```

### Port Already in Use
If port 5000 or 3000 is already in use:
- Kill the process using that port, OR
- Modify the port in `dashboard_api.py` (line 834) and `dashboard/src/services/dataService.js` (line 6)

## Accessing the Dashboard

- **Dashboard UI:** http://localhost:3000
- **Backend API:** http://localhost:5000/api
- **Available Endpoints:**
  - `GET /api/stocks` - List all available stocks
  - `GET /api/stock/<symbol>` - Get stock analysis data
  - `GET /api/candlestick/<symbol>` - Get candlestick chart data

## What You'll See

The dashboard displays:
- Real-time ML predictions for stock movements
- Backtesting results with performance metrics
- Interactive candlestick charts with ML annotations
- Day/Month/Year analysis with profit/loss breakdowns
- Trading statistics and drawdown analysis
- Multi-stock comparison views

## Data Requirements

Ensure you have CSV files in `data/raw/` directory with columns:
- Date, Open, High, Low, Close, Volume
- Symbol (if multi-stock file)

The system will automatically load and analyze available stock data.
