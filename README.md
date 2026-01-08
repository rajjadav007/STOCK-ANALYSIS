# Stock Analysis ML Trading Dashboard

A production-ready Machine Learning trading system with real-time backtesting dashboard for Indian stock market analysis.

## 🎯 Features

- **ML-Powered Predictions**: Multiple ML models (XGBoost, Random Forest, LSTM) for stock price prediction
- **Real-Time Dashboard**: Interactive React dashboard with live candlestick charts and performance metrics
- **Comprehensive Backtesting**: Walk-forward validation with realistic trading simulation
- **Multi-Stock Support**: Analyze 50+ NIFTY stocks simultaneously
- **Advanced Metrics**: ROI, drawdown analysis, win rate, trade statistics
- **ML Annotations**: Support/resistance levels, swing tops, breakout detection

## 📊 System Architecture

```
STOCK-ANALYSIS/
├── backend/                 # Flask API & ML Engine
│   ├── dashboard_api.py    # Main API server (Port 5000)
│   ├── ml_models.py        # ML model implementations
│   ├── data_loader.py      # Data processing pipeline
│   ├── ensemble_model.py   # Model ensemble logic
│   └── requirements.txt    # Python dependencies
│
├── frontend/               # React Dashboard
│   ├── src/
│   │   ├── components/    # Chart components
│   │   ├── services/      # API integration
│   │   └── App.jsx        # Main app
│   ├── public/
│   └── package.json
│
├── data/
│   ├── raw/               # Historical OHLCV data (CSV)
│   └── processed/         # Cleaned ML-ready data
│
├── models/                # Trained ML models (.joblib, .h5)
└── results/               # Backtest results & visualizations
```

## 🚀 Quick Start

### Prerequisites

**Backend:**
- Python 3.8+
- pip

**Frontend:**
- Node.js 14+
- npm or yarn

### Installation

#### 1. Clone & Setup Backend

```bash
# Navigate to backend
cd backend

# Install Python dependencies
pip install -r requirements.txt
```

#### 2. Setup Frontend

```bash
# Navigate to frontend
cd frontend

# Install Node dependencies
npm install
```

### Running the System

#### Method 1: Auto-Start (Windows)

```bash
# Double-click or run:
START_PROJECT.bat
```

This automatically starts both backend and frontend servers.

#### Method 2: Manual Start

**Terminal 1 - Backend:**
```bash
cd backend
python dashboard_api.py
```
Backend runs at: `http://localhost:5000`

**Terminal 2 - Frontend:**
```bash
cd frontend
npm start
```
Dashboard opens at: `http://localhost:3000`

## 📈 Using the Dashboard

### 1. Access Dashboard
Open browser: `http://localhost:3000`

### 2. Select Stock
Use dropdown to select from 50+ NIFTY stocks (RELIANCE, TCS, HDFCBANK, etc.)

### 3. View Analysis

**Summary Tab:**
- Capital allocation and P/L
- ROI and drawdown metrics
- Win rate and trade count
- Price chart with candlesticks

**ML Chart Tab:**
- Candlestick chart with ML annotations
- Support/resistance levels
- Swing tops and breakout points
- Entry/stop levels

**Analysis Tabs:**
- Day Analysis: Daily trading statistics
- Month Analysis: Monthly performance breakdown
- Year Analysis: Yearly aggregated results
- Trade Analysis: Individual trade details
- Drawdown Analysis: Risk metrics and recovery periods

## 🔧 Configuration

### Backend Configuration

Edit `backend/dashboard_api.py`:

```python
# Data paths
DATA_DIR = '../data'
RAW_DATA_DIR = '../data/raw'
MODELS_DIR = '../models'

# API settings
app.run(debug=True, port=5000)
```

### Frontend Configuration

Edit `frontend/src/services/dataService.js`:

```javascript
// API endpoint
this.apiUrl = 'http://localhost:5000/api';
```

## 📊 Data Format

### Input Data (CSV)

Required columns in `data/raw/SYMBOL.csv`:

```csv
Date,Open,High,Low,Close,Volume,Symbol
2000-01-03,237.5,251.7,237.5,251.7,4456424,RELIANCE
2000-01-04,248.75,258.75,243.75,248.75,9253158,RELIANCE
```

### API Endpoints

**Get Stock List:**
```
GET /api/stocks
Response: {"stocks": ["RELIANCE", "TCS", ...]}
```

**Get Stock Data:**
```
GET /api/stock/<symbol>
Response: {
  "strategy": {...},
  "summary": {"metrics": [...]},
  "performanceData": [...],
  "tradeAnalysis": {...},
  ...
}
```

**Get Candlestick Data:**
```
GET /api/candlestick/<symbol>
Response: {
  "candles": [{
    "time": 946857600,
    "date": "Jan 03 '00",
    "open": 237.5,
    "high": 251.7,
    "low": 237.5,
    "close": 251.7,
    "volume": 4456424
  }],
  "annotations": {...}
}
```

## 🧠 ML Pipeline

### 1. Data Processing
```bash
python load_and_clean_data.py
```
- Loads raw CSV data
- Handles missing values
- Engineers technical features

### 2. Model Training
```bash
python stock_ml_pipeline.py
```
- Trains multiple ML models
- Walk-forward validation
- Saves models to `models/`

### 3. Prediction & Backtesting
```bash
python predict_stocks.py
```
- Generates predictions
- Simulates trading
- Calculates performance metrics

### 4. Dashboard Visualization
Backend API automatically serves predictions to dashboard

## 🛠️ Troubleshooting

### Problem: Blank Dashboard

**Solution 1 - Check Backend:**
```bash
# Test API
curl http://localhost:5000/api/stocks
```
Should return stock list JSON.

**Solution 2 - Check Browser Console:**
Press F12 → Console tab → Look for errors

**Solution 3 - Verify Data:**
Ensure CSV files exist in `data/raw/`

### Problem: Module Not Found

```bash
# Backend
cd backend
pip install -r requirements.txt

# Frontend
cd frontend
npm install
```

### Problem: Port Already in Use

**Backend (Port 5000):**
```bash
# Windows
netstat -ano | findstr :5000
taskkill /PID <PID> /F

# Linux/Mac
lsof -ti:5000 | xargs kill -9
```

**Frontend (Port 3000):**
```bash
# Kill and restart
npm start
# Choose 'Y' when prompted to use different port
```

### Problem: Candlestick Chart Not Showing

1. Check browser console for errors
2. Verify candlestick API: `http://localhost:5000/api/candlestick/RELIANCE`
3. Ensure OHLC data has valid numeric values
4. Clear browser cache and refresh

## 📚 Project Files

### Core Scripts

- `stock_ml_pipeline.py` - Main ML training pipeline
- `predict_stocks.py` - Generate predictions
- `load_and_clean_data.py` - Data preprocessing
- `production_trading_system.py` - Production trading simulator
- `walk_forward_validation.py` - Time-series validation
- `multi_stock_backtest.py` - Multi-symbol backtesting

### Utilities

- `compare_models.py` - Model performance comparison
- `create_visualizations.py` - Generate charts
- `generate_report.py` - Performance reports
- `test_system.py` - System integration tests

### Configuration Files

- `requirements.txt` - Python dependencies
- `backend/requirements.txt` - Backend-specific packages
- `frontend/package.json` - Node dependencies

## 🔐 Production Deployment

### Backend (Flask API)

```bash
# Use production WSGI server
pip install gunicorn

# Run with gunicorn
gunicorn -w 4 -b 0.0.0.0:5000 backend.dashboard_api:app
```

### Frontend (React)

```bash
cd frontend
npm run build

# Serve with nginx or static hosting
```

## 📊 Performance Metrics

The system tracks:

- **ROI (Return on Investment)**: Overall profitability
- **Win Rate**: Percentage of profitable trades
- **Max Drawdown**: Largest peak-to-trough decline
- **Sharpe Ratio**: Risk-adjusted returns
- **Total Trades**: Number of executed trades
- **Average Trade P/L**: Mean profit/loss per trade

## 🤝 Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/improvement`)
3. Commit changes (`git commit -am 'Add feature'`)
4. Push to branch (`git push origin feature/improvement`)
5. Open Pull Request

## 📝 License

This project is for educational and research purposes only. Not financial advice.

## ⚠️ Disclaimer

This trading system is for backtesting and research purposes only. Past performance does not guarantee future results. Always conduct thorough testing and risk management before live trading.

## 🔗 Resources

- [Dashboard Quick Start](./DASHBOARD_QUICKSTART.md)
- [Dashboard Setup Guide](./DASHBOARD_SETUP.md)
- [How to Run Dashboard](./HOW_TO_RUN_DASHBOARD.md)
- [Project Structure](./PROJECT_STRUCTURE.md)

## 📧 Support

For issues and questions:
1. Check [Troubleshooting](#-troubleshooting) section
2. Review documentation files
3. Check browser console (F12) for errors
4. Verify both backend and frontend are running

---

**Built with:** Python, Flask, React, TensorFlow, Scikit-learn, XGBoost, Recharts, Lightweight Charts
