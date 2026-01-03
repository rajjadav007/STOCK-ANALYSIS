# Backend Flask API Server

This is the Python Flask backend that serves ML predictions and stock data.

## Start Backend

```bash
python dashboard_api.py
```

Server runs on **http://localhost:5000**

## API Endpoints

- `GET /api/stocks` - List all available stocks
- `GET /api/stock/<symbol>` - Get stock analysis data
- `GET /api/candlestick/<symbol>` - Get candlestick data

## Dependencies

Install with:
```bash
pip install -r requirements.txt
```

## Data Location

The API loads data from:
- `../data/raw/` - Stock CSV files
- `../models/` - Trained ML models
