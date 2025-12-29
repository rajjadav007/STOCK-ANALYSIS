"""
Backend API service for providing dashboard data.
Connects ML model outputs to the React dashboard.
"""

from flask import Flask, jsonify, request
from flask_cors import CORS
import pandas as pd
import json
import os
import glob
from datetime import datetime

app = Flask(__name__)
CORS(app)

# Paths
DATA_DIR = 'data'
RAW_DATA_DIR = os.path.join(DATA_DIR, 'raw')
PROCESSED_DATA_DIR = os.path.join(DATA_DIR, 'processed')
MODELS_DIR = 'models'
RESULTS_DIR = 'results'

def get_available_stocks():
    """Get list of available stocks from raw data directory"""
    try:
        csv_files = glob.glob(os.path.join(RAW_DATA_DIR, '*.csv'))
        stocks = [os.path.basename(f).replace('.csv', '') for f in csv_files 
                 if 'metadata' not in f.lower() and 'nifty50_all' not in f.lower()]
        stocks.sort()
        return stocks
    except Exception as e:
        print(f"Error getting stock list: {e}")
        return []

def load_stock_data(stock_symbol):
    """Load stock data from CSV files"""
    try:
        # Try processed data first
        processed_file = os.path.join(PROCESSED_DATA_DIR, f'ml_ready_{stock_symbol.lower()}.csv')
        if os.path.exists(processed_file):
            df = pd.read_csv(processed_file)
            return df
        
        # Fall back to raw data
        raw_file = os.path.join(RAW_DATA_DIR, f'{stock_symbol}.csv')
        if os.path.exists(raw_file):
            df = pd.read_csv(raw_file)
            return df
        
        return None
    except Exception as e:
        print(f"Error loading data for {stock_symbol}: {e}")
        return None

def format_strategy_data(stock_symbol, df):
    """Convert stock data to dashboard format"""
    try:
        # Ensure date column exists
        date_col = None
        for col in ['Date', 'date', 'Datetime', 'datetime']:
            if col in df.columns:
                date_col = col
                break
        
        if date_col is None:
            raise ValueError("No date column found")
        
        df[date_col] = pd.to_datetime(df[date_col])
        df = df.sort_values(date_col)
        
        # Calculate basic metrics
        initial_capital = 50000
        
        # Simulate trades based on price movements
        df['returns'] = df['Close'].pct_change()
        df['position'] = 75  # Fixed position size
        df['pnl'] = df['returns'] * df['position'] * df['Close']
        df['cumulative_pnl'] = df['pnl'].fillna(0).cumsum()
        df['equity'] = initial_capital + df['cumulative_pnl']
        
        # Calculate drawdown
        df['peak'] = df['equity'].cummax()
        df['drawdown'] = ((df['equity'] - df['peak']) / df['peak']) * 100
        
        total_pnl = df['pnl'].sum()
        roi = (total_pnl / initial_capital) * 100
        max_drawdown = df['drawdown'].min()
        
        # Get recent data for chart
        recent_df = df.tail(50)
        
        # Format performance data
        performance_data = [
            {
                "date": row[date_col].strftime("%b %d"),
                "equity": float(row['equity'])
            }
            for _, row in recent_df.iterrows()
        ]
        
        # Calculate trades (significant moves only)
        significant_moves = df[abs(df['pnl']) > 100].copy()
        winning_trades = len(significant_moves[significant_moves['pnl'] > 0])
        losing_trades = len(significant_moves[significant_moves['pnl'] < 0])
        total_trades = len(significant_moves)
        
        win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0
        
        # Build dashboard data
        dashboard_data = {
            "strategy": {
                "name": f"ML Trading Strategy - {stock_symbol}",
                "backtested": datetime.now().strftime("on %d %b %Y %H:%M"),
                "period": f"{df[date_col].min().strftime('%d %b %y')} to {df[date_col].max().strftime('%d %b %y')}",
                "created": "Today"
            },
            "summary": {
                "metrics": [
                    {"label": "Symbol", "value": stock_symbol, "type": "text"},
                    {"label": "Capital", "value": float(initial_capital), "type": "currency"},
                    {"label": "Profit/Loss", "value": float(total_pnl), "type": "currency"},
                    {"label": "ROI", "value": float(roi), "type": "percentage"},
                    {"label": "Drawdown", "value": f"{abs(max_drawdown):.2f}%", "type": "text"},
                    {"label": "Total Trades", "value": int(total_trades), "type": "text"},
                    {"label": "Win Rate", "value": f"{win_rate:.2f}%", "type": "text"},
                    {"label": "Type", "value": "ML Prediction", "type": "text"}
                ]
            },
            "performanceData": performance_data,
            "dayAnalysis": format_day_analysis(significant_moves, date_col),
            "monthAnalysis": {"stats": [], "tableData": []},
            "yearAnalysis": {"stats": [], "tableData": []},
            "tradeAnalysis": {
                "stats": [
                    {"label": "Total Trades", "value": int(total_trades), "type": "number"},
                    {"label": "Winning Trades", "value": int(winning_trades), "type": "number"},
                    {"label": "Losing Trades", "value": int(losing_trades), "type": "number"}
                ],
                "tableData": [],
                "isEmpty": True
            },
            "drawdownAnalysis": {
                "drawdownInfo": {
                    "drawdown": f"{abs(max_drawdown):.2f}%",
                    "downStartDate": "N/A",
                    "maxDownDate": "N/A",
                    "recoveryDate": "N/A",
                    "recoveryPeriod": "N/A"
                },
                "chartData": [
                    {
                        "date": row[date_col].strftime("%b %d"),
                        "drawdown": float(row['drawdown'])
                    }
                    for _, row in recent_df.iterrows()
                ]
            }
        }
        
        return dashboard_data
        
    except Exception as e:
        print(f"Error formatting data: {e}")
        raise

def format_day_analysis(trades_df, date_col):
    """Convert ML results to dashboard format"""
    return {
        "strategy": {
            "name": "ML Trading Strategy",
            "backtested": datetime.now().strftime("on %d %b %Y %H:%M"),
            "period": f"{results_df['date'].min()} to {results_df['date'].max()}",
            "created": "1 day ago"
        },
        "summary": {
            "metrics": [
                {"label": "Symbol", "value": results_df['symbol'].iloc[0], "type": "text"},
                {"label": "Capital", "value": float(results_df['initial_capital'].iloc[0]), "type": "currency"},
                {"label": "Profit/Loss", "value": float(results_df['total_pnl'].sum()), "type": "currency"},
                {"label": "ROI", "value": float(results_df['roi'].iloc[-1]), "type": "percentage"},
                {"label": "Drawdown", "value": float(results_df['max_drawdown'].max()), "type": "text"},
                {"label": "Risk Profile", "value": calculate_risk_profile(results_df), "type": "text"}
            ]
        },
        "performanceData": format_equity_curve(results_df),
        "dayAnalysis": format_day_analysis(results_df),
        "monthAnalysis": format_month_analysis(results_df)
    }

def format_day_analysis(trades_df, date_col):
    """Format day-wise analysis"""
    if len(trades_df) == 0:
        return {"stats": [], "tableData": [], "profitByDay": {}}
    
    try:
        trades_df['date_only'] = trades_df[date_col].dt.date
        daily = trades_df.groupby('date_only').agg({
            'pnl': ['sum', 'count']
        }).reset_index()
        daily.columns = ['date', 'pnl', 'trades']
        
        positive_days = len(daily[daily['pnl'] > 0])
        negative_days = len(daily[daily['pnl'] < 0])
        total_days = len(daily)
        
        return {
            "stats": [
                {"label": "Trading Days", "value": int(total_days), "type": "number"},
                {"label": "Positive Days", "value": f"{positive_days} ({positive_days/total_days*100:.2f}%)", "type": "text", "className": "positive"},
                {"label": "Negative Days", "value": f"{negative_days} ({negative_days/total_days*100:.2f}%)", "type": "text", "className": "negative"},
                {"label": "Day Average profit", "value": float(daily['pnl'].mean()), "type": "currency"}
            ],
            "profitByDay": {
                "Mon Profit": 0,
                "Tue Profit": 0,
                "Wed Profit": 0,
                "Thu Profit": 0,
                "Fri Profit": 0,
                "Sat Profit": 0,
                "Sun Profit": 0
            },
            "tableData": [
                {
                    "date": row['date'].strftime("%d-%m-%Y"),
                    "trades": int(row['trades']),
                    "targets": 0,
                    "stopLoss": 0,
                    "cover": 0,
                    "buyTrades": int(row['trades']),
                    "sellTrades": 0,
                    "qty": 0,
                    "profitLoss": float(row['pnl'])
                }
                for _, row in daily.tail(30).iterrows()
            ]
        }
    except Exception as e:
        print(f"Error in day analysis: {e}")
        return {"stats": [], "tableData": [], "profitByDay": {}}

@app.route('/api/stocks', methods=['GET'])
def get_stocks():
    """API endpoint for stock list"""
    try:
        stocks = get_available_stocks()
        return jsonify({"stocks": stocks})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/stock/<stock_symbol>', methods=['GET'])
def get_stock(stock_symbol):
    """API endpoint for specific stock data"""
    try:
        df = load_stock_data(stock_symbol)
        if df is None:
            return jsonify({"error": f"No data found for {stock_symbol}"}), 404
        
        dashboard_data = format_strategy_data(stock_symbol, df)
        return jsonify(dashboard_data)
    except Exception as e:
        print(f"Error in get_stock: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/backtest-results', methods=['GET'])
def get_backtest_results():
    """API endpoint for dashboard data"""
    try:
        # Default to RELIANCE if no stock specified
        stock = request.args.get('stock', 'RELIANCE')
        df = load_stock_data(stock)
        if df is None:
            return jsonify({"error": f"No data found for {stock}"}), 404
        
        dashboard_data = format_strategy_data(stock, df)
        return jsonify(dashboard_data)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    print("Starting Flask API server...")
    print(f"Available stocks: {len(get_available_stocks())}")
    app.run(debug=True, port=5000)
