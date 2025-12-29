"""
Backend API service for providing dashboard data.
Connects ML model outputs to the React dashboard.
"""

from flask import Flask, jsonify
from flask_cors import CORS
import pandas as pd
import json
from datetime import datetime

app = Flask(__name__)
CORS(app)

def load_backtest_results():
    """
    Load your ML model backtest results.
    Replace with actual data loading from your models.
    """
    # Example: Load from saved results
    # df = pd.read_csv('results/backtest_results.csv')
    # return process_results(df)
    
    pass

def format_strategy_data(results_df):
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

def format_equity_curve(results_df):
    """Format equity curve data for chart"""
    return [
        {
            "date": row['date'].strftime("%b %d"),
            "equity": float(row['equity'])
        }
        for _, row in results_df.iterrows()
    ]

def format_day_analysis(results_df):
    """Format day-wise analysis"""
    daily = results_df.groupby('date').agg({
        'trades': 'sum',
        'targets_hit': 'sum',
        'stop_loss_hit': 'sum',
        'pnl': 'sum',
        'quantity': 'sum'
    }).reset_index()
    
    return {
        "stats": [
            {"label": "Trading Days", "value": len(daily), "type": "number"},
            {"label": "Positive Days", "value": f"{len(daily[daily['pnl'] > 0])} ({len(daily[daily['pnl'] > 0])/len(daily)*100:.2f}%)", "type": "text", "className": "positive"},
            {"label": "Negative Days", "value": f"{len(daily[daily['pnl'] < 0])} ({len(daily[daily['pnl'] < 0])/len(daily)*100:.2f}%)", "type": "text", "className": "negative"},
            {"label": "Day Average profit", "value": float(daily['pnl'].mean()), "type": "currency"},
            {"label": "Day Max Profit", "value": float(daily['pnl'].max()), "type": "currency", "className": "positive"},
            {"label": "Day Max Loss", "value": float(daily['pnl'].min()), "type": "currency", "className": "negative"}
        ],
        "tableData": [
            {
                "date": row['date'].strftime("%d-%m-%Y"),
                "trades": int(row['trades']),
                "targets": int(row['targets_hit']),
                "stopLoss": int(row['stop_loss_hit']),
                "cover": 0,
                "buyTrades": int(row['trades']),
                "sellTrades": 0,
                "qty": int(row['quantity']),
                "profitLoss": float(row['pnl'])
            }
            for _, row in daily.iterrows()
        ]
    }

def format_month_analysis(results_df):
    """Format month-wise analysis"""
    results_df['month'] = pd.to_datetime(results_df['date']).dt.to_period('M')
    monthly = results_df.groupby('month').agg({
        'trades': 'sum',
        'targets_hit': 'sum',
        'stop_loss_hit': 'sum',
        'pnl': 'sum',
        'quantity': 'sum',
        'roi': 'last'
    }).reset_index()
    
    return {
        "stats": [
            {"label": "Total Months", "value": len(monthly), "type": "number"},
            {"label": "Positive Months", "value": f"{len(monthly[monthly['pnl'] > 0])} ({len(monthly[monthly['pnl'] > 0])/len(monthly)*100:.0f}%)", "type": "text", "className": "positive"},
            {"label": "Month Average Profit", "value": float(monthly['pnl'].mean()), "type": "currency"}
        ],
        "tableData": [
            {
                "month": row['month'].strftime("%b - %Y"),
                "trades": int(row['trades']),
                "targets": int(row['targets_hit']),
                "stopLoss": int(row['stop_loss_hit']),
                "cover": 0,
                "buyTrades": int(row['trades']),
                "sellTrades": 0,
                "qty": int(row['quantity']),
                "roi": float(row['roi']),
                "profitLoss": float(row['pnl'])
            }
            for _, row in monthly.iterrows()
        ]
    }

def calculate_risk_profile(results_df):
    """Calculate risk profile based on volatility and drawdown"""
    volatility = results_df['pnl'].std()
    max_dd = results_df['max_drawdown'].max()
    
    if volatility < 1000 and max_dd < 0.1:
        return "Conservative"
    elif volatility < 3000 and max_dd < 0.2:
        return "Moderate"
    else:
        return "Aggressive"

@app.route('/api/backtest-results', methods=['GET'])
def get_backtest_results():
    """API endpoint for dashboard data"""
    try:
        results_df = load_backtest_results()
        dashboard_data = format_strategy_data(results_df)
        return jsonify(dashboard_data)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/live-signals', methods=['GET'])
def get_live_signals():
    """API endpoint for live trading signals"""
    # Implement live signal fetching
    pass

if __name__ == '__main__':
    app.run(debug=True, port=5000)
