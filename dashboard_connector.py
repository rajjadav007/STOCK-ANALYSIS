"""
Data connector for integrating ML model outputs with the dashboard.
Processes model predictions and backtest results into dashboard format.
"""

import pandas as pd
import numpy as np
import json
from datetime import datetime
import joblib

class DashboardDataConnector:
    """Connects ML model outputs to dashboard data format"""
    
    def __init__(self, model_dir='models', results_dir='results'):
        self.model_dir = model_dir
        self.results_dir = results_dir
    
    def load_backtest_results(self, csv_path):
        """Load backtest results from CSV"""
        df = pd.read_csv(csv_path, parse_dates=['date'])
        return df
    
    def load_predictions(self, predictions_path):
        """Load model predictions"""
        return pd.read_csv(predictions_path)
    
    def calculate_metrics(self, trades_df):
        """Calculate summary metrics from trades"""
        total_trades = len(trades_df)
        winning_trades = len(trades_df[trades_df['pnl'] > 0])
        losing_trades = len(trades_df[trades_df['pnl'] < 0])
        
        total_pnl = trades_df['pnl'].sum()
        avg_win = trades_df[trades_df['pnl'] > 0]['pnl'].mean() if winning_trades > 0 else 0
        avg_loss = trades_df[trades_df['pnl'] < 0]['pnl'].mean() if losing_trades > 0 else 0
        
        win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0
        profit_factor = abs(avg_win / avg_loss) if avg_loss != 0 else 0
        
        return {
            'total_trades': total_trades,
            'winning_trades': winning_trades,
            'losing_trades': losing_trades,
            'total_pnl': total_pnl,
            'win_rate': win_rate,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'profit_factor': profit_factor
        }
    
    def calculate_equity_curve(self, trades_df, initial_capital=50000):
        """Calculate cumulative equity curve"""
        trades_df = trades_df.sort_values('date')
        trades_df['cumulative_pnl'] = trades_df['pnl'].cumsum()
        trades_df['equity'] = initial_capital + trades_df['cumulative_pnl']
        
        # Calculate drawdown
        trades_df['peak'] = trades_df['equity'].cummax()
        trades_df['drawdown'] = (trades_df['equity'] - trades_df['peak']) / trades_df['peak'] * 100
        
        return trades_df
    
    def calculate_roi(self, trades_df, initial_capital=50000):
        """Calculate return on investment"""
        total_pnl = trades_df['pnl'].sum()
        roi = (total_pnl / initial_capital) * 100
        return roi
    
    def calculate_max_drawdown(self, equity_series):
        """Calculate maximum drawdown"""
        peak = equity_series.cummax()
        drawdown = (equity_series - peak) / peak * 100
        return drawdown.min()
    
    def format_for_dashboard(self, trades_df, strategy_name, initial_capital=50000):
        """
        Format ML trading results for dashboard display.
        
        Args:
            trades_df: DataFrame with columns [date, pnl, signal, confidence, etc.]
            strategy_name: Name of the trading strategy
            initial_capital: Starting capital
            
        Returns:
            Dictionary matching dashboard data structure
        """
        # Calculate equity curve
        trades_df = self.calculate_equity_curve(trades_df, initial_capital)
        
        # Calculate metrics
        metrics = self.calculate_metrics(trades_df)
        roi = self.calculate_roi(trades_df, initial_capital)
        max_dd = self.calculate_max_drawdown(trades_df['equity'])
        
        # Format data
        dashboard_data = {
            "strategy": {
                "name": strategy_name,
                "backtested": f"on {datetime.now().strftime('%d %b %Y %H:%M')}",
                "period": f"{trades_df['date'].min().strftime('%d %b %y')} to {trades_df['date'].max().strftime('%d %b %y')}",
                "created": "Today"
            },
            "summary": {
                "metrics": [
                    {"label": "Symbol", "value": trades_df['symbol'].iloc[0] if 'symbol' in trades_df else "NIFTY50", "type": "text"},
                    {"label": "Capital", "value": initial_capital, "type": "currency"},
                    {"label": "Profit/Loss", "value": metrics['total_pnl'], "type": "currency"},
                    {"label": "ROI", "value": roi, "type": "percentage"},
                    {"label": "Drawdown", "value": f"{abs(max_dd):.2f}%", "type": "text", "sublabel": f"({max_dd:.2f}%)"},
                    {"label": "Risk Profile", "value": self._calculate_risk_profile(trades_df), "type": "text"},
                    {"label": "Win Rate", "value": f"{metrics['win_rate']:.2f}%", "type": "text"},
                    {"label": "Total Trades", "value": metrics['total_trades'], "type": "text"}
                ]
            },
            "performanceData": self._format_equity_curve(trades_df),
            "dayAnalysis": self._format_day_analysis(trades_df),
            "monthAnalysis": self._format_month_analysis(trades_df)
        }
        
        return dashboard_data
    
    def _format_equity_curve(self, trades_df):
        """Format equity curve for chart"""
        daily_equity = trades_df.groupby('date').agg({'equity': 'last'}).reset_index()
        return [
            {
                "date": row['date'].strftime("%b %d"),
                "equity": float(row['equity'])
            }
            for _, row in daily_equity.iterrows()
        ]
    
    def _format_day_analysis(self, trades_df):
        """Format day-wise analysis"""
        daily = trades_df.groupby('date').agg({
            'pnl': ['sum', 'count']
        }).reset_index()
        daily.columns = ['date', 'pnl', 'trades']
        
        # Calculate day of week profits
        trades_df['day_of_week'] = pd.to_datetime(trades_df['date']).dt.day_name()
        day_profits = trades_df.groupby('day_of_week')['pnl'].sum().to_dict()
        
        profit_by_day = {
            "Mon Profit": day_profits.get('Monday', 0),
            "Tue Profit": day_profits.get('Tuesday', 0),
            "Wed Profit": day_profits.get('Wednesday', 0),
            "Thu Profit": day_profits.get('Thursday', 0),
            "Fri Profit": day_profits.get('Friday', 0),
            "Sat Profit": 0,
            "Sun Profit": 0
        }
        
        positive_days = len(daily[daily['pnl'] > 0])
        negative_days = len(daily[daily['pnl'] < 0])
        total_days = len(daily)
        
        return {
            "stats": [
                {"label": "Trading Days", "value": total_days, "type": "number"},
                {"label": "Positive Days", "value": f"{positive_days} ({positive_days/total_days*100:.2f}%)", "type": "text", "className": "positive"},
                {"label": "Negative Days", "value": f"{negative_days} ({negative_days/total_days*100:.2f}%)", "type": "text", "className": "negative"},
                {"label": "Day Average profit", "value": float(daily['pnl'].mean()), "type": "currency"},
                {"label": "Day Max Profit", "value": float(daily['pnl'].max()), "type": "currency", "className": "positive"},
                {"label": "Day Max Loss", "value": float(daily['pnl'].min()), "type": "currency", "className": "negative"},
                {"label": "Day Average Trades", "value": int(daily['trades'].mean()), "type": "number"}
            ],
            "profitByDay": profit_by_day,
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
                for _, row in daily.iterrows()
            ]
        }
    
    def _format_month_analysis(self, trades_df):
        """Format month-wise analysis"""
        trades_df['month'] = pd.to_datetime(trades_df['date']).dt.to_period('M')
        monthly = trades_df.groupby('month').agg({
            'pnl': ['sum', 'count']
        }).reset_index()
        monthly.columns = ['month', 'pnl', 'trades']
        
        positive_months = len(monthly[monthly['pnl'] > 0])
        negative_months = len(monthly[monthly['pnl'] < 0])
        total_months = len(monthly)
        
        return {
            "stats": [
                {"label": "Total Months", "value": total_months, "type": "number"},
                {"label": "Positive Months", "value": f"{positive_months} ({positive_months/total_months*100:.0f}%)", "type": "text", "className": "positive"},
                {"label": "Negative Months", "value": f"{negative_months} ({negative_months/total_months*100:.0f}%)", "type": "text", "className": "negative"},
                {"label": "Month Average Profit", "value": float(monthly['pnl'].mean()), "type": "currency"},
                {"label": "Month Max Profit", "value": float(monthly['pnl'].max()), "type": "currency", "className": "positive"},
                {"label": "Month Average Trades", "value": int(monthly['trades'].mean()), "type": "number"}
            ],
            "tableData": [
                {
                    "month": row['month'].strftime("%b - %Y"),
                    "trades": int(row['trades']),
                    "targets": 0,
                    "stopLoss": 0,
                    "cover": 0,
                    "buyTrades": int(row['trades']),
                    "sellTrades": 0,
                    "qty": 0,
                    "roi": 0,
                    "profitLoss": float(row['pnl'])
                }
                for _, row in monthly.iterrows()
            ]
        }
    
    def _calculate_risk_profile(self, trades_df):
        """Calculate risk profile based on volatility"""
        volatility = trades_df['pnl'].std()
        max_dd_pct = abs(trades_df['drawdown'].min())
        
        if volatility < 500 and max_dd_pct < 5:
            return "Conservative"
        elif volatility < 1500 and max_dd_pct < 15:
            return "Moderate"
        else:
            return "Aggressive"
    
    def save_dashboard_data(self, dashboard_data, output_path):
        """Save formatted dashboard data to JSON"""
        with open(output_path, 'w') as f:
            json.dump(dashboard_data, f, indent=2, default=str)
        print(f"Dashboard data saved to {output_path}")


# Example usage
if __name__ == "__main__":
    # Initialize connector
    connector = DashboardDataConnector()
    
    # Example: Load your trading results
    # trades_df = pd.read_csv('results/backtest_trades.csv', parse_dates=['date'])
    
    # Or create sample data for testing
    dates = pd.date_range('2025-05-01', '2025-06-30', freq='D')
    sample_trades = pd.DataFrame({
        'date': dates,
        'symbol': 'NIFTY50',
        'pnl': np.random.randn(len(dates)) * 500 + 200,  # Random P&L
        'signal': np.random.choice(['BUY', 'SELL'], len(dates)),
        'confidence': np.random.uniform(0.5, 0.95, len(dates))
    })
    
    # Format for dashboard
    dashboard_data = connector.format_for_dashboard(
        sample_trades,
        strategy_name="ML Trading Strategy",
        initial_capital=50000
    )
    
    # Save to file
    connector.save_dashboard_data(dashboard_data, 'results/dashboard_data.json')
    
    print("Dashboard data generated successfully!")
    print(f"Total P&L: ₹{dashboard_data['summary']['metrics'][2]['value']:.2f}")
    print(f"ROI: {dashboard_data['summary']['metrics'][3]['value']:.2f}%")
