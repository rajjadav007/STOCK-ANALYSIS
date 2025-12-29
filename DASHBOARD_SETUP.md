# Dashboard Setup & Integration Guide

## Quick Start

### 1. Install Dashboard

```bash
cd dashboard
npm install
npm start
```

Dashboard runs at `http://localhost:3000`

### 2. Connect Your ML Model

#### Option A: Use Dashboard Connector (Recommended)

```python
from dashboard_connector import DashboardDataConnector

# Initialize
connector = DashboardDataConnector()

# Load your trading results
trades_df = pd.read_csv('results/my_backtest.csv')

# Format for dashboard
dashboard_data = connector.format_for_dashboard(
    trades_df,
    strategy_name="Your Strategy Name",
    initial_capital=50000
)

# Save to JSON
connector.save_dashboard_data(dashboard_data, 'results/dashboard_data.json')
```

Required columns in `trades_df`:
- `date`: Trade date
- `pnl`: Profit/Loss per trade
- `symbol`: Stock symbol (optional)

#### Option B: Use Flask API

```bash
# Install Flask
pip install flask flask-cors

# Run API server
python dashboard_api.py

# Update dashboard to fetch from API
# Edit dashboard/src/App.jsx:
# useEffect(() => {
#   fetch('http://localhost:5000/api/backtest-results')
#     .then(res => res.json())
#     .then(data => setDashboardData(data));
# }, []);
```

### 3. Customize Data

Replace sample data in `dashboard/src/App.jsx`:

```javascript
const sampleData = {
  strategy: {
    name: "Your Strategy",
    backtested: "on 27 Jun 2025 11:06",
    period: "01 Jan 25 to 30 Jun 25",
    created: "1 day ago"
  },
  summary: {
    metrics: [
      { label: "Capital", value: 50000, type: "currency" },
      { label: "Profit/Loss", value: 15000, type: "currency" },
      // ... add your metrics
    ]
  },
  performanceData: [
    { date: "Jan 1", equity: 50000 },
    // ... your equity curve data
  ]
};
```

## Dashboard Structure

```
dashboard/
├── src/
│   ├── components/
│   │   ├── Header.jsx              → Strategy title, metadata
│   │   ├── MetricsGrid.jsx         → Capital, P&L, ROI cards
│   │   ├── TabNavigation.jsx       → Tab switching
│   │   ├── PerformanceChart.jsx    → Equity curve
│   │   ├── DayAnalysisTable.jsx    → Daily breakdown
│   │   └── MonthAnalysisTable.jsx  → Monthly summary
│   ├── styles/
│   │   └── Dashboard.css           → Dark theme styling
│   └── App.jsx                     → Main component
└── package.json
```

## Data Format

### Performance Chart Data
```javascript
[
  { date: "Jun 1", equity: 55000 },
  { date: "Jun 2", equity: 56500 }
]
```

### Day Analysis Data
```javascript
{
  stats: [
    { label: "Trading Days", value: 30, type: "number" },
    { label: "Win Rate", value: 65.5, type: "percentage" }
  ],
  profitByDay: {
    "Mon Profit": 2500,
    "Tue Profit": -500
  },
  tableData: [
    {
      date: "24-06-2025",
      trades: 3,
      profitLoss: 250.00
    }
  ]
}
```

## Integration Examples

### From Existing Backtest

```python
# Load backtest results
results = pd.read_csv('results/backtest_results.csv')

# Convert to dashboard format
connector = DashboardDataConnector()
dashboard_data = connector.format_for_dashboard(
    results,
    strategy_name="MA Crossover",
    initial_capital=100000
)

# Use in dashboard
import json
with open('dashboard/src/data.json', 'w') as f:
    json.dump(dashboard_data, f)
```

### Live Data Updates

```javascript
// In App.jsx
const [data, setData] = useState(sampleData);

useEffect(() => {
  const interval = setInterval(() => {
    fetch('http://localhost:5000/api/live-data')
      .then(res => res.json())
      .then(newData => setData(newData));
  }, 5000); // Update every 5 seconds
  
  return () => clearInterval(interval);
}, []);
```

## Build for Production

```bash
cd dashboard
npm run build

# Deploy build/ folder to your web server
```

## Styling Customization

Edit `dashboard/src/styles/Dashboard.css`:

```css
/* Change accent color */
.tab-button.active {
  color: #your-color;
  border-bottom-color: #your-color;
}

/* Adjust chart colors */
<Area stroke="#your-color" fill="url(#yourGradient)" />
```

## Troubleshooting

**Issue**: Dashboard shows no data
- Verify `sampleData` structure matches expected format
- Check browser console for errors

**Issue**: Charts not rendering
- Ensure `recharts` is installed: `npm install recharts`
- Verify data has correct structure

**Issue**: API connection failed
- Check Flask server is running on port 5000
- Enable CORS: `pip install flask-cors`

## Next Steps

1. Connect to your ML model outputs
2. Customize metrics and labels
3. Add additional analysis tabs
4. Deploy to production server
