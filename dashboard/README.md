# Stock Trading Dashboard

Professional financial analytics dashboard for displaying ML model trading results.

## Setup

```bash
cd dashboard
npm install
npm start
```

The dashboard will open at `http://localhost:3000`

## Project Structure

```
dashboard/
├── src/
│   ├── components/
│   │   ├── Header.jsx              # Strategy header with metadata
│   │   ├── MetricsGrid.jsx         # Key metrics display
│   │   ├── TabNavigation.jsx       # Tab navigation component
│   │   ├── PerformanceChart.jsx    # Equity curve chart
│   │   ├── DayStats.jsx            # Day analysis statistics
│   │   ├── DayAnalysisTable.jsx    # Day-wise trading table
│   │   ├── MonthStats.jsx          # Month analysis statistics
│   │   └── MonthAnalysisTable.jsx  # Month-wise trading table
│   ├── styles/
│   │   └── Dashboard.css           # Dark theme styling
│   ├── App.jsx                     # Main dashboard component
│   └── index.js                    # React entry point
├── public/
│   └── index.html
└── package.json
```

## Data Integration

Replace sample data in `App.jsx` with your ML model outputs:

```javascript
const sampleData = {
  strategy: {
    name: "Your Strategy Name",
    backtested: "on DD MMM YYYY HH:MM",
    period: "DD MMM YY to DD MMM YY",
    created: "X days ago"
  },
  summary: {
    metrics: [
      { label: "Symbol", value: "STOCK", type: "text" },
      { label: "Capital", value: 50000, type: "currency" },
      { label: "Profit/Loss", value: 15000, type: "currency" },
      // ... more metrics
    ]
  },
  performanceData: [
    { date: "Jan 1", equity: 50000 },
    { date: "Jan 2", equity: 51500 },
    // ... more data points
  ],
  // ... rest of the data
};
```

## Components

### Header
- Strategy name
- Backtest metadata
- Action buttons

### MetricsGrid
- Capital, P&L, ROI, Drawdown
- Risk profile
- Color-coded values

### PerformanceChart
- Equity curve with gradient fill
- Time period filters (All, 1Y, 6M, 3M, 1M)
- Interactive tooltips

### Tables
- Day Analysis: Daily trading breakdown
- Month Analysis: Monthly aggregated stats
- Export functionality
- Sortable columns

## Styling

- Dark theme (#0a0e1a background)
- Green accent (#34d399)
- Professional trading platform aesthetic
- Responsive grid layouts
- Smooth animations

## Dependencies

- **React 18**: UI framework
- **Recharts**: Chart library for equity curve
- **react-scripts**: Build tooling

## Build for Production

```bash
npm run build
```

Output will be in `dashboard/build/` directory.

## Backend Integration

Connect to your ML backend by:
1. Creating an API service in `src/services/api.js`
2. Fetching data with `useEffect` in `App.jsx`
3. Updating state with backend responses

Example:
```javascript
useEffect(() => {
  fetch('/api/backtest-results')
    .then(res => res.json())
    .then(data => setBacktestData(data));
}, []);
```
