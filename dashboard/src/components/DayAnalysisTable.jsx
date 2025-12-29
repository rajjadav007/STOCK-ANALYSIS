import React, { useState, useMemo } from 'react';

export const DayAnalysisTable = ({ data, profitByDay }) => {
  const [timePeriod, setTimePeriod] = useState('Last 3 Month');

  const formatCurrency = (value) => {
    return new Intl.NumberFormat('en-IN', {
      minimumFractionDigits: 2,
      maximumFractionDigits: 2
    }).format(value);
  };

  // Filter data based on time period
  const filteredData = useMemo(() => {
    if (!data || data.length === 0) return [];
    
    let rowsToShow;
    switch (timePeriod) {
      case 'Last 3 Month':
        rowsToShow = Math.min(90, data.length); // ~3 months
        break;
      case 'Last 6 Month':
        rowsToShow = Math.min(180, data.length); // ~6 months
        break;
      case 'Last Year':
        rowsToShow = Math.min(365, data.length); // ~1 year
        break;
      case 'All Time':
      default:
        rowsToShow = data.length;
        break;
    }
    
    return data.slice(0, rowsToShow);
  }, [data, timePeriod]);

  return (
    <div className="table-section">
      <div className="profit-display">
        {profitByDay && Object.entries(profitByDay).map(([day, profit]) => (
          <div key={day} className="profit-day">
            <div className="profit-day-label">{day}</div>
            <div className={`profit-day-value ${profit > 0 ? 'positive' : profit < 0 ? 'negative' : ''}`}>
              {profit > 0 ? '+' : ''}{formatCurrency(profit)}
            </div>
          </div>
        ))}
      </div>

      <div className="table-header">
        <h3 className="table-title">Day Analysis</h3>
        <div className="table-controls">
          <select 
            className="table-dropdown"
            value={timePeriod}
            onChange={(e) => setTimePeriod(e.target.value)}
          >
            <option value="Last 3 Month">Last 3 Month</option>
            <option value="Last 6 Month">Last 6 Month</option>
            <option value="Last Year">Last Year</option>
            <option value="All Time">All Time</option>
          </select>
          <button className="btn-export">
            <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
              <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4M7 10l5 5 5-5M12 15V3"/>
            </svg>
            Export
          </button>
        </div>
      </div>

      <div className="table-container">
        <table className="data-table">
          <thead>
            <tr>
              <th>Date</th>
              <th className="align-center">Trades</th>
              <th className="align-center">Targets</th>
              <th className="align-center">Stop - Loss</th>
              <th className="align-center">Cover</th>
              <th className="align-center">Buy Trades</th>
              <th className="align-center">Sell Trades</th>
              <th className="align-center">Qty</th>
              <th className="align-right">Profit / Loss</th>
            </tr>
          </thead>
          <tbody>
            {filteredData.map((row, index) => (
              <tr key={index}>
                <td>{row.date}</td>
                <td className="align-center">{row.trades}</td>
                <td className="align-center">{row.targets}</td>
                <td className="align-center">{row.stopLoss}</td>
                <td className="align-center">{row.cover}</td>
                <td className="align-center">{row.buyTrades}</td>
                <td className="align-center">{row.sellTrades}</td>
                <td className="align-center">{row.qty}</td>
                <td className={`align-right ${row.profitLoss > 0 ? 'positive' : row.profitLoss < 0 ? 'negative' : ''}`}>
                  {row.profitLoss > 0 ? '+' : ''}{formatCurrency(row.profitLoss)}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
};

export default DayAnalysisTable;
