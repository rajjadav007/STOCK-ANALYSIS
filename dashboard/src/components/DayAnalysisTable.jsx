import React, { useState, useMemo } from 'react';

export const DayAnalysisTable = ({ data, profitByDay }) => {
  const [timePeriod, setTimePeriod] = useState('Last 3 Month');

  const formatCurrency = (value) => {
    return new Intl.NumberFormat('en-IN', {
      minimumFractionDigits: 2,
      maximumFractionDigits: 2
    }).format(value);
  };

  // Filter data based on time period - SLICE FROM END (most recent)
  const filteredData = useMemo(() => {
    if (!data || data.length === 0) return [];
    
    let rowsToShow;
    switch (timePeriod) {
      case 'Last 3 Month':
        rowsToShow = Math.min(66, data.length); // ~66 trading days in 3 months
        break;
      case 'Last 6 Month':
        rowsToShow = Math.min(132, data.length); // ~132 trading days in 6 months
        break;
      case 'Last Year':
        rowsToShow = Math.min(252, data.length); // ~252 trading days in 1 year
        break;
      case 'All Time':
      default:
        rowsToShow = data.length;
        break;
    }
    
    // Take LAST N rows (most recent dates) since data is sorted ascending
    return data.slice(-rowsToShow);
  }, [data, timePeriod]);

  // Recalculate stats from FILTERED data only
  const filteredStats = useMemo(() => {
    if (!filteredData || filteredData.length === 0) {
      return {
        tradingDays: 0,
        positiveDays: 0,
        negativeDays: 0,
        avgProfit: 0,
        maxProfit: 0,
        maxLoss: 0,
        avgTrades: 0
      };
    }

    const positiveDays = filteredData.filter(d => d.profitLoss > 0).length;
    const negativeDays = filteredData.filter(d => d.profitLoss < 0).length;
    const totalPnl = filteredData.reduce((sum, d) => sum + d.profitLoss, 0);
    const totalTrades = filteredData.reduce((sum, d) => sum + d.trades, 0);
    const maxProfit = Math.max(...filteredData.map(d => d.profitLoss));
    const maxLoss = Math.min(...filteredData.map(d => d.profitLoss));

    return {
      tradingDays: filteredData.length,
      positiveDays: positiveDays,
      positivePct: (positiveDays / filteredData.length * 100).toFixed(2),
      negativeDays: negativeDays,
      negativePct: (negativeDays / filteredData.length * 100).toFixed(2),
      avgProfit: totalPnl / filteredData.length,
      maxProfit: maxProfit,
      maxLoss: maxLoss,
      avgTrades: Math.round(totalTrades / filteredData.length)
    };
  }, [filteredData]);

  // Recalculate profit by day from filtered data
  const filteredProfitByDay = useMemo(() => {
    if (!filteredData || filteredData.length === 0) {
      return {
        "Mon Profit": 0,
        "Tue Profit": 0,
        "Wed Profit": 0,
        "Thu Profit": 0,
        "Fri Profit": 0,
        "Sat Profit": 0,
        "Sun Profit": 0
      };
    }

    const dayMap = {
      "Mon Profit": 0,
      "Tue Profit": 0,
      "Wed Profit": 0,
      "Thu Profit": 0,
      "Fri Profit": 0,
      "Sat Profit": 0,
      "Sun Profit": 0
    };

    filteredData.forEach(row => {
      if (row.date) {
        const dateParts = row.date.split('-');
        if (dateParts.length === 3) {
          const dateObj = new Date(`${dateParts[2]}-${dateParts[1]}-${dateParts[0]}`);
          const dayName = dateObj.toLocaleDateString('en-US', { weekday: 'short' });
          const key = `${dayName} Profit`;
          if (dayMap.hasOwnProperty(key)) {
            dayMap[key] += row.profitLoss;
          }
        }
      }
    });

    return dayMap;
  }, [filteredData]);

  return (
    <div className="table-section">
      <div className="stats-grid" style={{ marginBottom: '20px' }}>
        <div className="stat-card">
          <div className="stat-label">Trading Days</div>
          <div className="stat-value">{filteredStats.tradingDays}</div>
        </div>
        <div className="stat-card">
          <div className="stat-label">Positive Days</div>
          <div className="stat-value positive">{filteredStats.positiveDays} ({filteredStats.positivePct}%)</div>
        </div>
        <div className="stat-card">
          <div className="stat-label">Negative Days</div>
          <div className="stat-value negative">{filteredStats.negativeDays} ({filteredStats.negativePct}%)</div>
        </div>
        <div className="stat-card">
          <div className="stat-label">Day Average Profit</div>
          <div className="stat-value">{formatCurrency(filteredStats.avgProfit)}</div>
        </div>
        <div className="stat-card">
          <div className="stat-label">Day Max Profit</div>
          <div className="stat-value positive">{formatCurrency(filteredStats.maxProfit)}</div>
        </div>
        <div className="stat-card">
          <div className="stat-label">Day Max Loss</div>
          <div className="stat-value negative">{formatCurrency(filteredStats.maxLoss)}</div>
        </div>
        <div className="stat-card">
          <div className="stat-label">Day Average Trades</div>
          <div className="stat-value">{filteredStats.avgTrades}</div>
        </div>
      </div>

      <div className="profit-display">
        {filteredProfitByDay && Object.entries(filteredProfitByDay).map(([day, profit]) => (
          <div key={day} className="profit-day">
            <div className="profit-day-label">{day}</div>
            <div className={`profit-day-value ${ profit > 0 ? 'positive' : profit < 0 ? 'negative' : ''}`}>
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
