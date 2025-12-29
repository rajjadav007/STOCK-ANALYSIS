import React, { useState, useMemo } from 'react';

export const MonthAnalysisTable = ({ data }) => {
  const [selectedYear, setSelectedYear] = useState('All Years');

  const formatCurrency = (value) => {
    return new Intl.NumberFormat('en-IN', {
      minimumFractionDigits: 2,
      maximumFractionDigits: 2
    }).format(value);
  };

  const formatPercentage = (value) => {
    return `${value.toFixed(2)}%`;
  };

  // Get available years from data
  const availableYears = useMemo(() => {
    if (!data || data.length === 0) return ['All Years'];
    const years = [...new Set(data
      .filter(row => row.month !== 'Total')
      .map(row => row.month.split(' - ')[1])
      .filter(Boolean)
    )];
    return ['All Years', ...years.sort().reverse()];
  }, [data]);

  // Filter data based on selected year
  const filteredData = useMemo(() => {
    if (!data || data.length === 0) return [];
    if (selectedYear === 'All Years') return data;
    
    return data.filter(row => 
      row.month === 'Total' || row.month.includes(selectedYear)
    );
  }, [data, selectedYear]);

  return (
    <div className="table-section">
      <div className="table-header">
        <h3 className="table-title">Month Analysis</h3>
        <div className="table-controls">
          <select 
            className="table-dropdown"
            value={selectedYear}
            onChange={(e) => setSelectedYear(e.target.value)}
          >
            {availableYears.map(year => (
              <option key={year} value={year}>{year}</option>
            ))}
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
              <th>Month</th>
              <th className="align-center">Trades</th>
              <th className="align-center">Targets</th>
              <th className="align-center">Stop - Loss</th>
              <th className="align-center">Cover</th>
              <th className="align-center">Buy Trades</th>
              <th className="align-center">Sell Trades</th>
              <th className="align-center">Qty</th>
              <th className="align-right">ROI</th>
              <th className="align-right">Profit / Loss</th>
            </tr>
          </thead>
          <tbody>
            {filteredData.map((row, index) => (
              <tr key={index}>
                <td>{row.month}</td>
                <td className="align-center">{row.trades}</td>
                <td className="align-center">{row.targets}</td>
                <td className="align-center">{row.stopLoss}</td>
                <td className="align-center">{row.cover}</td>
                <td className="align-center">{row.buyTrades}</td>
                <td className="align-center">{row.sellTrades}</td>
                <td className="align-center">{row.qty}</td>
                <td className={`align-right ${row.roi > 0 ? 'positive' : row.roi < 0 ? 'negative' : ''}`}>
                  {formatPercentage(row.roi)}
                </td>
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

export default MonthAnalysisTable;
