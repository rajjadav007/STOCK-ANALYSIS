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

  // Filter data based on selected year and recalculate stats
  const { filteredData, filteredStats } = useMemo(() => {
    if (!data || data.length === 0) return { filteredData: [], filteredStats: null };
    
    // Separate total row from data rows
    const totalRow = data.find(row => row.month === 'Total');
    const dataRows = data.filter(row => row.month !== 'Total');
    
    // Filter by year
    let filtered = selectedYear === 'All Years' 
      ? dataRows 
      : dataRows.filter(row => row.month.includes(selectedYear));
    
    // Recalculate totals from filtered data
    if (filtered.length > 0) {
      const newTotal = {
        month: 'Total',
        trades: filtered.reduce((sum, row) => sum + row.trades, 0),
        targets: filtered.reduce((sum, row) => sum + row.targets, 0),
        stopLoss: filtered.reduce((sum, row) => sum + row.stopLoss, 0),
        cover: filtered.reduce((sum, row) => sum + row.cover, 0),
        buyTrades: filtered.reduce((sum, row) => sum + row.buyTrades, 0),
        sellTrades: filtered.reduce((sum, row) => sum + row.sellTrades, 0),
        qty: filtered.reduce((sum, row) => sum + row.qty, 0),
        roi: 0,
        profitLoss: filtered.reduce((sum, row) => sum + row.profitLoss, 0)
      };
      filtered = [...filtered, newTotal];
    }
    
    // Calculate stats from filtered data (excluding total row)
    const statsData = filtered.filter(row => row.month !== 'Total');
    const positiveMonths = statsData.filter(row => row.profitLoss > 0).length;
    const negativeMonths = statsData.filter(row => row.profitLoss < 0).length;
    const totalMonths = statsData.length;
    const avgProfit = totalMonths > 0 ? statsData.reduce((sum, row) => sum + row.profitLoss, 0) / totalMonths : 0;
    const avgRoi = totalMonths > 0 ? statsData.reduce((sum, row) => sum + row.roi, 0) / totalMonths : 0;
    const maxProfit = totalMonths > 0 ? Math.max(...statsData.map(row => row.profitLoss)) : 0;
    const avgTrades = totalMonths > 0 ? Math.round(statsData.reduce((sum, row) => sum + row.trades, 0) / totalMonths) : 0;
    
    const stats = {
      totalMonths,
      positiveMonths,
      positivePct: totalMonths > 0 ? (positiveMonths / totalMonths * 100).toFixed(0) : 0,
      negativeMonths,
      negativePct: totalMonths > 0 ? (negativeMonths / totalMonths * 100).toFixed(0) : 0,
      avgProfit,
      avgRoi,
      maxProfit,
      avgTrades
    };
    
    return { filteredData: filtered, filteredStats: stats };
  }, [data, selectedYear]);

  return (
    <div className="table-section">
      {filteredStats && (
        <div className="stats-grid" style={{ marginBottom: '20px' }}>
          <div className="stat-card">
            <div className="stat-label">Total Months</div>
            <div className="stat-value">{filteredStats.totalMonths}</div>
          </div>
          <div className="stat-card">
            <div className="stat-label">Positive Months</div>
            <div className="stat-value positive">{filteredStats.positiveMonths} ({filteredStats.positivePct}%)</div>
          </div>
          <div className="stat-card">
            <div className="stat-label">Negative Months</div>
            <div className="stat-value negative">{filteredStats.negativeMonths} ({filteredStats.negativePct}%)</div>
          </div>
          <div className="stat-card">
            <div className="stat-label">Month Average Profit</div>
            <div className="stat-value">{formatCurrency(filteredStats.avgProfit)}</div>
          </div>
          <div className="stat-card">
            <div className="stat-label">Month ROI</div>
            <div className="stat-value">{formatPercentage(filteredStats.avgRoi)}</div>
          </div>
          <div className="stat-card">
            <div className="stat-label">Month Max Profit</div>
            <div className="stat-value positive">{formatCurrency(filteredStats.maxProfit)}</div>
          </div>
          <div className="stat-card">
            <div className="stat-label">Month Average Trades</div>
            <div className="stat-value">{filteredStats.avgTrades}</div>
          </div>
        </div>
      )}

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
