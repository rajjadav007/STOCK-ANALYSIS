import React, { useMemo } from 'react';

export const YearAnalysisTable = ({ data }) => {
  const formatCurrency = (value) => {
    return new Intl.NumberFormat('en-IN', {
      minimumFractionDigits: 2,
      maximumFractionDigits: 2
    }).format(value);
  };

  const formatPercentage = (value) => {
    return `${value.toFixed(2)}%`;
  };

  // Calculate stats from data (excluding Total row)
  const stats = useMemo(() => {
    if (!data || data.length === 0) {
      return {
        totalYears: 0,
        positiveYears: 0,
        positivePct: 0,
        negativeYears: 0,
        negativePct: 0,
        avgProfit: 0,
        avgRoi: 0,
        maxProfit: 0,
        avgTrades: 0
      };
    }

    const dataRows = data.filter(row => row.year !== 'Total');
    const positiveYears = dataRows.filter(row => row.profitLoss > 0).length;
    const negativeYears = dataRows.filter(row => row.profitLoss < 0).length;
    const totalYears = dataRows.length;
    const avgProfit = totalYears > 0 ? dataRows.reduce((sum, row) => sum + row.profitLoss, 0) / totalYears : 0;
    const avgRoi = totalYears > 0 ? dataRows.reduce((sum, row) => sum + row.roi, 0) / totalYears : 0;
    const maxProfit = totalYears > 0 ? Math.max(...dataRows.map(row => row.profitLoss)) : 0;
    const avgTrades = totalYears > 0 ? Math.round(dataRows.reduce((sum, row) => sum + row.trades, 0) / totalYears) : 0;

    return {
      totalYears,
      positiveYears,
      positivePct: totalYears > 0 ? (positiveYears / totalYears * 100).toFixed(0) : 0,
      negativeYears,
      negativePct: totalYears > 0 ? (negativeYears / totalYears * 100).toFixed(0) : 0,
      avgProfit,
      avgRoi,
      maxProfit,
      avgTrades
    };
  }, [data]);

  return (
    <div className="table-section">
      <div className="stats-grid" style={{ marginBottom: '20px' }}>
        <div className="stat-card">
          <div className="stat-label">Total Years</div>
          <div className="stat-value">{stats.totalYears}</div>
        </div>
        <div className="stat-card">
          <div className="stat-label">Positive Years</div>
          <div className="stat-value positive">{stats.positiveYears} ({stats.positivePct}%)</div>
        </div>
        <div className="stat-card">
          <div className="stat-label">Negative Years</div>
          <div className="stat-value negative">{stats.negativeYears} ({stats.negativePct}%)</div>
        </div>
        <div className="stat-card">
          <div className="stat-label">Year Average Profit</div>
          <div className="stat-value">{formatCurrency(stats.avgProfit)}</div>
        </div>
        <div className="stat-card">
          <div className="stat-label">Year ROI</div>
          <div className="stat-value">{formatPercentage(stats.avgRoi)}</div>
        </div>
        <div className="stat-card">
          <div className="stat-label">Year Max Profit</div>
          <div className="stat-value positive">{formatCurrency(stats.maxProfit)}</div>
        </div>
        <div className="stat-card">
          <div className="stat-label">Year Average Trades</div>
          <div className="stat-value">{stats.avgTrades}</div>
        </div>
      </div>

      <div className="table-header">
        <h3 className="table-title">Year Analysis</h3>
        <div className="table-controls">
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
              <th>Year</th>
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
            {data.map((row, index) => (
              <tr key={index}>
                <td>{row.year}</td>
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

export default YearAnalysisTable;
