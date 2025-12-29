import React from 'react';

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

  return (
    <div className="table-section">
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
