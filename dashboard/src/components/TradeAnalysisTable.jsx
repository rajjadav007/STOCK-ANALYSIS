import React from 'react';

export const TradeAnalysisTable = ({ data, isEmpty }) => {
  const formatCurrency = (value) => {
    return new Intl.NumberFormat('en-IN', {
      minimumFractionDigits: 2,
      maximumFractionDigits: 2
    }).format(value);
  };

  return (
    <div className="table-section">
      <div className="table-header">
        <h3 className="table-title">Trade Analysis</h3>
        <div className="table-controls">
          <select className="table-dropdown">
            <option>Last Month</option>
            <option>Last 3 Months</option>
            <option>Last 6 Months</option>
            <option>All Time</option>
          </select>
          <select className="table-dropdown">
            <option>ALL</option>
            <option>Buy</option>
            <option>Sell</option>
          </select>
          <select className="table-dropdown">
            <option>ALL</option>
            <option>Profit</option>
            <option>Loss</option>
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
        {isEmpty ? (
          <div style={{ 
            padding: '80px 20px', 
            textAlign: 'center', 
            color: '#6b7280',
            fontSize: '15px'
          }}>
            Currently, no trade analysis has been found!
          </div>
        ) : (
          <table className="data-table">
            <thead>
              <tr>
                <th>Symbol</th>
                <th className="align-center">Side</th>
                <th className="align-center">Qty</th>
                <th className="align-center">Entry</th>
                <th className="align-right">Entry Price</th>
                <th className="align-right">Exit Price</th>
                <th className="align-center">Exit</th>
                <th className="align-right">Profit / Loss</th>
                <th className="align-center">Result</th>
              </tr>
            </thead>
            <tbody>
              {data.map((row, index) => (
                <tr key={index}>
                  <td>{row.symbol}</td>
                  <td className="align-center">{row.side}</td>
                  <td className="align-center">{row.qty}</td>
                  <td className="align-center">{row.entry}</td>
                  <td className="align-right">{formatCurrency(row.entryPrice)}</td>
                  <td className="align-right">{formatCurrency(row.exitPrice)}</td>
                  <td className="align-center">{row.exit}</td>
                  <td className={`align-right ${row.profitLoss > 0 ? 'positive' : row.profitLoss < 0 ? 'negative' : ''}`}>
                    {row.profitLoss > 0 ? '+' : ''}{formatCurrency(row.profitLoss)}
                  </td>
                  <td className="align-center">
                    <span style={{
                      padding: '4px 12px',
                      borderRadius: '6px',
                      fontSize: '11px',
                      fontWeight: 600,
                      background: row.result === 'Win' ? 'rgba(52, 211, 153, 0.15)' : 'rgba(248, 113, 113, 0.15)',
                      color: row.result === 'Win' ? '#34d399' : '#f87171'
                    }}>
                      {row.result}
                    </span>
                  </td>
                </tr>
              ))}
              {data.length > 0 && (
                <tr style={{ fontWeight: 600, background: 'rgba(15, 20, 35, 0.8)' }}>
                  <td colSpan="2">Total</td>
                  <td className="align-center">{data.reduce((sum, row) => sum + row.qty, 0)}</td>
                  <td colSpan="4"></td>
                  <td className={`align-right ${data.reduce((sum, row) => sum + row.profitLoss, 0) > 0 ? 'positive' : 'negative'}`}>
                    {formatCurrency(data.reduce((sum, row) => sum + row.profitLoss, 0))}
                  </td>
                  <td></td>
                </tr>
              )}
            </tbody>
          </table>
        )}
      </div>
    </div>
  );
};

export default TradeAnalysisTable;
