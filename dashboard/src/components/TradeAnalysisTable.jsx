import React, { useState, useMemo } from 'react';

export const TradeAnalysisTable = ({ data, isEmpty, selectedStock }) => {
  const [timePeriod, setTimePeriod] = useState('All Time');
  const [sideFilter, setSideFilter] = useState('ALL');
  const [resultFilter, setResultFilter] = useState('ALL');

  const formatCurrency = (value) => {
    return new Intl.NumberFormat('en-IN', {
      minimumFractionDigits: 2,
      maximumFractionDigits: 2
    }).format(value);
  };

  // Filter data based on all filters
  const filteredData = useMemo(() => {
    if (!data || data.length === 0) return [];
    
    let filtered = [...data];
    
    // Symbol is already filtered by backend - don't filter again
    
    if (sideFilter !== 'ALL') {
      filtered = filtered.filter(row => 
        row.side?.toUpperCase() === sideFilter.toUpperCase()
      );
    }
    
    if (resultFilter !== 'ALL') {
      filtered = filtered.filter(row => {
        if (resultFilter === 'Profit') return row.profitLoss > 0;
        if (resultFilter === 'Loss') return row.profitLoss < 0;
        return true;
      });
    }
    
    let rowsToShow = filtered.length;
    if (timePeriod !== 'All Time') {
      switch (timePeriod) {
        case 'Last Month':
          rowsToShow = Math.min(30, filtered.length);
          break;
        case 'Last 3 Months':
          rowsToShow = Math.min(90, filtered.length);
          break;
        case 'Last 6 Months':
          rowsToShow = Math.min(180, filtered.length);
          break;
      }
      filtered = filtered.slice(-rowsToShow);
    }
    
    return filtered;
  }, [data, timePeriod, sideFilter, resultFilter]);

  return (
    <div className="table-section">
      <div className="table-header">
        <h3 className="table-title">Trade Analysis</h3>
        <div className="table-controls">
          <select 
            className="table-dropdown"
            value={timePeriod}
            onChange={(e) => setTimePeriod(e.target.value)}
          >
            <option value="All Time">All Time</option>
            <option value="Last Month">Last Month</option>
            <option value="Last 3 Months">Last 3 Months</option>
            <option value="Last 6 Months">Last 6 Months</option>
          </select>
          <select 
            className="table-dropdown"
            value={sideFilter}
            onChange={(e) => setSideFilter(e.target.value)}
          >
            <option value="ALL">ALL</option>
            <option value="BUY">Buy</option>
            <option value="SELL">Sell</option>
          </select>
          <select 
            className="table-dropdown"
            value={resultFilter}
            onChange={(e) => setResultFilter(e.target.value)}
          >
            <option value="ALL">ALL</option>
            <option value="Profit">Profit</option>
            <option value="Loss">Loss</option>
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
        {(isEmpty === true && (!data || data.length === 0)) ? (
          <div style={{ 
            padding: '80px 20px', 
            textAlign: 'center', 
            color: '#6b7280',
            fontSize: '15px'
          }}>
            Currently, no trade analysis has been found!
          </div>
        ) : filteredData.length === 0 ? (
          <div style={{ 
            padding: '80px 20px', 
            textAlign: 'center', 
            color: '#6b7280',
            fontSize: '15px'
          }}>
            No trades match the selected filters
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
              {filteredData.map((row, index) => (
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
              {filteredData.length > 0 && (
                <tr style={{ fontWeight: 600, background: 'rgba(15, 20, 35, 0.8)' }}>
                  <td colSpan="2">Total</td>
                  <td className="align-center">{filteredData.reduce((sum, row) => sum + row.qty, 0)}</td>
                  <td colSpan="4"></td>
                  <td className={`align-right ${filteredData.reduce((sum, row) => sum + row.profitLoss, 0) > 0 ? 'positive' : 'negative'}`}>
                    {formatCurrency(filteredData.reduce((sum, row) => sum + row.profitLoss, 0))}
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
