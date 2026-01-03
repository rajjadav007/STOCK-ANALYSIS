import React from 'react';

export const Header = ({ strategyName, backtested, period, created, stocks, selectedStock, onStockChange }) => {
  return (
    <div className="dashboard-header">
      <div>
        <h1 className="strategy-title">{strategyName}</h1>
        <div className="strategy-meta">
          <span>
            <span className="label">Backtested:</span>
            <span className="value">{backtested}</span>
          </span>
          <span>
            <span className="label">Period:</span>
            <span className="value">{period}</span>
          </span>
          <span>
            <span className="label">Created:</span>
            <span className="value">{created}</span>
          </span>
        </div>
      </div>
      <div className="header-actions">
        <div className="ml-prediction-badge" style={{
          background: 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)',
          color: 'white',
          padding: '8px 16px',
          borderRadius: '8px',
          fontWeight: '600',
          fontSize: '14px',
          marginRight: '12px',
          display: 'flex',
          alignItems: 'center',
          gap: '8px',
          boxShadow: '0 2px 8px rgba(102, 126, 234, 0.3)'
        }}>
          <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
            <circle cx="12" cy="12" r="10" />
            <path d="M12 6v6l4 2" />
          </svg>
          ML Prediction
        </div>
        <select
          className="stock-selector"
          value={selectedStock}
          onChange={(e) => onStockChange(e.target.value)}
        >
          {stocks.map(stock => (
            <option key={stock} value={stock}>{stock}</option>
          ))}
        </select>
        <button className="btn-icon" title="Download">
          <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
            <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4M7 10l5 5 5-5M12 15V3" />
          </svg>
        </button>
        <button className="btn-icon" title="Close">
          <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
            <path d="M18 6L6 18M6 6l12 12" />
          </svg>
        </button>
      </div>
    </div>
  );
};

export default Header;
