import React from 'react';

export const TradeStats = ({ stats }) => {
  const formatValue = (value, type) => {
    if (type === 'currency') {
      return new Intl.NumberFormat('en-IN', {
        minimumFractionDigits: 2,
        maximumFractionDigits: 2
      }).format(value);
    }
    if (type === 'percentage') {
      return `${value}`;
    }
    return value;
  };

  return (
    <div className="stats-grid">
      {stats.map((stat, index) => (
        <div key={index} className="stat-card">
          <div className="stat-label">{stat.label}</div>
          <div className={`stat-value ${stat.className || ''}`}>
            {formatValue(stat.value, stat.type)}
          </div>
        </div>
      ))}
    </div>
  );
};

export default TradeStats;
