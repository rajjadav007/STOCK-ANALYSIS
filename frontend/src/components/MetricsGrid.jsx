import React from 'react';

export const MetricsGrid = ({ metrics }) => {
  const formatValue = (value, type) => {
    if (type === 'currency') {
      return new Intl.NumberFormat('en-IN', {
        style: 'currency',
        currency: 'INR',
        minimumFractionDigits: 2
      }).format(value).replace('₹', '₹');
    }
    if (type === 'percentage') {
      return `${value}%`;
    }
    return value;
  };

  const getValueClass = (value, type) => {
    if (type === 'currency' || type === 'percentage') {
      return value > 0 ? 'positive' : value < 0 ? 'negative' : '';
    }
    return '';
  };

  return (
    <div className="metrics-grid">
      {metrics.map((metric, index) => (
        <div key={index} className="metric-card">
          <div className="metric-label">{metric.label}</div>
          <div className={`metric-value ${getValueClass(metric.value, metric.type)}`}>
            {formatValue(metric.value, metric.type)}
          </div>
          {metric.sublabel && (
            <div className="metric-sublabel">{metric.sublabel}</div>
          )}
        </div>
      ))}
    </div>
  );
};

export default MetricsGrid;
