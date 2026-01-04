import React from 'react';

export const MetricsGrid = ({ metrics }) => {
  console.log('[MetricsGrid] Rendering with metrics:', metrics?.length || 0);
  
  if (!metrics || metrics.length === 0) {
    return (
      <div className="metrics-grid" style={{ 
        display: 'flex', 
        alignItems: 'center', 
        justifyContent: 'center',
        color: '#9ca3af',
        fontSize: '14px',
        padding: '2rem'
      }}>
        No metrics data available
      </div>
    );
  }

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
