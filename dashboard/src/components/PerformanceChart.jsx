import React, { useMemo } from 'react';
import { ComposedChart, Bar, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Cell } from 'recharts';

export const PerformanceChart = ({ data, timeFilter, onTimeFilterChange }) => {
  // Filter data based on time period - using actual trade counts
  const filteredData = useMemo(() => {
    if (!data || data.length === 0) return [];
    if (timeFilter === 'All') return data;

    // Calculate exact number of trades to show based on time period
    let tradesToShow;
    switch (timeFilter) {
      case '1M':
        tradesToShow = Math.min(22, data.length); // ~22 trades in a month
        break;
      case '3M':
        tradesToShow = Math.min(66, data.length); // ~66 trades in 3 months
        break;
      case '6M':
        tradesToShow = Math.min(132, data.length); // ~132 trades in 6 months
        break;
      case '1Y':
        tradesToShow = Math.min(252, data.length); // ~252 trades in a year
        break;
      default:
        tradesToShow = data.length;
    }

    // Return the last N trade points
    return data.slice(-tradesToShow);
  }, [data, timeFilter]);

  const CustomTooltip = ({ active, payload }) => {
    if (active && payload && payload.length) {
      const data = payload[0].payload;
      return (
        <div style={{
          background: 'rgba(15, 20, 35, 0.95)',
          border: `1px solid ${data.type === 'win' ? 'rgba(52, 211, 153, 0.5)' : data.type === 'loss' ? 'rgba(239, 68, 68, 0.5)' : 'rgba(52, 211, 153, 0.3)'}`,
          borderRadius: '8px',
          padding: '12px',
          fontSize: '13px'
        }}>
          <div style={{ color: '#9ca3af', marginBottom: '6px' }}>{data.date}</div>
          <div style={{ color: '#34d399', fontWeight: 600, marginBottom: '4px' }}>
            ₹{data.equity.toLocaleString('en-IN', { minimumFractionDigits: 2 })}
          </div>
          {data.type !== 'start' && (
            <div style={{ 
              color: data.pnl > 0 ? '#34d399' : '#ef4444',
              fontSize: '12px',
              marginTop: '4px'
            }}>
              P&L: {data.pnl > 0 ? '+' : ''}₹{data.pnl.toLocaleString('en-IN', { minimumFractionDigits: 2 })}
            </div>
          )}
        </div>
      );
    }
    return null;
  };

  return (
    <div className="chart-section">
      <div className="chart-header">
        <h3 className="chart-title">Performance Chart</h3>
        <div className="time-filter">
          {['All', '1Y', '6M', '3M', '1M'].map((filter) => (
            <button
              key={filter}
              className={timeFilter === filter ? 'active' : ''}
              onClick={() => onTimeFilterChange(filter)}
            >
              {filter}
            </button>
          ))}
        </div>
      </div>
      <div className="chart-container">
        <ResponsiveContainer width="100%" height="100%">
          <ComposedChart data={filteredData} margin={{ top: 10, right: 30, left: 0, bottom: 0 }}>
            <defs>
              <linearGradient id="colorProfit" x1="0" y1="0" x2="0" y2="1">
                <stop offset="5%" stopColor="#34d399" stopOpacity={0.8}/>
                <stop offset="95%" stopColor="#34d399" stopOpacity={0.2}/>
              </linearGradient>
              <linearGradient id="colorLoss" x1="0" y1="0" x2="0" y2="1">
                <stop offset="5%" stopColor="#ef4444" stopOpacity={0.8}/>
                <stop offset="95%" stopColor="#ef4444" stopOpacity={0.2}/>
              </linearGradient>
            </defs>
            <CartesianGrid strokeDasharray="3 3" stroke="rgba(52, 211, 153, 0.1)" />
            <XAxis 
              dataKey="date" 
              stroke="#6b7280" 
              style={{ fontSize: '12px' }}
              tick={{ fill: '#9ca3af' }}
            />
            <YAxis 
              stroke="#6b7280" 
              style={{ fontSize: '12px' }}
              tick={{ fill: '#9ca3af' }}
              tickFormatter={(value) => `₹${(value / 1000).toFixed(0)}k`}
            />
            <Tooltip content={<CustomTooltip />} />
            <Bar dataKey="pnl" fill="#34d399" radius={[4, 4, 0, 0]}>
              {filteredData.map((entry, index) => (
                <Cell 
                  key={`cell-${index}`} 
                  fill={entry.type === 'win' ? '#34d399' : entry.type === 'loss' ? '#ef4444' : 'transparent'}
                  fillOpacity={0.6}
                />
              ))}
            </Bar>
            <Line 
              type="stepAfter" 
              dataKey="equity" 
              stroke="#34d399" 
              strokeWidth={2}
              dot={{ fill: '#34d399', r: 3 }}
              activeDot={{ r: 5 }}
            />
          </ComposedChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
};

export default PerformanceChart;
