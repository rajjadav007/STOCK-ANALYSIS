import React from 'react';
import { AreaChart, Area, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from 'recharts';

export const DrawdownChart = ({ data, drawdownInfo, timeFilter, onTimeFilterChange }) => {
  const CustomTooltip = ({ active, payload }) => {
    if (active && payload && payload.length) {
      return (
        <div style={{
          background: 'rgba(15, 20, 35, 0.95)',
          border: '1px solid rgba(248, 113, 113, 0.3)',
          borderRadius: '8px',
          padding: '12px',
          fontSize: '13px'
        }}>
          <div style={{ color: '#9ca3af', marginBottom: '4px' }}>{payload[0].payload.date}</div>
          <div style={{ color: '#f87171', fontWeight: 600 }}>
            {payload[0].value.toFixed(2)}%
          </div>
        </div>
      );
    }
    return null;
  };

  return (
    <>
      <div className="stats-grid" style={{ marginBottom: '32px' }}>
        <div className="stat-card">
          <div className="stat-label">Drawdown</div>
          <div className="stat-value negative">
            {drawdownInfo.drawdown}
          </div>
        </div>
        <div className="stat-card">
          <div className="stat-label">Down Start Date</div>
          <div className="stat-value">
            {drawdownInfo.downStartDate}
          </div>
        </div>
        <div className="stat-card">
          <div className="stat-label">Max Down Date</div>
          <div className="stat-value">
            {drawdownInfo.maxDownDate}
          </div>
        </div>
        <div className="stat-card">
          <div className="stat-label">Recovery Date</div>
          <div className="stat-value">
            {drawdownInfo.recoveryDate}
          </div>
        </div>
        <div className="stat-card">
          <div className="stat-label">Recovery Period</div>
          <div className="stat-value">
            {drawdownInfo.recoveryPeriod}
          </div>
        </div>
      </div>

      <div className="chart-section">
        <div className="chart-header">
          <h3 className="chart-title">Drawdown Chart</h3>
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
            <AreaChart data={data} margin={{ top: 10, right: 30, left: 0, bottom: 0 }}>
              <defs>
                <linearGradient id="colorDrawdown" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="5%" stopColor="#f87171" stopOpacity={0.3}/>
                  <stop offset="95%" stopColor="#f87171" stopOpacity={0}/>
                </linearGradient>
              </defs>
              <CartesianGrid strokeDasharray="3 3" stroke="rgba(248, 113, 113, 0.1)" />
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
                tickFormatter={(value) => `${value}%`}
                domain={['dataMin', 0]}
              />
              <Tooltip content={<CustomTooltip />} />
              <Area 
                type="monotone" 
                dataKey="drawdown" 
                stroke="#f87171" 
                strokeWidth={2}
                fillOpacity={1} 
                fill="url(#colorDrawdown)" 
              />
            </AreaChart>
          </ResponsiveContainer>
        </div>
      </div>
    </>
  );
};

export default DrawdownChart;
