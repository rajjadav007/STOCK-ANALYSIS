import React, { useMemo } from 'react';
import { ComposedChart, Bar, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Cell, Rectangle } from 'recharts';

// Custom Candlestick Shape
const CandlestickShape = (props) => {
  const { x, y, width, height, payload } = props;
  
  if (!payload || !payload.open || !payload.close || !payload.high || !payload.low) {
    return null;
  }
  
  const isUpward = payload.close >= payload.open;
  const color = isUpward ? '#34d399' : '#ef4444';
  const bodyHeight = Math.abs(payload.close - payload.open);
  const bodyY = Math.min(payload.close, payload.open);
  
  // Wick (high-low line)
  const wickX = x + width / 2;
  
  return (
    <g>
      {/* Upper wick */}
      <line
        x1={wickX}
        y1={y + height - ((payload.high - payload.low) * (height / (payload.high - payload.low)))}
        x2={wickX}
        y2={y + height - ((bodyY + (isUpward ? bodyHeight : 0) - payload.low) * (height / (payload.high - payload.low)))}
        stroke={color}
        strokeWidth={1}
      />
      {/* Lower wick */}
      <line
        x1={wickX}
        y1={y + height - ((bodyY + (isUpward ? 0 : bodyHeight) - payload.low) * (height / (payload.high - payload.low)))}
        x2={wickX}
        y2={y + height}
        stroke={color}
        strokeWidth={1}
      />
      {/* Body */}
      <rect
        x={x + width * 0.2}
        y={y + height - ((bodyY + bodyHeight - payload.low) * (height / (payload.high - payload.low)))}
        width={width * 0.6}
        height={Math.max(1, (bodyHeight * height) / (payload.high - payload.low))}
        fill={color}
        stroke={color}
        strokeWidth={1}
      />
    </g>
  );
};

export const PerformanceChart = ({ data, timeFilter, onTimeFilterChange, candlestickData }) => {
  // Filter data based on time period - using actual trade counts or candlestick data
  const filteredData = useMemo(() => {
    // If we have candlestick data, use that instead
    if (candlestickData && candlestickData.length > 0) {
      if (timeFilter === 'All') return candlestickData;
      
      let candlesToShow;
      switch (timeFilter) {
        case '1M':
          candlesToShow = Math.min(22, candlestickData.length);
          break;
        case '3M':
          candlesToShow = Math.min(66, candlestickData.length);
          break;
        case '6M':
          candlesToShow = Math.min(132, candlestickData.length);
          break;
        case '1Y':
          candlesToShow = Math.min(252, candlestickData.length);
          break;
        default:
          candlesToShow = candlestickData.length;
      }
      return candlestickData.slice(-candlesToShow);
    }
    
    // Otherwise use equity/performance data
    if (!data || data.length === 0) return [];
    if (timeFilter === 'All') return data;

    // Calculate exact number of trades to show based on time period
    let tradesToShow;
    switch (timeFilter) {
      case '1M':
        tradesToShow = Math.min(22, data.length);
        break;
      case '3M':
        tradesToShow = Math.min(66, data.length);
        break;
      case '6M':
        tradesToShow = Math.min(132, data.length);
        break;
      case '1Y':
        tradesToShow = Math.min(252, data.length);
        break;
      default:
        tradesToShow = data.length;
    }

    // Return the last N trade points
    return data.slice(-tradesToShow);
  }, [data, candlestickData, timeFilter]);

  const CustomTooltip = ({ active, payload }) => {
    if (active && payload && payload.length) {
      const data = payload[0].payload;
      
      // Check if this is candlestick data or performance data
      if (data.open !== undefined && data.high !== undefined) {
        // Candlestick tooltip
        const change = data.close - data.open;
        const changePercent = ((change / data.open) * 100).toFixed(2);
        
        return (
          <div style={{
            background: 'rgba(15, 20, 35, 0.95)',
            border: `1px solid ${data.close >= data.open ? 'rgba(52, 211, 153, 0.5)' : 'rgba(239, 68, 68, 0.5)'}`,
            borderRadius: '8px',
            padding: '12px',
            fontSize: '13px'
          }}>
            <div style={{ color: '#9ca3af', marginBottom: '8px' }}>{data.time || data.date}</div>
            <div style={{ display: 'grid', gridTemplateColumns: 'auto auto', gap: '8px 16px', fontSize: '12px' }}>
              <span style={{ color: '#6b7280' }}>Open:</span>
              <span style={{ color: '#e5e7eb', fontWeight: 600 }}>₹{data.open.toFixed(2)}</span>
              
              <span style={{ color: '#6b7280' }}>High:</span>
              <span style={{ color: '#34d399', fontWeight: 600 }}>₹{data.high.toFixed(2)}</span>
              
              <span style={{ color: '#6b7280' }}>Low:</span>
              <span style={{ color: '#ef4444', fontWeight: 600 }}>₹{data.low.toFixed(2)}</span>
              
              <span style={{ color: '#6b7280' }}>Close:</span>
              <span style={{ color: '#e5e7eb', fontWeight: 600 }}>₹{data.close.toFixed(2)}</span>
              
              <span style={{ color: '#6b7280' }}>Change:</span>
              <span style={{ 
                color: change >= 0 ? '#34d399' : '#ef4444',
                fontWeight: 600 
              }}>
                {change >= 0 ? '+' : ''}₹{change.toFixed(2)} ({changePercent}%)
              </span>
            </div>
          </div>
        );
      }
      
      // Performance/equity tooltip
      const equity = data.equity ?? 0;
      const pnl = data.pnl ?? 0;
      
      return (
        <div style={{
          background: 'rgba(15, 20, 35, 0.95)',
          border: `1px solid ${data.type === 'win' ? 'rgba(52, 211, 153, 0.5)' : data.type === 'loss' ? 'rgba(239, 68, 68, 0.5)' : 'rgba(52, 211, 153, 0.3)'}`,
          borderRadius: '8px',
          padding: '12px',
          fontSize: '13px'
        }}>
          <div style={{ color: '#9ca3af', marginBottom: '6px' }}>{data.date || 'N/A'}</div>
          <div style={{ color: '#34d399', fontWeight: 600, marginBottom: '4px' }}>
            ₹{equity.toLocaleString('en-IN', { minimumFractionDigits: 2 })}
          </div>
          {data.type !== 'start' && (
            <div style={{ 
              color: pnl > 0 ? '#34d399' : '#ef4444',
              fontSize: '12px',
              marginTop: '4px'
            }}>
              P&L: {pnl > 0 ? '+' : ''}₹{pnl.toLocaleString('en-IN', { minimumFractionDigits: 2 })}
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
        <h3 className="chart-title">
          {candlestickData && candlestickData.length > 0 ? 'Price Chart (Candlestick)' : 'Performance Chart'}
        </h3>
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
          {candlestickData && candlestickData.length > 0 ? (
            // Candlestick Chart
            <ComposedChart data={filteredData} margin={{ top: 10, right: 30, left: 0, bottom: 0 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="rgba(52, 211, 153, 0.1)" />
              <XAxis 
                dataKey="time" 
                stroke="#6b7280" 
                style={{ fontSize: '12px' }}
                tick={{ fill: '#9ca3af' }}
                tickFormatter={(value) => {
                  const date = new Date(value * 1000);
                  return date.toLocaleDateString('en-IN', { month: 'short', day: 'numeric' });
                }}
              />
              <YAxis 
                stroke="#6b7280" 
                style={{ fontSize: '12px' }}
                tick={{ fill: '#9ca3af' }}
                domain={['dataMin - 10', 'dataMax + 10']}
                tickFormatter={(value) => `₹${value.toFixed(0)}`}
              />
              <Tooltip content={<CustomTooltip />} />
              {/* Render candlesticks */}
              {filteredData.map((entry, index) => {
                const isUpward = entry.close >= entry.open;
                const color = isUpward ? '#34d399' : '#ef4444';
                const bodyTop = Math.max(entry.open, entry.close);
                const bodyBottom = Math.min(entry.open, entry.close);
                const bodyHeight = bodyTop - bodyBottom;
                
                return (
                  <g key={index}>
                    {/* This will be rendered using custom shape */}
                  </g>
                );
              })}
              <Bar 
                dataKey="high"
                fill="transparent"
                shape={(props) => {
                  const { x, width, payload } = props;
                  if (!payload) return null;
                  
                  const isUpward = payload.close >= payload.open;
                  const color = isUpward ? '#34d399' : '#ef4444';
                  const wickX = x + width / 2;
                  
                  // Calculate Y positions - need to scale based on domain
                  const chart = props.yAxis;
                  const domain = chart.domain;
                  const range = chart.range;
                  const scale = (range[0] - range[1]) / (domain[1] - domain[0]);
                  
                  const getY = (value) => range[1] + (domain[1] - value) * scale;
                  
                  const highY = getY(payload.high);
                  const lowY = getY(payload.low);
                  const openY = getY(payload.open);
                  const closeY = getY(payload.close);
                  const bodyTop = Math.min(openY, closeY);
                  const bodyHeight = Math.abs(closeY - openY);
                  
                  return (
                    <g>
                      {/* Wick */}
                      <line
                        x1={wickX}
                        y1={highY}
                        x2={wickX}
                        y2={lowY}
                        stroke={color}
                        strokeWidth={1.5}
                      />
                      {/* Body */}
                      <rect
                        x={x + width * 0.15}
                        y={bodyTop}
                        width={width * 0.7}
                        height={Math.max(1, bodyHeight)}
                        fill={color}
                        stroke={color}
                        strokeWidth={1}
                        opacity={0.9}
                      />
                    </g>
                  );
                }}
              />
            </ComposedChart>
          ) : (
            // Performance/Equity Chart
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
          )}
        </ResponsiveContainer>
      </div>
    </div>
  );
};

export default PerformanceChart;
