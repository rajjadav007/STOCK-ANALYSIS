import React, { useEffect, useRef, useState } from 'react';
import { createChart } from 'lightweight-charts';

export const CandlestickChart = ({ data, annotations }) => {
  const chartContainerRef = useRef();
  const chartRef = useRef();
  const candleSeriesRef = useRef();
  const [selectedTimeframe, setSelectedTimeframe] = useState('ALL');

  useEffect(() => {
    if (!chartContainerRef.current || !data || data.length === 0) return;

    // Create chart
    const chart = createChart(chartContainerRef.current, {
      width: chartContainerRef.current.clientWidth,
      height: 500,
      layout: {
        background: { color: '#0f1423' },
        textColor: '#9ca3af',
      },
      grid: {
        vertLines: { color: 'rgba(52, 211, 153, 0.1)' },
        horzLines: { color: 'rgba(52, 211, 153, 0.1)' },
      },
      crosshair: {
        mode: 1,
      },
      rightPriceScale: {
        borderColor: 'rgba(52, 211, 153, 0.3)',
      },
      timeScale: {
        borderColor: 'rgba(52, 211, 153, 0.3)',
        timeVisible: true,
      },
    });

    const candleSeries = chart.addCandlestickSeries({
      upColor: '#34d399',
      downColor: '#ef4444',
      borderUpColor: '#34d399',
      borderDownColor: '#ef4444',
      wickUpColor: '#34d399',
      wickDownColor: '#ef4444',
    });

    // Filter data based on timeframe
    let filteredData = data;
    if (selectedTimeframe !== 'ALL') {
      const counts = {
        '1M': 22,
        '3M': 66,
        '6M': 132,
        '1Y': 252
      };
      const count = counts[selectedTimeframe] || data.length;
      filteredData = data.slice(-count);
    }

    candleSeries.setData(filteredData);

    // Add annotations
    if (annotations) {
      // Support lines
      if (annotations.support) {
        annotations.support.forEach(level => {
          candleSeries.createPriceLine({
            price: level.price,
            color: '#3b82f6',
            lineWidth: 2,
            lineStyle: 2,
            axisLabelVisible: true,
            title: 'Support',
          });
        });
      }

      // Resistance lines
      if (annotations.resistance) {
        annotations.resistance.forEach(level => {
          candleSeries.createPriceLine({
            price: level.price,
            color: '#ef4444',
            lineWidth: 2,
            lineStyle: 2,
            axisLabelVisible: true,
            title: 'Resistance',
          });
        });
      }

      // Stop level
      if (annotations.stopLevel) {
        candleSeries.createPriceLine({
          price: annotations.stopLevel.price,
          color: '#f59e0b',
          lineWidth: 2,
          lineStyle: 0,
          axisLabelVisible: true,
          title: 'Stop Level',
        });
      }

      // Entry level
      if (annotations.entry) {
        candleSeries.createPriceLine({
          price: annotations.entry.price,
          color: '#8b5cf6',
          lineWidth: 2,
          lineStyle: 0,
          axisLabelVisible: true,
          title: 'Entry',
        });
      }

      // Add markers for swing tops and breakouts
      const markers = [];
      
      if (annotations.swingTops) {
        annotations.swingTops.forEach((top, index) => {
          markers.push({
            time: top.time,
            position: 'aboveBar',
            color: '#fbbf24',
            shape: 'arrowDown',
            text: `Top ${index + 1}`,
          });
        });
      }

      if (annotations.breakout) {
        markers.push({
          time: annotations.breakout.time,
          position: 'belowBar',
          color: '#34d399',
          shape: 'arrowUp',
          text: 'Breakout',
        });
      }

      if (markers.length > 0) {
        candleSeries.setMarkers(markers);
      }
    }

    chart.timeScale().fitContent();

    chartRef.current = chart;
    candleSeriesRef.current = candleSeries;

    // Handle resize
    const handleResize = () => {
      if (chartContainerRef.current) {
        chart.applyOptions({
          width: chartContainerRef.current.clientWidth,
        });
      }
    };

    window.addEventListener('resize', handleResize);

    return () => {
      window.removeEventListener('resize', handleResize);
      chart.remove();
    };
  }, [data, annotations, selectedTimeframe]);

  return (
    <div className="chart-section">
      <div className="chart-header">
        <h3 className="chart-title">ML Trading Chart</h3>
        <div className="time-filter">
          {['ALL', '1Y', '6M', '3M', '1M'].map((filter) => (
            <button
              key={filter}
              className={selectedTimeframe === filter ? 'active' : ''}
              onClick={() => setSelectedTimeframe(filter)}
            >
              {filter}
            </button>
          ))}
        </div>
      </div>
      <div ref={chartContainerRef} style={{ position: 'relative', width: '100%', height: '500px' }} />
      
      {annotations && (
        <div style={{ 
          marginTop: '20px', 
          padding: '15px', 
          background: 'rgba(15, 20, 35, 0.5)', 
          borderRadius: '8px',
          display: 'flex',
          flexWrap: 'wrap',
          gap: '15px'
        }}>
          {annotations.trend && (
            <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
              <span style={{ color: '#9ca3af', fontSize: '14px' }}>Trend:</span>
              <span style={{ 
                color: annotations.trend === 'Uptrend' ? '#34d399' : '#ef4444',
                fontWeight: 600,
                fontSize: '14px'
              }}>
                {annotations.trend}
              </span>
            </div>
          )}
          <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
            <div style={{ width: '20px', height: '2px', background: '#3b82f6', borderStyle: 'dashed' }}></div>
            <span style={{ color: '#9ca3af', fontSize: '13px' }}>Support</span>
          </div>
          <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
            <div style={{ width: '20px', height: '2px', background: '#ef4444', borderStyle: 'dashed' }}></div>
            <span style={{ color: '#9ca3af', fontSize: '13px' }}>Resistance</span>
          </div>
          <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
            <div style={{ width: '20px', height: '2px', background: '#f59e0b' }}></div>
            <span style={{ color: '#9ca3af', fontSize: '13px' }}>Stop Level</span>
          </div>
          <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
            <div style={{ width: '20px', height: '2px', background: '#8b5cf6' }}></div>
            <span style={{ color: '#9ca3af', fontSize: '13px' }}>Entry</span>
          </div>
        </div>
      )}
    </div>
  );
};

export default CandlestickChart;
