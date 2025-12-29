import React, { useState } from 'react';
import Header from './components/Header';
import MetricsGrid from './components/MetricsGrid';
import TabNavigation from './components/TabNavigation';
import PerformanceChart from './components/PerformanceChart';
import DayStats from './components/DayStats';
import DayAnalysisTable from './components/DayAnalysisTable';
import MonthStats from './components/MonthStats';
import MonthAnalysisTable from './components/MonthAnalysisTable';
import YearStats from './components/YearStats';
import YearAnalysisTable from './components/YearAnalysisTable';
import TradeStats from './components/TradeStats';
import TradeAnalysisTable from './components/TradeAnalysisTable';
import DrawdownChart from './components/DrawdownChart';
import './styles/Dashboard.css';

// Sample data - replace with your ML model outputs
const sampleData = {
  strategy: {
    name: "Moving Average Cross Over- Multi Leg",
    backtested: "on 27 Jun 2025 11:06",
    period: "29 May 25 to 24 Jun 25",
    created: "24 days ago"
  },
  summary: {
    metrics: [
      { label: "Symbol", value: "Nifty 50", type: "text" },
      { label: "Capital", value: 55293.75, type: "currency" },
      { label: "Profit/Loss", value: 15140.00, type: "currency" },
      { label: "ROI", value: 384.39, type: "percentage" },
      { label: "Drawdown", value: 5293.75, type: "text", sublabel: "(9.57%)" },
      { label: "Risk Profile", value: "Conservative", type: "text" },
      { label: "Recovery Ratio", value: "Fast", type: "text" },
      { label: "Type", value: "Intraday", type: "text" }
    ]
  },
  performanceData: [
    { date: "May 30", equity: 55293.75 },
    { date: "Jun 2", equity: 58564.35 },
    { date: "Jun 3", equity: 62745.60 },
    { date: "Jun 4", equity: 66285.60 },
    { date: "Jun 9", equity: 69826.60 },
    { date: "Jun 10", equity: 64782.85 },
    { date: "Jun 12", equity: 68033.85 },
    { date: "Jun 17", equity: 71534.85 },
    { date: "Jun 18", equity: 71284.85 },
    { date: "Jun 23", equity: 72743.60 },
    { date: "Jun 24", equity: 70433.75 }
  ],
  dayAnalysis: {
    stats: [
      { label: "Trading Days", value: 13, type: "number" },
      { label: "Positive Days", value: "7 (53.85%)", type: "text", className: "positive" },
      { label: "Negative Days", value: "6 (46.15%)", type: "text", className: "negative" },
      { label: "Day Average profit", value: 1164.62, type: "currency" },
      { label: "Day ROI", value: 2.11, type: "percentage" },
      { label: "Day Max Profit", value: 4953.75, type: "currency", className: "positive" },
      { label: "Day Max Loss", value: -5043.75, type: "currency", className: "negative" },
      { label: "Day Min Profit", value: 78.75, type: "currency" },
      { label: "Consecutive Positive Days", value: 4, type: "number" },
      { label: "Consecutive Negative Days", value: 2, type: "number" },
      { label: "Day Average Trades", value: 3, type: "number" }
    ],
    profitByDay: {
      "Mon Profit": 5679.00,
      "Tue Profit": -1613.00,
      "Wed Profit": -750.00,
      "Thu Profit": 3330.00,
      "Fri Profit": 8494.00,
      "Sat Profit": 0,
      "Sun Profit": 0
    },
    tableData: [
      { date: "24-06-2025", trades: 3, targets: 0, stopLoss: 3, cover: 0, buyTrades: 3, sellTrades: 0, qty: 225, profitLoss: -250.00 },
      { date: "23-06-2025", trades: 3, targets: 0, stopLoss: 3, cover: 0, buyTrades: 3, sellTrades: 0, qty: 225, profitLoss: 2458.75 },
      { date: "18-06-2025", trades: 3, targets: 0, stopLoss: 3, cover: 0, buyTrades: 3, sellTrades: 0, qty: 225, profitLoss: -250.00 },
      { date: "17-06-2025", trades: 3, targets: 0, stopLoss: 3, cover: 0, buyTrades: 3, sellTrades: 0, qty: 225, profitLoss: -500.00 },
      { date: "12-06-2025", trades: 3, targets: 0, stopLoss: 3, cover: 0, buyTrades: 3, sellTrades: 0, qty: 225, profitLoss: 3251.25 },
      { date: "10-06-2025", trades: 3, targets: 0, stopLoss: 0, cover: 3, buyTrades: 3, sellTrades: 0, qty: 225, profitLoss: -5043.75 },
      { date: "09-06-2025", trades: 3, targets: 0, stopLoss: 3, cover: 0, buyTrades: 3, sellTrades: 0, qty: 225, profitLoss: -250.00 },
      { date: "04-06-2025", trades: 3, targets: 0, stopLoss: 3, cover: 0, buyTrades: 3, sellTrades: 0, qty: 225, profitLoss: 3540.00 },
      { date: "04-06-2025", trades: 3, targets: 0, stopLoss: 3, cover: 0, buyTrades: 3, sellTrades: 0, qty: 225, profitLoss: -500.00 },
      { date: "03-06-2025", trades: 3, targets: 0, stopLoss: 3, cover: 0, buyTrades: 3, sellTrades: 0, qty: 225, profitLoss: 4181.25 },
      { date: "02-06-2025", trades: 3, targets: 0, stopLoss: 3, cover: 0, buyTrades: 3, sellTrades: 0, qty: 225, profitLoss: 3270.00 },
      { date: "30-05-2025", trades: 3, targets: 0, stopLoss: 3, cover: 0, buyTrades: 3, sellTrades: 0, qty: 225, profitLoss: 4953.75 }
    ]
  },
  monthAnalysis: {
    stats: [
      { label: "Total Months", value: 2, type: "number" },
      { label: "Positive Months", value: "2 (100%)", type: "text", className: "positive" },
      { label: "Negative Months", value: "0 (0%)", type: "text", className: "negative" },
      { label: "Month Average Profit", value: 7570.00, type: "currency" },
      { label: "Month ROI", value: 31.59, type: "percentage" },
      { label: "Month Max Profit", value: 10107.50, type: "currency", className: "positive" },
      { label: "Month Max Loss", value: -6032.50, type: "currency", className: "negative" },
      { label: "Month Min Profit", value: 5032.50, type: "currency" },
      { label: "Month Average Trades", value: 19, type: "number" }
    ],
    tableData: [
      { month: "Jun - 2025", trades: 33, targets: 0, stopLoss: 30, cover: 3, buyTrades: 33, sellTrades: 0, qty: 2475, roi: 18.28, profitLoss: 10107.50 },
      { month: "May - 2025", trades: 6, targets: 0, stopLoss: 6, cover: 0, buyTrades: 6, sellTrades: 0, qty: 450, roi: 9.1, profitLoss: 5032.50 },
      { month: "Total", trades: 39, targets: 0, stopLoss: 36, cover: 3, buyTrades: 39, sellTrades: 0, qty: 2925, roi: 0, profitLoss: 15140.00 }
    ]
  },
  yearAnalysis: {
    stats: [
      { label: "Total Years", value: 1, type: "number" },
      { label: "Positive Years", value: "1 (100%)", type: "text", className: "positive" },
      { label: "Negative Years", value: "0 (0%)", type: "text", className: "negative" },
      { label: "Year Average Profit", value: 15140.00, type: "currency" },
      { label: "Year ROI", value: 384.39, type: "percentage" },
      { label: "Year Max Profit", value: 15140.00, type: "currency", className: "positive" },
      { label: "Year Max Loss", value: 15140.00, type: "currency", className: "negative" },
      { label: "Year Min Profit", value: 15140.00, type: "currency" },
      { label: "Year Average Trades", value: 39, type: "number" }
    ],
    tableData: [
      { year: "2025", trades: 39, targets: 0, stopLoss: 36, cover: 3, buyTrades: 39, sellTrades: 0, qty: 2925, roi: 27.38, profitLoss: 15140.00 },
      { year: "Total", trades: 39, targets: 0, stopLoss: 36, cover: 3, buyTrades: 39, sellTrades: 0, qty: 2925, roi: 0, profitLoss: 15140.00 }
    ]
  },
  tradeAnalysis: {
    stats: [
      { label: "Total Trades", value: 39, type: "number" },
      { label: "Positive Trades", value: "23 (58.97%)", type: "text", className: "positive" },
      { label: "Negative Trades", value: "16 (41.03%)", type: "text", className: "negative" },
      { label: "Cover Trades", value: "39 (100%)", type: "text" },
      { label: "Target Trades", value: "0 (0%)", type: "text" },
      { label: "Stop Loss Trades", value: "0 (0%)", type: "text" },
      { label: "Consecutive Target Trades", value: 0, type: "number" },
      { label: "Consecutive Stop Loss Trades", value: 3, type: "number" },
      { label: "No of Master Target", value: 0, type: "number" },
      { label: "No of Master SL", value: 6, type: "number" },
      { label: "BUY Trades", value: 39, type: "number" },
      { label: "SELL Trades", value: 0, type: "number" }
    ],
    tableData: [],
    isEmpty: true
  },
  drawdownAnalysis: {
    drawdownInfo: {
      drawdown: "5293.75 (9.57%)",
      downStartDate: "09-06-2025",
      maxDownDate: "10-06-2025",
      recoveryDate: "10-06-2025",
      recoveryPeriod: "1 Day(s)"
    },
    chartData: [
      { date: "May 30", drawdown: 0 },
      { date: "Jun 2", drawdown: 0 },
      { date: "Jun 3", drawdown: 0 },
      { date: "Jun 4", drawdown: 0 },
      { date: "Jun 9", drawdown: 0 },
      { date: "Jun 10", drawdown: -7.22 },
      { date: "Jun 12", drawdown: -2.68 },
      { date: "Jun 17", drawdown: 0 },
      { date: "Jun 18", drawdown: -0.35 },
      { date: "Jun 23", drawdown: 0 },
      { date: "Jun 24", drawdown: -3.17 }
    ]
  }
};

function App() {
  const [activeTab, setActiveTab] = useState('summary');
  const [timeFilter, setTimeFilter] = useState('All');

  const tabs = [
    { id: 'summary', label: 'Summary' },
    { id: 'day', label: 'Day Analysis' },
    { id: 'month', label: 'Month Analysis' },
    { id: 'year', label: 'Year Analysis' },
    { id: 'trade', label: 'Trade Analysis' },
    { id: 'drawdown', label: 'Drawdown Analysis' },
    { id: 'parameters', label: 'Parameters' }
  ];

  const renderTabContent = () => {
    switch (activeTab) {
      case 'summary':
        return (
          <>
            <PerformanceChart 
              data={sampleData.performanceData}
              timeFilter={timeFilter}
              onTimeFilterChange={setTimeFilter}
            />
            <div className="disclaimer">
              <div className="disclaimer-title">Disclaimer</div>
              <div className="disclaimer-text">
                The past performance results presented herein are intended solely for illustrative purposes and are based on simulations, 
                which inherently carry limitations. It is crucial to understand that no representation is made that any trading account will 
                achieve similar profits or losses. Real trading outcomes often vary significantly from simulation results due to the dynamic 
                nature of markets and the presence of actual financial risk in simulated trading scenarios. Simulated results, being retrospective, 
                do not fully account for the complexities of actual financial risk or real-time market conditions. Various factors, including 
                but not limited to the clarity in message losses or others is a specific trading strategy, can significantly impact actual trading results, 
                leading them to differ from simulated outcomes.
              </div>
            </div>
          </>
        );
      case 'day':
        return (
          <>
            <DayStats stats={sampleData.dayAnalysis.stats} />
            <DayAnalysisTable 
              data={sampleData.dayAnalysis.tableData}
              profitByDay={sampleData.dayAnalysis.profitByDay}
            />
          </>
        );
      case 'month':
        return (
          <>
            <MonthStats stats={sampleData.monthAnalysis.stats} />
            <MonthAnalysisTable data={sampleData.monthAnalysis.tableData} />
          </>
        );
      case 'year':
        return (
          <>
            <YearStats stats={sampleData.yearAnalysis.stats} />
            <YearAnalysisTable data={sampleData.yearAnalysis.tableData} />
          </>
        );
      case 'trade':
        return (
          <>
            <TradeStats stats={sampleData.tradeAnalysis.stats} />
            <TradeAnalysisTable 
              data={sampleData.tradeAnalysis.tableData}
              isEmpty={sampleData.tradeAnalysis.isEmpty}
            />
          </>
        );
      case 'drawdown':
        return (
          <DrawdownChart 
            data={sampleData.drawdownAnalysis.chartData}
            drawdownInfo={sampleData.drawdownAnalysis.drawdownInfo}
            timeFilter={timeFilter}
            onTimeFilterChange={setTimeFilter}
          />
        );
      case 'parameters':
        return <div style={{ padding: '40px', textAlign: 'center', color: '#9ca3af' }}>Parameters - Coming Soon</div>;
      default:
        return null;
    }
  };

  return (
    <div className="dashboard-container">
      <div className="dashboard-card">
        <Header 
          strategyName={sampleData.strategy.name}
          backtested={sampleData.strategy.backtested}
          period={sampleData.strategy.period}
          created={sampleData.strategy.created}
        />
        <TabNavigation 
          tabs={tabs}
          activeTab={activeTab}
          onTabChange={setActiveTab}
        />
        {activeTab === 'summary' && (
          <MetricsGrid metrics={sampleData.summary.metrics} />
        )}
        <div className="dashboard-content">
          {renderTabContent()}
        </div>
      </div>
    </div>
  );
}

export default App;
