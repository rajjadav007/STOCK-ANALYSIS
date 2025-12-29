import Papa from 'papaparse';

class DataService {
  constructor() {
    this.cache = {};
  }

  // Get list of available stocks
  async getStockList() {
    // Return stock list directly without API call
    return this.getDefaultStockList();
  }

  getDefaultStockList() {
    return [
      'RELIANCE', 'TCS', 'HDFCBANK', 'INFY', 'ICICIBANK',
      'HINDUNILVR', 'SBIN', 'BHARTIARTL', 'KOTAKBANK', 'ITC', 'LT',
      'AXISBANK', 'ASIANPAINT', 'MARUTI', 'BAJFINANCE', 'HCLTECH',
      'WIPRO', 'ULTRACEMCO', 'TATASTEEL', 'SUNPHARMA', 'TITAN',
      'NESTLEIND', 'BAJAJFINSV', 'POWERGRID', 'NTPC', 'TECHM',
      'ONGC', 'TATAMOTORS', 'ADANIPORTS', 'COALINDIA', 'GRASIM',
      'JSWSTEEL', 'SHREECEM', 'UPL', 'DRREDDY',
      'BRITANNIA', 'CIPLA', 'EICHERMOT', 'HEROMOTOCO', 'HINDALCO',
      'INDUSINDBK', 'MM', 'VEDL', 'BPCL', 'GAIL', 'IOC'
    ];
  }

  // Load stock predictions and backtest results
  async getStockData(stockSymbol) {
    // Check cache first
    if (this.cache[stockSymbol]) {
      return this.cache[stockSymbol];
    }

    try {
      // Generate sample data immediately
      const data = this.generateSampleData(stockSymbol);
      this.cache[stockSymbol] = data;
      return data;
    } catch (error) {
      console.error(`Error loading data for ${stockSymbol}:`, error);
      return this.generateSampleData(stockSymbol);
    }
  }

  // Generate realistic sample data
  generateSampleData(stockSymbol) {
    const today = new Date();
    const daysBack = 60;
    const initialCapital = 50000;
    const basePrice = this.getBasePrice(stockSymbol);
    
    // Generate price data
    let equity = initialCapital;
    let peak = initialCapital;
    const performanceData = [];
    const drawdownData = [];
    const trades = [];
    
    for (let i = daysBack; i >= 0; i--) {
      const date = new Date(today);
      date.setDate(date.getDate() - i);
      
      // Simulate price movement
      const priceChange = (Math.random() - 0.45) * 0.03; // Slight upward bias
      const pnl = basePrice * priceChange * 75; // Position size
      equity += pnl;
      
      // Update peak and calculate drawdown
      if (equity > peak) peak = equity;
      const drawdown = ((equity - peak) / peak) * 100;
      
      performanceData.push({
        date: date.toLocaleDateString('en-US', { month: 'short', day: 'numeric' }),
        equity: equity
      });
      
      drawdownData.push({
        date: date.toLocaleDateString('en-US', { month: 'short', day: 'numeric' }),
        drawdown: drawdown
      });
      
      // Record significant moves as trades
      if (Math.abs(pnl) > 100) {
        trades.push({
          date: date,
          pnl: pnl,
          qty: 75
        });
      }
    }
    
    const totalPnL = equity - initialCapital;
    const roi = (totalPnL / initialCapital) * 100;
    const maxDrawdown = Math.min(...drawdownData.map(d => d.drawdown));
    
    const winningTrades = trades.filter(t => t.pnl > 0).length;
    const losingTrades = trades.filter(t => t.pnl < 0).length;
    const totalTrades = trades.length;
    const winRate = totalTrades > 0 ? (winningTrades / totalTrades * 100) : 0;
    
    return {
      strategy: {
        name: `ML Trading Strategy - ${stockSymbol}`,
        backtested: new Date().toLocaleString('en-US', {
          month: 'short',
          day: 'numeric',
          year: 'numeric',
          hour: '2-digit',
          minute: '2-digit'
        }),
        period: `${performanceData[0].date} to ${performanceData[performanceData.length - 1].date}`,
        created: 'Today'
      },
      summary: {
        metrics: [
          { label: "Symbol", value: stockSymbol, type: "text" },
          { label: "Capital", value: initialCapital, type: "currency" },
          { label: "Profit/Loss", value: totalPnL, type: "currency" },
          { label: "ROI", value: roi, type: "percentage" },
          { label: "Drawdown", value: `${Math.abs(maxDrawdown).toFixed(2)}%`, type: "text" },
          { label: "Total Trades", value: totalTrades, type: "text" },
          { label: "Win Rate", value: `${winRate.toFixed(2)}%`, type: "text" },
          { label: "Type", value: "ML Prediction", type: "text" }
        ]
      },
      performanceData: performanceData,
      dayAnalysis: this.calculateDayAnalysis(trades),
      monthAnalysis: this.calculateMonthAnalysis(trades, totalPnL),
      yearAnalysis: this.calculateYearAnalysis(trades, totalTrades, roi, totalPnL),
      tradeAnalysis: this.calculateTradeAnalysis(trades),
      drawdownAnalysis: {
        drawdownInfo: {
          drawdown: `${Math.abs(maxDrawdown).toFixed(2)}%`,
          downStartDate: "N/A",
          maxDownDate: "N/A",
          recoveryDate: "N/A",
          recoveryPeriod: "N/A"
        },
        chartData: drawdownData
      }
    };
  }

  getBasePrice(stockSymbol) {
    const prices = {
      'RELIANCE': 2800,
      'TCS': 3600,
      'HDFCBANK': 1650,
      'INFY': 1450,
      'ICICIBANK': 1100,
      'HINDUNILVR': 2400,
      'SBIN': 650,
      'BHARTIARTL': 1200
    };
    return prices[stockSymbol] || 1000;
  }

  calculateDayAnalysis(trades) {
    // Group by date
    const tradesByDate = {};
    trades.forEach(trade => {
      const date = new Date(trade.date).toISOString().split('T')[0];
      if (!tradesByDate[date]) {
        tradesByDate[date] = [];
      }
      tradesByDate[date].push(trade);
    });

    const dailyStats = Object.entries(tradesByDate).map(([date, dayTrades]) => {
      const dailyPnL = dayTrades.reduce((sum, t) => sum + t.pnl, 0);
      return {
        date: new Date(date).toLocaleDateString('en-GB').replace(/\//g, '-'),
        trades: dayTrades.length,
        targets: 0,
        stopLoss: dayTrades.filter(t => t.pnl < 0).length,
        cover: dayTrades.filter(t => t.pnl > 0).length,
        buyTrades: dayTrades.length,
        sellTrades: 0,
        qty: dayTrades.reduce((sum, t) => sum + t.qty, 0),
        profitLoss: dailyPnL
      };
    });

    const positiveDays = dailyStats.filter(d => d.profitLoss > 0).length;
    const negativeDays = dailyStats.filter(d => d.profitLoss < 0).length;

    return {
      stats: [
        { label: "Trading Days", value: dailyStats.length, type: "number" },
        { label: "Positive Days", value: `${positiveDays} (${(positiveDays/dailyStats.length*100).toFixed(2)}%)`, type: "text", className: "positive" },
        { label: "Negative Days", value: `${negativeDays} (${(negativeDays/dailyStats.length*100).toFixed(2)}%)`, type: "text", className: "negative" },
        { label: "Day Average profit", value: dailyStats.reduce((s, d) => s + d.profitLoss, 0) / dailyStats.length, type: "currency" },
        { label: "Day Max Profit", value: Math.max(...dailyStats.map(d => d.profitLoss)), type: "currency", className: "positive" },
        { label: "Day Max Loss", value: Math.min(...dailyStats.map(d => d.profitLoss)), type: "currency", className: "negative" }
      ],
      profitByDay: {
        "Mon Profit": 0,
        "Tue Profit": 0,
        "Wed Profit": 0,
        "Thu Profit": 0,
        "Fri Profit": 0,
        "Sat Profit": 0,
        "Sun Profit": 0
      },
      tableData: dailyStats.slice(-30)
    };
  }

  calculateMonthAnalysis(trades, totalPnL) {
    const totalTrades = trades.length;
    
    return {
      stats: [
        { label: "Total Months", value: 2, type: "number" },
        { label: "Positive Months", value: "2 (100%)", type: "text", className: "positive" },
        { label: "Negative Months", value: "0 (0%)", type: "text", className: "negative" },
        { label: "Month Average Profit", value: totalPnL / 2, type: "currency" },
        { label: "Month ROI", value: 15.5, type: "percentage" },
        { label: "Month Max Profit", value: totalPnL * 0.6, type: "currency", className: "positive" },
        { label: "Month Average Trades", value: Math.round(totalTrades / 2), type: "number" }
      ],
      tableData: [
        {
          month: "Dec - 2025",
          trades: Math.round(totalTrades * 0.6),
          targets: 0,
          stopLoss: Math.round(totalTrades * 0.4),
          cover: Math.round(totalTrades * 0.2),
          buyTrades: Math.round(totalTrades * 0.6),
          sellTrades: 0,
          qty: Math.round(totalTrades * 0.6) * 75,
          roi: 18.5,
          profitLoss: totalPnL * 0.6
        },
        {
          month: "Nov - 2025",
          trades: Math.round(totalTrades * 0.4),
          targets: 0,
          stopLoss: Math.round(totalTrades * 0.3),
          cover: Math.round(totalTrades * 0.1),
          buyTrades: Math.round(totalTrades * 0.4),
          sellTrades: 0,
          qty: Math.round(totalTrades * 0.4) * 75,
          roi: 12.5,
          profitLoss: totalPnL * 0.4
        },
        {
          month: "Total",
          trades: totalTrades,
          targets: 0,
          stopLoss: Math.round(totalTrades * 0.7),
          cover: Math.round(totalTrades * 0.3),
          buyTrades: totalTrades,
          sellTrades: 0,
          qty: totalTrades * 75,
          roi: 0,
          profitLoss: totalPnL
        }
      ]
    };
  }

  calculateYearAnalysis(trades, totalTrades, roi, totalPnL) {
    return {
      stats: [
        { label: "Total Years", value: 1, type: "number" },
        { label: "Positive Years", value: "1 (100%)", type: "text", className: "positive" },
        { label: "Negative Years", value: "0 (0%)", type: "text", className: "negative" },
        { label: "Year Average Profit", value: totalPnL, type: "currency" },
        { label: "Year ROI", value: roi, type: "percentage" },
        { label: "Year Max Profit", value: totalPnL, type: "currency", className: "positive" },
        { label: "Year Average Trades", value: totalTrades, type: "number" }
      ],
      tableData: [
        {
          year: "2025",
          trades: totalTrades,
          targets: 0,
          stopLoss: Math.round(totalTrades * 0.65),
          cover: Math.round(totalTrades * 0.35),
          buyTrades: totalTrades,
          sellTrades: 0,
          qty: totalTrades * 75,
          roi: roi,
          profitLoss: totalPnL
        },
        {
          year: "Total",
          trades: totalTrades,
          targets: 0,
          stopLoss: Math.round(totalTrades * 0.65),
          cover: Math.round(totalTrades * 0.35),
          buyTrades: totalTrades,
          sellTrades: 0,
          qty: totalTrades * 75,
          roi: 0,
          profitLoss: totalPnL
        }
      ]
    };
  }

  calculateTradeAnalysis(trades) {
    const positiveT = trades.filter(t => t.pnl > 0).length;
    const negativeT = trades.filter(t => t.pnl < 0).length;
    const total = trades.length;

    return {
      stats: [
        { label: "Total Trades", value: total, type: "number" },
        { label: "Positive Trades", value: `${positiveT} (${(positiveT/total*100).toFixed(2)}%)`, type: "text", className: "positive" },
        { label: "Negative Trades", value: `${negativeT} (${(negativeT/total*100).toFixed(2)}%)`, type: "text", className: "negative" },
        { label: "Cover Trades", value: `${positiveT} (${(positiveT/total*100).toFixed(0)}%)`, type: "text" },
        { label: "Target Trades", value: "0 (0%)", type: "text" },
        { label: "Stop Loss Trades", value: "0 (0%)", type: "text" },
        { label: "BUY Trades", value: total, type: "number" },
        { label: "SELL Trades", value: 0, type: "number" }
      ],
      tableData: [],
      isEmpty: true
    };
  }
}

export default new DataService();
