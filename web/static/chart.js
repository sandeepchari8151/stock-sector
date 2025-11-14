// Real-Time Chart & Pattern Analysis for Swing Trading
class ChartPatternAnalyzer {
  constructor() {
    this.sector = window.__SECTOR__ || '';
    this.symbol = window.__SYMBOL__;
    this.chart = null;
    this.currentData = null;
    this.patterns = [];
    this.indicators = {
      ema20: true,
      ema50: true,
      rsi: true,
      macd: true,
      bollinger: true,
      volume: true
    };
    this.showPatterns = true;
    this.liveTimer = null;
    this.symbolTimeout = null;
    this.init();
  }

  init() {
    this.bindEvents();
    this.loadInitialData();
    this.enableTooltips();
  }

  bindEvents() {
    console.log('Binding events for Chart Pattern Analyzer');
    
    document.getElementById('analyzeBtn')?.addEventListener('click', () => this.analyzePatterns());
    document.getElementById('togglePatterns')?.addEventListener('click', () => this.togglePatterns());
    document.getElementById('liveToggle')?.addEventListener('click', () => this.toggleLive());
    
    // Add event listeners for filter changes
    const symbolInput = document.getElementById('symbolInput');
    const periodSelect = document.getElementById('periodSelect');
    const timeframeSelect = document.getElementById('timeframeSelect');
    
    console.log('Filter elements found:', {
      symbolInput: !!symbolInput,
      periodSelect: !!periodSelect,
      timeframeSelect: !!timeframeSelect
    });
    
    symbolInput?.addEventListener('change', () => {
      console.log('Symbol changed to:', symbolInput.value);
      this.loadInitialData();
    });
    periodSelect?.addEventListener('change', () => {
      console.log('Period changed to:', periodSelect.value);
      this.loadInitialData();
    });
    timeframeSelect?.addEventListener('change', () => {
      console.log('Timeframe changed to:', timeframeSelect.value);
      this.loadInitialData();
    });
    
    // Add input event for symbol input (real-time updates)
    symbolInput?.addEventListener('input', () => {
      clearTimeout(this.symbolTimeout);
      this.symbolTimeout = setTimeout(() => {
        console.log('Symbol input timeout triggered:', symbolInput.value);
        this.loadInitialData();
      }, 1000);
    });

    // Add event listeners for individual indicator filters
    this.bindIndicatorFilters();
  }

  bindIndicatorFilters() {
    // Individual indicator checkboxes
    const indicatorFilters = {
      'filterEMA20': 'ema20',
      'filterEMA50': 'ema50',
      'filterRSI': 'rsi',
      'filterMACD': 'macd',
      'filterBollinger': 'bollinger',
      'filterVolume': 'volume'
    };

    Object.entries(indicatorFilters).forEach(([filterId, indicatorKey]) => {
      const checkbox = document.getElementById(filterId);
      if (checkbox) {
        checkbox.addEventListener('change', () => {
          this.indicators[indicatorKey] = checkbox.checked;
          console.log(`${indicatorKey} filter changed to:`, checkbox.checked);
          this.renderChart();
        });
      }
    });

    // Select all indicators
    document.getElementById('selectAllIndicators')?.addEventListener('click', () => {
      Object.values(indicatorFilters).forEach(key => {
        this.indicators[key] = true;
      });
      Object.keys(indicatorFilters).forEach(filterId => {
        const checkbox = document.getElementById(filterId);
        if (checkbox) checkbox.checked = true;
      });
      this.renderChart();
    });

    // Deselect all indicators
    document.getElementById('deselectAllIndicators')?.addEventListener('click', () => {
      Object.values(indicatorFilters).forEach(key => {
        this.indicators[key] = false;
      });
      Object.keys(indicatorFilters).forEach(filterId => {
        const checkbox = document.getElementById(filterId);
        if (checkbox) checkbox.checked = false;
      });
      this.renderChart();
    });
  }

  async loadInitialData() {
    const symbolInput = document.getElementById('symbolInput');
    const periodSelect = document.getElementById('periodSelect');
    const timeframeSelect = document.getElementById('timeframeSelect');
    
    const symbol = symbolInput?.value || this.symbol;
    let period = periodSelect?.value || '3mo';
    const interval = timeframeSelect?.value || '1d';

    // Clamp period for intraday intervals due to yfinance limits
    const intraday = ['1m','2m','5m','15m','30m','90m','1h'].includes(interval);
    if (intraday) {
      // yfinance supports up to 30 days for <=1h intervals
      const allowed = new Set(['1d','5d','7d','30d']);
      if (!allowed.has(period)) period = '7d';
    }
    
    if (!symbol) {
      console.warn('No symbol provided');
      return;
    }
    
    try {
      this.showLoading();
      const { data } = await axios.get('/api/candles', {
        params: { symbols: symbol, period, interval }
      });
      
      if (data.series && data.series.length > 0) {
        this.currentData = data.series[0];
        this.symbol = symbol;
        this.renderChart();
        this.updateKeyStats();
        this.analyzePatterns();
      } else {
        this.showError('No data available for this symbol');
      }
    } catch (error) {
      console.error('Error loading data:', error);
      this.showError(`Failed to load market data: ${error.message}`);
    }
  }

  async analyzePatterns() {
    if (!this.currentData) return;

    this.showLoading();
    
    try {
      // Simulate pattern analysis (in real implementation, this would be server-side)
      const patterns = this.detectPatterns(this.currentData);
      const signals = this.generateSignals(patterns);
      const riskMetrics = this.calculateRiskMetrics(this.currentData);
      
      this.displayPatternAnalysis(patterns);
      this.displayTradingSignals(signals);
      this.displayRiskAnalysis(riskMetrics);
      
    } catch (error) {
      console.error('Error analyzing patterns:', error);
      this.showError('Pattern analysis failed');
    }
  }

  detectPatterns(data) {
    const patterns = [];
    const prices = data.close;
    const highs = data.high;
    const lows = data.low;
    const volumes = data.volume;

    // Calculate indicators for signal detection
    const ema20 = this.calculateEMA(prices, 20);
    const ema50 = this.calculateEMA(prices, 50);
    const rsi = this.calculateRSI(prices, 14);
    const macd = this.calculateMACD(prices, 12, 26, 9);
    const bb = this.calculateBollingerBands(prices, 20, 2);

    // Trend Analysis (EMA 20/50)
    const trendSignal = this.analyzeTrend(ema20, ema50, prices);
    if (trendSignal) patterns.push(trendSignal);

    // RSI Entry Timing (30-70 zones)
    const rsiSignal = this.analyzeRSI(rsi, prices);
    if (rsiSignal) patterns.push(rsiSignal);

    // MACD Momentum Confirmation
    const macdSignal = this.analyzeMACD(macd, prices);
    if (macdSignal) patterns.push(macdSignal);

    // Bollinger Bands Breakout Confirmation
    const bbSignal = this.analyzeBollingerBands(bb, prices, volumes);
    if (bbSignal) patterns.push(bbSignal);

    // Volume Confirmation
    const volumeSignal = this.analyzeVolume(volumes, prices);
    if (volumeSignal) patterns.push(volumeSignal);

    // Traditional Pattern Detection
    const doubleBottom = this.findDoubleBottom(lows, volumes);
    if (doubleBottom) patterns.push({...doubleBottom, type: 'Double Bottom', signal: 'BUY'});

    const doubleTop = this.findDoubleTop(highs, volumes);
    if (doubleTop) patterns.push({...doubleTop, type: 'Double Top', signal: 'SELL'});

    const headShoulders = this.findHeadShoulders(highs, volumes);
    if (headShoulders) patterns.push({...headShoulders, type: 'Head & Shoulders', signal: 'SELL'});

    const bullFlag = this.findBullFlag(prices, volumes);
    if (bullFlag) patterns.push({...bullFlag, type: 'Bull Flag', signal: 'BUY'});

    const triangle = this.findTriangle(highs, lows);
    if (triangle) patterns.push({...triangle, type: 'Triangle', signal: 'NEUTRAL'});

    return patterns;
  }

  analyzeTrend(ema20, ema50, prices) {
    const currentPrice = prices[prices.length - 1];
    const currentEMA20 = ema20[ema20.length - 1];
    const currentEMA50 = ema50[ema50.length - 1];
    
    if (!currentEMA20 || !currentEMA50) return null;

    // Bullish trend: Price > EMA20 > EMA50
    if (currentPrice > currentEMA20 && currentEMA20 > currentEMA50) {
      return {
        type: 'Bullish Trend',
        signal: 'BUY',
        confidence: 0.8,
        entry: currentPrice,
        stopLoss: currentEMA20 * 0.98,
        target: currentPrice * 1.05
      };
    }
    
    // Bearish trend: Price < EMA20 < EMA50
    if (currentPrice < currentEMA20 && currentEMA20 < currentEMA50) {
      return {
        type: 'Bearish Trend',
        signal: 'SELL',
        confidence: 0.8,
        entry: currentPrice,
        stopLoss: currentEMA20 * 1.02,
        target: currentPrice * 0.95
      };
    }
    
    return null;
  }

  analyzeRSI(rsi, prices) {
    const currentRSI = rsi[rsi.length - 1];
    const currentPrice = prices[prices.length - 1];
    
    if (!currentRSI) return null;

    // RSI in 30-70 zone (good entry timing)
    if (currentRSI >= 30 && currentRSI <= 70) {
      return {
        type: 'RSI Entry Zone',
        signal: currentRSI < 50 ? 'BUY' : 'NEUTRAL',
        confidence: 0.6,
        entry: currentPrice,
        stopLoss: currentPrice * (currentRSI < 50 ? 0.97 : 1.03),
        target: currentPrice * (currentRSI < 50 ? 1.04 : 0.96)
      };
    }
    
    // RSI oversold (strong buy signal)
    if (currentRSI < 30) {
      return {
        type: 'RSI Oversold',
        signal: 'BUY',
        confidence: 0.9,
        entry: currentPrice,
        stopLoss: currentPrice * 0.95,
        target: currentPrice * 1.08
      };
    }
    
    // RSI overbought (strong sell signal)
    if (currentRSI > 70) {
      return {
        type: 'RSI Overbought',
        signal: 'SELL',
        confidence: 0.9,
        entry: currentPrice,
        stopLoss: currentPrice * 1.05,
        target: currentPrice * 0.92
      };
    }
    
    return null;
  }

  analyzeMACD(macd, prices) {
    const currentMACD = macd.macd[macd.macd.length - 1];
    const currentSignal = macd.signal[macd.signal.length - 1];
    const currentHistogram = macd.histogram[macd.histogram.length - 1];
    const currentPrice = prices[prices.length - 1];
    
    if (!currentMACD || !currentSignal) return null;

    // MACD bullish crossover
    if (currentMACD > currentSignal && currentHistogram > 0) {
      return {
        type: 'MACD Bullish',
        signal: 'BUY',
        confidence: 0.7,
        entry: currentPrice,
        stopLoss: currentPrice * 0.98,
        target: currentPrice * 1.06
      };
    }
    
    // MACD bearish crossover
    if (currentMACD < currentSignal && currentHistogram < 0) {
      return {
        type: 'MACD Bearish',
        signal: 'SELL',
        confidence: 0.7,
        entry: currentPrice,
        stopLoss: currentPrice * 1.02,
        target: currentPrice * 0.94
      };
    }
    
    return null;
  }

  analyzeBollingerBands(bb, prices, volumes) {
    const currentPrice = prices[prices.length - 1];
    const currentUpper = bb.upper[bb.upper.length - 1];
    const currentLower = bb.lower[bb.lower.length - 1];
    const currentMiddle = bb.middle[bb.middle.length - 1];
    const currentVolume = volumes[volumes.length - 1];
    const avgVolume = volumes.slice(-20).reduce((a, b) => a + b, 0) / 20;
    
    if (!currentUpper || !currentLower || !currentMiddle) return null;

    // Bollinger Band breakout with volume confirmation
    if (currentPrice > currentUpper && currentVolume > avgVolume * 1.5) {
      return {
        type: 'BB Breakout',
        signal: 'BUY',
        confidence: 0.8,
        entry: currentPrice,
        stopLoss: currentMiddle,
        target: currentPrice + (currentPrice - currentMiddle) * 2
      };
    }
    
    // Bollinger Band breakdown with volume confirmation
    if (currentPrice < currentLower && currentVolume > avgVolume * 1.5) {
      return {
        type: 'BB Breakdown',
        signal: 'SELL',
        confidence: 0.8,
        entry: currentPrice,
        stopLoss: currentMiddle,
        target: currentPrice - (currentMiddle - currentPrice) * 2
      };
    }
    
    return null;
  }

  analyzeVolume(volumes, prices) {
    const currentVolume = volumes[volumes.length - 1];
    const avgVolume = volumes.slice(-20).reduce((a, b) => a + b, 0) / 20;
    const currentPrice = prices[prices.length - 1];
    
    // High volume spike
    if (currentVolume > avgVolume * 2) {
      return {
        type: 'Volume Spike',
        signal: 'NEUTRAL',
        confidence: 0.5,
        entry: currentPrice,
        stopLoss: currentPrice * 0.99,
        target: currentPrice * 1.02
      };
    }
    
    return null;
  }

  findDoubleBottom(lows, volumes) {
    // Simplified double bottom detection
    const recentLows = lows.slice(-20);
    const minLow = Math.min(...recentLows);
    const secondMin = Math.min(...recentLows.filter(l => l !== minLow));
    
    if (Math.abs(minLow - secondMin) / minLow < 0.02) { // Within 2%
      return {
        confidence: 0.8,
        entry: Math.max(minLow, secondMin) * 1.02,
        stopLoss: Math.min(minLow, secondMin) * 0.98,
        target: Math.max(minLow, secondMin) * 1.06
      };
    }
    return null;
  }

  findDoubleTop(highs, volumes) {
    // Simplified double top detection
    const recentHighs = highs.slice(-20);
    const maxHigh = Math.max(...recentHighs);
    const secondMax = Math.max(...recentHighs.filter(h => h !== maxHigh));
    
    if (Math.abs(maxHigh - secondMax) / maxHigh < 0.02) { // Within 2%
      return {
        confidence: 0.8,
        entry: Math.min(maxHigh, secondMax) * 0.98,
        stopLoss: Math.max(maxHigh, secondMax) * 1.02,
        target: Math.min(maxHigh, secondMax) * 0.94
      };
    }
    return null;
  }

  findHeadShoulders(highs, volumes) {
    // Simplified H&S detection
    const recentHighs = highs.slice(-30);
    const maxHigh = Math.max(...recentHighs);
    const maxIndex = recentHighs.indexOf(maxHigh);
    
    if (maxIndex > 5 && maxIndex < recentHighs.length - 5) {
      const leftShoulder = Math.max(...recentHighs.slice(0, maxIndex));
      const rightShoulder = Math.max(...recentHighs.slice(maxIndex + 1));
      
      if (leftShoulder > rightShoulder && rightShoulder > maxHigh * 0.95) {
        return {
          confidence: 0.7,
          entry: rightShoulder * 0.98,
          stopLoss: rightShoulder * 1.02,
          target: rightShoulder * 0.92
        };
      }
    }
    return null;
  }

  findBullFlag(prices, volumes) {
    // Simplified bull flag detection
    const recentPrices = prices.slice(-15);
    const recentVolumes = volumes.slice(-15);
    
    const firstHalf = recentPrices.slice(0, 7);
    const secondHalf = recentPrices.slice(7);
    
    const firstHalfAvg = firstHalf.reduce((a, b) => a + b, 0) / firstHalf.length;
    const secondHalfAvg = secondHalf.reduce((a, b) => a + b, 0) / secondHalf.length;
    
    if (firstHalfAvg > secondHalfAvg && secondHalfAvg > firstHalfAvg * 0.95) {
      return {
        confidence: 0.6,
        entry: Math.max(...secondHalf) * 1.01,
        stopLoss: Math.min(...secondHalf) * 0.99,
        target: Math.max(...secondHalf) * 1.05
      };
    }
    return null;
  }

  findTriangle(highs, lows) {
    // Simplified triangle detection
    const recentHighs = highs.slice(-20);
    const recentLows = lows.slice(-20);
    
    const highTrend = this.calculateTrend(recentHighs);
    const lowTrend = this.calculateTrend(recentLows);
    
    if (Math.abs(highTrend) < 0.001 && Math.abs(lowTrend) < 0.001) {
      return {
        confidence: 0.5,
        entry: null,
        stopLoss: null,
        target: null
      };
    }
    return null;
  }

  calculateTrend(data) {
    const n = data.length;
    const x = Array.from({length: n}, (_, i) => i);
    const sumX = x.reduce((a, b) => a + b, 0);
    const sumY = data.reduce((a, b) => a + b, 0);
    const sumXY = x.reduce((sum, xi, i) => sum + xi * data[i], 0);
    const sumXX = x.reduce((sum, xi) => sum + xi * xi, 0);
    
    return (n * sumXY - sumX * sumY) / (n * sumXX - sumX * sumX);
  }

  generateSignals(patterns) {
    const signals = [];
    
    patterns.forEach(pattern => {
      if (pattern.signal === 'BUY') {
        signals.push({
          type: 'BUY',
          pattern: pattern.type,
          confidence: pattern.confidence,
          entry: pattern.entry,
          stopLoss: pattern.stopLoss,
          target: pattern.target,
          riskReward: ((pattern.target - pattern.entry) / (pattern.entry - pattern.stopLoss)).toFixed(2)
        });
      } else if (pattern.signal === 'SELL') {
        signals.push({
          type: 'SELL',
          pattern: pattern.type,
          confidence: pattern.confidence,
          entry: pattern.entry,
          stopLoss: pattern.stopLoss,
          target: pattern.target,
          riskReward: ((pattern.entry - pattern.target) / (pattern.stopLoss - pattern.entry)).toFixed(2)
        });
      }
    });
    
    return signals;
  }

  calculateRiskMetrics(data) {
    const prices = data.close;
    const volumes = data.volume;
    
    // Calculate ATR for volatility
    const atr = this.calculateATR(data.high, data.low, data.close, 14);
    const currentPrice = prices[prices.length - 1];
    
    // Calculate position sizing
    const riskPerTrade = 0.02; // 2%
    const stopDistance = atr * 2;
    const positionSize = (riskPerTrade * 10000) / stopDistance; // Assuming $10k account
    
    return {
      atr: atr.toFixed(2),
      volatility: ((atr / currentPrice) * 100).toFixed(2) + '%',
      positionSize: Math.round(positionSize),
      stopDistance: stopDistance.toFixed(2),
      riskPerTrade: (riskPerTrade * 100).toFixed(1) + '%'
    };
  }

  calculateATR(highs, lows, closes, period) {
    const tr = [];
    for (let i = 1; i < highs.length; i++) {
      tr.push(Math.max(
        highs[i] - lows[i],
        Math.abs(highs[i] - closes[i-1]),
        Math.abs(lows[i] - closes[i-1])
      ));
    }
    
    return tr.slice(-period).reduce((a, b) => a + b, 0) / period;
  }

  renderChart() {
    if (!this.currentData) return;

    // Calculate dynamic chart height based on active indicators
    const activeIndicators = this.getActiveIndicators();
    const chartHeight = this.calculateChartHeight(activeIndicators);
    
    // Update chart container height
    const chartContainer = document.getElementById('mainChart');
    if (chartContainer) {
      chartContainer.style.height = `${chartHeight}px`;
    }

    const traces = [{
      type: 'candlestick',
      x: this.currentData.dates,
      open: this.currentData.open,
      high: this.currentData.high,
      low: this.currentData.low,
      close: this.currentData.close,
      name: 'Price'
    }];

    // Add EMA 20 (Trend)
    if (this.indicators.ema20) {
      const ema20 = this.calculateEMA(this.currentData.close, 20);
      traces.push({
        type: 'scatter',
        x: this.currentData.dates,
        y: ema20,
        mode: 'lines',
        name: 'EMA 20',
        line: { color: '#FF6B6B', width: 2 }
      });
    }

    // Add EMA 50 (Trend)
    if (this.indicators.ema50) {
      const ema50 = this.calculateEMA(this.currentData.close, 50);
      traces.push({
        type: 'scatter',
        x: this.currentData.dates,
        y: ema50,
        mode: 'lines',
        name: 'EMA 50',
        line: { color: '#4ECDC4', width: 2 }
      });
    }

    // Add Bollinger Bands (Breakout Confirmation)
    if (this.indicators.bollinger) {
      const bb = this.calculateBollingerBands(this.currentData.close, 20, 2);
      traces.push({
        type: 'scatter',
        x: this.currentData.dates,
        y: bb.upper,
        mode: 'lines',
        name: 'BB Upper',
        line: { color: '#95A5A6', width: 1, dash: 'dash' },
        showlegend: false
      });
      traces.push({
        type: 'scatter',
        x: this.currentData.dates,
        y: bb.middle,
        mode: 'lines',
        name: 'BB Middle',
        line: { color: '#95A5A6', width: 1, dash: 'dot' },
        showlegend: false
      });
      traces.push({
        type: 'scatter',
        x: this.currentData.dates,
        y: bb.lower,
        mode: 'lines',
        name: 'Bollinger Bands',
        line: { color: '#95A5A6', width: 1, dash: 'dash' },
        fill: 'tonexty',
        fillcolor: 'rgba(149, 165, 166, 0.1)'
      });
    }

    // Add volume bars (Breakout Confirmation)
    if (this.indicators.volume && this.currentData.volume && this.currentData.volume.length) {
      traces.push({
        x: this.currentData.dates,
        y: this.currentData.volume,
        type: 'bar',
        name: 'Volume',
        marker: { color: '#a8dadc' },
        opacity: 0.7,
        yaxis: 'y2'
      });
    }

    // Add RSI (Entry Timing) - Secondary subplot
    if (this.indicators.rsi) {
      const rsi = this.calculateRSI(this.currentData.close, 14);
      traces.push({
        type: 'scatter',
        x: this.currentData.dates,
        y: rsi,
        mode: 'lines',
        name: 'RSI 14',
        line: { color: '#9B59B6', width: 2 },
        yaxis: 'y3'
      });
      
      // Add RSI zones
      traces.push({
        type: 'scatter',
        x: this.currentData.dates,
        y: Array(this.currentData.dates.length).fill(70),
        mode: 'lines',
        name: 'RSI Overbought',
        line: { color: '#E74C3C', width: 1, dash: 'dash' },
        yaxis: 'y3',
        showlegend: false
      });
      traces.push({
        type: 'scatter',
        x: this.currentData.dates,
        y: Array(this.currentData.dates.length).fill(30),
        mode: 'lines',
        name: 'RSI Oversold',
        line: { color: '#27AE60', width: 1, dash: 'dash' },
        yaxis: 'y3',
        showlegend: false
      });
    }

    // Add MACD (Momentum Confirmation) - Secondary subplot
    if (this.indicators.macd) {
      const macd = this.calculateMACD(this.currentData.close, 12, 26, 9);
      traces.push({
        type: 'scatter',
        x: this.currentData.dates,
        y: macd.macd,
        mode: 'lines',
        name: 'MACD',
        line: { color: '#F39C12', width: 2 },
        yaxis: 'y4'
      });
      traces.push({
        type: 'scatter',
        x: this.currentData.dates,
        y: macd.signal,
        mode: 'lines',
        name: 'MACD Signal',
        line: { color: '#E67E22', width: 2 },
        yaxis: 'y4'
      });
      traces.push({
        type: 'bar',
        x: this.currentData.dates,
        y: macd.histogram,
        name: 'MACD Histogram',
        marker: { color: macd.histogram.map(h => h >= 0 ? '#27AE60' : '#E74C3C') },
        yaxis: 'y4'
      });
  }

    // Calculate dynamic layout based on active indicators
    const layout = this.calculateDynamicLayout(activeIndicators);

    Plotly.newPlot('mainChart', traces, layout, { responsive: true, displaylogo: false });
  }

  calculateSMA(data, period) {
    const sma = [];
    for (let i = period - 1; i < data.length; i++) {
      const sum = data.slice(i - period + 1, i + 1).reduce((a, b) => a + b, 0);
      sma.push(sum / period);
    }
    return Array(period - 1).fill(null).concat(sma);
  }

  calculateEMA(data, period) {
    const ema = [];
    const multiplier = 2 / (period + 1);
    
    // First EMA value is SMA
  let sum = 0;
    for (let i = 0; i < period; i++) {
      sum += data[i];
    }
    ema[period - 1] = sum / period;
    
    // Calculate EMA for remaining values
    for (let i = period; i < data.length; i++) {
      ema[i] = (data[i] * multiplier) + (ema[i - 1] * (1 - multiplier));
    }
    
    return Array(period - 1).fill(null).concat(ema.slice(period - 1));
  }

  calculateRSI(data, period = 14) {
    const rsi = Array(data.length).fill(null);
    let gains = 0, losses = 0;
    
    for (let i = 1; i < data.length; i++) {
      const change = data[i] - data[i - 1];
      const gain = Math.max(change, 0);
      const loss = Math.max(-change, 0);
      
      if (i <= period) {
        gains += gain;
        losses += loss;
        if (i === period) {
          const rs = (gains / period) / (losses / period || 1e-9);
          rsi[i] = 100 - 100 / (1 + rs);
        }
      } else {
        gains = (gains * (period - 1) + gain) / period;
        losses = (losses * (period - 1) + loss) / period;
        const rs = gains / (losses || 1e-9);
        rsi[i] = 100 - 100 / (1 + rs);
      }
    }
    
    return rsi;
  }

  calculateMACD(data, fastPeriod = 12, slowPeriod = 26, signalPeriod = 9) {
    const emaFast = this.calculateEMA(data, fastPeriod);
    const emaSlow = this.calculateEMA(data, slowPeriod);
    
    const macd = [];
    const signal = [];
    const histogram = [];
    
    // Calculate MACD line
    for (let i = 0; i < data.length; i++) {
      if (emaFast[i] !== null && emaSlow[i] !== null) {
        macd[i] = emaFast[i] - emaSlow[i];
      } else {
        macd[i] = null;
      }
    }
    
    // Calculate signal line (EMA of MACD)
    const macdValues = macd.filter(v => v !== null);
    if (macdValues.length > 0) {
      const signalEMA = this.calculateEMA(macdValues, signalPeriod);
      let signalIndex = 0;
      
      for (let i = 0; i < macd.length; i++) {
        if (macd[i] !== null) {
          signal[i] = signalEMA[signalIndex];
          histogram[i] = macd[i] - signal[i];
          signalIndex++;
        } else {
          signal[i] = null;
          histogram[i] = null;
        }
      }
    }
    
    return { macd, signal, histogram };
  }

  calculateBollingerBands(data, period = 20, stdDev = 2) {
    const sma = this.calculateSMA(data, period);
    const upper = [];
    const lower = [];
    const middle = [];
    
    for (let i = period - 1; i < data.length; i++) {
      if (sma[i] !== null) {
        // Calculate standard deviation
        const slice = data.slice(i - period + 1, i + 1);
        const mean = sma[i];
        const variance = slice.reduce((sum, val) => sum + Math.pow(val - mean, 2), 0) / period;
        const std = Math.sqrt(variance);
        
        upper[i] = mean + (stdDev * std);
        lower[i] = mean - (stdDev * std);
        middle[i] = mean;
      } else {
        upper[i] = null;
        lower[i] = null;
        middle[i] = null;
      }
    }
    
    return {
      upper: Array(period - 1).fill(null).concat(upper.slice(period - 1)),
      middle: Array(period - 1).fill(null).concat(middle.slice(period - 1)),
      lower: Array(period - 1).fill(null).concat(lower.slice(period - 1))
    };
  }

  getActiveIndicators() {
    const active = [];
    if (this.indicators.ema20) active.push('ema20');
    if (this.indicators.ema50) active.push('ema50');
    if (this.indicators.rsi) active.push('rsi');
    if (this.indicators.macd) active.push('macd');
    if (this.indicators.bollinger) active.push('bollinger');
    if (this.indicators.volume) active.push('volume');
    return active;
  }

  calculateChartHeight(activeIndicators) {
    const baseHeight = 300; // Base height for price chart only
    const subplotHeight = 120; // Height per additional subplot
    
    let subplots = 0;
    
    // Count subplots based on active indicators
    if (activeIndicators.includes('volume')) subplots++;
    if (activeIndicators.includes('rsi')) subplots++;
    if (activeIndicators.includes('macd')) subplots++;
    
    // If no subplots, use compact height
    if (subplots === 0) {
      return Math.max(baseHeight, 400);
    }
    
    // Calculate total height
    const totalHeight = baseHeight + (subplots * subplotHeight);
    
    // Set reasonable limits
    const minHeight = 400;
    const maxHeight = 800;
    
    return Math.max(minHeight, Math.min(maxHeight, totalHeight));
  }

  calculateDynamicLayout(activeIndicators) {
    const hasVolume = activeIndicators.includes('volume');
    const hasRSI = activeIndicators.includes('rsi');
    const hasMACD = activeIndicators.includes('macd');
    
    let yaxis = { title: 'Price', domain: [0.5, 1] };
    let yaxis2 = null;
    let yaxis3 = null;
    let yaxis4 = null;
    
    // Calculate domains based on active subplots
    if (hasVolume && hasRSI && hasMACD) {
      // All subplots active
      yaxis = { title: 'Price', domain: [0.6, 1] };
      yaxis2 = { title: 'Volume', domain: [0.45, 0.6] };
      yaxis3 = { title: 'RSI', domain: [0.3, 0.45], range: [0, 100] };
      yaxis4 = { title: 'MACD', domain: [0, 0.3] };
    } else if (hasVolume && hasRSI) {
      // Volume + RSI
      yaxis = { title: 'Price', domain: [0.55, 1] };
      yaxis2 = { title: 'Volume', domain: [0.4, 0.55] };
      yaxis3 = { title: 'RSI', domain: [0.2, 0.4], range: [0, 100] };
    } else if (hasVolume && hasMACD) {
      // Volume + MACD
      yaxis = { title: 'Price', domain: [0.55, 1] };
      yaxis2 = { title: 'Volume', domain: [0.4, 0.55] };
      yaxis4 = { title: 'MACD', domain: [0, 0.4] };
    } else if (hasRSI && hasMACD) {
      // RSI + MACD
      yaxis = { title: 'Price', domain: [0.6, 1] };
      yaxis3 = { title: 'RSI', domain: [0.3, 0.6], range: [0, 100] };
      yaxis4 = { title: 'MACD', domain: [0, 0.3] };
    } else if (hasVolume) {
      // Only Volume
      yaxis = { title: 'Price', domain: [0.4, 1] };
      yaxis2 = { title: 'Volume', domain: [0, 0.4] };
    } else if (hasRSI) {
      // Only RSI
      yaxis = { title: 'Price', domain: [0.4, 1] };
      yaxis3 = { title: 'RSI', domain: [0, 0.4], range: [0, 100] };
    } else if (hasMACD) {
      // Only MACD
      yaxis = { title: 'Price', domain: [0.4, 1] };
      yaxis4 = { title: 'MACD', domain: [0, 0.4] };
    } else {
      // No subplots - full height for price
      yaxis = { title: 'Price', domain: [0, 1] };
    }
    
    const layout = {
      title: `${window.__NAME__ || this.symbol} - Advanced Technical Analysis`,
      xaxis: { rangeslider: { visible: false } },
      yaxis: yaxis,
      template: 'plotly_white',
      margin: { t: 30, r: 20, b: 40, l: 50 },
      paper_bgcolor: '#f5f7fb',
      plot_bgcolor: '#ffffff',
      legend: { orientation: 'h', y: -0.1 }
    };
    
    // Add subplot axes if they exist
    if (yaxis2) layout.yaxis2 = yaxis2;
    if (yaxis3) layout.yaxis3 = yaxis3;
    if (yaxis4) layout.yaxis4 = yaxis4;
    
    return layout;
  }

  updateKeyStats() {
    if (!this.currentData) return;
    
    const prices = this.currentData.close;
    const volumes = this.currentData.volume;
    const currentPrice = prices[prices.length - 1];
    const prevPrice = prices[prices.length - 2];
    const dailyChange = ((currentPrice - prevPrice) / prevPrice * 100).toFixed(2);
    
    // Calculate volatility (simplified)
    const recentPrices = prices.slice(-20);
    const avgPrice = recentPrices.reduce((a, b) => a + b, 0) / recentPrices.length;
    const volatility = (Math.sqrt(recentPrices.reduce((sum, price) => sum + Math.pow(price - avgPrice, 2), 0) / recentPrices.length) / avgPrice * 100).toFixed(1);
    
  const el = document.getElementById('keyStats');
    if (el) {
  el.innerHTML = '';
  const items = [
        { label: 'Last Price', value: `$${currentPrice.toFixed(2)}` },
        { label: 'Daily Change', value: `${dailyChange > 0 ? '+' : ''}${dailyChange}%` },
        { label: 'Volatility', value: `${volatility}%` },
        { label: 'Volume', value: this.formatNumber(volumes[volumes.length - 1]) }
      ];
      
  for (const it of items) {
    const div = document.createElement('div');
    div.className = 'col-6';
        div.innerHTML = `<div class="small text-muted">${it.label}</div><div class="fw-semibold">${it.value}</div>`;
    el.appendChild(div);
  }
}
  }

  formatNumber(num) {
    if (num >= 1e9) return (num / 1e9).toFixed(1) + 'B';
    if (num >= 1e6) return (num / 1e6).toFixed(1) + 'M';
    if (num >= 1e3) return (num / 1e3).toFixed(1) + 'K';
    return num.toFixed(0);
  }


  displayPatternAnalysis(patterns) {
    const container = document.getElementById('patternAnalysis');
    if (!container) return;
    
    if (patterns.length === 0) {
      container.innerHTML = '<div class="text-center text-muted"><p>No patterns detected</p></div>';
      return;
    }

    let html = '<div class="pattern-list">';
    patterns.forEach(pattern => {
      const confidenceColor = pattern.confidence > 0.7 ? 'success' : pattern.confidence > 0.5 ? 'warning' : 'danger';
      html += `
        <div class="pattern-item mb-2 p-2 border rounded">
          <div class="d-flex justify-content-between align-items-center">
            <span class="fw-bold">${pattern.type}</span>
            <span class="badge bg-${confidenceColor}">${(pattern.confidence * 100).toFixed(0)}%</span>
          </div>
          <div class="small text-muted">Signal: ${pattern.signal}</div>
        </div>
      `;
    });
    html += '</div>';
    
    container.innerHTML = html;
  }

  displayTradingSignals(signals) {
    const container = document.getElementById('tradingSignals');
    if (!container) return;
    
    if (signals.length === 0) {
      container.innerHTML = '<div class="text-center text-muted"><p>No trading signals</p></div>';
      return;
    }

    let html = '<div class="signals-list">';
    signals.forEach(signal => {
      const signalColor = signal.type === 'BUY' ? 'success' : 'danger';
      html += `
        <div class="signal-item mb-3 p-3 border rounded">
          <div class="d-flex justify-content-between align-items-center mb-2">
            <span class="badge bg-${signalColor}">${signal.type}</span>
            <span class="small">${signal.pattern}</span>
          </div>
          <div class="row g-2 small">
            <div class="col-6"><strong>Entry:</strong> $${signal.entry?.toFixed(2) || 'N/A'}</div>
            <div class="col-6"><strong>Stop:</strong> $${signal.stopLoss?.toFixed(2) || 'N/A'}</div>
            <div class="col-6"><strong>Target:</strong> $${signal.target?.toFixed(2) || 'N/A'}</div>
            <div class="col-6"><strong>R:R:</strong> ${signal.riskReward || 'N/A'}</div>
          </div>
        </div>
      `;
    });
    html += '</div>';
    
    container.innerHTML = html;
  }

  displayRiskAnalysis(metrics) {
    const container = document.getElementById('riskAnalysis');
    if (!container) return;
    
    container.innerHTML = `
      <div class="risk-metrics">
        <div class="metric-item mb-2" data-bs-toggle="tooltip" title="Average True Range: measures recent price volatility in points">
          <strong>ATR:</strong> ${metrics.atr}
        </div>
        <div class="metric-item mb-2" data-bs-toggle="tooltip" title="Volatility as a percentage of price (higher = larger swings)">
          <strong>Volatility:</strong> ${metrics.volatility}
        </div>
        <div class="metric-item mb-2" data-bs-toggle="tooltip" title="Approximate number of shares based on 1–2% risk rule">
          <strong>Position Size:</strong> ${metrics.positionSize} shares
        </div>
        <div class="metric-item mb-2" data-bs-toggle="tooltip" title="Suggested stop loss distance based on ATR">
          <strong>Stop Distance:</strong> $${metrics.stopDistance}
        </div>
        <div class="metric-item mb-2" data-bs-toggle="tooltip" title="Maximum capital risked on one trade">
          <strong>Risk per Trade:</strong> ${metrics.riskPerTrade}
        </div>
      </div>
    `;

    this.enableTooltips();
  }


  togglePatterns() {
    this.showPatterns = !this.showPatterns;
    // In a real implementation, this would toggle pattern overlays on the chart
  }

  toggleLive() {
    const liveToggle = document.getElementById('liveToggle');
  const isOn = liveToggle.getAttribute('data-live') === 'on';
    
  if (isOn) {
      clearInterval(this.liveTimer);
      this.liveTimer = null;
    liveToggle.setAttribute('data-live', 'off');
    liveToggle.textContent = 'Live: Off';
    liveToggle.classList.remove('btn-danger');
    liveToggle.classList.add('btn-success');
  } else {
      // poll every 5 seconds for live updates
      this.liveTimer = setInterval(() => {
        this.loadInitialData();
      }, 5000);
    liveToggle.setAttribute('data-live', 'on');
    liveToggle.textContent = 'Live: On';
    liveToggle.classList.remove('btn-success');
    liveToggle.classList.add('btn-danger');
  }
  }

  showLoading() {
    const container = document.getElementById('patternAnalysis');
    if (container) {
      container.innerHTML = '<div class="text-center"><div class="spinner-border text-primary" role="status"></div><p>Loading data...</p></div>';
    }
  }

  showError(message) {
    const container = document.getElementById('patternAnalysis');
    if (container) {
      container.innerHTML = `<div class="text-center text-danger"><p>${message}</p></div>`;
    }
    console.error('Chart Pattern Analyzer Error:', message);
  }

  enableTooltips() {
    try {
      // Initialize Bootstrap tooltips for any element with data-bs-toggle="tooltip"
      const tooltipTriggerList = [].slice.call(document.querySelectorAll('[data-bs-toggle="tooltip"]'));
      tooltipTriggerList.forEach(el => {
        try { new bootstrap.Tooltip(el); } catch (e) {}
      });
      // Also enable title tooltips inside dropdown items
      const titleItems = document.querySelectorAll('.dropdown-menu [title]');
      titleItems.forEach(el => {
        try { new bootstrap.Tooltip(el); } catch (e) {}
      });
    } catch (e) {
      // bootstrap might not be available in some contexts; ignore
    }
  }
}

// Initialize the chart pattern analyzer when the page loads
document.addEventListener('DOMContentLoaded', () => {
  new ChartPatternAnalyzer();
});