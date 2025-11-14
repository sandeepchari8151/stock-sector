const sector = window.__SECTOR__;
const minPrice = document.getElementById('minPrice');
const maxPrice = document.getElementById('maxPrice');
const minADV = document.getElementById('minADV');
const limit = document.getElementById('limit');
const period = document.getElementById('period');
const searchBtn = document.getElementById('searchBtn');
const resultsBody = document.getElementById('resultsBody');
const chartPeriod = document.getElementById('chartPeriod');
const indicatorSel = document.getElementById('indicator');
// Removed advanced technical filter elements
const exportCsv = document.getElementById('exportCsv');

const btStrategy = document.getElementById('btStrategy');
const btSymbol = document.getElementById('btSymbol');
const btPeriod = document.getElementById('btPeriod');
const btFast = document.getElementById('btFast');
const btSlow = document.getElementById('btSlow');
const runBacktest = document.getElementById('runBacktest');
const btMetrics = document.getElementById('btMetrics');

const alertSymbol = document.getElementById('alertSymbol');
const alertDirection = document.getElementById('alertDirection');
const createAlert = document.getElementById('createAlert');
const checkAlerts = document.getElementById('checkAlerts');
const alertStatus = document.getElementById('alertStatus');

let __searchSeq = 0;
let __controller = null;

async function search() {
  setResultsLoading(true);
  toggleSearchDisabled(true);
  try {
  // cancel previous in-flight request
  try { __controller?.abort(); } catch(e) {}
  __controller = new AbortController();
  const seq = ++__searchSeq;
  const params = {
    sector,
    min_price: Number(minPrice.value || 0),
    max_price: Number(maxPrice.value || 0),
    min_adv: Number(minADV.value || 0),
    limit: Number(limit.value || 20),
    period: period.value,
  };
  console.debug('Screener params', params);
  const { data } = await axios.get('/api/stocks/search_advanced', { params, signal: __controller.signal });

  // if a newer request was started, ignore this result
  if (seq !== __searchSeq) return;

  resultsBody.innerHTML = '';
  data.items.forEach((r) => {
    const tr = document.createElement('tr');
    tr.innerHTML = `
      <td><span class="badge text-bg-secondary">${r.symbol}</span></td>
      <td>${r.name || ''}</td>
      <td class="text-end">${Number(r.lastPrice).toFixed(2)}</td>
      <td class="text-end">$${formatCompact(r.avgDollarVolume)}</td>
      <td class="text-end">${r.ret1m == null ? '-' : (r.ret1m * 100).toFixed(2) + '%'}</td>
      <td class="text-end"><a class="btn btn-sm btn-primary" href="/chart/${r.symbol}?sector=${encodeURIComponent(sector)}">Explore</a></td>
    `;
    resultsBody.appendChild(tr);
  });
  if (!data.items || data.items.length === 0) {
    resultsBody.innerHTML = `<tr><td colspan="6" class="text-center p-4">No results. Adjust filters and try again.</td></tr>`;
  }
  } finally {
    setResultsLoading(false);
    toggleSearchDisabled(false);
  }
}

function formatCompact(num) {
  try { return Intl.NumberFormat('en', { notation: 'compact', maximumFractionDigits: 2 }).format(num); }
  catch { return String(Math.round(num)); }
}

async function loadChart(symbol) {
  const { data } = await axios.get('/api/candles', {
    params: { sector, symbols: symbol, period: chartPeriod.value, interval: '1d' }
  });

  const s = data.series[0];
  const traces = [
    {
      x: s.dates, open: s.open, high: s.high, low: s.low, close: s.close,
      type: 'candlestick', name: symbol
    }
  ];

  // Optional indicators
  if (indicatorSel.value === 'sma20' || indicatorSel.value === 'ema20' || indicatorSel.value === 'bbands') {
    const closes = s.close;
    const sma = movingAverage(closes, 20);
    traces.push({ x: s.dates, y: sma, type: 'scatter', mode: 'lines', name: indicatorSel.value.toUpperCase() });

    if (indicatorSel.value === 'bbands') {
      const std20 = rollingStd(closes, 20);
      const upper = sma.map((v, i) => v != null && std20[i] != null ? v + 2 * std20[i] : null);
      const lower = sma.map((v, i) => v != null && std20[i] != null ? v - 2 * std20[i] : null);
      traces.push({ x: s.dates, y: upper, type: 'scatter', mode: 'lines', name: 'BB Upper' });
      traces.push({ x: s.dates, y: lower, type: 'scatter', mode: 'lines', name: 'BB Lower' });
    }
  }

  Plotly.newPlot('candles', traces, {
    margin: { t: 30, r: 20, b: 40, l: 50 }, template: 'plotly_white',
    paper_bgcolor: '#f5f7fb', plot_bgcolor: '#ffffff',
    xaxis: { rangeslider: { visible: false } },
    colorway: ['#0a2540', '#4bb3fd', '#e63946', '#1d3557', '#a8dadc']
  }, { responsive: true, displaylogo: false });

  // RSI overlay (secondary area) — simplified
  if (indicatorSel.value === 'rsi14') {
    const rsi = computeRSI(s.close, 14);
    Plotly.addTraces('candles', [{ x: s.dates, y: rsi, yaxis: 'y2', type: 'scatter', mode: 'lines', name: 'RSI 14' }]);
    Plotly.relayout('candles', {
      grid: { rows: 2, columns: 1, pattern: 'independent' },
      yaxis: { domain: [0.35, 1] },
      yaxis2: { domain: [0, 0.3] }
    });
  }
}

function movingAverage(arr, window) {
  const out = Array(arr.length).fill(null);
  let sum = 0;
  for (let i = 0; i < arr.length; i++) {
    sum += arr[i];
    if (i >= window) sum -= arr[i - window];
    if (i >= window - 1) out[i] = sum / window;
  }
  return out;
}

function rollingStd(arr, window) {
  const out = Array(arr.length).fill(null);
  const mean = movingAverage(arr, window);
  for (let i = 0; i < arr.length; i++) {
    if (i >= window - 1 && mean[i] != null) {
      let s = 0;
      for (let j = i - window + 1; j <= i; j++) s += Math.pow(arr[j] - mean[i], 2);
      out[i] = Math.sqrt(s / window);
    }
  }
  return out;
}

function computeRSI(closes, period = 14) {
  const rsi = Array(closes.length).fill(null);
  let gains = 0, losses = 0;
  for (let i = 1; i < closes.length; i++) {
    const change = closes[i] - closes[i - 1];
    const gain = Math.max(change, 0);
    const loss = Math.max(-change, 0);
    if (i <= period) {
      gains += gain; losses += loss;
      if (i === period) {
        const rs = (gains / period) / (losses / period || 1e-9);
        rsi[i] = 100 - 100 / (1 + rs);
      }
    } else {
      gains = (gains * (period - 1) + gain) / period;
      losses = (losses * (period - 1) + loss) / period;
      const rs = (gains) / (losses || 1e-9);
      rsi[i] = 100 - 100 / (1 + rs);
    }
  }
  return rsi;
}

searchBtn?.addEventListener('click', search);
period?.addEventListener('change', search);

// Quick nav helpers
document.getElementById('quickTop10')?.addEventListener('click', async () => {
  limit.value = 10; await search();
});
document.getElementById('quickTop20')?.addEventListener('click', async () => {
  limit.value = 20; await search();
});

document.getElementById('runNow')?.addEventListener('click', async () => {
  await search();
});

function setResultsLoading(isLoading) {
  if (isLoading) {
    resultsBody.innerHTML = `<tr><td colspan="6" class="text-center p-4"><div class="spinner-border text-primary me-2" role="status"></div> Loading...</td></tr>`;
  }
}

function toggleSearchDisabled(disabled) {
  [searchBtn, exportCsv].forEach(el => { 
    if (el) { 
      el.disabled = disabled; 
      el.classList.toggle('disabled', disabled); 
    } 
  });
}

// Auto-search on filter changes
[
  minPrice, maxPrice, minADV, limit, period
].forEach(el => {
  el?.addEventListener('change', () => search());
  el?.addEventListener('input', () => {
    // Debounced search for inputs
    clearTimeout(el.__t);
    el.__t = setTimeout(() => search(), 500);
  });
});

exportCsv?.addEventListener('click', async () => {
  const { data } = await axios.get('/api/stocks/search_advanced', {
    params: {
      sector,
      min_price: minPrice?.value || 0,
      max_price: maxPrice?.value || 0,
      min_adv: minADV?.value || 0,
      limit: limit?.value,
      period: period?.value,
    }
  });
  const headers = ['symbol','name','lastPrice','avgDollarVolume','ret1m'];
  const rows = [headers.join(',')].concat(data.items.map(it => headers.map(h => it[h]).join(',')));
  const blob = new Blob([rows.join('\n')], { type: 'text/csv;charset=utf-8;' });
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url; a.download = `screener_${sector}_${Date.now()}.csv`; a.click();
  URL.revokeObjectURL(url);
});

runBacktest?.addEventListener('click', async () => {
  const symbol = btSymbol?.value?.trim();
  if (!symbol) return;
  const payload = {
    symbol,
    strategy: btStrategy?.value,
    period: btPeriod?.value,
    interval: '1d',
    params: btStrategy?.value === 'sma_cross' ? { fast: Number(btFast?.value||10), slow: Number(btSlow?.value||20) } : { period: 14, buy: 30, sell: 50 }
  };
  const { data } = await axios.post('/api/backtest', payload);
  Plotly.newPlot('equityCurve', [{ x: data.dates, y: data.equity, type: 'scatter', mode: 'lines', name: 'Equity' }], {
    template: 'plotly_white', paper_bgcolor: '#f5f7fb', plot_bgcolor: '#ffffff', margin: { t: 20, r: 20, b: 40, l: 50 }
  }, { responsive: true, displaylogo: false });
  if (btMetrics) btMetrics.textContent = `Return: ${(data.metrics.totalReturn*100).toFixed(1)}% | MaxDD: ${(data.metrics.maxDrawdown*100).toFixed(1)}% | Trades: ${data.trades.length}`;
});

createAlert?.addEventListener('click', async () => {
  const symbol = alertSymbol?.value?.trim();
  if (!symbol) return;
  await axios.post('/api/alerts', { symbol, type: 'price_cross_sma20', direction: alertDirection?.value });
  if (alertStatus) alertStatus.textContent = 'Alert created';
  setTimeout(() => { if (alertStatus) alertStatus.textContent = ''; }, 2000);
});

checkAlerts?.addEventListener('click', async () => {
  const { data } = await axios.post('/api/alerts/check');
  if (data.fired && data.fired.length) {
    if (alertStatus) alertStatus.textContent = `Alerts fired: ${data.fired.map(a=>a.symbol).join(', ')}`;
  } else {
    if (alertStatus) alertStatus.textContent = 'No alerts fired';
  }
  setTimeout(() => { if (alertStatus) alertStatus.textContent = ''; }, 3000);
});

// initial load
try { if (limit) limit.value = '10'; } catch(e) {}
search();


