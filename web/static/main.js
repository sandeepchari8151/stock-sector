const sectorSelect = document.getElementById('sectorSelect');
const stockSelect = document.getElementById('stockSelect');
const loadBtn = document.getElementById('loadBtn');
const topTableBody = document.getElementById('topTableBody');
const refreshTopBtn = document.getElementById('refreshTopBtn');

async function loadSectors() {
  const { data } = await axios.get('/api/sectors');
  sectorSelect.innerHTML = '';
  data.sectors.forEach((s) => {
    const opt = document.createElement('option');
    opt.value = s.id;
    opt.textContent = s.name;
    sectorSelect.appendChild(opt);
  });
  if (data.sectors.length) {
    sectorSelect.value = data.sectors[0].id;
    await loadStocks(data.sectors[0].id);
  }
}

async function loadStocks(sectorId) {
  const { data } = await axios.get(`/api/stocks?sector=${encodeURIComponent(sectorId)}`);
  stockSelect.innerHTML = '';
  data.stocks.forEach((stk) => {
    const opt = document.createElement('option');
    opt.value = stk.symbol;
    opt.textContent = `${stk.symbol} — ${stk.name}`;
    stockSelect.appendChild(opt);
  });
}

sectorSelect.addEventListener('change', async (e) => {
  await loadStocks(e.target.value);
  await loadTopTradables();
});

loadBtn.addEventListener('click', async () => {
  const sectorId = sectorSelect.value;
  const selected = Array.from(stockSelect.selectedOptions).map((o) => o.value);
  if (!selected.length) return;

  const { data } = await axios.get('/api/candles', {
    params: { sector: sectorId, symbols: selected.join(','), period: '6mo', interval: '1d' }
  });

  renderCandles(data);
});

function renderCandles(apiData) {
  const traces = apiData.series.map((s) => ({
    x: s.dates,
    open: s.open,
    high: s.high,
    low: s.low,
    close: s.close,
    type: 'candlestick',
    name: s.symbol,
  }));

  const layout = {
    dragmode: 'zoom',
    margin: { t: 30, r: 30, b: 40, l: 50 },
    xaxis: { rangeslider: { visible: false } },
    yaxis: { fixedrange: false },
    legend: { orientation: 'h' },
    template: 'plotly_white',
    paper_bgcolor: '#f5f7fb',
    plot_bgcolor: '#ffffff',
    colorway: ['#0a2540', '#4bb3fd', '#e63946', '#1d3557', '#a8dadc']
  };

  Plotly.newPlot('candles', traces, layout, { responsive: true, displaylogo: false });
}

async function loadTopTradables() {
  const sectorId = sectorSelect.value || '';
  const { data } = await axios.get('/api/top-stocks', {
    params: { sector: sectorId, period: '3mo', interval: '1d', lookback_days: 60 }
  });
  topTableBody.innerHTML = '';
  data.items.forEach((item) => {
    const tr = document.createElement('tr');
    tr.innerHTML = `
      <td><span class="badge text-bg-secondary">${item.symbol}</span></td>
      <td>${item.name || ''}</td>
      <td class="text-end">$${formatCompact(item.avgDollarVolume)}</td>
      <td class="text-end">${Number(item.lastPrice).toFixed(2)}</td>
      <td class="text-end">
        <button class="btn btn-sm btn-primary" data-sym="${item.symbol}">Chart</button>
      </td>
    `;
    tr.querySelector('button').addEventListener('click', async (e) => {
      const sym = e.currentTarget.getAttribute('data-sym');
      const { data } = await axios.get('/api/candles', {
        params: { sector: sectorId, symbols: sym, period: '6mo', interval: '1d' }
      });
      renderCandles(data);
    });
    topTableBody.appendChild(tr);
  });
}

function formatCompact(num) {
  try {
    return Intl.NumberFormat('en', { notation: 'compact', maximumFractionDigits: 2 }).format(num);
  } catch {
    return String(Math.round(num));
  }
}

refreshTopBtn.addEventListener('click', loadTopTradables);

// init
loadSectors().then(loadTopTradables);


