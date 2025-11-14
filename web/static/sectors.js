const periodSel = document.getElementById('period');
const refreshBtn = document.getElementById('refresh');
const overviewBody = document.getElementById('overviewBody');
const chartSelector = document.getElementById('chartSelector');

// Chart elements
const trendsChart = document.getElementById('trendsChart');
const returnsChart = document.getElementById('returnsChart');
const distributionChart = document.getElementById('distributionChart');
const riskChart = document.getElementById('riskChart');
const performanceChart = document.getElementById('performanceChart');
const interactiveChart = document.getElementById('interactiveChart');

// Shared summary section
const summarySection = document.getElementById('summarySection');

let liveTimer = null;

async function loadOverview() {
  const { data } = await axios.get('/api/sectors/overview', {
    params: { period: periodSel.value, interval: '1d' }
  });

  // Populate the shared overview table
  const sortedData = data.overview.sort((a, b) => b.cagr - a.cagr);
  
  if (overviewBody) {
    overviewBody.innerHTML = '';
    sortedData.forEach((row) => {
      const tr = document.createElement('tr');
      tr.innerHTML = `
        <td>${row.sector}</td>
        <td class="text-end">${(row.volatility * 100).toFixed(2)}%</td>
        <td class="text-end">${(row.cagr * 100).toFixed(2)}%</td>
        <td class="text-end"><a class="btn btn-sm btn-primary" href="/stocks/${row.sector}">Explore</a></td>
      `;
      overviewBody.appendChild(tr);
    });
  }

  // Chart: Cumulative trends (line)
  const traces = data.cum.map((s) => ({
    x: s.dates,
    y: s.cum,
    type: 'scatter',
    mode: 'lines',
    name: s.sector
  }));
  Plotly.newPlot('cumChart', traces, {
    margin: { t: 30, r: 20, b: 40, l: 50 },
    template: 'plotly_white',
    paper_bgcolor: '#f5f7fb',
    plot_bgcolor: '#ffffff',
    yaxis: { title: 'Cumulative Index (x)' },
    xaxis: { title: 'Date' },
    colorway: ['#0a2540', '#4bb3fd', '#e63946', '#1d3557', '#a8dadc']
  }, { responsive: true, displaylogo: false });

  // Chart: Sector Trends (duplicate of cumChart but separate container)
  Plotly.newPlot('chartTrends', traces, {
    margin: { t: 30, r: 20, b: 40, l: 50 },
    template: 'plotly_white',
    paper_bgcolor: '#f5f7fb',
    plot_bgcolor: '#ffffff',
    yaxis: { title: 'Cumulative Index (x)' },
    xaxis: { title: 'Date' },
    showlegend: true
  }, { responsive: true, displaylogo: false });

  // Chart: Average Returns (bar)
  const returnsData = data.overview.map(o => ({ sector: o.sector, cagr: o.cagr }));
  returnsData.sort((a,b) => b.cagr - a.cagr);
  Plotly.newPlot('chartReturns', [{
    x: returnsData.map(r => r.sector),
    y: returnsData.map(r => +(r.cagr * 100).toFixed(2)),
    type: 'bar'
  }], {
    margin: { t: 30, r: 20, b: 60, l: 50 },
    template: 'plotly_white',
    yaxis: { title: 'CAGR %' }
  }, { responsive: true, displaylogo: false });

  // Chart: Investment Distribution (pie) - weight by cumLast as proxy
  const weights = data.overview.map(o => o.cumLast || 1);
  Plotly.newPlot('chartDistribution', [{
    labels: data.overview.map(o => o.sector),
    values: weights,
    type: 'pie',
    textinfo: 'label+percent'
  }], {
    margin: { t: 30, r: 20, b: 20, l: 20 },
    template: 'plotly_white'
  }, { responsive: true, displaylogo: false });

  // Chart: Risk vs Growth (scatter)
  Plotly.newPlot('chartRisk', [{
    x: data.overview.map(o => +(o.volatility * 100).toFixed(2)),
    y: data.overview.map(o => +(o.cagr * 100).toFixed(2)),
    mode: 'markers+text',
    type: 'scatter',
    text: data.overview.map(o => o.sector),
    textposition: 'top center'
  }], {
    margin: { t: 30, r: 20, b: 60, l: 60 },
    template: 'plotly_white',
    xaxis: { title: 'Volatility %' },
    yaxis: { title: 'CAGR %' }
  }, { responsive: true, displaylogo: false });

  // Chart: Performance Summary (bar by cumulative last)
  const perfSorted = data.overview.slice().sort((a,b) => b.cumLast - a.cumLast);
  Plotly.newPlot('chartPerformance', [{
    x: perfSorted.map(o => o.sector),
    y: perfSorted.map(o => +o.cumLast.toFixed(3)),
    type: 'bar'
  }], {
    margin: { t: 30, r: 20, b: 60, l: 50 },
    template: 'plotly_white',
    yaxis: { title: 'Cumulative (x)' }
  }, { responsive: true, displaylogo: false });
}

// Function to show/hide charts based on selection
function toggleCharts() {
  const selectedChart = chartSelector.value;
  
  // Hide all charts first
  trendsChart.style.display = 'none';
  returnsChart.style.display = 'none';
  distributionChart.style.display = 'none';
  riskChart.style.display = 'none';
  performanceChart.style.display = 'none';
  interactiveChart.style.display = 'none';
  
  // Show selected chart(s) and always show summary
  switch(selectedChart) {
    case 'all':
      trendsChart.style.display = 'block';
      returnsChart.style.display = 'block';
      distributionChart.style.display = 'block';
      riskChart.style.display = 'block';
      performanceChart.style.display = 'block';
      interactiveChart.style.display = 'block';
      summarySection.style.display = 'block';
      break;
    case 'trends':
      trendsChart.style.display = 'block';
      summarySection.style.display = 'block';
      break;
    case 'returns':
      returnsChart.style.display = 'block';
      summarySection.style.display = 'block';
      break;
    case 'distribution':
      distributionChart.style.display = 'block';
      summarySection.style.display = 'block';
      break;
    case 'risk':
      riskChart.style.display = 'block';
      summarySection.style.display = 'block';
      break;
    case 'performance':
      performanceChart.style.display = 'block';
      summarySection.style.display = 'block';
      break;
    case 'interactive':
      interactiveChart.style.display = 'block';
      summarySection.style.display = 'block';
      break;
  }
}

// Event listeners
refreshBtn.addEventListener('click', loadOverview);
periodSel.addEventListener('change', loadOverview);
chartSelector.addEventListener('change', toggleCharts);

// Initialize with polling every 30s for real-time feel
function startPolling() {
  if (liveTimer) clearInterval(liveTimer);
  liveTimer = setInterval(loadOverview, 30000);
}

loadOverview();
startPolling();
// Set default to show only the first chart (Sector Trends)
chartSelector.value = 'trends';
toggleCharts();


