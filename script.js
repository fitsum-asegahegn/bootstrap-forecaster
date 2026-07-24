/* ---------------------------------------------------------------
   Bootstrap Small-Sample Forecaster
   Implements: OLS trend-vs-mean selection, circular block bootstrap
   of residuals, refit-and-extrapolate reconstruction, percentile
   prediction intervals. See accompanying paper Section 3 for the
   full derivation; this client-side version uses percentile
   intervals as a transparent proxy for the paper's BCa intervals.
------------------------------------------------------------------ */

const SAMPLE_SERIES = [82000, 79500, 91000, 88200, 95000, 101000, 97500, 106000, 112000, 108500, 115000, 121000, 118500, 126000, 131000];

let state = {
  horizon: 3,
  target: 'next',
};

// ---------- stats helpers ----------

function mean(arr) { return arr.reduce((a, b) => a + b, 0) / arr.length; }

function olsFit(x, y) {
  const n = x.length;
  const xBar = mean(x), yBar = mean(y);
  let Sxx = 0, Sxy = 0;
  for (let i = 0; i < n; i++) {
    Sxx += (x[i] - xBar) ** 2;
    Sxy += (x[i] - xBar) * (y[i] - yBar);
  }
  const slope = Sxx === 0 ? 0 : Sxy / Sxx;
  const intercept = yBar - slope * xBar;
  return { intercept, slope, Sxx };
}

const TREND_T_THRESHOLD = 1.5; // heuristic cutoff (~p 0.13-0.20 for small n), not a formal 0.05 test
const TREND_MARGINAL_T = 2.5;  // above this, treat the trend call as comfortably confident rather than borderline

function trendTStat(x, y) {
  const n = x.length;
  if (n < 5) return 0;
  const { intercept, slope, Sxx } = olsFit(x, y);
  let sse = 0;
  for (let i = 0; i < n; i++) {
    const fitted = intercept + slope * x[i];
    sse += (y[i] - fitted) ** 2;
  }
  const sigma2 = sse / Math.max(n - 2, 1);
  const seSlope = Sxx === 0 ? Infinity : Math.sqrt(sigma2 / Sxx);
  return seSlope === 0 ? 0 : slope / seSlope;
}

function decompose(y) {
  const n = y.length;
  const t = y.map((_, i) => i);
  const tStat = trendTStat(t, y);
  const useTrend = Math.abs(tStat) > TREND_T_THRESHOLD;
  let intercept, slope;
  if (useTrend) {
    ({ intercept, slope } = olsFit(t, y));
  } else {
    intercept = mean(y);
    slope = 0;
  }
  const fitted = t.map(ti => intercept + slope * ti);
  const residuals = y.map((yi, i) => yi - fitted[i]);
  return { useTrend, tStat, intercept, slope, fitted, residuals, t };
}

function circularBlockBootstrap(residuals, n, blockLen) {
  const out = [];
  while (out.length < n) {
    const start = Math.floor(Math.random() * n);
    for (let i = 0; i < blockLen && out.length < n; i++) {
      out.push(residuals[(start + i) % n]);
    }
  }
  return out;
}

function percentile(sortedArr, p) {
  const n = sortedArr.length;
  if (n === 0) return NaN;
  const idx = p * (n - 1);
  const lo = Math.floor(idx), hi = Math.ceil(idx);
  if (lo === hi) return sortedArr[lo];
  const frac = idx - lo;
  return sortedArr[lo] * (1 - frac) + sortedArr[hi] * frac;
}

function runBootstrapForecast(y, horizon, B = 3000) {
  const n = y.length;
  const { useTrend, tStat, fitted, residuals, t } = decompose(y);
  const blockLen = Math.max(1, Math.floor(n / 3));

  // paths[b][k] = forecast for horizon step k (0-indexed) in replicate b
  const paths = new Array(B);

  for (let b = 0; b < B; b++) {
    const bootResid = circularBlockBootstrap(residuals, n, blockLen);
    const yBoot = fitted.map((f, i) => f + bootResid[i]);

    let bIntercept, bSlope;
    if (useTrend) {
      ({ intercept: bIntercept, slope: bSlope } = olsFit(t, yBoot));
    } else {
      bIntercept = mean(yBoot);
      bSlope = 0;
    }

    const row = new Array(horizon);
    for (let k = 1; k <= horizon; k++) {
      const muFuture = bIntercept + bSlope * (n - 1 + k);
      const futureResid = bootResid[Math.floor(Math.random() * n)];
      row[k - 1] = muFuture + futureResid;
    }
    paths[b] = row;
  }

  // per-horizon quantiles
  const perHorizon = [];
  for (let k = 0; k < horizon; k++) {
    const col = paths.map(r => r[k]).sort((a, b) => a - b);
    perHorizon.push({
      p5: percentile(col, 0.05),
      p10: percentile(col, 0.10),
      p25: percentile(col, 0.25),
      p50: percentile(col, 0.50),
      p75: percentile(col, 0.75),
      p90: percentile(col, 0.90),
      p95: percentile(col, 0.95),
    });
  }

  const cumulative = paths.map(r => r.reduce((a, b) => a + b, 0)).sort((a, b) => a - b);

  return { paths, perHorizon, cumulative, useTrend, tStat, fitted, residuals, blockLen, n };
}

// ---------- formatting ----------

function fmtBirr(v) {
  return Math.round(v).toLocaleString('en-US') + ' Birr';
}

// ---------- parsing ----------

function parseSeries(text) {
  const nums = text
    .split(/[\s,]+/)
    .map(s => s.trim())
    .filter(s => s.length > 0)
    .map(Number)
    .filter(v => !Number.isNaN(v));
  return nums;
}

// ---------- rendering ----------

function renderFanChart(y, result, horizon) {
  const n = y.length;
  const monthsHist = Array.from({ length: n }, (_, i) => i + 1);
  const monthsFuture = Array.from({ length: horizon }, (_, i) => n + i + 1);

  const p5 = result.perHorizon.map(h => h.p5);
  const p95 = result.perHorizon.map(h => h.p95);
  const p10 = result.perHorizon.map(h => h.p10);
  const p90 = result.perHorizon.map(h => h.p90);
  const p25 = result.perHorizon.map(h => h.p25);
  const p75 = result.perHorizon.map(h => h.p75);
  const p50 = result.perHorizon.map(h => h.p50);

  const anchorX = [n, ...monthsFuture];
  const lastActual = y[n - 1];

  const band = (lo, hi, color) => ([
    { x: anchorX, y: [lastActual, ...hi], mode: 'lines', line: { width: 0 }, showlegend: false, hoverinfo: 'skip' },
    { x: anchorX, y: [lastActual, ...lo], mode: 'lines', line: { width: 0 }, fill: 'tonexty', fillcolor: color, showlegend: false, hoverinfo: 'skip' },
  ]);

  const traces = [
    { x: monthsHist, y: y, mode: 'lines+markers', line: { color: '#eef1f6', width: 2 }, marker: { size: 5 }, name: 'History' },
    ...band(p5, p95, 'rgba(230,165,60,0.10)'),
    ...band(p10, p90, 'rgba(230,165,60,0.16)'),
    ...band(p25, p75, 'rgba(230,165,60,0.30)'),
    { x: anchorX, y: [lastActual, ...p50], mode: 'lines', line: { color: '#e6a53c', width: 2.5, dash: 'solid' }, name: 'Median forecast' },
  ];

  const layout = {
    paper_bgcolor: 'transparent',
    plot_bgcolor: 'transparent',
    font: { color: '#a9b1c4', family: 'IBM Plex Mono, monospace', size: 11 },
    margin: { l: 60, r: 20, t: 10, b: 40 },
    xaxis: { title: 'Month', gridcolor: '#232d40', zeroline: false },
    yaxis: { title: 'Birr', gridcolor: '#232d40', zeroline: false, tickformat: ',.0f' },
    showlegend: true,
    legend: { orientation: 'h', y: -0.2, font: { color: '#a9b1c4' } },
  };

  Plotly.newPlot('fan-chart', traces, layout, { displayModeBar: false, responsive: true });
}

function renderHistogram(values, threshold, label) {
  const trace = {
    x: values,
    type: 'histogram',
    marker: { color: 'rgba(230,165,60,0.55)', line: { color: '#e6a53c', width: 0.5 } },
    nbinsx: 40,
  };
  const shapes = [];
  if (!Number.isNaN(threshold)) {
    shapes.push({
      type: 'line', x0: threshold, x1: threshold, y0: 0, y1: 1, yref: 'paper',
      line: { color: '#f2a65a', width: 2, dash: 'dash' },
    });
  }
  const layout = {
    paper_bgcolor: 'transparent',
    plot_bgcolor: 'transparent',
    font: { color: '#a9b1c4', family: 'IBM Plex Mono, monospace', size: 11 },
    margin: { l: 50, r: 20, t: 10, b: 40 },
    xaxis: { title: label, gridcolor: '#232d40', tickformat: ',.0f' },
    yaxis: { title: 'Simulated futures', gridcolor: '#232d40' },
    shapes,
  };
  Plotly.newPlot('hist-chart', [trace], layout, { displayModeBar: false, responsive: true });
}

function buildPlainLanguage(y, result, horizon, target, threshold) {
  const n = y.length;
  const items = [];

  const nextMonthDist = result.paths.map(r => r[0]).sort((a, b) => a - b);
  const nextMedian = percentile(nextMonthDist, 0.5);

  items.push(
    `Based on your ${n} months of data, the median forecast for next month is <strong>${fmtBirr(nextMedian)}</strong>.`
  );

  if (!Number.isNaN(threshold)) {
    let distribution, label;
    if (target === 'next') {
      distribution = nextMonthDist;
      label = 'next month';
    } else {
      distribution = result.cumulative;
      label = `the next ${horizon} month${horizon > 1 ? 's' : ''} combined`;
    }
    const exceedCount = distribution.filter(v => v > threshold).length;
    const prob = Math.round((exceedCount / distribution.length) * 100);
    items.push(
      `There is a <strong>${prob}%</strong> chance that ${label} will exceed <strong>${fmtBirr(threshold)}</strong>.`
    );
  }

  const worstCumulative = percentile(result.cumulative, 0.05);
  const bestCumulative = percentile(result.cumulative, 0.95);
  items.push(
    `Across the full ${horizon}-month horizon, the worst-case (5th percentile) total is <strong>${fmtBirr(worstCumulative)}</strong>, and the best-case (95th percentile) total is <strong>${fmtBirr(bestCumulative)}</strong>.`
  );

  const absT = Math.abs(result.tStat);
  const tDisplay = absT.toFixed(2);
  const isMarginal = result.useTrend && absT < TREND_MARGINAL_T;
  const marginalTag = isMarginal ? ' <strong>(marginal call — close to the cutoff)</strong>' : '';

  items.push(
    result.useTrend
      ? `A trend was detected in your series (t = ${tDisplay}, threshold: ${TREND_T_THRESHOLD})${marginalTag} and extrapolated forward — re-estimated on every one of the 3,000 bootstrap replicates to reflect estimation uncertainty, not just noise. This threshold is deliberately looser than a standard 5% significance test, to avoid missing weak drift in short series; that trade-off means it will occasionally call a trend on what is really just noise.`
      : `No trend was detected in your series (t = ${tDisplay}, threshold: ${TREND_T_THRESHOLD}), so the forecast is centered on the historical average rather than an extrapolated slope.`
  );

  return items;
}

function updateReadout(y, result, horizon, target, threshold) {
  document.getElementById('readout-empty').hidden = true;
  const content = document.getElementById('readout-content');
  content.hidden = false;

  const distribution = target === 'next'
    ? result.paths.map(r => r[0]).sort((a, b) => a - b)
    : result.cumulative;

  const label = target === 'next' ? "Next month" : `${horizon}-month total`;

  document.getElementById('prob-label').textContent = Number.isNaN(threshold)
    ? 'P(exceeds threshold)'
    : `P(${label.toLowerCase()} > threshold)`;
  document.getElementById('median-label').textContent = `Median — ${label.toLowerCase()}`;
  document.getElementById('worst-label').textContent = `Worst case (5th pct)`;
  document.getElementById('best-label').textContent = `Best case (95th pct)`;

  if (Number.isNaN(threshold)) {
    document.getElementById('prob-value').textContent = '—';
  } else {
    const exceedCount = distribution.filter(v => v > threshold).length;
    const prob = Math.round((exceedCount / distribution.length) * 100);
    document.getElementById('prob-value').textContent = prob + '%';
  }

  document.getElementById('median-value').textContent = fmtBirr(percentile(distribution, 0.5));
  document.getElementById('worst-value').textContent = fmtBirr(percentile(distribution, 0.05));
  document.getElementById('best-value').textContent = fmtBirr(percentile(distribution, 0.95));

  const list = document.getElementById('plain-language');
  list.innerHTML = '';
  buildPlainLanguage(y, result, horizon, target, threshold).forEach(text => {
    const li = document.createElement('li');
    li.innerHTML = text;
    list.appendChild(li);
  });

  document.getElementById('hist-sub').textContent =
    `3,000 simulated futures for ${label.toLowerCase()}.`;
}

// ---------- main run ----------

function showWarning(msg) {
  const el = document.getElementById('input-warning');
  if (!msg) { el.hidden = true; el.textContent = ''; return; }
  el.hidden = false;
  el.textContent = msg;
}

function runForecast() {
  const text = document.getElementById('series-input').value;
  const y = parseSeries(text);

  if (y.length < 6) {
    showWarning('Enter at least 6 monthly values to run a forecast (12+ recommended).');
    return;
  }
  showWarning(y.length < 12 ? `Only ${y.length} points — the tool will run, but intervals will be wide and less reliable.` : '');

  const horizon = state.horizon;
  const target = state.target;
  const thresholdRaw = document.getElementById('threshold-input').value;
  const threshold = thresholdRaw === '' ? NaN : Number(thresholdRaw);

  const result = runBootstrapForecast(y, horizon, 3000);

  renderFanChart(y, result, horizon);

  const histValues = target === 'next' ? result.paths.map(r => r[0]) : result.cumulative;
  const histLabel = target === 'next' ? 'Next month (Birr)' : `${horizon}-month total (Birr)`;
  renderHistogram(histValues, threshold, histLabel);

  updateReadout(y, result, horizon, target, threshold);
}

// ---------- UI wiring ----------

function wireSegmented(containerId, stateKey, isNumber) {
  const container = document.getElementById(containerId);
  container.querySelectorAll('button').forEach(btn => {
    btn.addEventListener('click', () => {
      container.querySelectorAll('button').forEach(b => b.classList.remove('is-active'));
      btn.classList.add('is-active');
      state[stateKey] = isNumber ? Number(btn.dataset.value) : btn.dataset.value;
    });
  });
}

document.addEventListener('DOMContentLoaded', () => {
  wireSegmented('horizon-select', 'horizon', true);
  wireSegmented('target-select', 'target', false);

  document.getElementById('run-btn').addEventListener('click', runForecast);

  document.getElementById('sample-btn').addEventListener('click', () => {
    document.getElementById('series-input').value = SAMPLE_SERIES.join(', ');
    document.getElementById('threshold-input').value = 135000;
    showWarning('');
  });

  document.getElementById('reset-btn').addEventListener('click', () => {
    document.getElementById('series-input').value = '';
    document.getElementById('threshold-input').value = '';
    document.getElementById('readout-content').hidden = true;
    document.getElementById('readout-empty').hidden = false;
    Plotly.purge('fan-chart');
    Plotly.purge('hist-chart');
    showWarning('');
  });
});
