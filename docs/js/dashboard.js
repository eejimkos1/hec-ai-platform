/**
 * HEC AI Platform — Multi-page Dashboard
 * Sidebar navigation with full page replacement
 */

var HEC_COLORS = {
    navy: '#0D1B2A',
    navyLight: '#1B2838',
    teal: '#1B9AAA',
    tealLight: '#23B5C6',
    text: '#1A202C',
    textMuted: '#64748B',
    border: '#E2E8F0',
    success: '#059669',
    warning: '#D97706',
    danger: '#DC2626',
    colorway: ['#1B9AAA', '#0D1B2A', '#D97706', '#059669', '#7C3AED', '#DC2626', '#2563EB', '#0891B2', '#78350F', '#475569']
};

var PLOTLY_LAYOUT = {
    font: { family: "Inter, -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif", size: 12, color: HEC_COLORS.text },
    paper_bgcolor: 'rgba(0,0,0,0)',
    plot_bgcolor: 'rgba(0,0,0,0)',
    colorway: HEC_COLORS.colorway,
    margin: { t: 44, r: 24, b: 44, l: 60 },
    xaxis: { gridcolor: '#F1F5F9', zerolinecolor: '#E2E8F0', gridwidth: 1 },
    yaxis: { gridcolor: '#F1F5F9', zerolinecolor: '#E2E8F0', gridwidth: 1 },
    hoverlabel: { bgcolor: HEC_COLORS.navy, font: { color: 'white', size: 12, family: 'Inter, sans-serif' }, bordercolor: 'transparent' },
};

var DATA = {};
var rendered = { separation: false, fleet: false };

// Page titles for the topbar
var PAGE_TITLES = {
    separation: 'AI-Optimized Separation Process Control',
    fleet: 'Predictive Fleet Routing & Demand Forecasting',
    roadmap: 'Future AI Topics — Use Cases 3-7'
};

// ============================
// DATA LOADING
// ============================
async function loadData() {
    var files = ['separation', 'fleet', 'models', 'financial'];
    for (var i = 0; i < files.length; i++) {
        var resp = await fetch('data/' + files[i] + '.json');
        DATA[files[i]] = await resp.json();
    }
}

// ============================
// FORMATTERS
// ============================
function fmtEur(val) { return '€' + Math.round(val).toLocaleString('en'); }
function fmtPct(val) { return val.toFixed(1) + '%'; }
function fmtNum(val) { return Math.round(val).toLocaleString('en'); }

// ============================
// NAVIGATION
// ============================
function switchPage(pageName) {
    // Hide all pages
    var pages = document.querySelectorAll('.page');
    for (var i = 0; i < pages.length; i++) {
        pages[i].style.display = 'none';
    }

    // Show target page
    var target = document.getElementById('page-' + pageName);
    if (target) {
        target.style.display = 'block';
        target.style.animation = 'none';
        target.offsetHeight; // trigger reflow
        target.style.animation = 'fadeIn 0.3s ease';
    }

    // Update sidebar active state
    var links = document.querySelectorAll('.sidebar-link');
    for (var i = 0; i < links.length; i++) {
        links[i].classList.remove('active');
        if (links[i].getAttribute('data-page') === pageName) {
            links[i].classList.add('active');
        }
    }

    // Update topbar title
    document.getElementById('topbar-title').textContent = PAGE_TITLES[pageName] || '';

    // Render charts on first visit
    if (pageName === 'separation' && !rendered.separation) {
        renderSeparation();
        rendered.separation = true;
    }
    if (pageName === 'fleet' && !rendered.fleet) {
        renderFleet();
        rendered.fleet = true;
    }

    // Trigger resize for Plotly
    window.dispatchEvent(new Event('resize'));

    // Close mobile sidebar
    document.getElementById('sidebar').classList.remove('open');
}

// ============================
// SEPARATION CHARTS
// ============================
function renderSeparation() {
    var d = DATA.separation;
    var m = DATA.models.yield || {};
    var fin = DATA.financial;

    // Header stat
    var headerR2 = document.getElementById('sep-header-r2');
    if (headerR2) headerR2.textContent = (m.r2 || 0).toFixed(3);

    // KPI cards
    document.getElementById('sep-kpis').innerHTML =
        '<div class="kpi-card"><div class="kpi-label">Yield Model R²</div><div class="kpi-value">' + (m.r2 || 0).toFixed(3) + '</div></div>' +
        '<div class="kpi-card accent-navy"><div class="kpi-label">Model MAE</div><div class="kpi-value">' + (m.mae || 0).toFixed(1) + ' pp</div></div>' +
        '<div class="kpi-card accent-success"><div class="kpi-label">Total Batches</div><div class="kpi-value">' + fmtNum(d.total_batches) + '</div></div>' +
        '<div class="kpi-card accent-warning"><div class="kpi-label">Avg Yield</div><div class="kpi-value">' + fmtPct(d.avg_yield) + '</div></div>' +
        '<div class="kpi-card"><div class="kpi-label">Avg Margin/Batch</div><div class="kpi-value">' + fmtEur(d.avg_margin) + '</div></div>' +
        '<div class="kpi-card accent-success"><div class="kpi-label">Quality Pass Rate</div><div class="kpi-value">' + fmtPct(fin.separation.quality_rate) + '</div></div>';

    // AI opportunity
    document.getElementById('sep-opportunity').innerHTML =
        '<div class="money-callout">' +
        '<div class="mc-label">AI Optimization Opportunity — Separation Yield Improvement</div>' +
        '<div class="mc-value">' + fmtEur(fin.ai_opportunity.separation_savings) + ' / year</div>' +
        '<div class="mc-desc">+1.07 pp yield improvement × €1,500/batch × 12,000 batches/year. Each additional 1% yield = €18M/year across all facilities.</div>' +
        '</div>';

    // Feature importance
    var fi = d.feature_importances;
    Plotly.newPlot('chart-feat-imp', [{
        type: 'bar', orientation: 'h',
        y: fi.features.slice().reverse(),
        x: fi.importances.slice().reverse(),
        marker: { color: HEC_COLORS.teal, line: { width: 0 } },
        hovertemplate: '%{y}: %{x:.1%}<extra></extra>'
    }], Object.assign({}, PLOTLY_LAYOUT, {
        title: { text: 'Top 15 Feature Importances', font: { size: 14, color: HEC_COLORS.navy } },
        margin: { t: 44, r: 20, b: 40, l: 180 },
        height: 440
    }), { responsive: true, displayModeBar: false });

    // Actual vs Predicted scatter
    var ps = d.predictions_scatter;
    var minVal = Math.min.apply(null, ps.actual.concat(ps.predicted));
    var maxVal = Math.max.apply(null, ps.actual.concat(ps.predicted));
    Plotly.newPlot('chart-pred-scatter', [
        { type: 'scatter', mode: 'markers', x: ps.actual, y: ps.predicted,
          marker: { color: HEC_COLORS.teal, size: 5, opacity: 0.5 },
          hovertemplate: 'Actual: %{x:.1f}%<br>Predicted: %{y:.1f}%<extra></extra>', name: 'Predictions' },
        { type: 'scatter', mode: 'lines', x: [minVal, maxVal], y: [minVal, maxVal],
          line: { color: HEC_COLORS.navy, dash: 'dash', width: 1.5 }, showlegend: false }
    ], Object.assign({}, PLOTLY_LAYOUT, {
        title: { text: 'Actual vs Predicted Yield (%)', font: { size: 14, color: HEC_COLORS.navy } },
        xaxis: { gridcolor: '#F1F5F9', title: { text: 'Actual Yield (%)', font: { size: 11 } } },
        yaxis: { gridcolor: '#F1F5F9', title: { text: 'Predicted Yield (%)', font: { size: 11 } } },
        height: 400
    }), { responsive: true, displayModeBar: false });

    // Yield by facility
    var boxTraces = Object.entries(d.yield_by_facility).map(function(entry, i) {
        return { type: 'box', y: entry[1], name: entry[0], marker: { color: HEC_COLORS.colorway[i % HEC_COLORS.colorway.length] } };
    });
    Plotly.newPlot('chart-yield-fac', boxTraces, Object.assign({}, PLOTLY_LAYOUT, {
        title: { text: 'Yield Distribution by Facility', font: { size: 14, color: HEC_COLORS.navy } },
        showlegend: false, height: 380
    }), { responsive: true, displayModeBar: false });

    // Quality donut
    var qb = d.quality_breakdown;
    Plotly.newPlot('chart-quality', [{
        type: 'pie', values: [qb.pass, qb.fail], labels: ['Pass (' + qb.pass + ')', 'Fail (' + qb.fail + ')'],
        marker: { colors: [HEC_COLORS.success, HEC_COLORS.danger] },
        hole: 0.55, textinfo: 'label+percent',
        hovertemplate: '%{label}<br>Count: %{value:,}<extra></extra>'
    }], Object.assign({}, PLOTLY_LAYOUT, {
        title: { text: 'Quality Pass/Fail', font: { size: 14, color: HEC_COLORS.navy } },
        height: 380, margin: { t: 44, b: 20, l: 20, r: 20 }
    }), { responsive: true, displayModeBar: false });

    // Monthly yield trend
    var mt = d.monthly_trends;
    Plotly.newPlot('chart-sep-yield-monthly', [{
        type: 'scatter', mode: 'lines+markers',
        x: mt.map(function(r) { return r.processing_date; }),
        y: mt.map(function(r) { return r.avg_yield; }),
        line: { color: HEC_COLORS.teal, width: 2.5 },
        marker: { size: 5, color: HEC_COLORS.teal },
        fill: 'tozeroy', fillcolor: 'rgba(27,154,170,0.05)',
        hovertemplate: '%{x}<br>Avg Yield: %{y:.1f}%<extra></extra>'
    }], Object.assign({}, PLOTLY_LAYOUT, {
        title: { text: 'Monthly Average Yield', font: { size: 14, color: HEC_COLORS.navy } },
        xaxis: { gridcolor: '#F1F5F9' },
        yaxis: { gridcolor: '#F1F5F9', title: { text: 'Yield (%)', font: { size: 11 } } },
        height: 340
    }), { responsive: true, displayModeBar: false });

    // Monthly margin
    Plotly.newPlot('chart-sep-margin-monthly', [{
        type: 'bar',
        x: mt.map(function(r) { return r.processing_date; }),
        y: mt.map(function(r) { return r.total_margin; }),
        marker: { color: HEC_COLORS.navy, line: { width: 0 } },
        hovertemplate: '%{x}<br>Margin: €%{y:,.0f}<extra></extra>'
    }], Object.assign({}, PLOTLY_LAYOUT, {
        title: { text: 'Monthly Total Margin', font: { size: 14, color: HEC_COLORS.navy } },
        xaxis: { gridcolor: '#F1F5F9' },
        yaxis: { gridcolor: '#F1F5F9', title: { text: 'EUR', font: { size: 11 } }, tickprefix: '€' },
        height: 340
    }), { responsive: true, displayModeBar: false });

    // Model summary table
    var qm = DATA.models.quality || {};
    document.getElementById('sep-model-summary').innerHTML =
        '<table class="data-table"><thead><tr><th>Model</th><th>Type</th><th>Metric</th><th>Value</th><th>Train Size</th><th>Test Size</th></tr></thead><tbody>' +
        '<tr><td>Yield Predictor</td><td>XGBoost Regressor</td><td>R²</td><td>' + (m.r2 || 0).toFixed(3) + '</td><td>' + fmtNum(m.train_size || 0) + '</td><td>' + fmtNum(m.test_size || 0) + '</td></tr>' +
        '<tr><td>Quality Classifier</td><td>Random Forest</td><td>Accuracy</td><td>' + fmtPct((qm.accuracy || 0) * 100) + '</td><td>' + fmtNum(m.train_size || 0) + '</td><td>' + fmtNum(m.test_size || 0) + '</td></tr>' +
        '<tr><td>Profitability Model</td><td>XGBoost Regressor</td><td>R²</td><td>' + ((DATA.models.profitability || {}).r2 || 0).toFixed(3) + '</td><td>' + fmtNum((DATA.models.profitability || {}).train_rows || 0) + '</td><td>' + fmtNum((DATA.models.profitability || {}).test_rows || 0) + '</td></tr>' +
        '</tbody></table>';
}

// ============================
// FLEET CHARTS
// ============================
function renderFleet() {
    var d = DATA.fleet;
    var m = DATA.models.demand || {};
    var fin = DATA.financial;

    // Header stat
    var headerR2 = document.getElementById('fleet-header-r2');
    if (headerR2) headerR2.textContent = (m.r2 || 0).toFixed(3);

    // KPIs
    document.getElementById('fleet-kpis').innerHTML =
        '<div class="kpi-card"><div class="kpi-label">Forecast R²</div><div class="kpi-value">' + (m.r2 || 0).toFixed(3) + '</div></div>' +
        '<div class="kpi-card accent-navy"><div class="kpi-label">MAPE</div><div class="kpi-value">' + fmtPct(m.mape || 0) + '</div></div>' +
        '<div class="kpi-card accent-success"><div class="kpi-label">Total Voyages</div><div class="kpi-value">' + fmtNum(d.total_voyages) + '</div></div>' +
        '<div class="kpi-card accent-warning"><div class="kpi-label">Total Revenue</div><div class="kpi-value">' + fmtEur(d.total_revenue) + '</div></div>' +
        '<div class="kpi-card"><div class="kpi-label">Avg Voyage Margin</div><div class="kpi-value">' + fmtEur(d.avg_voyage_margin) + '</div></div>' +
        '<div class="kpi-card accent-navy"><div class="kpi-label">Total Fuel Cost</div><div class="kpi-value">' + fmtEur(d.total_fuel_cost) + '</div></div>';

    // AI opportunity
    document.getElementById('fleet-opportunity').innerHTML =
        '<div class="money-callout">' +
        '<div class="mc-label">AI Optimization Opportunity — Fleet Route Optimization</div>' +
        '<div class="mc-value">' + fmtEur(fin.ai_opportunity.fleet_savings) + ' / year</div>' +
        '<div class="mc-desc">10% fuel cost reduction through optimized routing = ' + fmtEur(fin.ai_opportunity.fleet_savings) + ' annual savings. Additional revenue from better demand-driven scheduling.</div>' +
        '</div>';

    // Demand scatter
    var ds = d.demand_scatter;
    if (ds && ds.actual && ds.actual.length) {
        var mn = Math.min.apply(null, ds.actual.concat(ds.predicted));
        var mx = Math.max.apply(null, ds.actual.concat(ds.predicted));
        Plotly.newPlot('chart-demand-scatter', [
            { type: 'scatter', mode: 'markers', x: ds.actual, y: ds.predicted,
              marker: { color: HEC_COLORS.teal, size: 5, opacity: 0.45 },
              hovertemplate: 'Actual: %{x:.0f} m³<br>Predicted: %{y:.0f} m³<extra></extra>', name: 'Predictions' },
            { type: 'scatter', mode: 'lines', x: [mn, mx], y: [mn, mx],
              line: { color: HEC_COLORS.navy, dash: 'dash', width: 1.5 }, showlegend: false }
        ], Object.assign({}, PLOTLY_LAYOUT, {
            title: { text: 'Demand: Actual vs Predicted (m³)', font: { size: 14, color: HEC_COLORS.navy } },
            xaxis: { gridcolor: '#F1F5F9', title: { text: 'Actual Volume (m³)', font: { size: 11 } } },
            yaxis: { gridcolor: '#F1F5F9', title: { text: 'Predicted Volume (m³)', font: { size: 11 } } },
            height: 400
        }), { responsive: true, displayModeBar: false });
    }

    // Voyage margin by class
    if (d.voyage_by_class && d.voyage_by_class.length) {
        Plotly.newPlot('chart-voyage-class', [{
            type: 'bar',
            x: d.voyage_by_class.map(function(v) { return v.vessel_class; }),
            y: d.voyage_by_class.map(function(v) { return v.avg_margin; }),
            marker: { color: HEC_COLORS.colorway.slice(0, d.voyage_by_class.length), line: { width: 0 } },
            hovertemplate: '%{x}<br>Avg Margin: €%{y:,.0f}<extra></extra>'
        }], Object.assign({}, PLOTLY_LAYOUT, {
            title: { text: 'Avg Voyage Margin by Vessel Class', font: { size: 14, color: HEC_COLORS.navy } },
            yaxis: { gridcolor: '#F1F5F9', title: { text: 'EUR', font: { size: 11 } }, tickprefix: '€' },
            height: 380
        }), { responsive: true, displayModeBar: false });
    }

    // Monthly demand (top 5 ports)
    var md = d.monthly_demand_top5;
    if (md) {
        var traces = Object.entries(md).map(function(entry, i) {
            return {
                type: 'scatter', mode: 'lines', name: entry[0],
                x: entry[1].map(function(r) { return r.date; }),
                y: entry[1].map(function(r) { return r.waste_volume_collected_m3; }),
                line: { color: HEC_COLORS.colorway[i], width: 2 }
            };
        });
        if (traces.length) {
            Plotly.newPlot('chart-demand-monthly', traces, Object.assign({}, PLOTLY_LAYOUT, {
                title: { text: 'Monthly Waste Demand — Top 5 Ports', font: { size: 14, color: HEC_COLORS.navy } },
                xaxis: { gridcolor: '#F1F5F9' },
                yaxis: { gridcolor: '#F1F5F9', title: { text: 'Volume (m³)', font: { size: 11 } } },
                height: 400, legend: { orientation: 'h', y: -0.2, font: { size: 11 } }
            }), { responsive: true, displayModeBar: false });
        }
    }

    // Port metrics table
    if (d.port_metrics && d.port_metrics.length) {
        var rows = d.port_metrics.map(function(p) {
            return '<tr><td>' + p.port_code + '</td><td>' + fmtPct(p.mape) + '</td><td>' + p.rmse.toFixed(1) + '</td><td>' + p.r2.toFixed(3) + '</td><td>' + p.n_samples + '</td></tr>';
        }).join('');
        document.getElementById('fleet-port-table').innerHTML =
            '<table class="data-table"><thead><tr><th>Port</th><th>MAPE</th><th>RMSE (m³)</th><th>R²</th><th>Samples</th></tr></thead><tbody>' + rows + '</tbody></table>';
    }

    // Revenue vs Fuel
    if (d.voyage_by_class && d.voyage_by_class.length) {
        Plotly.newPlot('chart-fleet-economics', [{
            type: 'bar', name: 'Avg Revenue',
            x: d.voyage_by_class.map(function(v) { return v.vessel_class; }),
            y: d.voyage_by_class.map(function(v) { return v.avg_revenue; }),
            marker: { color: HEC_COLORS.teal }
        }, {
            type: 'bar', name: 'Avg Fuel Cost',
            x: d.voyage_by_class.map(function(v) { return v.vessel_class; }),
            y: d.voyage_by_class.map(function(v) { return v.avg_fuel; }),
            marker: { color: HEC_COLORS.danger }
        }], Object.assign({}, PLOTLY_LAYOUT, {
            title: { text: 'Revenue vs Fuel by Vessel Class', font: { size: 14, color: HEC_COLORS.navy } },
            barmode: 'group',
            yaxis: { gridcolor: '#F1F5F9', title: { text: 'EUR', font: { size: 11 } }, tickprefix: '€' },
            height: 380, legend: { orientation: 'h', y: -0.15, font: { size: 11 } }
        }), { responsive: true, displayModeBar: false });
    }

    // Fuel breakdown donut
    Plotly.newPlot('chart-fleet-fuel', [{
        type: 'pie',
        values: [fin.fleet.total_fuel_cost, fin.fleet.total_revenue - fin.fleet.total_fuel_cost],
        labels: ['Fuel Cost', 'Net After Fuel'],
        marker: { colors: [HEC_COLORS.danger, HEC_COLORS.success] },
        hole: 0.55, textinfo: 'label+percent',
        hovertemplate: '%{label}<br>€%{value:,.0f}<extra></extra>'
    }], Object.assign({}, PLOTLY_LAYOUT, {
        title: { text: 'Fleet Revenue Breakdown', font: { size: 14, color: HEC_COLORS.navy } },
        height: 380, margin: { t: 44, b: 20, l: 20, r: 20 }
    }), { responsive: true, displayModeBar: false });
}

// ============================
// INITIALIZATION
// ============================
document.addEventListener('DOMContentLoaded', async function() {
    // Load data
    await loadData();
    document.getElementById('loading').style.display = 'none';
    document.getElementById('pages-container').style.display = 'block';

    // Render initial page
    switchPage('separation');

    // Wire up sidebar navigation
    var links = document.querySelectorAll('.sidebar-link');
    for (var i = 0; i < links.length; i++) {
        links[i].addEventListener('click', function(e) {
            e.preventDefault();
            var page = this.getAttribute('data-page');
            switchPage(page);
        });
    }

    // Mobile toggle
    var toggle = document.getElementById('sidebar-toggle');
    if (toggle) {
        toggle.addEventListener('click', function() {
            document.getElementById('sidebar').classList.toggle('open');
        });
    }
});
