/**
 * HEC AI Platform — Static Dashboard (3 tabs)
 * Tab 1: Separation Process Control (UC1)
 * Tab 2: Fleet Routing & Demand (UC2)
 * Tab 3: Future AI Topics (UC3-7) — static HTML, no charts needed
 */

const HEC_COLORS = {
    navy: '#0D1B2A',
    navyLight: '#1B2838',
    teal: '#1B9AAA',
    tealLight: '#23B5C6',
    text: '#0D1B2A',
    textMuted: '#546E7A',
    border: '#E0E4E8',
    success: '#2E7D32',
    warning: '#F57C00',
    danger: '#C62828',
    colorway: ['#1B9AAA', '#0D1B2A', '#F57C00', '#2E7D32', '#7B1FA2', '#C62828', '#1565C0', '#00838F', '#4E342E', '#455A64']
};

const PLOTLY_LAYOUT = {
    font: { family: "-apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif", color: HEC_COLORS.text },
    paper_bgcolor: 'rgba(0,0,0,0)',
    plot_bgcolor: 'rgba(0,0,0,0)',
    colorway: HEC_COLORS.colorway,
    margin: { t: 40, r: 20, b: 40, l: 60 },
    xaxis: { gridcolor: HEC_COLORS.border, zerolinecolor: HEC_COLORS.border },
    yaxis: { gridcolor: HEC_COLORS.border, zerolinecolor: HEC_COLORS.border },
    hoverlabel: { bgcolor: HEC_COLORS.navy, font: { color: 'white', size: 12 } },
};

// --- Data store ---
let DATA = {};

async function loadData() {
    const files = ['separation', 'fleet', 'models', 'financial'];
    for (const name of files) {
        const resp = await fetch('data/' + name + '.json');
        DATA[name] = await resp.json();
    }
}

// --- Formatters ---
function fmtEur(val) { return '€' + Math.round(val).toLocaleString('en'); }
function fmtPct(val) { return val.toFixed(1) + '%'; }
function fmtNum(val) { return Math.round(val).toLocaleString('en'); }

// =========================================================================
// TAB 1: SEPARATION PROCESS CONTROL
// =========================================================================
function renderSeparation() {
    var d = DATA.separation;
    var m = DATA.models.yield || {};
    var fin = DATA.financial;

    // KPI cards — model + financial
    document.getElementById('sep-kpis').innerHTML =
        '<div class="kpi-card"><div class="kpi-label">Yield Model R²</div><div class="kpi-value">' + (m.r2 || 0).toFixed(3) + '</div></div>' +
        '<div class="kpi-card accent-navy"><div class="kpi-label">Model MAE</div><div class="kpi-value">' + (m.mae || 0).toFixed(1) + ' pp</div></div>' +
        '<div class="kpi-card accent-success"><div class="kpi-label">Total Batches</div><div class="kpi-value">' + fmtNum(d.total_batches) + '</div></div>' +
        '<div class="kpi-card accent-warning"><div class="kpi-label">Avg Yield</div><div class="kpi-value">' + fmtPct(d.avg_yield) + '</div></div>' +
        '<div class="kpi-card"><div class="kpi-label">Avg Margin/Batch</div><div class="kpi-value">' + fmtEur(d.avg_margin) + '</div></div>' +
        '<div class="kpi-card accent-success"><div class="kpi-label">Quality Pass Rate</div><div class="kpi-value">' + fmtPct(fin.separation.quality_rate) + '</div></div>';

    // AI opportunity callout
    document.getElementById('sep-opportunity').innerHTML =
        '<div class="money-callout">' +
        '<div class="mc-label">AI Optimization Opportunity — Separation Yield Improvement</div>' +
        '<div class="mc-value">' + fmtEur(fin.ai_opportunity.separation_savings) + ' / year</div>' +
        '<div class="mc-desc">+1.07 pp yield improvement × €1,500/batch × 12,000 batches/year. Each additional 1% yield = €18M/year across all facilities.</div>' +
        '</div>';

    // Feature importance (horizontal bar)
    var fi = d.feature_importances;
    Plotly.newPlot('chart-feat-imp', [{
        type: 'bar', orientation: 'h',
        y: fi.features.slice().reverse(),
        x: fi.importances.slice().reverse(),
        marker: { color: HEC_COLORS.teal },
        hovertemplate: '%{y}: %{x:.1%}<extra></extra>'
    }], Object.assign({}, PLOTLY_LAYOUT, {
        title: 'Top 15 Feature Importances (Yield Model)',
        margin: { t: 40, r: 20, b: 40, l: 200 },
        height: 450
    }), { responsive: true });

    // Actual vs Predicted scatter
    var ps = d.predictions_scatter;
    var minVal = Math.min.apply(null, ps.actual.concat(ps.predicted));
    var maxVal = Math.max.apply(null, ps.actual.concat(ps.predicted));
    Plotly.newPlot('chart-pred-scatter', [
        { type: 'scatter', mode: 'markers', x: ps.actual, y: ps.predicted,
          marker: { color: HEC_COLORS.teal, size: 5, opacity: 0.6 },
          hovertemplate: 'Actual: %{x:.1f}%<br>Predicted: %{y:.1f}%<extra></extra>', name: 'Predictions' },
        { type: 'scatter', mode: 'lines', x: [minVal, maxVal], y: [minVal, maxVal],
          line: { color: HEC_COLORS.navy, dash: 'dash', width: 1 }, showlegend: false, name: 'Perfect' }
    ], Object.assign({}, PLOTLY_LAYOUT, {
        title: 'Actual vs Predicted Yield (%)',
        xaxis: { gridcolor: HEC_COLORS.border, title: 'Actual Yield (%)' },
        yaxis: { gridcolor: HEC_COLORS.border, title: 'Predicted Yield (%)' },
        height: 400
    }), { responsive: true });

    // Yield by facility (box plots)
    var boxTraces = Object.entries(d.yield_by_facility).map(function(entry, i) {
        return { type: 'box', y: entry[1], name: entry[0], marker: { color: HEC_COLORS.colorway[i % HEC_COLORS.colorway.length] } };
    });
    Plotly.newPlot('chart-yield-fac', boxTraces, Object.assign({}, PLOTLY_LAYOUT, {
        title: 'Yield Distribution by Facility', showlegend: false, height: 380
    }), { responsive: true });

    // Quality donut
    var qb = d.quality_breakdown;
    Plotly.newPlot('chart-quality', [{
        type: 'pie', values: [qb.pass, qb.fail], labels: ['Pass (' + qb.pass + ')', 'Fail (' + qb.fail + ')'],
        marker: { colors: [HEC_COLORS.success, HEC_COLORS.danger] },
        hole: 0.5, textinfo: 'label+percent',
        hovertemplate: '%{label}<br>Count: %{value:,}<extra></extra>'
    }], Object.assign({}, PLOTLY_LAYOUT, {
        title: 'Quality Pass/Fail Breakdown', height: 380, margin: { t: 40, b: 20, l: 20, r: 20 }
    }), { responsive: true });

    // Monthly yield trend
    var mt = d.monthly_trends;
    Plotly.newPlot('chart-sep-yield-monthly', [{
        type: 'scatter', mode: 'lines+markers',
        x: mt.map(function(r) { return r.processing_date; }),
        y: mt.map(function(r) { return r.avg_yield; }),
        line: { color: HEC_COLORS.teal, width: 2 }, marker: { size: 4 },
        hovertemplate: '%{x}<br>Avg Yield: %{y:.1f}%<extra></extra>'
    }], Object.assign({}, PLOTLY_LAYOUT, {
        title: 'Monthly Average Yield Trend',
        xaxis: { gridcolor: HEC_COLORS.border, rangeslider: { visible: true } },
        yaxis: { gridcolor: HEC_COLORS.border, title: 'Yield (%)' },
        height: 360
    }), { responsive: true });

    // Monthly margin trend
    Plotly.newPlot('chart-sep-margin-monthly', [{
        type: 'bar',
        x: mt.map(function(r) { return r.processing_date; }),
        y: mt.map(function(r) { return r.total_margin; }),
        marker: { color: HEC_COLORS.navy },
        hovertemplate: '%{x}<br>Total Margin: €%{y:,.0f}<extra></extra>'
    }], Object.assign({}, PLOTLY_LAYOUT, {
        title: 'Monthly Total Margin (€)',
        xaxis: { gridcolor: HEC_COLORS.border },
        yaxis: { gridcolor: HEC_COLORS.border, title: 'EUR', tickprefix: '€' },
        height: 360
    }), { responsive: true });

    // Model summary table
    var qm = DATA.models.quality || {};
    document.getElementById('sep-model-summary').innerHTML =
        '<table class="data-table"><thead><tr><th>Model</th><th>Type</th><th>Metric</th><th>Value</th><th>Train Size</th><th>Test Size</th></tr></thead><tbody>' +
        '<tr><td>Yield Predictor</td><td>XGBoost Regressor</td><td>R²</td><td>' + (m.r2 || 0).toFixed(3) + '</td><td>' + fmtNum(m.train_size || 0) + '</td><td>' + fmtNum(m.test_size || 0) + '</td></tr>' +
        '<tr><td>Quality Classifier</td><td>Random Forest</td><td>Accuracy</td><td>' + fmtPct((qm.accuracy || 0) * 100) + '</td><td>' + fmtNum(m.train_size || 0) + '</td><td>' + fmtNum(m.test_size || 0) + '</td></tr>' +
        '<tr><td>Profitability Model</td><td>XGBoost Regressor</td><td>R²</td><td>' + ((DATA.models.profitability || {}).r2 || 0).toFixed(3) + '</td><td>' + fmtNum((DATA.models.profitability || {}).train_rows || 0) + '</td><td>' + fmtNum((DATA.models.profitability || {}).test_rows || 0) + '</td></tr>' +
        '</tbody></table>';
}

// =========================================================================
// TAB 2: FLEET ROUTING & DEMAND FORECASTING
// =========================================================================
function renderFleet() {
    var d = DATA.fleet;
    var m = DATA.models.demand || {};
    var fin = DATA.financial;

    // KPI cards — model + financial
    document.getElementById('fleet-kpis').innerHTML =
        '<div class="kpi-card"><div class="kpi-label">Forecast R²</div><div class="kpi-value">' + (m.r2 || 0).toFixed(3) + '</div></div>' +
        '<div class="kpi-card accent-navy"><div class="kpi-label">MAPE</div><div class="kpi-value">' + fmtPct(m.mape || 0) + '</div></div>' +
        '<div class="kpi-card accent-success"><div class="kpi-label">Total Voyages</div><div class="kpi-value">' + fmtNum(d.total_voyages) + '</div></div>' +
        '<div class="kpi-card accent-warning"><div class="kpi-label">Total Revenue</div><div class="kpi-value">' + fmtEur(d.total_revenue) + '</div></div>' +
        '<div class="kpi-card"><div class="kpi-label">Avg Voyage Margin</div><div class="kpi-value">' + fmtEur(d.avg_voyage_margin) + '</div></div>' +
        '<div class="kpi-card accent-navy"><div class="kpi-label">Total Fuel Cost</div><div class="kpi-value">' + fmtEur(d.total_fuel_cost) + '</div></div>';

    // AI opportunity callout
    document.getElementById('fleet-opportunity').innerHTML =
        '<div class="money-callout">' +
        '<div class="mc-label">AI Optimization Opportunity — Fleet Route Optimization</div>' +
        '<div class="mc-value">' + fmtEur(fin.ai_opportunity.fleet_savings) + ' / year</div>' +
        '<div class="mc-desc">10% fuel cost reduction through optimized routing = ' + fmtEur(fin.ai_opportunity.fleet_savings) + ' annual savings. Additional revenue from better demand-driven scheduling.</div>' +
        '</div>';

    // Demand scatter
    var ds = d.demand_scatter;
    if (ds.actual.length) {
        var mn = Math.min.apply(null, ds.actual.concat(ds.predicted));
        var mx = Math.max.apply(null, ds.actual.concat(ds.predicted));
        Plotly.newPlot('chart-demand-scatter', [
            { type: 'scatter', mode: 'markers', x: ds.actual, y: ds.predicted,
              marker: { color: HEC_COLORS.teal, size: 5, opacity: 0.5 },
              hovertemplate: 'Actual: %{x:.0f} m³<br>Predicted: %{y:.0f} m³<extra></extra>', name: 'Predictions' },
            { type: 'scatter', mode: 'lines', x: [mn, mx], y: [mn, mx],
              line: { color: HEC_COLORS.navy, dash: 'dash', width: 1 }, showlegend: false }
        ], Object.assign({}, PLOTLY_LAYOUT, {
            title: 'Demand Forecast: Actual vs Predicted (m³)',
            xaxis: { gridcolor: HEC_COLORS.border, title: 'Actual Volume (m³)' },
            yaxis: { gridcolor: HEC_COLORS.border, title: 'Predicted Volume (m³)' },
            height: 400
        }), { responsive: true });
    }

    // Voyage margin by class
    if (d.voyage_by_class && d.voyage_by_class.length) {
        Plotly.newPlot('chart-voyage-class', [{
            type: 'bar',
            x: d.voyage_by_class.map(function(v) { return v.vessel_class; }),
            y: d.voyage_by_class.map(function(v) { return v.avg_margin; }),
            marker: { color: HEC_COLORS.colorway.slice(0, d.voyage_by_class.length) },
            hovertemplate: '%{x}<br>Avg Margin: €%{y:,.0f}<extra></extra>'
        }], Object.assign({}, PLOTLY_LAYOUT, {
            title: 'Average Voyage Margin by Vessel Class',
            yaxis: { gridcolor: HEC_COLORS.border, title: 'EUR', tickprefix: '€' },
            height: 380
        }), { responsive: true });
    }

    // Monthly demand (top 5 ports)
    var md = d.monthly_demand_top5;
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
            title: 'Monthly Waste Demand — Top 5 Ports',
            xaxis: { gridcolor: HEC_COLORS.border, rangeslider: { visible: true } },
            yaxis: { gridcolor: HEC_COLORS.border, title: 'Volume (m³)' },
            height: 400, legend: { orientation: 'h', y: -0.25 }
        }), { responsive: true });
    }

    // Port metrics table
    if (d.port_metrics && d.port_metrics.length) {
        var rows = d.port_metrics.map(function(p) {
            return '<tr><td>' + p.port_code + '</td><td>' + fmtPct(p.mape) + '</td><td>' + p.rmse.toFixed(1) + '</td><td>' + p.r2.toFixed(3) + '</td><td>' + p.n_samples + '</td></tr>';
        }).join('');
        document.getElementById('fleet-port-table').innerHTML =
            '<table class="data-table"><thead><tr><th>Port Code</th><th>MAPE</th><th>RMSE (m³)</th><th>R²</th><th>Test Samples</th></tr></thead><tbody>' + rows + '</tbody></table>';
    }

    // Fleet economics — revenue vs fuel bar
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
            title: 'Revenue vs Fuel Cost by Vessel Class',
            barmode: 'group',
            yaxis: { gridcolor: HEC_COLORS.border, title: 'EUR', tickprefix: '€' },
            height: 380
        }), { responsive: true });
    }

    // Fuel breakdown donut
    Plotly.newPlot('chart-fleet-fuel', [{
        type: 'pie',
        values: [fin.fleet.total_fuel_cost, fin.fleet.total_revenue - fin.fleet.total_fuel_cost],
        labels: ['Fuel Cost', 'Net After Fuel'],
        marker: { colors: [HEC_COLORS.danger, HEC_COLORS.success] },
        hole: 0.5, textinfo: 'label+percent',
        hovertemplate: '%{label}<br>€%{value:,.0f}<extra></extra>'
    }], Object.assign({}, PLOTLY_LAYOUT, {
        title: 'Fleet Revenue Breakdown', height: 380, margin: { t: 40, b: 20, l: 20, r: 20 }
    }), { responsive: true });
}

// =========================================================================
// TAB SWITCHING & INIT
// =========================================================================
function initTabs() {
    var rendered = { separation: false, fleet: false, roadmap: true };

    document.querySelectorAll('[data-bs-toggle="tab"]').forEach(function(tab) {
        tab.addEventListener('shown.bs.tab', function(e) {
            var target = e.target.getAttribute('data-bs-target').replace('#tab-', '');
            if (!rendered[target]) {
                if (target === 'separation') renderSeparation();
                if (target === 'fleet') renderFleet();
                rendered[target] = true;
            }
            window.dispatchEvent(new Event('resize'));
        });
    });

    // Render first tab immediately
    renderSeparation();
    rendered.separation = true;
}

document.addEventListener('DOMContentLoaded', async function() {
    document.getElementById('loading').style.display = 'block';
    await loadData();
    document.getElementById('loading').style.display = 'none';
    document.getElementById('main-content').style.display = 'block';
    initTabs();
});
