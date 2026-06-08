/**
 * HEC AI Platform — Dashboard Charts (Plotly.js)
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

// --- Data loading ---
let DATA = {};

async function loadData() {
    const files = ['separation', 'fleet', 'models', 'financial'];
    for (const name of files) {
        const resp = await fetch(`data/${name}.json`);
        DATA[name] = await resp.json();
    }
}

// --- Formatters ---
function fmtEur(val) { return '€' + Math.round(val).toLocaleString('en'); }
function fmtPct(val) { return val.toFixed(1) + '%'; }
function fmtNum(val) { return Math.round(val).toLocaleString('en'); }

// --- Tab 1: Separation ---
function renderSeparation() {
    const d = DATA.separation;
    const m = DATA.models.yield || {};

    // KPIs
    document.getElementById('sep-kpis').innerHTML = `
        <div class="kpi-card"><div class="kpi-label">Model R²</div><div class="kpi-value">${(m.r2 || 0).toFixed(3)}</div></div>
        <div class="kpi-card accent-navy"><div class="kpi-label">MAE</div><div class="kpi-value">${(m.mae || 0).toFixed(1)}</div></div>
        <div class="kpi-card accent-success"><div class="kpi-label">Total Batches</div><div class="kpi-value">${fmtNum(d.total_batches)}</div></div>
        <div class="kpi-card accent-warning"><div class="kpi-label">Avg Yield</div><div class="kpi-value">${fmtPct(d.avg_yield)}</div></div>
    `;

    // Feature importance
    const fi = d.feature_importances;
    Plotly.newPlot('chart-feat-imp', [{
        type: 'bar', orientation: 'h',
        y: fi.features.slice().reverse(),
        x: fi.importances.slice().reverse(),
        marker: { color: HEC_COLORS.teal },
        hovertemplate: '%{y}: %{x:.1%}<extra></extra>'
    }], { ...PLOTLY_LAYOUT, title: 'Top 15 Feature Importances (Yield Model)', margin: { ...PLOTLY_LAYOUT.margin, l: 200 }, height: 420 }, { responsive: true });

    // Actual vs Predicted
    const ps = d.predictions_scatter;
    const minVal = Math.min(...ps.actual, ...ps.predicted);
    const maxVal = Math.max(...ps.actual, ...ps.predicted);
    Plotly.newPlot('chart-pred-scatter', [
        { type: 'scatter', mode: 'markers', x: ps.actual, y: ps.predicted, marker: { color: HEC_COLORS.teal, size: 5, opacity: 0.6 }, hovertemplate: 'Actual: %{x:.1f}%<br>Predicted: %{y:.1f}%<extra></extra>' },
        { type: 'scatter', mode: 'lines', x: [minVal, maxVal], y: [minVal, maxVal], line: { color: HEC_COLORS.navy, dash: 'dash', width: 1 }, showlegend: false }
    ], { ...PLOTLY_LAYOUT, title: 'Actual vs Predicted Yield (%)', xaxis: { ...PLOTLY_LAYOUT.xaxis, title: 'Actual' }, yaxis: { ...PLOTLY_LAYOUT.yaxis, title: 'Predicted' }, height: 380 }, { responsive: true });

    // Yield by facility (box plots)
    const boxTraces = Object.entries(d.yield_by_facility).map(([fac, vals], i) => ({
        type: 'box', y: vals, name: fac, marker: { color: HEC_COLORS.colorway[i % HEC_COLORS.colorway.length] }
    }));
    Plotly.newPlot('chart-yield-fac', boxTraces, { ...PLOTLY_LAYOUT, title: 'Yield Distribution by Facility', showlegend: false, height: 360 }, { responsive: true });

    // Quality donut
    const qb = d.quality_breakdown;
    Plotly.newPlot('chart-quality', [{
        type: 'pie', values: [qb.pass, qb.fail], labels: ['Pass', 'Fail'],
        marker: { colors: [HEC_COLORS.success, HEC_COLORS.danger] },
        hole: 0.5, textinfo: 'label+percent',
        hovertemplate: '%{label}: %{value:,}<extra></extra>'
    }], { ...PLOTLY_LAYOUT, title: 'Quality Pass/Fail', height: 340, margin: { t: 40, b: 20, l: 20, r: 20 } }, { responsive: true });

    // Monthly trend
    const mt = d.monthly_trends;
    Plotly.newPlot('chart-sep-monthly', [{
        type: 'scatter', mode: 'lines+markers',
        x: mt.map(r => r.processing_date), y: mt.map(r => r.avg_yield),
        line: { color: HEC_COLORS.teal, width: 2 }, marker: { size: 4 },
        hovertemplate: '%{x}<br>Avg Yield: %{y:.1f}%<extra></extra>'
    }], { ...PLOTLY_LAYOUT, title: 'Monthly Average Yield Trend', xaxis: { ...PLOTLY_LAYOUT.xaxis, rangeslider: { visible: true } }, yaxis: { ...PLOTLY_LAYOUT.yaxis, title: 'Yield (%)' }, height: 360 }, { responsive: true });
}

// --- Tab 2: Fleet ---
function renderFleet() {
    const d = DATA.fleet;
    const m = DATA.models.demand || {};

    // KPIs
    document.getElementById('fleet-kpis').innerHTML = `
        <div class="kpi-card"><div class="kpi-label">Forecast R²</div><div class="kpi-value">${(m.r2 || 0).toFixed(3)}</div></div>
        <div class="kpi-card accent-navy"><div class="kpi-label">MAPE</div><div class="kpi-value">${fmtPct(m.mape || 0)}</div></div>
        <div class="kpi-card accent-success"><div class="kpi-label">Total Voyages</div><div class="kpi-value">${fmtNum(d.total_voyages)}</div></div>
        <div class="kpi-card accent-warning"><div class="kpi-label">Avg Voyage Margin</div><div class="kpi-value">${fmtEur(d.avg_voyage_margin)}</div></div>
    `;

    // Port metrics table
    if (d.port_metrics && d.port_metrics.length) {
        let rows = d.port_metrics.map(p => `<tr><td>${p.port_code}</td><td>${fmtPct(p.mape)}</td><td>${p.rmse.toFixed(1)}</td><td>${p.r2.toFixed(3)}</td><td>${p.n_samples}</td></tr>`).join('');
        document.getElementById('fleet-port-table').innerHTML = `
            <table class="data-table"><thead><tr><th>Port</th><th>MAPE</th><th>RMSE</th><th>R²</th><th>Samples</th></tr></thead><tbody>${rows}</tbody></table>
        `;
    }

    // Demand scatter
    const ds = d.demand_scatter;
    if (ds.actual.length) {
        const mn = Math.min(...ds.actual, ...ds.predicted);
        const mx = Math.max(...ds.actual, ...ds.predicted);
        Plotly.newPlot('chart-demand-scatter', [
            { type: 'scatter', mode: 'markers', x: ds.actual, y: ds.predicted, marker: { color: HEC_COLORS.teal, size: 5, opacity: 0.5 }, hovertemplate: 'Actual: %{x:.0f} m³<br>Predicted: %{y:.0f} m³<extra></extra>' },
            { type: 'scatter', mode: 'lines', x: [mn, mx], y: [mn, mx], line: { color: HEC_COLORS.navy, dash: 'dash', width: 1 }, showlegend: false }
        ], { ...PLOTLY_LAYOUT, title: 'Demand: Actual vs Predicted (m³)', xaxis: { ...PLOTLY_LAYOUT.xaxis, title: 'Actual' }, yaxis: { ...PLOTLY_LAYOUT.yaxis, title: 'Predicted' }, height: 380 }, { responsive: true });
    }

    // Voyage by class
    if (d.voyage_by_class && d.voyage_by_class.length) {
        Plotly.newPlot('chart-voyage-class', [{
            type: 'bar',
            x: d.voyage_by_class.map(v => v.vessel_class),
            y: d.voyage_by_class.map(v => v.avg_margin),
            marker: { color: HEC_COLORS.colorway.slice(0, d.voyage_by_class.length) },
            hovertemplate: '%{x}<br>Avg Margin: €%{y:,.0f}<extra></extra>'
        }], { ...PLOTLY_LAYOUT, title: 'Average Voyage Margin by Vessel Class', yaxis: { ...PLOTLY_LAYOUT.yaxis, title: 'EUR', tickprefix: '€' }, height: 360 }, { responsive: true });
    }

    // Monthly demand (top 5 ports)
    const md = d.monthly_demand_top5;
    const traces = Object.entries(md).map(([port, data], i) => ({
        type: 'scatter', mode: 'lines', name: port,
        x: data.map(r => r.date), y: data.map(r => r.waste_volume_collected_m3),
        line: { color: HEC_COLORS.colorway[i], width: 2 }
    }));
    if (traces.length) {
        Plotly.newPlot('chart-demand-monthly', traces, {
            ...PLOTLY_LAYOUT, title: 'Monthly Demand — Top 5 Ports',
            xaxis: { ...PLOTLY_LAYOUT.xaxis, rangeslider: { visible: true } },
            yaxis: { ...PLOTLY_LAYOUT.yaxis, title: 'Volume (m³)' },
            height: 380, legend: { orientation: 'h', y: -0.25 }
        }, { responsive: true });
    }
}

// --- Tab 3: Financial ---
function renderFinancial() {
    const d = DATA.financial;

    // Big callout
    document.getElementById('fin-opportunity').innerHTML = `
        <div class="money-callout">
            <div class="mc-label">Estimated Annual AI Optimization Opportunity</div>
            <div class="mc-value">${fmtEur(d.ai_opportunity.total)}</div>
            <div class="mc-desc">Separation yield improvement (${fmtEur(d.ai_opportunity.separation_savings)}) + Fleet route optimization (${fmtEur(d.ai_opportunity.fleet_savings)})</div>
        </div>
    `;

    // KPIs
    document.getElementById('fin-kpis').innerHTML = `
        <div class="kpi-card"><div class="kpi-label">Avg Yield</div><div class="kpi-value">${fmtPct(d.separation.avg_yield)}</div></div>
        <div class="kpi-card accent-navy"><div class="kpi-label">Avg Margin/Batch</div><div class="kpi-value">${fmtEur(d.separation.avg_margin)}</div></div>
        <div class="kpi-card accent-success"><div class="kpi-label">Fleet Revenue</div><div class="kpi-value">${fmtEur(d.fleet.total_revenue)}</div></div>
        <div class="kpi-card accent-warning"><div class="kpi-label">Total Fuel Cost</div><div class="kpi-value">${fmtEur(d.fleet.total_fuel_cost)}</div></div>
    `;

    // Monthly combined margin
    const sepM = d.monthly_separation;
    const fltM = d.monthly_fleet;
    Plotly.newPlot('chart-fin-monthly', [
        { type: 'scatter', mode: 'lines+markers', name: 'Separation Margin', x: sepM.map(r => r.date), y: sepM.map(r => r.separation_margin), line: { color: HEC_COLORS.navy, width: 2 }, marker: { size: 4 } },
        { type: 'scatter', mode: 'lines+markers', name: 'Fleet Margin', x: fltM.map(r => r.date), y: fltM.map(r => r.fleet_margin), line: { color: HEC_COLORS.teal, width: 2 }, marker: { size: 4 } }
    ], {
        ...PLOTLY_LAYOUT, title: 'Monthly Margin Trend',
        xaxis: { ...PLOTLY_LAYOUT.xaxis, rangeslider: { visible: true } },
        yaxis: { ...PLOTLY_LAYOUT.yaxis, title: 'EUR', tickprefix: '€', separatethousands: true },
        legend: { orientation: 'h', y: 1.1 }, height: 420
    }, { responsive: true });

    // Savings breakdown donut
    Plotly.newPlot('chart-fin-breakdown', [{
        type: 'pie', values: [d.ai_opportunity.separation_savings, d.ai_opportunity.fleet_savings],
        labels: ['Separation Yield Improvement', 'Fleet Route Optimization'],
        marker: { colors: [HEC_COLORS.navy, HEC_COLORS.teal] },
        hole: 0.5, textinfo: 'label+percent',
        hovertemplate: '%{label}<br>€%{value:,.0f}<extra></extra>'
    }], { ...PLOTLY_LAYOUT, title: 'AI Savings Breakdown', height: 340, margin: { t: 40, b: 20, l: 20, r: 20 } }, { responsive: true });
}

// --- Tab switching ---
function initTabs() {
    const rendered = { separation: false, fleet: false, financial: false, roadmap: false };

    document.querySelectorAll('[data-bs-toggle="tab"]').forEach(tab => {
        tab.addEventListener('shown.bs.tab', (e) => {
            const target = e.target.getAttribute('data-bs-target').replace('#tab-', '');
            if (!rendered[target]) {
                if (target === 'separation') renderSeparation();
                if (target === 'fleet') renderFleet();
                if (target === 'financial') renderFinancial();
                rendered[target] = true;
            }
            // Resize plotly charts on tab switch
            window.dispatchEvent(new Event('resize'));
        });
    });

    // Render first tab
    renderSeparation();
    rendered.separation = true;
}

// --- Init ---
document.addEventListener('DOMContentLoaded', async () => {
    document.getElementById('loading').style.display = 'block';
    await loadData();
    document.getElementById('loading').style.display = 'none';
    document.getElementById('main-content').style.display = 'block';
    initTabs();
});
