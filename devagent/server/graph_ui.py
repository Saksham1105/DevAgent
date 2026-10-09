"""On-demand graph visualisation UI — v2.

The HTML is built in memory and served at GET / by fastapi_app.py.
Nothing is ever written to disk — the visualisation exists only while
the server is running and disappears automatically when it stops.
"""

from __future__ import annotations

# Placeholder replaced at serve-time with the real origin.
_PLACEHOLDER = "___BASE_URL___"

# Regular string (not f-string) so JS template literals need no escaping.
_TEMPLATE = """\
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width,initial-scale=1">
  <title>DevAgent — Session Graph</title>
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap" rel="stylesheet">
  <script src="https://cdnjs.cloudflare.com/ajax/libs/d3/7.9.0/d3.min.js"></script>
  <style>
    /* ── design tokens ───────────────────────────────────────────── */
    :root {
      --bg:          #0a0d14;
      --surface:     #111520;
      --surface2:    #161d2e;
      --border:      #1e2d45;
      --text:        #f1f5f9;
      --text2:       #94a3b8;
      --muted:       #475569;
      --accent:      #6366f1;
      --accent-dim:  rgba(99,102,241,.15);
      --running:     #10b981;
      --running-dim: rgba(16,185,129,.12);
      --failed:      #f43f5e;
      --failed-dim:  rgba(244,63,94,.12);
      --done-text:   #64748b;
    }

    /* ── reset ───────────────────────────────────────────────────── */
    *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
    a { color: var(--accent); text-decoration: none; }
    a:hover { text-decoration: underline; }

    /* ── base ────────────────────────────────────────────────────── */
    body {
      background: var(--bg);
      color: var(--text);
      font-family: 'Inter', system-ui, -apple-system, sans-serif;
      font-size: 13px;
      line-height: 1.5;
      height: 100vh;
      display: flex;
      flex-direction: column;
      overflow: hidden;
    }

    /* ── header ──────────────────────────────────────────────────── */
    header {
      height: 48px;
      background: var(--surface);
      border-bottom: 1px solid var(--border);
      display: flex;
      align-items: center;
      padding: 0 16px;
      gap: 14px;
      flex-shrink: 0;
      z-index: 30;
    }
    .logo {
      display: flex;
      align-items: center;
      gap: 8px;
      font-weight: 700;
      font-size: 14px;
      letter-spacing: -.3px;
    }
    .logo-mark {
      width: 26px; height: 26px;
      background: linear-gradient(135deg, #6366f1, #8b5cf6);
      border-radius: 7px;
      display: flex; align-items: center; justify-content: center;
      font-size: 13px; line-height: 1;
    }
    .hdr-sep { flex: 1; }
    .live-chip {
      display: flex; align-items: center; gap: 6px;
      padding: 4px 10px;
      border-radius: 20px;
      font-size: 11px; font-weight: 600;
      background: var(--running-dim);
      color: var(--running);
      border: 1px solid rgba(16,185,129,.25);
      cursor: default;
      transition: background .2s, color .2s, border-color .2s;
    }
    .live-chip.disconnected {
      background: var(--failed-dim);
      color: var(--failed);
      border-color: rgba(244,63,94,.35);
    }
    .pulse-dot {
      width: 6px; height: 6px; border-radius: 50%;
      background: var(--running);
      animation: pulse 2s ease-in-out infinite;
    }
    .live-chip.disconnected .pulse-dot {
      background: var(--failed);
      animation: none;
    }
    @keyframes pulse { 0%,100%{opacity:1;transform:scale(1)} 50%{opacity:.35;transform:scale(.75)} }
    #conn-banner {
      display: none;
      align-items: center;
      justify-content: space-between;
      gap: 8px;
      margin-bottom: 10px;
      padding: 6px 10px;
      border-radius: 6px;
      font-size: 11px;
      background: var(--failed-dim);
      border: 1px solid rgba(244,63,94,.35);
      color: var(--failed);
    }
    #conn-banner .retry-btn {
      background: transparent;
      border: 1px solid rgba(244,63,94,.5);
      color: var(--failed);
      border-radius: 4px;
      padding: 2px 7px;
      font-size: 10px;
      font-weight: 600;
      cursor: pointer;
      line-height: 1.2;
    }
    #conn-banner .retry-btn:hover {
      background: rgba(244,63,94,.15);
    }
    .hdr-countdown { font-size: 11px; color: var(--muted); min-width: 52px; text-align: right; }
    .hdr-link {
      font-size: 11px; color: var(--text2);
      padding: 4px 8px; border-radius: 4px;
      transition: background .15s, color .15s;
    }
    .hdr-link:hover { background: var(--surface2); color: var(--text); text-decoration: none; }

    /* ── main layout ─────────────────────────────────────────────── */
    main { flex: 1; display: flex; overflow: hidden; }

    /* ── graph area ──────────────────────────────────────────────── */
    #graph-area { flex: 1; position: relative; overflow: hidden; }
    #graph-area svg { width: 100%; height: 100%; display: block; }

    /* ── filter chips ────────────────────────────────────────────── */
    #filters {
      position: absolute; top: 14px; left: 14px;
      display: flex; gap: 6px; z-index: 10;
    }
    .chip {
      padding: 4px 12px; border-radius: 20px;
      font-size: 11px; font-weight: 500;
      border: 1px solid var(--border);
      background: var(--surface); color: var(--text2);
      cursor: pointer; transition: all .15s; user-select: none;
    }
    .chip:hover { border-color: var(--accent); color: var(--text); }
    .chip.active { background: var(--accent-dim); border-color: var(--accent); color: var(--accent); }

    /* ── tooltip ─────────────────────────────────────────────────── */
    #tooltip {
      position: fixed; pointer-events: none; z-index: 100;
      background: var(--surface2);
      border: 1px solid var(--border);
      border-radius: 10px; padding: 12px 14px;
      min-width: 200px; display: none;
      box-shadow: 0 12px 32px rgba(0,0,0,.55);
      backdrop-filter: blur(10px);
      -webkit-backdrop-filter: blur(10px);
    }
    .tt-title { font-weight: 600; font-size: 13px; margin-bottom: 3px; }
    .tt-id    { color: var(--muted); font-size: 10px; font-family: ui-monospace,monospace; margin-bottom: 8px; }
    .tt-row   { display: flex; justify-content: space-between; gap: 16px; margin-top: 3px; font-size: 12px; }
    .tt-key   { color: var(--text2); }
    .tt-val   { font-weight: 500; }
    .tt-footer{ margin-top: 10px; font-size: 10px; color: var(--muted); }

    /* ── status badges ───────────────────────────────────────────── */
    .sbadge {
      display: inline-flex; align-items: center; gap: 4px;
      padding: 2px 8px; border-radius: 10px;
      font-size: 10px; font-weight: 600;
      text-transform: uppercase; letter-spacing: .4px;
    }
    .sbadge.running   { background: var(--running-dim);          color: var(--running); }
    .sbadge.completed { background: rgba(71,85,105,.2);           color: var(--done-text); }
    .sbadge.failed    { background: var(--failed-dim);            color: var(--failed); }

    /* ── empty state ─────────────────────────────────────────────── */
    #empty {
      position: absolute; top: 50%; left: 50%;
      transform: translate(-50%,-50%);
      text-align: center; color: var(--muted);
      display: none;
    }
    #empty .empty-icon {
      width: 52px; height: 52px; border-radius: 14px;
      background: var(--surface2); border: 1px solid var(--border);
      display: flex; align-items: center; justify-content: center;
      margin: 0 auto 14px; font-size: 24px;
    }
    #empty h3 { font-size: 14px; color: var(--text2); margin-bottom: 6px; }
    #empty p  { font-size: 12px; }
    #empty code {
      font-family: ui-monospace,monospace; color: var(--accent);
      background: var(--accent-dim); padding: 1px 6px; border-radius: 3px;
    }

    /* ── sidebar ─────────────────────────────────────────────────── */
    aside {
      width: 264px; flex-shrink: 0;
      background: var(--surface);
      border-left: 1px solid var(--border);
      display: flex; flex-direction: column;
      overflow-y: auto;
    }
    .s-section { padding: 14px 16px; border-bottom: 1px solid var(--border); }
    .s-section:last-child { border-bottom: none; }
    .s-title {
      font-size: 10px; font-weight: 600; color: var(--muted);
      text-transform: uppercase; letter-spacing: .7px; margin-bottom: 10px;
    }

    /* stat cards */
    .stat-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 8px; }
    .stat-card {
      background: var(--surface2);
      border: 1px solid var(--border);
      border-radius: 8px; padding: 10px 12px;
    }
    .stat-lbl { font-size: 10px; color: var(--muted); margin-bottom: 5px; font-weight: 500; }
    .stat-val { font-size: 19px; font-weight: 700; line-height: 1; color: var(--text); }
    .stat-val.accent  { color: var(--accent); }
    .stat-val.running { color: var(--running); }

    /* sparkline */
    #sparkline { height: 44px; width: 100%; display: block; }

    /* model rows */
    .model-row {
      padding: 5px 0;
      border-bottom: 1px solid rgba(30,45,69,.5);
      font-size: 11px;
    }
    .model-row:last-child { border-bottom: none; }
    .model-top { display: flex; justify-content: space-between; align-items: baseline; margin-bottom: 3px; }
    .model-name  { color: var(--text2); overflow: hidden; text-overflow: ellipsis; white-space: nowrap; max-width: 170px; }
    .model-toks  { color: var(--text); font-weight: 500; flex-shrink: 0; margin-left: 6px; }
    .bar-track   { height: 2px; background: var(--border); border-radius: 1px; }
    .bar-fill    { height: 2px; background: var(--accent); border-radius: 1px; transition: width .4s; }

    /* legend */
    .legend-item { display: flex; align-items: center; gap: 8px; padding: 3px 0; font-size: 12px; color: var(--text2); }
    .l-swatch    { width: 10px; height: 10px; border-radius: 50%; flex-shrink: 0; }
    .legend-hint { font-size: 10px; color: var(--muted); margin-top: 10px; line-height: 1.7; }

    /* ── D3 node helpers ─────────────────────────────────────────── */
    .node { cursor: pointer; }
    .node .ring-circle { fill: none; }
    .node .bg-circle   { transition: r .3s; }
    .node .lbl         { pointer-events: none; font-family: 'Inter',system-ui,sans-serif; }
    .node.selected .bg-circle { stroke-width: 2.5 !important; }
  </style>
</head>
<body>

<!-- ── header ────────────────────────────────────────────────────────── -->
<header>
  <div class="logo">
    <div class="logo-mark">&#11041;</div>
    DevAgent
  </div>
  <div class="hdr-sep"></div>
  <div class="live-chip" id="live-chip">
    <span class="pulse-dot"></span>
    <span id="live-label">Live</span>
  </div>
  <span class="hdr-countdown" id="hdr-cd"></span>
  <a class="hdr-link" href="___BASE_URL___/api/docs" target="_blank">API docs &#8599;</a>
</header>

<!-- ── main ──────────────────────────────────────────────────────────── -->
<main>
  <!-- graph -->
  <div id="graph-area">
    <div id="filters">
      <div class="chip active" data-filter="all">All</div>
      <div class="chip" data-filter="running">Running</div>
      <div class="chip" data-filter="completed">Completed</div>
      <div class="chip" data-filter="failed">Failed</div>
    </div>
    <svg>
      <defs>
        <filter id="glow-r" x="-60%" y="-60%" width="220%" height="220%">
          <feGaussianBlur in="SourceGraphic" stdDeviation="5" result="b"/>
          <feMerge><feMergeNode in="b"/><feMergeNode in="SourceGraphic"/></feMerge>
        </filter>
        <filter id="glow-f" x="-60%" y="-60%" width="220%" height="220%">
          <feGaussianBlur in="SourceGraphic" stdDeviation="3.5" result="b"/>
          <feMerge><feMergeNode in="b"/><feMergeNode in="SourceGraphic"/></feMerge>
        </filter>
      </defs>
    </svg>
    <div id="empty">
      <div class="empty-icon">&#128202;</div>
      <h3>No sessions yet</h3>
      <p>Run <code>devagent run</code> to create one</p>
    </div>
  </div>

  <!-- sidebar -->
  <aside>
    <div class="s-section">
      <div class="s-title">Overview</div>
      <div id="conn-banner">
        <span id="conn-banner-msg">Cannot reach server</span>
        <button class="retry-btn" id="conn-retry-btn" type="button">Retry</button>
      </div>
      <div class="stat-grid">
        <div class="stat-card"><div class="stat-lbl">Active</div><div class="stat-val running" id="m-active">&#8212;</div></div>
        <div class="stat-card"><div class="stat-lbl">Total</div><div class="stat-val" id="m-total">&#8212;</div></div>
        <div class="stat-card"><div class="stat-lbl">Tokens 24h</div><div class="stat-val accent" id="m-tokens">&#8212;</div></div>
        <div class="stat-card"><div class="stat-lbl">Cost 24h</div><div class="stat-val" id="m-cost">&#8212;</div></div>
      </div>
    </div>

    <div class="s-section">
      <div class="s-title">Tokens / hour (24h)</div>
      <svg id="sparkline"></svg>
    </div>

    <div class="s-section">
      <div class="s-title">By model</div>
      <div id="m-models"><div style="color:var(--muted);font-size:11px">Loading&hellip;</div></div>
    </div>

    <div class="s-section">
      <div class="s-title">Legend</div>
      <div class="legend-item"><div class="l-swatch" style="background:var(--running)"></div>Running</div>
      <div class="legend-item"><div class="l-swatch" style="background:var(--done-text)"></div>Completed</div>
      <div class="legend-item"><div class="l-swatch" style="background:var(--failed)"></div>Failed</div>
      <div class="legend-hint">
        Node size &#x221d; tokens used<br>
        Drag nodes to reposition<br>
        Scroll / pinch to zoom<br>
        Hover for details &middot; R to refresh<br>
        Esc to dismiss tooltip
      </div>
    </div>
  </aside>
</main>

<!-- floating tooltip -->
<div id="tooltip"></div>

<script>
(function () {
"use strict";

const BASE = "___BASE_URL___";

// ── state ─────────────────────────────────────────────────────────────
let allSessions = [];
let filter      = "all";
let isPinned    = false;
let countdown   = 30;

// ── formatters ────────────────────────────────────────────────────────
const fmt     = n => n >= 1e6 ? (n/1e6).toFixed(1)+'M' : n >= 1e3 ? (n/1e3).toFixed(1)+'k' : String(n||0);
const fmtCost = c => (!c || c < 1e-4) ? '$0' : '$' + c.toFixed(c < 0.01 ? 4 : 2);

// ── node colour helpers ───────────────────────────────────────────────
function nodeStroke(s) {
  if (s === 'running')   return 'var(--running)';
  if (s === 'failed')    return 'var(--failed)';
  return 'var(--done-text)';
}
function nodeFill(s) {
  if (s === 'running')   return 'rgba(16,185,129,.14)';
  if (s === 'failed')    return 'rgba(244,63,94,.12)';
  return 'rgba(100,116,139,.12)';
}
function nodeFilter(s) {
  if (s === 'running')  return 'url(#glow-r)';
  if (s === 'failed')   return 'url(#glow-f)';
  return null;
}

// ── rScale ────────────────────────────────────────────────────────────
const rScale = d3.scaleSqrt().domain([0, 80000]).range([10, 40]).clamp(true);

// ── SVG / zoom ────────────────────────────────────────────────────────
const svg  = d3.select('#graph-area svg');
const W    = () => svg.node().clientWidth  || 600;
const H    = () => svg.node().clientHeight || 400;

const root  = svg.append('g');
const linkG = root.append('g').attr('class', 'links');
const nodeG = root.append('g').attr('class', 'nodes');

const zoomer = d3.zoom().scaleExtent([0.08, 8])
  .on('zoom', e => root.attr('transform', e.transform));
svg.call(zoomer);

// ── simulation ────────────────────────────────────────────────────────
const sim = d3.forceSimulation()
  .force('link',      d3.forceLink().id(d => d.id).distance(150))
  .force('charge',    d3.forceManyBody().strength(-340))
  .force('center',    d3.forceCenter(W()/2, H()/2))
  .force('collision', d3.forceCollide(d => rScale(d.tokens||0) + 12))
  .force('x',         d3.forceX(W()/2).strength(0.03))
  .force('y',         d3.forceY(H()/2).strength(0.03));

sim.on('tick', () => {
  nodeG.selectAll('.node').attr('transform', d => `translate(${d.x||0},${d.y||0})`);
  linkG.selectAll('line')
    .attr('x1', d => d.source.x).attr('y1', d => d.source.y)
    .attr('x2', d => d.target.x).attr('y2', d => d.target.y);
});

window.addEventListener('resize', () => {
  sim.force('center', d3.forceCenter(W()/2, H()/2))
     .force('x', d3.forceX(W()/2).strength(0.03))
     .force('y', d3.forceY(H()/2).strength(0.03))
     .restart();
});

// ── render graph ──────────────────────────────────────────────────────
function render() {
  const sess = filter === 'all'
    ? allSessions
    : allSessions.filter(s => s.status === filter);

  document.getElementById('empty').style.display = sess.length ? 'none' : 'block';

  // update live chip
  if (!connState.hasError) {
    const nRun = allSessions.filter(s => s.status === 'running').length;
    document.getElementById('live-label').textContent =
      nRun ? `${nRun} running` : 'Live';
  }

  const nodes = sess.map(s => ({
    id:      s.id,
    label:   (s.title || s.id || '').slice(0, 14),
    tokens:  (s.token_input||0) + (s.token_output||0),
    status:  s.status || 'completed',
    model:   s.model  || '—',
    cost:    s.cost_usd || 0,
    updated: s.updated_at,
  }));

  // preserve positions across updates
  const pos = {};
  nodeG.selectAll('.node').each(d => { pos[d.id] = {x: d.x, y: d.y}; });
  nodes.forEach(n => { if (pos[n.id]) { n.x = pos[n.id].x; n.y = pos[n.id].y; } });

  // data join
  const sel = nodeG.selectAll('.node').data(nodes, d => d.id);

  const ent = sel.enter().append('g').attr('class', 'node')
    .attr('transform', `translate(${W()/2},${H()/2})`)
    .style('opacity', 0)
    .call(d3.drag()
      .on('start', (e, d) => { if (!e.active) sim.alphaTarget(0.3).restart(); d.fx = d.x; d.fy = d.y; })
      .on('drag',  (e, d) => { d.fx = e.x;   d.fy = e.y; })
      .on('end',   (e, d) => { if (!e.active) sim.alphaTarget(0); d.fx = null; d.fy = null; })
    )
    .on('click',     (_, d) => pinTooltip(d))
    .on('mouseover', (e, d) => { if (!isPinned) showTip(e, d); })
    .on('mousemove', e       => { if (!isPinned) moveTip(e); })
    .on('mouseout',  ()      => { if (!isPinned) hideTip(); });

  ent.append('circle').attr('class', 'ring-circle');
  ent.append('circle').attr('class', 'bg-circle');
  ent.append('text').attr('class', 'lbl').attr('text-anchor', 'middle');

  ent.transition().duration(350).style('opacity', 1);

  sel.exit().transition().duration(250).style('opacity', 0).remove();

  const all = nodeG.selectAll('.node');

  all.select('.ring-circle')
    .attr('r',      d => rScale(d.tokens) + 7)
    .attr('stroke', d => d.status === 'running' ? 'var(--running)' : 'none')
    .attr('stroke-width', 1.5)
    .attr('stroke-opacity', 0);   // animated via RAF

  all.select('.bg-circle')
    .attr('r',      d => rScale(d.tokens))
    .attr('fill',   d => nodeFill(d.status))
    .attr('stroke', d => nodeStroke(d.status))
    .attr('stroke-width', 1.5)
    .attr('filter', d => nodeFilter(d.status));

  all.select('.lbl')
    .attr('y',         d => rScale(d.tokens) + 14)
    .attr('font-size', '10')
    .attr('fill',      'var(--text2)')
    .text(d => d.label);

  all.classed('is-running', d => d.status === 'running');

  sim.nodes(nodes);
  sim.force('link').links([]);
  sim.alpha(0.35).restart();
}

// ── running ring animation (RAF) ──────────────────────────────────────
let ringPhase = 0;
(function animRings() {
  ringPhase += 0.04;
  const opacity = 0.18 + 0.22 * Math.abs(Math.sin(ringPhase));
  const extra   = 2 + 2 * Math.abs(Math.sin(ringPhase));
  nodeG.selectAll('.node.is-running .ring-circle')
    .attr('r',              d => rScale(d.tokens) + 6 + extra)
    .attr('stroke-opacity', opacity);
  requestAnimationFrame(animRings);
}());

// ── tooltip ───────────────────────────────────────────────────────────
const tipEl = document.getElementById('tooltip');

function statusBadge(s) {
  const cls = s === 'running' ? 'running' : s === 'failed' ? 'failed' : 'completed';
  return `<span class="sbadge ${cls}">${s}</span>`;
}

function tipHTML(d) {
  const tok  = (d.tokens || 0).toLocaleString();
  const cost = fmtCost(d.cost || 0);
  const when = d.updated ? new Date(d.updated * 1000).toLocaleTimeString() : '—';
  const mid  = d.id ? d.id.slice(0, 20) + '&hellip;' : '—';
  return `
    <div class="tt-title">${d.label || d.id}</div>
    <div class="tt-id">${mid}</div>
    <div style="margin-bottom:8px">${statusBadge(d.status)}</div>
    <div class="tt-row"><span class="tt-key">Model</span><span class="tt-val">${(d.model||'—').slice(0,24)}</span></div>
    <div class="tt-row"><span class="tt-key">Tokens</span><span class="tt-val">${tok}</span></div>
    <div class="tt-row"><span class="tt-key">Cost</span><span class="tt-val">${cost}</span></div>
    <div class="tt-row"><span class="tt-key">Updated</span><span class="tt-val">${when}</span></div>
    <div class="tt-footer">
      <a href="${BASE}/api/v1/sessions/${d.id}" target="_blank">View in API &#8599;</a>
    </div>`;
}

function showTip(e, d) {
  tipEl.style.pointerEvents = 'none';
  tipEl.innerHTML = tipHTML(d);
  tipEl.style.display = 'block';
  moveTip(e);
}
function moveTip(e) {
  const tw = tipEl.offsetWidth, th = tipEl.offsetHeight;
  let x = e.clientX + 14, y = e.clientY - 10;
  if (x + tw > window.innerWidth  - 12) x = e.clientX - tw - 14;
  if (y + th > window.innerHeight - 12) y = window.innerHeight - th - 12;
  tipEl.style.left = x + 'px'; tipEl.style.top = y + 'px';
  tipEl.style.bottom = '';
}
function hideTip() { tipEl.style.display = 'none'; }

function pinTooltip(d) {
  isPinned = true;
  tipEl.style.pointerEvents = 'auto';
  tipEl.innerHTML = tipHTML(d) +
    `<div style="margin-top:10px;cursor:pointer;font-size:11px;color:var(--muted)"
          onclick="window.__unpinTip()">&#10005; close</div>`;
  tipEl.style.left = '16px'; tipEl.style.top = ''; tipEl.style.bottom = '14px';
  tipEl.style.display = 'block';
}
window.__unpinTip = () => {
  isPinned = false;
  tipEl.style.pointerEvents = 'none';
  tipEl.style.display = 'none';
  tipEl.style.bottom = '';
};

// ── sparkline ─────────────────────────────────────────────────────────
function renderSparkline(hourly) {
  const sp = d3.select('#sparkline');
  const w  = sp.node().clientWidth || 232;
  const h  = 44;
  sp.selectAll('*').remove();
  if (!hourly || hourly.length < 2) {
    sp.append('text').attr('x', w/2).attr('y', 24).attr('text-anchor', 'middle')
      .attr('fill', 'var(--muted)').attr('font-size', 11).text('No data yet');
    return;
  }
  const vals = hourly.map(d => d.tokens || 0);
  const xS = d3.scaleLinear().domain([0, vals.length - 1]).range([0, w]);
  const yS = d3.scaleLinear().domain([0, d3.max(vals) || 1]).range([h - 2, 4]);

  const area = d3.area()
    .x((_, i) => xS(i)).y0(h).y1(v => yS(v)).curve(d3.curveCatmullRom);
  const line = d3.line()
    .x((_, i) => xS(i)).y(v => yS(v)).curve(d3.curveCatmullRom);

  const gid = 'sg' + Math.random().toString(36).slice(2);
  const grad = sp.append('defs').append('linearGradient')
    .attr('id', gid).attr('gradientUnits', 'userSpaceOnUse')
    .attr('x1', 0).attr('y1', 0).attr('x2', 0).attr('y2', h);
  grad.append('stop').attr('offset', '0%').attr('stop-color', '#6366f1').attr('stop-opacity', .45);
  grad.append('stop').attr('offset', '100%').attr('stop-color', '#6366f1').attr('stop-opacity', 0);

  sp.append('path').datum(vals).attr('d', area).attr('fill', `url(#${gid})`);
  sp.append('path').datum(vals).attr('d', line)
    .attr('fill', 'none').attr('stroke', '#6366f1').attr('stroke-width', 1.5);
}

// ── connection state ──────────────────────────────────────────────────
const connState = {
  metricsError: null,
  sessionsError: null,
  get hasError() { return Boolean(this.metricsError || this.sessionsError); },
  get errorMessage() { return this.metricsError || this.sessionsError || ''; }
};

function updateConnUI() {
  const chip = document.getElementById('live-chip');
  const label = document.getElementById('live-label');
  const banner = document.getElementById('conn-banner');
  const bannerMsg = document.getElementById('conn-banner-msg');

  if (connState.hasError) {
    if (chip) {
      chip.classList.add('disconnected');
      chip.title = connState.errorMessage;
    }
    if (label) label.textContent = 'Disconnected';
    if (banner) banner.style.display = 'flex';
    if (bannerMsg) bannerMsg.textContent = connState.errorMessage;
  } else {
    if (chip) {
      chip.classList.remove('disconnected');
      chip.title = '';
    }
    if (banner) banner.style.display = 'none';
    const nRun = allSessions.filter(s => s.status === 'running').length;
    if (label) label.textContent = nRun ? `${nRun} running` : 'Live';
  }
}

function setEndpointStatus(endpoint, err) {
  if (endpoint === 'metrics') {
    connState.metricsError = err;
  } else if (endpoint === 'sessions') {
    connState.sessionsError = err;
  }
  updateConnUI();
}

async function refreshAll() {
  await Promise.all([loadSessions(), loadMetrics()]);
}

// ── sidebar metrics ───────────────────────────────────────────────────
async function loadMetrics() {
  try {
    const r = await fetch(BASE + '/api/v1/metrics?period=24h');
    if (!r.ok) {
      throw new Error(`Metrics request failed (HTTP ${r.status})`);
    }
    const d = await r.json();
    document.getElementById('m-active').textContent = d.active_sessions ?? '—';
    document.getElementById('m-total').textContent  = d.sessions_count  ?? '—';
    document.getElementById('m-tokens').textContent = fmt(d.total_tokens || 0);
    document.getElementById('m-cost').textContent   = fmtCost(d.total_cost_usd || 0);

    renderSparkline(d.by_hour || []);

    const maxT = Math.max(1, ...((d.by_model || []).map(m => m.tokens || 0)));
    document.getElementById('m-models').innerHTML =
      (d.by_model || []).slice(0, 7).map(m => {
        const pct = Math.round((m.tokens || 0) / maxT * 100);
        return `
          <div class="model-row">
            <div class="model-top">
              <span class="model-name">${m.model || '—'}</span>
              <span class="model-toks">${fmt(m.tokens || 0)}</span>
            </div>
            <div class="bar-track"><div class="bar-fill" style="width:${pct}%"></div></div>
          </div>`;
      }).join('') || '<div style="color:var(--muted);font-size:11px">No model data yet</div>';

    setEndpointStatus('metrics', null);
  } catch (err) {
    const msg = err && err.message ? err.message : 'Cannot reach server';
    console.error('loadMetrics error:', err);
    setEndpointStatus('metrics', msg);
  }
}

// ── sessions ──────────────────────────────────────────────────────────
async function loadSessions() {
  try {
    const r = await fetch(BASE + '/api/v1/sessions?limit=80');
    if (!r.ok) {
      throw new Error(`Sessions request failed (HTTP ${r.status})`);
    }
    const d = await r.json();
    allSessions = d.sessions || [];
    setEndpointStatus('sessions', null);
    render();
  } catch (err) {
    const msg = err && err.message ? err.message : 'Cannot reach server';
    console.error('loadSessions error:', err);
    setEndpointStatus('sessions', msg);
  }
}

// ── retry button ──────────────────────────────────────────────────────
const retryBtn = document.getElementById('conn-retry-btn');
if (retryBtn) {
  retryBtn.addEventListener('click', () => {
    refreshAll();
  });
}

// ── filter chips ──────────────────────────────────────────────────────
document.querySelectorAll('.chip').forEach(c => {
  c.addEventListener('click', () => {
    filter = c.dataset.filter;
    document.querySelectorAll('.chip').forEach(x => x.classList.remove('active'));
    c.classList.add('active');
    render();
  });
});

// ── countdown ─────────────────────────────────────────────────────────
const cdEl = document.getElementById('hdr-cd');
setInterval(() => {
  countdown--;
  if (countdown <= 0) { countdown = 30; refreshAll(); }
  cdEl.textContent = '↻ ' + countdown + 's';
}, 1000);

// ── keyboard ──────────────────────────────────────────────────────────
document.addEventListener('keydown', e => {
  if (e.target.tagName === 'INPUT') return;
  if (e.key === 'r' || e.key === 'R') { countdown = 1; refreshAll(); }
  if (e.key === 'Escape') { window.__unpinTip(); }
  if (e.key === 'f' || e.key === 'F') {
    svg.transition().duration(420).call(
      zoomer.transform, d3.zoomIdentity.translate(W()/2, H()/2)
    );
  }
});

// ── boot ──────────────────────────────────────────────────────────────
refreshAll();
}());
</script>
</body>
</html>
"""


def build_html(host: str = "127.0.0.1", port: int = 7331) -> str:
    """Return the graph UI HTML with the correct API base URL injected."""
    base = f"http://{host}:{port}"
    return _TEMPLATE.replace(_PLACEHOLDER, base)
