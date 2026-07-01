"""Self-contained HTML rendering for the divergence visualizer.

Takes the per-token analysis records produced by ``divergence_viz`` and turns
them into a single standalone HTML page: the rollout rendered as large monospace
tokens, each tinted with its own hue and given a background opacity proportional
to the Jensen-Shannon divergence between the master's and novice's distribution.
Hovering a token pops up a chart comparing the two distributions.

This module owns every presentation concern (the palette, the opacity mapping and
the page template); it has no dependency on torch or the models.
"""

from __future__ import annotations

import html
import json


# A palette of pleasant, distinct hues (degrees) cycled per token so neighbours
# are visually separable — opacity (not hue) carries the divergence signal.
HUE_PALETTE = [262, 145, 38, 0, 205, 280, 96, 178, 320, 18]

# Background opacity is tweened across this range by normalized divergence, so
# even a zero-divergence token stays visibly tinted.
MIN_ALPHA, MAX_ALPHA = 0.15, 1.0


def _decorate_tokens(tokens: list[dict]) -> None:
    """Add ``hue`` and ``alpha`` to each token record (in place).

    Opacity is the JS divergence normalized across this rollout's generated
    tokens; forced prompt tokens (no ``js``) stay transparent.
    """
    js_values = [t["js"] for t in tokens if "js" in t]
    js_min = min(js_values) if js_values else 0.0
    js_max = max(js_values) if js_values else 0.0
    span = js_max - js_min if js_max > js_min else 1.0
    for t in tokens:
        if "js" in t:
            norm = (t["js"] - js_min) / span
            t["alpha"] = round(MIN_ALPHA + (MAX_ALPHA - MIN_ALPHA) * norm, 3)
        else:
            t["alpha"] = 0.0
        t["hue"] = HUE_PALETTE[t["i"] % len(HUE_PALETTE)]


def render_html(reports: list[dict], model_name: str, rank_fraction: float) -> str:
    """Render one self-contained page holding one or more rollouts.

    Each entry of ``reports`` is ``{"meta": ..., "stats": ..., "tokens": ...}``;
    the page shows a selector to switch between them when there is more than one.
    """
    for report in reports:
        _decorate_tokens(report["tokens"])
    data = {"model": model_name, "rank_fraction": rank_fraction, "reports": reports}
    payload = json.dumps(data, ensure_ascii=False)
    title = html.escape(f"Divergence — {model_name} (rf={rank_fraction})")
    return _HTML_TEMPLATE.replace("/*__DATA__*/", payload).replace("__TITLE__", title)


_HTML_TEMPLATE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>__TITLE__</title>
<style>
  :root {
    --bg: #ffffff; --fg: #1d1d1f; --muted: #86868b; --panel: #fbfbfd;
    --border: #e0e0e5; --master: #2563eb; --novice: #dc2626;
  }
  * { box-sizing: border-box; }
  body {
    margin: 0; background: var(--bg); color: var(--fg);
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    -webkit-font-smoothing: antialiased;
  }
  .wrap { max-width: 1100px; margin: 0 auto; padding: 32px 28px 120px; }
  header h1 { font-size: 15px; font-weight: 600; margin: 0 0 4px; }
  header .sub { color: var(--muted); font-size: 13px; margin-bottom: 24px; }
  .selectbar { display: flex; align-items: center; gap: 10px; margin-bottom: 22px; }
  .selectbar label { font-size: 13px; font-weight: 600; }
  .selectbar select {
    font: inherit; font-size: 13px; padding: 5px 10px; border-radius: 8px;
    border: 1px solid var(--border); background: #fff; color: var(--fg); cursor: pointer;
  }
  .selectbar .muted { color: var(--muted); font-size: 12px; }
  .statbar { display: flex; gap: 40px; margin-bottom: 8px; flex-wrap: wrap; }
  .stat .label { font-size: 13px; font-weight: 600; color: var(--fg); }
  .stat .value { font-size: 30px; font-weight: 600; letter-spacing: -0.5px; }
  .stat .value.small { font-size: 22px; }
  .problem {
    background: var(--panel); border: 1px solid var(--border); border-radius: 12px;
    padding: 14px 18px; font-size: 13px; color: #3a3a3c; margin: 18px 0 6px;
    white-space: pre-wrap; line-height: 1.5;
  }
  .legend { font-size: 12px; color: var(--muted); margin: 14px 2px 6px; display: flex;
            gap: 18px; align-items: center; flex-wrap: wrap; }
  .legend .sw { display: inline-block; width: 11px; height: 11px; border-radius: 3px;
                vertical-align: -1px; margin-right: 5px; }
  .stream {
    background: var(--panel); border: 1px solid var(--border); border-radius: 14px;
    padding: 22px 20px; margin-top: 8px;
    font-family: "SF Mono", ui-monospace, "JetBrains Mono", Menlo, monospace;
    font-size: 21px; line-height: 1.9; white-space: pre-wrap; word-break: break-word;
  }
  .tok {
    border-radius: 4px; padding: 1px 0; cursor: pointer;
    transition: outline-color .1s; outline: 2px solid transparent;
  }
  .tok:hover { outline-color: rgba(0,0,0,.35); }
  .tok.pinned { outline-color: #1d1d1f; }
  .tok.gen-start { box-shadow: -3px 0 0 0 #1d1d1f; }
  .ctx { color: #a1a1a6; }  /* forced prompt tokens: plain context, no interaction */

  /* popover */
  #pop {
    position: absolute; z-index: 50; display: none; width: 340px;
    background: #fff; border: 1px solid var(--border); border-radius: 12px;
    box-shadow: 0 12px 40px rgba(0,0,0,.18); padding: 14px 14px 12px;
    font-family: -apple-system, BlinkMacSystemFont, sans-serif;
  }
  #pop.show { display: block; }
  #pop .phead { font-size: 12px; color: var(--muted); margin-bottom: 2px; }
  #pop .ptok {
    font-family: "SF Mono", ui-monospace, Menlo, monospace; font-size: 17px;
    font-weight: 600; word-break: break-all;
  }
  #pop .pjs { font-size: 12px; color: var(--fg); margin: 6px 0 10px; }
  #pop .pjs b { font-variant-numeric: tabular-nums; }
  #pop .detail {
    min-height: 30px; font-size: 11.5px; color: #3a3a3c; background: var(--panel);
    border: 1px solid var(--border); border-radius: 8px; padding: 6px 8px; margin-bottom: 10px;
    font-variant-numeric: tabular-nums; line-height: 1.45;
  }
  #pop .detail .ph { color: var(--muted); }
  .row { display: grid; grid-template-columns: 92px 1fr; gap: 8px; align-items: center;
         padding: 2px 3px; border-radius: 5px; cursor: default; }
  .row:hover { background: #f0f0f4; }
  .row.chosen .lab { font-weight: 700; }
  .row.chosen .lab::after { content: " ◀"; color: var(--muted); font-weight: 400; }
  .lab { font-family: "SF Mono", ui-monospace, Menlo, monospace; font-size: 12px;
         white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
  .bars { display: flex; flex-direction: column; gap: 2px; }
  .bar { height: 7px; border-radius: 3px; background: #ececf0; position: relative; overflow: hidden; }
  .bar > span { position: absolute; left: 0; top: 0; bottom: 0; border-radius: 3px; }
  .bar.m > span { background: var(--master); }
  .bar.n > span { background: var(--novice); }
  .foot { font-size: 11px; color: var(--muted); margin-top: 8px; text-align: right; }
</style>
</head>
<body>
<div class="wrap">
  <header>
    <h1 id="h-title"></h1>
    <div class="sub" id="h-sub"></div>
  </header>
  <div class="selectbar" id="selectbar">
    <label for="sel">Rollout</label>
    <select id="sel"></select>
    <span class="muted" id="sel-info"></span>
  </div>
  <div class="statbar" id="statbar"></div>
  <div class="problem" id="problem"></div>
  <div class="legend">
    <span><span class="sw" style="background:var(--master)"></span>Master (full rank)</span>
    <span><span class="sw" style="background:var(--novice)"></span>Novice (truncated)</span>
    <span>Opacity ∝ Jensen-Shannon divergence (normalized)</span>
    <span><span class="sw" style="background:#1d1d1f"></span>▌ generation starts</span>
  </div>
  <div class="stream" id="stream"></div>
  <div class="foot">Hover a token for its distribution · click to pin · hover bars for exact values · click outside to close</div>
</div>
<div id="pop">
  <div class="phead">prediction that produced</div>
  <div class="ptok" id="p-tok"></div>
  <div class="pjs" id="p-js"></div>
  <div class="detail" id="p-detail"></div>
  <div id="p-rows"></div>
</div>

<script>
const DATA = /*__DATA__*/;
const pop = document.getElementById('pop');
const elDetail = document.getElementById('p-detail');
const stream = document.getElementById('stream');
let pinned = null, active = 0;

function esc(s){ const d=document.createElement('div'); d.textContent=s; return d.innerHTML; }
function pct(x){ return (x*100).toFixed(x>=0.1?1:2)+'%'; }

// header
document.getElementById('h-title').textContent = 'Master vs Novice — next-token divergence';
document.getElementById('h-sub').textContent =
  DATA.model + '  ·  rank fraction ' + DATA.rank_fraction + '  ·  ' + DATA.reports.length + ' rollout(s)';

// rollout selector
const sel = document.getElementById('sel');
DATA.reports.forEach((R, i) => {
  const o = document.createElement('option');
  o.value = i;
  const m = R.meta;
  o.textContent = `Problem ${m.index}` +
    (m.subject ? ` — ${m.subject}${m.level!=null?` (L${m.level})`:''}` : '');
  sel.appendChild(o);
});
sel.addEventListener('change', () => { unpin(); hidePop(); active = +sel.value; renderReport(); });
if (DATA.reports.length <= 1) document.getElementById('selectbar').style.display = 'none';

function renderReport(){
  const R = DATA.reports[active];
  const S = R.stats;
  document.getElementById('sel-info').textContent =
    R.meta.answer != null ? ('answer: ' + R.meta.answer) : '';
  const stats = [
    ['Tokens', S.tokens, false], ['Characters', S.chars, false],
    ['Generated', S.gen_len, false],
    ['Mean JS (bits)', S.js_mean.toFixed(4), true], ['Max JS', S.js_max.toFixed(4), true],
  ];
  document.getElementById('statbar').innerHTML = stats.map(([l,v,sm]) =>
    `<div class="stat"><div class="label">${l}</div><div class="value${sm?' small':''}">${v}</div></div>`).join('');
  document.getElementById('problem').textContent = R.meta.problem || '(raw prompt)';

  stream.innerHTML = '';
  R.tokens.forEach(tk => {
    const sp = document.createElement('span');
    sp.textContent = tk.t.length ? tk.t : '';
    sp.dataset.i = tk.i;
    if (tk.cand) {
      sp.className = 'tok' + (tk.i === S.prompt_len ? ' gen-start' : '');
      sp.style.background = `hsla(${tk.hue},68%,72%,${tk.alpha})`;
      sp.addEventListener('mouseenter', () => { if (!pinned) showPop(tk, sp); });
      sp.addEventListener('mouseleave', () => { if (!pinned) hidePop(); });
      sp.addEventListener('click', (e) => {
        e.stopPropagation();
        if (pinned === sp) { unpin(); hidePop(); }
        else { unpin(); pinned = sp; sp.classList.add('pinned'); showPop(tk, sp); }
      });
    } else {
      sp.className = 'ctx';  // forced prompt token
    }
    stream.appendChild(sp);
  });
}
renderReport();

function unpin(){ if (pinned){ pinned.classList.remove('pinned'); pinned = null; } }
function hidePop(){ pop.classList.remove('show'); }

function setDetail(r){
  if (!r){ elDetail.innerHTML = '<span class="ph">hover a bar for exact probability, logit &amp; rank</span>'; return; }
  elDetail.innerHTML =
    `<b>"${esc(r.tok)}"</b><br>` +
    `<span style="color:var(--master)">master</span> ${pct(r.pm)} · logit ${r.lm} · rank ${r.rm}<br>` +
    `<span style="color:var(--novice)">novice</span> ${pct(r.pn)} · logit ${r.ln} · rank ${r.rn}`;
}

function showPop(tk, sp){
  document.getElementById('p-tok').textContent = '"' + tk.chosen + '"';
  document.getElementById('p-js').innerHTML = 'JS divergence: <b>' + tk.js.toFixed(4) + '</b> bits';
  setDetail(null);
  const maxp = Math.max(...tk.cand.map(r => Math.max(r.pm, r.pn)), 1e-6);
  const rows = document.getElementById('p-rows');
  rows.innerHTML = '';
  tk.cand.forEach(r => {
    const row = document.createElement('div');
    row.className = 'row' + (r.chosen ? ' chosen' : '');
    row.innerHTML =
      `<div class="lab" title="${esc(r.tok)}">${esc(r.tok)}</div>` +
      `<div class="bars">` +
        `<div class="bar m"><span style="width:${(r.pm/maxp*100).toFixed(1)}%"></span></div>` +
        `<div class="bar n"><span style="width:${(r.pn/maxp*100).toFixed(1)}%"></span></div>` +
      `</div>`;
    row.addEventListener('mouseenter', () => setDetail(r));
    rows.appendChild(row);
  });
  positionPop(sp);
  pop.classList.add('show');
}

function positionPop(sp){
  pop.classList.add('show');
  pop.style.visibility = 'hidden';
  const r = sp.getBoundingClientRect();
  const pw = pop.offsetWidth, ph = pop.offsetHeight;
  let left = window.scrollX + r.left;
  left = Math.min(left, window.scrollX + document.documentElement.clientWidth - pw - 12);
  left = Math.max(left, window.scrollX + 8);
  let top = window.scrollY + r.top - ph - 8;
  if (top < window.scrollY + 4) top = window.scrollY + r.bottom + 8; // flip below
  pop.style.left = left + 'px';
  pop.style.top = top + 'px';
  pop.style.visibility = 'visible';
}

pop.addEventListener('click', e => e.stopPropagation());
pop.addEventListener('mouseleave', () => { if (!pinned) hidePop(); });
document.addEventListener('click', () => { unpin(); hidePop(); });
</script>
</body>
</html>
"""
