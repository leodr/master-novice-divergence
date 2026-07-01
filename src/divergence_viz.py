"""Master-vs-Novice next-token divergence visualizer.

The **master** (the full-rank base model) owns the rollout: it greedily (or by
sampling) generates a continuation of a prompt. For every position in the
resulting sequence we then compare the master's next-token distribution against
the **novice's** (the rank-truncated disentangled model scoring the *same*
tokens). The two are identical everywhere except the MLP layers, which the
novice reconstructs at a low rank — so the divergence isolates what the lost MLP
capacity was doing.

Output is a single self-contained HTML file: the rollout rendered as large
monospace tokens, each token tinted with its own hue (so adjacent tokens are
easy to tell apart) and given a background opacity proportional to the
Jensen-Shannon divergence between the master's and novice's distribution for the
prediction that produced it. Hovering a token pops up a chart comparing the two
distributions over the top candidate tokens; clicking pins the popover so you
can hover its bars for exact probabilities / logits / ranks.

Models are loaded **sequentially** (master generates + is scored, is freed, then
the novice scores the fixed sequence) so peak memory stays near a single model.

Example::

    uv run python src/divergence_viz.py \
        --novice-dir novice-qwen3.5-4b-rf0.1 \
        --problem-indices 0,1,2 --max-new-tokens 256

Multiple ``--problem-indices`` are scored in one novice load and written into a
single HTML file (switch between them with the in-page selector). Reports land
in ``reports/`` (gitignored) by default.
"""

from __future__ import annotations

import argparse
import gc
import html
import json
import os
import time

import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import StoppingCriteria, StoppingCriteriaList

try:
    from .novice_model import NoviceModel, MANIFEST_NAME
except ImportError:
    from novice_model import NoviceModel, MANIFEST_NAME


# A palette of pleasant, distinct hues (degrees) cycled per token so neighbours
# are visually separable — opacity (not hue) carries the divergence signal.
HUE_PALETTE = [262, 145, 38, 0, 205, 280, 96, 178, 320, 18]

# Background opacity is tweened across this range by normalized divergence, so
# even a zero-divergence token stays visibly tinted.
MIN_ALPHA, MAX_ALPHA = 0.15, 1.0


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def resolve_device(spec: str) -> str:
    if spec != "auto":
        return spec
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def resolve_dtype(spec: str) -> torch.dtype:
    return {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[spec]


def free_memory(device: str) -> None:
    gc.collect()
    if device == "cuda":
        torch.cuda.empty_cache()
    elif device == "mps":
        torch.mps.empty_cache()


class TokenProgress(StoppingCriteria):
    """Ticks a tqdm bar once per generated token (never stops generation)."""

    def __init__(self, total: int, *, desc: str) -> None:
        self.bar = tqdm(
            total=total, desc=desc, unit="tok", leave=True, dynamic_ncols=True, mininterval=1.0
        )

    def __call__(self, input_ids, scores, **kwargs) -> bool:
        self.bar.update(1)
        return False

    def close(self) -> None:
        self.bar.close()


def incremental_pieces(tokenizer, ids: list[int]) -> list[str]:
    """The exact substring each token contributes to the decoded text.

    Decoding the growing prefix and diffing is the standard way to recover
    per-token display strings (handles byte-level BPE spaces / partial UTF-8).
    """
    pieces, prev = [], ""
    for i in range(len(ids)):
        cur = tokenizer.decode(
            ids[: i + 1], skip_special_tokens=False, clean_up_tokenization_spaces=False
        )
        pieces.append(cur[len(prev):])
        prev = cur
    return pieces


def label_token(tokenizer, token_id: int) -> str:
    """A readable single-token label with visible whitespace markers."""
    s = tokenizer.decode([token_id], skip_special_tokens=False, clean_up_tokenization_spaces=False)
    return s.replace("\n", "↵").replace("\t", "⇥").replace(" ", "·")


def js_divergence_bits(p: np.ndarray, q: np.ndarray) -> float:
    """Jensen-Shannon divergence in bits (base-2), bounded in [0, 1]."""
    m = 0.5 * (p + q)

    def _kl(a: np.ndarray) -> float:
        mask = a > 0
        return float(np.sum(a[mask] * np.log2(a[mask] / m[mask])))

    return 0.5 * _kl(p) + 0.5 * _kl(q)


def candidate_rows(
    p_master: np.ndarray,
    p_novice: np.ndarray,
    logit_master: np.ndarray,
    logit_novice: np.ndarray,
    chosen_id: int,
    tokenizer,
    top_k: int,
) -> list[dict]:
    """Union of each model's top-k tokens (plus the chosen one), with per-model
    probability, logit and rank — sorted by master probability."""
    top_m = np.argpartition(p_master, -top_k)[-top_k:]
    top_n = np.argpartition(p_novice, -top_k)[-top_k:]
    ids = sorted(set(top_m.tolist()) | set(top_n.tolist()) | {chosen_id})
    rows = []
    for i in ids:
        rows.append(
            {
                "id": int(i),
                "tok": label_token(tokenizer, i),
                "pm": float(p_master[i]),
                "pn": float(p_novice[i]),
                "lm": round(float(logit_master[i]), 2),
                "ln": round(float(logit_novice[i]), 2),
                "rm": int((p_master > p_master[i]).sum()) + 1,
                "rn": int((p_novice > p_novice[i]).sum()) + 1,
                "chosen": i == chosen_id,
            }
        )
    rows.sort(key=lambda r: r["pm"], reverse=True)
    return rows


def softmax_np(logits: torch.Tensor) -> np.ndarray:
    """Row-wise softmax of a [T, V] logit tensor -> float32 numpy [T, V]."""
    return torch.softmax(logits.float(), dim=-1).numpy()


# --------------------------------------------------------------------------- #
# generation + scoring
# --------------------------------------------------------------------------- #
def build_input_ids(tokenizer, problem: str, *, use_chat: bool, think: bool) -> tuple[torch.Tensor, str]:
    """Return (input_ids[1, P], rendered_prompt_text)."""
    if use_chat and tokenizer.chat_template:
        messages = [{"role": "user", "content": problem}]
        try:
            text = tokenizer.apply_chat_template(
                messages, add_generation_prompt=True, tokenize=False, enable_thinking=think
            )
        except TypeError:
            text = tokenizer.apply_chat_template(
                messages, add_generation_prompt=True, tokenize=False
            )
        # The template already emits all special tokens, so don't add more.
        ids = tokenizer(text, return_tensors="pt", add_special_tokens=False).input_ids
        return ids, text
    ids = tokenizer(problem, return_tensors="pt").input_ids
    return ids, problem


@torch.no_grad()
def rollout_and_score(model, input_ids: torch.Tensor, *, max_new_tokens: int, do_sample: bool,
                      pad_token_id: int | None, eos_token_id: int | None,
                      temperature: float, top_p: float, top_k_sample: int, min_p: float,
                      repetition_penalty: float, device: str,
                      progress_desc: str) -> tuple[torch.Tensor, int, np.ndarray, torch.Tensor]:
    """Generate with ``model`` and score the full sequence in one forward pass.

    Returns (full_ids[T], prompt_len, probs[T, V], logits[T, V]) on CPU.
    """
    input_ids = input_ids.to(device)
    prompt_len = input_ids.shape[1]
    attention_mask = torch.ones_like(input_ids, device=device)
    gen_kwargs = dict(max_new_tokens=max_new_tokens, num_beams=1, do_sample=do_sample,
                      pad_token_id=pad_token_id, eos_token_id=eos_token_id,
                      repetition_penalty=repetition_penalty)
    if do_sample:
        gen_kwargs.update(temperature=temperature, top_p=top_p, top_k=top_k_sample, min_p=min_p)
    progress = TokenProgress(max_new_tokens, desc=progress_desc)
    try:
        generated = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            stopping_criteria=StoppingCriteriaList([progress]),
            **gen_kwargs,
        )
    finally:
        progress.close()
    full_ids = generated[0].detach().cpu()

    logits = model(generated).logits[0].detach().to("cpu")
    probs = softmax_np(logits)
    return full_ids, prompt_len, probs, logits


@torch.no_grad()
def score_sequence(model, full_ids: torch.Tensor, *, device: str) -> tuple[np.ndarray, torch.Tensor]:
    """Teacher-forced forward of a fixed sequence -> (probs[T, V], logits[T, V]) on CPU."""
    logits = model(full_ids.unsqueeze(0).to(device)).logits[0].detach().to("cpu")
    return softmax_np(logits), logits


# --------------------------------------------------------------------------- #
# HTML rendering
# --------------------------------------------------------------------------- #
def build_token_data(
    tokenizer,
    full_ids: torch.Tensor,
    prompt_len: int,
    master_probs: np.ndarray,
    novice_probs: np.ndarray,
    master_logits: torch.Tensor,
    novice_logits: torch.Tensor,
    top_k: int,
) -> tuple[list[dict], dict]:
    """Per-token visualization records + summary stats."""
    ids = full_ids.tolist()
    pieces = incremental_pieces(tokenizer, ids)
    master_logits_np = master_logits.float().numpy()
    novice_logits_np = novice_logits.float().numpy()

    entries: list[dict] = []
    js_values: list[float] = []
    for j in range(len(ids)):
        entry = {"i": j, "t": pieces[j], "gen": j >= prompt_len}
        # Only the generated tokens are real decisions the master sampled; prompt
        # tokens are forced inputs, so we leave them as plain context (no color /
        # no popover) and don't compute divergence for them.
        if j >= prompt_len:
            pos = j - 1  # the distribution that predicted token j
            js = js_divergence_bits(master_probs[pos], novice_probs[pos])
            entry["js"] = round(js, 5)
            entry["chosen"] = label_token(tokenizer, ids[j])
            entry["cand"] = candidate_rows(
                master_probs[pos], novice_probs[pos],
                master_logits_np[pos], novice_logits_np[pos],
                ids[j], tokenizer, top_k,
            )
            js_values.append(js)
        entries.append(entry)

    js_arr = np.array(js_values) if js_values else np.array([0.0])
    js_min, js_max = float(js_arr.min()), float(js_arr.max())
    span = js_max - js_min if js_max > js_min else 1.0
    for entry in entries:
        if "js" in entry:
            norm = (entry["js"] - js_min) / span
            entry["alpha"] = round(MIN_ALPHA + (MAX_ALPHA - MIN_ALPHA) * norm, 3)
        else:
            entry["alpha"] = 0.0
        entry["hue"] = HUE_PALETTE[entry["i"] % len(HUE_PALETTE)]

    stats = {
        "tokens": len(ids),
        "chars": sum(len(p) for p in pieces),
        "prompt_len": prompt_len,
        "gen_len": len(ids) - prompt_len,
        "js_mean": round(float(js_arr.mean()), 5),
        "js_max": round(js_max, 5),
    }
    return entries, stats


def render_html(reports: list[dict], model_name: str, rank_fraction: float) -> str:
    """Render one self-contained page holding one or more rollouts.

    Each entry of ``reports`` is ``{"meta": ..., "stats": ..., "tokens": ...}``;
    the page shows a selector to switch between them when there is more than one.
    """
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


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--novice-dir", required=True, help="Sharded novice checkpoint dir (has manifest.pt).")
    parser.add_argument("--problem-indices", default="0", help="Comma-separated MATH-500 indices.")
    parser.add_argument("--dataset", default="HuggingFaceH4/MATH-500")
    parser.add_argument("--split", default="test")
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--rank-fraction", type=float, default=None, help="Override the manifest rank.")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    parser.add_argument("--sample", action="store_true", help="Sample instead of greedy decoding.")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--sample-top-k", type=int, default=20,
                        help="Generation top_k; named separately from --top-k, which controls HTML candidates.")
    parser.add_argument("--min-p", type=float, default=0.0)
    parser.add_argument("--repetition-penalty", type=float, default=1.0)
    parser.add_argument("--no-chat", action="store_true", help="Feed the raw problem text (skip chat template).")
    parser.add_argument("--think", action="store_true", help="Enable thinking mode if the chat template supports it.")
    parser.add_argument("--out", default="reports",
                        help="Output .html file, or a directory to write a timestamped report into "
                             "(default: reports/, which is gitignored).")
    args = parser.parse_args()

    device = resolve_device(args.device)
    dtype = resolve_dtype(args.dtype)
    indices = [int(x) for x in args.problem_indices.split(",") if x.strip() != ""]
    print(f"device={device} dtype={args.dtype} problems={indices}", flush=True)

    manifest = torch.load(os.path.join(args.novice_dir, MANIFEST_NAME), map_location="cpu", weights_only=False)
    model_name = manifest["model_name"]
    rank_fraction = args.rank_fraction if args.rank_fraction is not None else manifest["rank_fraction"]

    from datasets import load_dataset
    ds = load_dataset(args.dataset, split=args.split)
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    use_chat = (not args.no_chat) and bool(tokenizer.chat_template)
    print(f"model={model_name} rank_fraction={rank_fraction} chat_template={'yes' if use_chat else 'no'}", flush=True)
    print(f"decoding={'sampling' if args.sample else 'greedy'} temperature={args.temperature} "
          f"top_p={args.top_p} top_k={args.sample_top_k} min_p={args.min_p} "
          f"repetition_penalty={args.repetition_penalty}",
          flush=True)

    # ---- phase A: master generates + scores each rollout, then is freed ----
    print("Loading master (full rank) ...", flush=True)
    t0 = time.perf_counter()
    master = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True, dtype=dtype).to(device).eval()
    print(f"  loaded in {time.perf_counter()-t0:.0f}s", flush=True)

    runs: list[dict] = []
    for idx in tqdm(
        indices, desc="master rollouts", unit="problem", dynamic_ncols=True, mininterval=1.0
    ):
        problem = ds[idx]["problem"]
        meta = {
            "index": idx, "problem": problem,
            "subject": ds[idx].get("subject"), "level": ds[idx].get("level"),
            "answer": ds[idx].get("answer"),
        }
        input_ids, _ = build_input_ids(tokenizer, problem, use_chat=use_chat, think=args.think)
        pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
        t0 = time.perf_counter()
        full_ids, prompt_len, m_probs, m_logits = rollout_and_score(
            master, input_ids, max_new_tokens=args.max_new_tokens, do_sample=args.sample,
            pad_token_id=pad_id, eos_token_id=tokenizer.eos_token_id, device=device,
            temperature=args.temperature, top_p=args.top_p, top_k_sample=args.sample_top_k,
            min_p=args.min_p, repetition_penalty=args.repetition_penalty,
            progress_desc=f"master p{idx}",
        )
        gen_n = full_ids.shape[0] - prompt_len
        print(f"[problem {idx}] generated {gen_n} tokens in {time.perf_counter()-t0:.0f}s "
              f"(seq len {full_ids.shape[0]})", flush=True)
        runs.append({"meta": meta, "full_ids": full_ids, "prompt_len": prompt_len,
                     "m_probs": m_probs.astype(np.float16), "m_logits": m_logits})

    del master
    free_memory(device)

    # ---- phase B: novice scores each fixed sequence ----
    print("Loading novice (truncated) ...", flush=True)
    t0 = time.perf_counter()
    novice = NoviceModel.from_sharded_checkpoint(args.novice_dir, rank_fraction=rank_fraction, device=device)
    novice.model.to(dtype=dtype).eval()
    print(f"  loaded in {time.perf_counter()-t0:.0f}s", flush=True)

    reports = []
    for run in tqdm(
        runs, desc="novice scoring", unit="problem", dynamic_ncols=True, mininterval=1.0
    ):
        n_probs, n_logits = score_sequence(novice, run["full_ids"], device=device)
        entries, stats = build_token_data(
            tokenizer, run["full_ids"], run["prompt_len"],
            run["m_probs"].astype(np.float32), n_probs,
            run["m_logits"], n_logits, args.top_k,
        )
        reports.append({"meta": run["meta"], "stats": stats, "tokens": entries})
        print(f"[problem {run['meta']['index']}] mean JS {stats['js_mean']:.4f} bits, "
              f"max {stats['js_max']:.4f}", flush=True)

    del novice
    free_memory(device)

    # One page holds every rollout (switchable via the in-page selector).
    if args.out.endswith(".html"):
        path = args.out
        out_dir = os.path.dirname(path)
    else:
        out_dir = args.out
        idxs = "-".join(str(r["meta"]["index"]) for r in reports)
        path = os.path.join(out_dir, f"divergence_p{idxs}_{time.strftime('%Y%m%d_%H%M%S')}.html")
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    with open(path, "w", encoding="utf-8") as fh:
        fh.write(render_html(reports, model_name, rank_fraction))
    print(f"Done. Wrote {len(reports)} rollout(s) -> {path}", flush=True)


if __name__ == "__main__":
    main()
