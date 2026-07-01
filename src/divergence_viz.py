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
import os
import time

import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import LogitsProcessorList
from transformers import StoppingCriteriaList

try:
    from .sampling import (
        MIN_P,
        PRESENCE_PENALTY,
        REPETITION_PENALTY,
        TEMPERATURE,
        TOP_K,
        TOP_P,
        PresencePenaltyLogitsProcessor,
        TokenProgress,
    )
    from .novice_model import NoviceModel, MANIFEST_NAME
    from .divergence_html import render_html
except ImportError:
    from sampling import (
        MIN_P,
        PRESENCE_PENALTY,
        REPETITION_PENALTY,
        TEMPERATURE,
        TOP_K,
        TOP_P,
        PresencePenaltyLogitsProcessor,
        TokenProgress,
    )
    from novice_model import NoviceModel, MANIFEST_NAME
    from divergence_html import render_html


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
def build_input_ids(tokenizer, problem: str) -> tuple[torch.Tensor, str]:
    """Render the problem through the chat template (thinking enabled) -> (input_ids[1, P], text)."""
    messages = [{"role": "user", "content": problem}]
    try:
        text = tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=False, enable_thinking=True
        )
    except TypeError:
        text = tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=False
        )
    # The template already emits all special tokens, so don't add more.
    ids = tokenizer(text, return_tensors="pt", add_special_tokens=False).input_ids
    return ids, text


@torch.no_grad()
def rollout_and_score(model, input_ids: torch.Tensor, *, max_new_tokens: int, do_sample: bool,
                      pad_token_id: int | None, eos_token_id: int | None,
                      temperature: float, top_p: float, top_k_sample: int, min_p: float,
                      repetition_penalty: float, presence_penalty: float, device: str,
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
    if presence_penalty != 0.0:
        gen_kwargs["logits_processor"] = LogitsProcessorList([
            PresencePenaltyLogitsProcessor(
                presence_penalty=presence_penalty,
                prompt_lengths=prompt_len,
            )
        ])
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
# token data assembly
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

    # Opacity/hue tinting is a presentation concern handled by the renderer
    # (``divergence_html``); here we only emit the raw analysis data.
    js_arr = np.array(js_values) if js_values else np.array([0.0])
    stats = {
        "tokens": len(ids),
        "chars": sum(len(p) for p in pieces),
        "prompt_len": prompt_len,
        "gen_len": len(ids) - prompt_len,
        "js_mean": round(float(js_arr.mean()), 5),
        "js_max": round(float(js_arr.max()), 5),
    }
    return entries, stats


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
    parser.add_argument("--temperature", type=float, default=TEMPERATURE)
    parser.add_argument("--top-p", type=float, default=TOP_P)
    parser.add_argument("--sample-top-k", type=int, default=TOP_K,
                        help="Generation top_k; named separately from --top-k, which controls HTML candidates.")
    parser.add_argument("--min-p", type=float, default=MIN_P)
    parser.add_argument("--repetition-penalty", type=float, default=REPETITION_PENALTY)
    parser.add_argument("--presence-penalty", type=float, default=PRESENCE_PENALTY,
                        help="OpenAI/vLLM-style presence penalty over generated tokens only.")
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
    print(f"model={model_name} rank_fraction={rank_fraction}", flush=True)
    print(f"decoding={'sampling' if args.sample else 'greedy'} temperature={args.temperature} "
          f"top_p={args.top_p} top_k={args.sample_top_k} min_p={args.min_p} "
          f"repetition_penalty={args.repetition_penalty} presence_penalty={args.presence_penalty}",
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
        input_ids, _ = build_input_ids(tokenizer, problem)
        pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
        t0 = time.perf_counter()
        full_ids, prompt_len, m_probs, m_logits = rollout_and_score(
            master, input_ids, max_new_tokens=args.max_new_tokens, do_sample=args.sample,
            pad_token_id=pad_id, eos_token_id=tokenizer.eos_token_id, device=device,
            temperature=args.temperature, top_p=args.top_p, top_k_sample=args.sample_top_k,
            min_p=args.min_p, repetition_penalty=args.repetition_penalty,
            presence_penalty=args.presence_penalty,
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
