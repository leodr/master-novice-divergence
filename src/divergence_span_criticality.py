"""Span-level (reasoning-step) criticality test for master/novice disagreement.

Why spans
---------
The token-level test (``divergence_criticality.py``) found that one-token
perturbations barely move the final answer (answer_logprob_drop ~0 for everyone) —
single tokens are the wrong granularity for reasoning credit. This test perturbs a
whole **reasoning step** (a newline-delimited span of tokens) and lets the master
free-run to the answer, so the intervention is large enough to actually change the
outcome, and the reasoning downstream is free to adapt.

Question: are steps that *contain more novice/master disagreement* (higher flip
density) more outcome-critical than depth-matched low-disagreement steps — and does
that survive controlling for the step's own entropy?

Interventions (each followed by a master free-run to completion):
  ``span_rank2_flip``    high-flip-density step, replaced by the master's 2nd-choice
                         tokens (generic, matched corruption)
  ``span_rank2_control`` depth-matched low-density step, same generic corruption
  ``span_novice_flip``   high-density step, replaced by the NOVICE's version of it
                         (its argmax token at each position) — the real counterfactual

Comparisons:
  rank2_flip vs rank2_control  -> position leverage (does step disagreement mark
                                  outcome-critical steps?); entropy-controlled via OLS.
  novice_flip vs rank2_flip    -> same step, so entropy-matched; does the novice's
                                  actual version break the answer more than a generic
                                  corruption of the same step?

Metrics per trial:
  outcome_flip          master free-run no longer reaches the correct answer (the
                        primary, reasoning-free-to-change outcome; capped for cost).
  answer_logprob_drop   drop in log-prob of the master's own answer tokens with the
                        post-span reasoning teacher-forced (cheap graded complement).
  span_entropy          mean master entropy over the step (the covariate).

Example::

    uv run python src/divergence_span_criticality.py \\
        --novice-dir novice-qwen3.5-4b-rf0.1_shards --rank-fraction 0.25 \\
        --problems 50 --spans-per-problem 3 --free-run-cap 150
"""

from __future__ import annotations

import argparse
import json
import os
import random

import numpy as np
import torch

try:
    from .divergence_viz import (
        resolve_device, resolve_dtype, free_memory, build_input_ids,
        incremental_pieces, score_sequence,
    )
    from .divergence_leverage import (
        extract_boxed, answers_equal, greedy_generate, flip_mask, BOX_INSTRUCTION,
    )
    from .divergence_criticality import (
        master_stats, perturbed_tok_logp, answer_token_indices, ols_is_flip,
    )
    from .novice_model import NoviceModel, MANIFEST_NAME
except ImportError:
    from divergence_viz import (
        resolve_device, resolve_dtype, free_memory, build_input_ids,
        incremental_pieces, score_sequence,
    )
    from divergence_leverage import (
        extract_boxed, answers_equal, greedy_generate, flip_mask, BOX_INSTRUCTION,
    )
    from divergence_criticality import (
        master_stats, perturbed_tok_logp, answer_token_indices, ols_is_flip,
    )
    from novice_model import NoviceModel, MANIFEST_NAME


def segment_spans(tokenizer, ids: list[int], prompt_len: int, min_tokens: int = 3) -> list[dict]:
    """Split the generated region into newline-delimited token spans (reasoning
    steps). Drops spans shorter than ``min_tokens`` or whitespace-only."""
    pieces = incremental_pieces(tokenizer, ids)
    spans, cur = [], []
    for j in range(prompt_len, len(ids)):
        cur.append(j)
        if "\n" in pieces[j]:
            spans.append(cur)
            cur = []
    if cur:
        spans.append(cur)
    out = []
    for sp in spans:
        text = "".join(pieces[k] for k in sp)
        if len(sp) >= min_tokens and text.strip():
            out.append({"positions": sp, "start": sp[0], "end": sp[-1], "text": text})
    return out


def annotate_and_pair(spans: list[dict], flips: list[dict], prompt_len: int, T: int,
                      k: int, rng: random.Random) -> list[tuple[dict, dict]]:
    """Tag each span with flip density, then pair the top-k densest steps with
    depth-matched low-density steps (each control used once)."""
    fd = {r["position"]: r for r in flips}
    for sp in spans:
        pos = sp["positions"]
        nflip = sum(fd[p]["is_flip"] for p in pos if p in fd)
        sp["flip_count"] = int(nflip)
        sp["flip_frac"] = nflip / len(pos)
        sp["depth"] = (sp["start"] - prompt_len) / max(1, T - prompt_len)
        sp["novice_tokens"] = [int(fd[p]["novice_argmax"]) for p in pos]
        sp["entropy"] = None  # filled by caller (needs master entropy array)
    ranked = sorted(spans, key=lambda s: -s["flip_frac"])
    dense = [s for s in ranked if s["flip_frac"] > 0][:k]
    sparse = sorted([s for s in ranked if s["flip_frac"] < (dense[-1]["flip_frac"] if dense else 1)],
                    key=lambda s: s["flip_frac"])
    pairs, used = [], set()
    for f in dense:
        cands = [s for s in sparse if s["start"] not in used]
        if not cands:
            break
        ctrl = min(cands, key=lambda s: abs(s["depth"] - f["depth"]))
        used.add(ctrl["start"])
        pairs.append((f, ctrl))
    return pairs


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--novice-dir", required=True)
    p.add_argument("--dataset", default="HuggingFaceH4/MATH-500")
    p.add_argument("--split", default="test")
    p.add_argument("--problems", type=int, default=50)
    p.add_argument("--max-new-tokens", type=int, default=700)
    p.add_argument("--spans-per-problem", type=int, default=3, help="Flip-dense steps per problem (each paired with a control).")
    p.add_argument("--min-span-tokens", type=int, default=4)
    p.add_argument("--free-run-cap", type=int, default=150, help="Max trials to run the free-run binary outcome on (0=skip).")
    p.add_argument("--complete-tokens", type=int, default=0)
    p.add_argument("--rank-fraction", type=float, default=None)
    p.add_argument("--device", default="auto")
    p.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    p.add_argument("--no-chat", action="store_true")
    p.add_argument("--think", action="store_true")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="reports/span_criticality_n50")
    args = p.parse_args()

    rng = random.Random(args.seed)
    device = resolve_device(args.device)
    dtype = resolve_dtype(args.dtype)
    os.makedirs(args.out, exist_ok=True)

    manifest = torch.load(os.path.join(args.novice_dir, MANIFEST_NAME), map_location="cpu", weights_only=False)
    model_name = manifest["model_name"]
    rank_fraction = args.rank_fraction if args.rank_fraction is not None else manifest["rank_fraction"]

    from datasets import load_dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer

    ds = load_dataset(args.dataset, split=args.split)
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    use_chat = (not args.no_chat) and bool(tokenizer.chat_template)
    indices = list(range(min(args.problems, len(ds))))
    print(f"device={device} dtype={args.dtype} model={model_name} rf={rank_fraction} "
          f"problems={len(indices)} chat={use_chat}", flush=True)

    # ---- phase 1 (master): rollout, keep solved, clean stats + answer span ----- #
    print("Loading master ...", flush=True)
    master = AutoModelForCausalLM.from_pretrained(
        model_name, trust_remote_code=True, dtype=dtype).to(device).eval()
    kept, n_correct = [], 0
    for idx in indices:
        problem, gold = ds[idx]["problem"], ds[idx].get("answer")
        prompt = problem + BOX_INSTRUCTION if use_chat else problem
        input_ids, _ = build_input_ids(tokenizer, prompt, use_chat=use_chat, think=args.think)
        full = greedy_generate(master, input_ids, max_new_tokens=args.max_new_tokens,
                               pad_token_id=pad_id, device=device)
        plen = input_ids.shape[1]
        ids = full.tolist()
        correct = answers_equal(extract_boxed(tokenizer.decode(full[plen:], skip_special_tokens=True)), gold)
        n_correct += correct
        ans_idx = answer_token_indices(tokenizer, ids, plen) if correct else []
        print(f"[master {idx}] len={len(ids)-plen} correct={correct} answer_tokens={len(ans_idx)}", flush=True)
        if correct and ans_idx:
            top2, ent, tok_lp = master_stats(master, ids, device)
            kept.append({"idx": idx, "gold": gold, "ids": ids, "prompt_len": plen,
                         "top2": top2, "entropy": ent, "tok_lp": tok_lp, "answer_idx": ans_idx})
    del master
    free_memory(device)
    print(f"master solved {n_correct}/{len(indices)}; using {len(kept)} rollouts", flush=True)
    if not kept:
        raise SystemExit("Nothing usable; try more problems or --max-new-tokens.")

    # ---- phase 2 (novice): flips + segment into steps + pick pairs ------------- #
    print("Loading novice ...", flush=True)
    novice = NoviceModel.from_sharded_checkpoint(
        args.novice_dir, rank_fraction=rank_fraction, device=device, baked=True)
    novice.model.to(dtype=dtype).eval()
    total_trials = 0
    for rec in kept:
        n_probs, _ = score_sequence(novice.model, torch.tensor(rec["ids"]), device=device)
        flips = flip_mask(n_probs, rec["ids"], rec["prompt_len"])
        spans = segment_spans(tokenizer, rec["ids"], rec["prompt_len"], args.min_span_tokens)
        pairs = annotate_and_pair(spans, flips, rec["prompt_len"], len(rec["ids"]),
                                  args.spans_per_problem, rng)
        for f, c in pairs:  # fill span entropy now that we have the master entropy array
            for sp in (f, c):
                sp["entropy"] = float(np.mean([rec["entropy"][q - 1] for q in sp["positions"]]))
        rec["pairs"] = pairs
        total_trials += len(pairs) * 3
    del novice
    free_memory(device)
    free_run_set = set(rng.sample(range(total_trials), min(args.free_run_cap, total_trials))) \
        if args.free_run_cap > 0 else set()
    print(f"{total_trials} span trials; free-run binary on {len(free_run_set)} of them", flush=True)

    # ---- phase 3 (master reloaded): corrupt whole step, free-run + graded ------ #
    print("Reloading master for span perturbations ...", flush=True)
    master = AutoModelForCausalLM.from_pretrained(
        model_name, trust_remote_code=True, dtype=dtype).to(device).eval()

    trials, ti = [], 0
    for rec in kept:
        ids, gold, T, plen = rec["ids"], rec["gold"], len(rec["ids"]), rec["prompt_len"]
        tok_lp, ans_idx, top2 = rec["tok_lp"], rec["answer_idx"], rec["top2"]
        for si, (f, ctrl) in enumerate(rec["pairs"]):
            pair_id = f"{rec['idx']}:{si}"
            rank2 = lambda sp: [int(top2[q - 1][1]) for q in sp["positions"]]
            jobs = [("span_rank2_flip", f, rank2(f)),
                    ("span_rank2_control", ctrl, rank2(ctrl)),
                    ("span_novice_flip", f, f["novice_tokens"])]
            for arm, sp, corrupt in jobs:
                do_free = ti in free_run_set
                ti += 1
                s0, s1 = sp["start"], sp["end"]
                ans_pos = [q for q in ans_idx if q > s1]
                seq = ids[:s0] + corrupt + ids[s1 + 1:]
                if ans_pos:
                    pert = perturbed_tok_logp(master, seq, ans_pos, device)
                    ans_drop = float(np.mean([tok_lp[q] - pert[q] for q in ans_pos]))
                else:
                    ans_drop = np.nan
                recovered = np.nan
                if do_free:
                    budget = args.complete_tokens or ((T - s1) + 48)
                    out = greedy_generate(master, torch.tensor(ids[:s0] + corrupt).unsqueeze(0),
                                          max_new_tokens=budget, pad_token_id=pad_id, device=device)
                    recovered = bool(answers_equal(
                        extract_boxed(tokenizer.decode(out[plen:], skip_special_tokens=True)), gold))
                trials.append({"problem": rec["idx"], "pair_id": pair_id, "arm": arm,
                               "start": s0, "end": s1, "span_len": len(sp["positions"]),
                               "depth": sp["depth"], "flip_frac": sp["flip_frac"],
                               "is_flip": arm != "span_rank2_control", "span_entropy": sp["entropy"],
                               "answer_logprob_drop": ans_drop, "free_run": do_free, "recovered": recovered})
        print(f"[span {rec['idx']}] {len(rec['pairs'])} pairs done", flush=True)
    del master
    free_memory(device)

    # ---- analysis -------------------------------------------------------------- #
    import pandas as pd
    df = pd.DataFrame(trials)
    df.to_parquet(os.path.join(args.out, "span_trials.parquet"))

    def arm_stats(a):
        s = df[df.arm == a]
        fr = s[s.free_run]
        return {"n": int(len(s)),
                "mean_flip_frac": float(s["flip_frac"].mean()),
                "mean_span_entropy": float(s["span_entropy"].mean()),
                "mean_answer_logprob_drop": float(s["answer_logprob_drop"].mean()),
                "n_free_run": int(len(fr)),
                "outcome_flip_rate": float((~fr["recovered"].astype(bool)).mean()) if len(fr) else None}

    # position leverage: flip vs control step (rank2), entropy-controlled
    r2 = df[df.arm.isin(["span_rank2_flip", "span_rank2_control"])]
    def controlled(col):
        sub = r2.dropna(subset=[col, "span_entropy"])
        if len(sub) < 4:
            return {"n": int(len(sub))}
        raw = float(sub[sub.is_flip][col].mean() - sub[~sub.is_flip][col].mean())
        ols = ols_is_flip(sub[col].to_numpy(), sub["is_flip"].to_numpy(), sub["span_entropy"].to_numpy())
        return {"raw_flip_minus_control": raw, "entropy_controlled_ols": ols}
    # binary outcome position leverage (free-run subset)
    frr2 = r2[r2.free_run]
    def outc(mask):
        s = frr2[mask]
        return float((~s["recovered"].astype(bool)).mean()) if len(s) else None
    pos_outcome = {"flip_outcome_flip_rate": outc(frr2.is_flip), "flip_n": int(frr2.is_flip.sum()),
                   "control_outcome_flip_rate": outc(~frr2.is_flip), "control_n": int((~frr2.is_flip).sum())}

    # step effect: novice_flip vs rank2_flip on the SAME step (entropy-matched)
    def paired(col):
        piv = df[df.arm.isin(["span_novice_flip", "span_rank2_flip"])].pivot_table(
            index=["problem", "start"], columns="arm", values=col).dropna()
        if not len(piv):
            return {"n_pairs": 0}
        d = piv["span_novice_flip"] - piv["span_rank2_flip"]
        return {"n_pairs": int(len(piv)), "mean_diff_novice_minus_rank2": float(d.mean()),
                "frac_novice_larger": float((d > 0).mean())}

    summary = {
        "model_name": model_name, "rank_fraction": rank_fraction,
        "problems_seen": len(indices), "problems_master_correct": int(n_correct),
        "rollouts_used": len(kept), "total_trials": len(df),
        "by_arm": {a: arm_stats(a) for a in ("span_rank2_flip", "span_rank2_control", "span_novice_flip")},
        "position_leverage_answer_drop": controlled("answer_logprob_drop"),
        "position_leverage_outcome": pos_outcome,
        "step_effect_novice_vs_rank2": {"answer_logprob_drop": paired("answer_logprob_drop")},
    }
    with open(os.path.join(args.out, "span_criticality_summary.json"), "w") as fh:
        json.dump(summary, fh, indent=2)

    print("\n=== span criticality summary ===", flush=True)
    print(json.dumps(summary, indent=2), flush=True)
    print("\nDecisive checks:\n"
          "  position_leverage_outcome  -> do flip-dense STEPS break the answer more when corrupted?\n"
          "  position_leverage_answer_drop.entropy_controlled_ols.is_flip_t  -> does it survive entropy control?\n"
          "  step_effect_novice_vs_rank2  -> does the novice's actual version of a step break it more "
          "than a generic corruption of the same step?", flush=True)
    print(f"Wrote artifacts to {args.out}/", flush=True)


if __name__ == "__main__":
    main()
