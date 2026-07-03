"""Master-perturbation criticality test (scaled): does a one-token perturbation at a
master/novice *disagreement* position matter more than at a matched non-flip — and
does that survive controlling for the master's own uncertainty (entropy)?

Why this test
-------------
The release-from-here test (``divergence_leverage.py``) was too blunt: a whole weak
novice free-runs after the handoff, so one flip token is swamped by the shared tail.
Here we perturb a *single* token and let the strong **master** continue, isolating a
step's causal effect. The pilot showed the master does NOT recover from every
perturbation (so the test is viable) and that flip positions look more disruptive —
but only in a *local* metric and without controlling for entropy. This scaled version
adds the two things needed to draw a real conclusion.

Metrics per perturbation trial (all but the last are forward-only, so cheap at scale):

- **local_disruption** — mean drop in the master's log-prob of its own next-K tokens
  under the perturbed vs clean prefix. Local surprise; had a flip>control gap in the
  pilot but is confounded by entropy.
- **answer_logprob_drop** — mean drop in the master's log-prob of its own *correct
  answer tokens* (the \\boxed{} span), with the intervening reasoning teacher-forced.
  **Outcome-tied**: it measures the perturbation's residual effect that reaches the
  answer, not just local surprise.
- **master_entropy** — the master's entropy at the perturbed position, the covariate
  we regress out. If the flip effect vanishes once entropy is included, the
  disagreement adds nothing over a free single-model signal.
- **outcome_flip** (capped subset) — the master free-runs after the perturbation; did
  the final answer actually change? The "real" outcome, sampled to bound runtime.

Conditions (depth-matched flip/control pairs):
  ``flip_novice``   flip position, splice the NOVICE's token (the real counterfactual)
  ``flip_rank2``    flip position, splice the master's 2nd-choice token (generic)
  ``control_rank2`` matched non-flip, splice the master's 2nd-choice token (baseline)

Key comparisons:
  flip_rank2 vs control_rank2      — position leverage (confounded by entropy → OLS)
  flip_novice vs flip_rank2        — same position, so automatically entropy-matched;
                                     isolates whether the novice's *specific* token bites

Example::

    uv run python src/divergence_criticality.py \\
        --novice-dir novice-qwen3.5-4b-rf0.1_shards --rank-fraction 0.25 \\
        --problems 50 --pairs-per-problem 5 --k-local 8 --free-run-cap 90
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
        resolve_device, resolve_dtype, free_memory, build_input_ids, incremental_pieces,
    )
    from .divergence_leverage import (
        extract_boxed, answers_equal, greedy_generate, flip_mask,
        pick_release_positions, BOX_INSTRUCTION,
    )
    from .novice_model import NoviceModel, MANIFEST_NAME
except ImportError:
    from divergence_viz import (
        resolve_device, resolve_dtype, free_memory, build_input_ids, incremental_pieces,
    )
    from divergence_leverage import (
        extract_boxed, answers_equal, greedy_generate, flip_mask,
        pick_release_positions, BOX_INSTRUCTION,
    )
    from novice_model import NoviceModel, MANIFEST_NAME


# --------------------------------------------------------------------------- #
# clean master statistics (one forward per rollout; keeps only O(T) arrays)
# --------------------------------------------------------------------------- #
@torch.no_grad()
def master_stats(model, ids: list[int], device: str):
    """Return (top2[T,2], entropy[T], tok_logp[T]) from a single teacher-forced pass.

    ``tok_logp[j]`` is log P(ids[j] | ids[:j]) — the realized-token log-prob — so any
    span's clean log-prob is just a sum of these, no [T,V] matrix retained.
    """
    logits = model(torch.tensor(ids).unsqueeze(0).to(device)).logits[0]
    logp = torch.log_softmax(logits.float(), dim=-1)
    T = len(ids)
    top2 = torch.topk(logits.float(), 2, dim=-1).indices.cpu().numpy()
    ent = (-(logp.exp() * logp).sum(-1)).cpu().numpy()
    idx = torch.tensor(ids, device=logp.device)
    tok_lp = np.full(T, np.nan, dtype=np.float64)
    tok_lp[1:] = logp[torch.arange(T - 1, device=logp.device), idx[1:]].cpu().numpy()
    return top2, ent, tok_lp


@torch.no_grad()
def perturbed_tok_logp(model, seq: list[int], positions: list[int], device: str) -> dict:
    """log P(seq[j] | seq[:j]) for each j in ``positions`` (perturbed prefix)."""
    logits = model(torch.tensor(seq).unsqueeze(0).to(device)).logits[0]
    logp = torch.log_softmax(logits.float(), dim=-1)
    return {j: float(logp[j - 1, seq[j]]) for j in positions}


def answer_token_indices(tokenizer, ids: list[int], prompt_len: int) -> list[int]:
    """Full-sequence token indices covering the content of the last ``\\boxed{...}``
    in the generated (assistant) region, via char-span overlap. [] if none."""
    pieces = incremental_pieces(tokenizer, ids)
    gen = pieces[prompt_len:]
    text = "".join(gen)
    start = text.rfind(r"\boxed")
    if start == -1:
        return []
    i = start
    while i < len(text) and text[i] != "{":
        i += 1
    depth, cstart, cend = 0, None, None
    for k in range(i, len(text)):
        if text[k] == "{":
            depth += 1
            if depth == 1:
                cstart = k + 1
        elif text[k] == "}":
            depth -= 1
            if depth == 0:
                cend = k
                break
    if cstart is None or cend is None:
        return []
    idxs, off = [], 0
    for jj, pc in enumerate(gen):
        a, b = off, off + len(pc)
        if a < cend and b > cstart:
            idxs.append(prompt_len + jj)
        off = b
    return idxs


def ols_is_flip(y: np.ndarray, is_flip: np.ndarray, entropy: np.ndarray) -> dict:
    """OLS of ``y`` on [1, is_flip, entropy]; report the is_flip and entropy terms."""
    X = np.column_stack([np.ones(len(y)), is_flip.astype(float), entropy])
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    dof = max(1, len(y) - X.shape[1])
    cov = (resid @ resid) / dof * np.linalg.inv(X.T @ X)
    se = np.sqrt(np.diag(cov))
    return {"is_flip_coef": float(beta[1]), "is_flip_se": float(se[1]),
            "is_flip_t": float(beta[1] / se[1]) if se[1] else float("nan"),
            "entropy_coef": float(beta[2]), "entropy_t": float(beta[2] / se[2]) if se[2] else float("nan"),
            "n": int(len(y))}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--novice-dir", required=True)
    p.add_argument("--dataset", default="HuggingFaceH4/MATH-500")
    p.add_argument("--split", default="test")
    p.add_argument("--problems", type=int, default=50)
    p.add_argument("--max-new-tokens", type=int, default=700)
    p.add_argument("--pairs-per-problem", type=int, default=5)
    p.add_argument("--k-local", type=int, default=8)
    p.add_argument("--free-run-cap", type=int, default=90,
                   help="Max trials to also run the (expensive) free-run binary outcome on (0=skip).")
    p.add_argument("--complete-tokens", type=int, default=0)
    p.add_argument("--rank-fraction", type=float, default=None)
    p.add_argument("--device", default="auto")
    p.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    p.add_argument("--no-chat", action="store_true")
    p.add_argument("--think", action="store_true")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="reports/criticality_n50")
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
    print(f"master solved {n_correct}/{len(indices)}; using {len(kept)} rollouts (correct + locatable answer)", flush=True)
    if not kept:
        raise SystemExit("Nothing usable; try more problems or --max-new-tokens.")

    # ---- phase 2 (novice): flips + novice's token per position ----------------- #
    print("Loading novice ...", flush=True)
    from divergence_viz import score_sequence
    novice = NoviceModel.from_sharded_checkpoint(
        args.novice_dir, rank_fraction=rank_fraction, device=device, baked=True)
    novice.model.to(dtype=dtype).eval()
    total_trials = 0
    for rec in kept:
        n_probs, _ = score_sequence(novice.model, torch.tensor(rec["ids"]), device=device)
        rec["flips"] = flip_mask(n_probs, rec["ids"], rec["prompt_len"])
        rec["pairs"] = pick_release_positions(rec["flips"], args.pairs_per_problem, rng)
        total_trials += len(rec["pairs"]) * 3
    del novice
    free_memory(device)
    free_run_set = set(rng.sample(range(total_trials), min(args.free_run_cap, total_trials))) \
        if args.free_run_cap > 0 else set()
    print(f"{total_trials} perturbation trials; free-run binary on {len(free_run_set)} of them", flush=True)

    # ---- phase 3 (master reloaded): perturb one token, measure effect ---------- #
    print("Reloading master for perturbations ...", flush=True)
    master = AutoModelForCausalLM.from_pretrained(
        model_name, trust_remote_code=True, dtype=dtype).to(device).eval()

    trials, ti = [], 0
    for rec in kept:
        ids, gold, T, plen = rec["ids"], rec["gold"], len(rec["ids"]), rec["prompt_len"]
        tok_lp, ent, ans_idx = rec["tok_lp"], rec["entropy"], rec["answer_idx"]
        for pi, (f, ctrl) in enumerate(rec["pairs"]):
            pair_id = f"{rec['idx']}:{pi}"
            fp, cp = f["position"], ctrl["position"]
            jobs = [("flip_novice", fp, f["novice_argmax"]),
                    ("flip_rank2", fp, int(rec["top2"][fp - 1][1])),
                    ("control_rank2", cp, int(rec["top2"][cp - 1][1]))]
            for arm, pos, perturb in jobs:
                do_free = ti in free_run_set
                ti += 1
                local_pos = [j for j in range(pos + 1, min(pos + 1 + args.k_local, T))]
                ans_pos = [j for j in ans_idx if j > pos]
                need = sorted(set(local_pos) | set(ans_pos))
                if not need:
                    continue
                seq = ids[:pos] + [perturb] + ids[pos + 1:]
                pert = perturbed_tok_logp(master, seq, need, device)
                local_disr = float(np.mean([tok_lp[j] - pert[j] for j in local_pos])) if local_pos else np.nan
                ans_drop = float(np.mean([tok_lp[j] - pert[j] for j in ans_pos])) if ans_pos else np.nan
                recovered = np.nan
                if do_free:
                    budget = args.complete_tokens or ((T - pos) + 48)
                    out = greedy_generate(master, torch.tensor(ids[:pos] + [perturb]).unsqueeze(0),
                                          max_new_tokens=budget, pad_token_id=pad_id, device=device)
                    recovered = bool(answers_equal(
                        extract_boxed(tokenizer.decode(out[plen:], skip_special_tokens=True)), gold))
                trials.append({"problem": rec["idx"], "pair_id": pair_id, "arm": arm, "position": pos,
                               "depth": (f if arm != "control_rank2" else ctrl)["depth"],
                               "is_flip": arm != "control_rank2", "master_entropy": float(ent[pos - 1]),
                               "local_disruption": local_disr, "answer_logprob_drop": ans_drop,
                               "free_run": do_free, "recovered": recovered})
        print(f"[perturb {rec['idx']}] {len(rec['pairs'])} pairs done", flush=True)
    del master
    free_memory(device)

    # ---- analysis -------------------------------------------------------------- #
    import pandas as pd
    df = pd.DataFrame(trials)
    df.to_parquet(os.path.join(args.out, "criticality_trials.parquet"))

    def arm_stats(a):
        s = df[df.arm == a]
        fr = s[s.free_run]
        return {"n": int(len(s)),
                "mean_local_disruption": float(s["local_disruption"].mean()),
                "mean_answer_logprob_drop": float(s["answer_logprob_drop"].mean()),
                "mean_master_entropy": float(s["master_entropy"].mean()),
                "n_free_run": int(len(fr)),
                "outcome_flip_rate": float((~fr["recovered"].astype(bool)).mean()) if len(fr) else None}

    # entropy-controlled position effect (flip_rank2 vs control_rank2, same perturbation type)
    r2 = df[df.arm.isin(["flip_rank2", "control_rank2"])]
    def controlled(col):
        sub = r2.dropna(subset=[col, "master_entropy"])
        raw_gap = float(sub[sub.is_flip][col].mean() - sub[~sub.is_flip][col].mean())
        ols = ols_is_flip(sub[col].to_numpy(), sub["is_flip"].to_numpy(), sub["master_entropy"].to_numpy())
        corr = float(np.corrcoef(sub["master_entropy"], sub[col])[0, 1]) if len(sub) > 2 else float("nan")
        # entropy-tercile view: flip vs control within each tercile
        try:
            terc = pd.qcut(sub["master_entropy"], 3, labels=[0, 1, 2], duplicates="drop")
            tview = {int(t): {"flip": float(sub[(terc == t) & sub.is_flip][col].mean()),
                              "control": float(sub[(terc == t) & ~sub.is_flip][col].mean())}
                     for t in sorted(set(terc.dropna()))}
        except (ValueError, IndexError):
            tview = {}
        return {"raw_flip_minus_control": raw_gap, "entropy_controlled_ols": ols,
                "corr_entropy_vs_metric": corr, "by_entropy_tercile": tview}

    # token effect: flip_novice vs flip_rank2 at the SAME position (auto entropy-matched)
    def paired_token_effect(col):
        piv = df[df.arm.isin(["flip_novice", "flip_rank2"])].pivot_table(
            index=["problem", "position"], columns="arm", values=col).dropna()
        if not len(piv):
            return {"n_pairs": 0, "mean_diff_novice_minus_rank2": float("nan")}
        d = piv["flip_novice"] - piv["flip_rank2"]
        return {"n_pairs": int(len(piv)), "mean_diff_novice_minus_rank2": float(d.mean()),
                "frac_novice_larger": float((d > 0).mean())}

    summary = {
        "model_name": model_name, "rank_fraction": rank_fraction,
        "problems_seen": len(indices), "problems_master_correct": int(n_correct),
        "rollouts_used": len(kept), "total_trials": len(df), "k_local": args.k_local,
        "by_arm": {a: arm_stats(a) for a in ("flip_novice", "flip_rank2", "control_rank2")},
        "position_leverage": {"local_disruption": controlled("local_disruption"),
                              "answer_logprob_drop": controlled("answer_logprob_drop")},
        "token_effect_novice_vs_rank2": {"local_disruption": paired_token_effect("local_disruption"),
                                         "answer_logprob_drop": paired_token_effect("answer_logprob_drop")},
    }
    with open(os.path.join(args.out, "criticality_summary.json"), "w") as fh:
        json.dump(summary, fh, indent=2)

    print("\n=== criticality (scaled) summary ===", flush=True)
    print(json.dumps(summary, indent=2), flush=True)
    print("\nDecisive checks:\n"
          "  position_leverage.*.entropy_controlled_ols.is_flip_t  -> is the flip effect real "
          "AFTER regressing out master entropy? (|t|>~2 = yes; ~0 = 'just use entropy')\n"
          "  answer_logprob_drop gaps  -> does the effect reach the OUTCOME, not only local surprise?\n"
          "  token_effect_novice_vs_rank2  -> does the novice's specific token bite harder "
          "than a generic 2nd choice, at the same position?", flush=True)
    print(f"Wrote artifacts to {args.out}/", flush=True)


if __name__ == "__main__":
    main()
