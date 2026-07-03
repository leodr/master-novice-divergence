"""D1a — release-from-here leverage test for master/novice top-1 disagreement.

Research question this answers
------------------------------
We define **divergence** as a *position-aligned top-1 flip*: the master (full-rank
base model) rolls out a good solution; the novice (rank-truncated model) is
teacher-forced over the master's exact tokens, and at each generated position we
ask "would the novice's argmax have emitted a different token here?"
(``novice_argmax != master_token``). The hypothesis behind weighting these steps
more in RL is that they carry disproportionate **outcome leverage** — they are the
places where the student can't reproduce the expert and where letting it take
over would derail the answer.

The test (release-from-here)
----------------------------
For each problem the *master solves*, we pick release positions ``p`` and let the
novice take over from there: it is fed the master's tokens ``ids[:p]`` as context,
then free-runs to completion, and we check whether its final ``\\boxed{}`` answer
is still correct. We compare **flip positions** against **depth-matched non-flip
positions** (matching on how far into the rollout ``p`` is, because later releases
are trivially easier). If flips are the high-leverage steps, releasing at a flip
should fail markedly more often than releasing at a depth-matched non-flip.

Outputs (under ``--out``)
-------------------------
- ``flip_tokens.parquet`` — one row per generated token: flip mask + confidence.
- ``release_trials.parquet`` — one row per release trial: (problem, position,
  depth, is_flip, success).
- ``leverage_summary.json`` — success rates (flip vs control), overall and
  depth-binned, with the paired difference.
- ``leverage.png`` — the flip-vs-control success comparison.

This reuses the scoring primitives in ``divergence_viz`` and loads the novice via
``NoviceModel.from_sharded_checkpoint`` (baked, so free-running generation is
fast). It is a *diagnostic*, not training: it validates whether the flip signal is
worth wiring into an RL objective before you pay for a training run.

Example::

    uv run python src/divergence_leverage.py \\
        --novice-dir novice-qwen3.5-4b-rf0.1_shards \\
        --problems 50 --release-per-problem 4
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import time

import numpy as np
import torch

try:  # match divergence_viz's dual import style (run as script or as a module)
    from .divergence_viz import (
        resolve_device, resolve_dtype, free_memory, build_input_ids, score_sequence,
    )
    from .novice_model import NoviceModel, MANIFEST_NAME
except ImportError:
    from divergence_viz import (
        resolve_device, resolve_dtype, free_memory, build_input_ids, score_sequence,
    )
    from novice_model import NoviceModel, MANIFEST_NAME


# Standard MATH-eval instruction: without it the model answers in prose and never
# emits \boxed{}, so the grader (and the answer-location the release test needs)
# has nothing to latch onto.
BOX_INSTRUCTION = "\n\nPlease reason step by step, and put your final answer within \\boxed{}."


# --------------------------------------------------------------------------- #
# MATH-500 answer checking (pragmatic; not a full CAS equivalence checker)
# --------------------------------------------------------------------------- #
def extract_boxed(text: str) -> str | None:
    """Return the content of the LAST ``\\boxed{...}`` in ``text`` (balanced
    braces), or ``None`` if there is no boxed answer."""
    marker = r"\boxed"
    start = text.rfind(marker)
    if start == -1:
        return None
    i = start + len(marker)
    while i < len(text) and text[i] != "{":
        i += 1
    if i >= len(text):
        return None
    depth, i = 0, i
    out = []
    for ch in text[i:]:
        if ch == "{":
            depth += 1
            if depth == 1:
                continue
        elif ch == "}":
            depth -= 1
            if depth == 0:
                break
        out.append(ch)
    return "".join(out) if depth == 0 else None


_STRIP_TOKENS = [r"\left", r"\right", r"\!", r"\,", r"\;", r"\:", r"\ ", "$", " ", "\n"]


def normalize_answer(s: str | None) -> str | None:
    """Light LaTeX normalization so string comparison catches common equivalents.

    Deliberately conservative: it fixes formatting noise (\\left/\\right, spacing,
    \\dfrac vs \\frac, \\text{}, trailing punctuation, degrees/percent) but does
    not attempt symbolic equivalence. Consistent bias here affects the flip and
    control arms equally, so relative success rates remain meaningful.
    """
    if s is None:
        return None
    s = s.strip()
    # drop a leading/trailing math-mode delimiter and \text wrappers
    s = re.sub(r"\\text\{([^{}]*)\}", r"\1", s)
    s = s.replace(r"\dfrac", r"\frac").replace(r"\tfrac", r"\frac")
    s = s.replace(r"^{\circ}", "").replace(r"\circ", "").replace("%", "")
    s = s.replace("\\$", "").replace("$", "")
    for tok in _STRIP_TOKENS:
        s = s.replace(tok, "")
    s = s.rstrip(".")
    # \frac{a}{b} -> a/b ; {x} -> x for single groups
    s = re.sub(r"\\frac\{([^{}]+)\}\{([^{}]+)\}", r"(\1)/(\2)", s)
    return s.strip()


def answers_equal(pred: str | None, gold: str | None) -> bool:
    """True if ``pred`` matches ``gold`` after normalization, or as floats."""
    np_, ng = normalize_answer(pred), normalize_answer(gold)
    if np_ is None or ng is None:
        return False
    if np_ == ng:
        return True
    try:  # numeric fallback for plain numbers
        return abs(float(np_) - float(ng)) < 1e-6
    except (ValueError, TypeError):
        return False


# --------------------------------------------------------------------------- #
# generation helpers
# --------------------------------------------------------------------------- #
@torch.no_grad()
def greedy_generate(model, input_ids: torch.Tensor, *, max_new_tokens: int,
                    pad_token_id: int | None, device: str) -> torch.Tensor:
    """Greedy continuation; returns the full 1-D id sequence (prompt + gen) on CPU."""
    input_ids = input_ids.to(device)
    out = model.generate(
        input_ids=input_ids, attention_mask=torch.ones_like(input_ids),
        max_new_tokens=max_new_tokens, num_beams=1, do_sample=False,
        pad_token_id=pad_token_id,
    )
    return out[0].detach().cpu()


def flip_mask(novice_probs: np.ndarray, ids: list[int], prompt_len: int) -> list[dict]:
    """Per generated token: is the novice's argmax != the master's emitted token,
    and how confident is that disagreement (novice prob gap between its own pick
    and the master's token)."""
    rows = []
    for j in range(prompt_len, len(ids)):
        dist = novice_probs[j - 1]           # distribution that predicts token j
        n_arg = int(dist.argmax())
        master_tok = ids[j]
        gap = float(dist[n_arg] - dist[master_tok])  # >=0; 0 when they agree
        rows.append({
            "position": j,
            "depth": (j - prompt_len) / max(1, len(ids) - prompt_len),  # 0..1
            "is_flip": n_arg != master_tok,
            "novice_argmax": n_arg,
            "master_token": master_tok,
            "confidence_gap": gap,
        })
    return rows


def pick_release_positions(rows: list[dict], k: int, rng: random.Random) -> list[tuple[dict, dict]]:
    """Pair each of the top-``k`` most-confident flips with the nearest-depth
    non-flip (each control used once), so the two arms are matched on depth."""
    flips = sorted([r for r in rows if r["is_flip"]], key=lambda r: -r["confidence_gap"])
    nonflips = [r for r in rows if not r["is_flip"]]
    rng.shuffle(nonflips)
    pairs, used = [], set()
    for f in flips[:k]:
        cands = [r for r in nonflips if r["position"] not in used]
        if not cands:
            break
        ctrl = min(cands, key=lambda r: abs(r["position"] - f["position"]))
        used.add(ctrl["position"])
        pairs.append((f, ctrl))
    return pairs


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--novice-dir", required=True, help="Sharded novice checkpoint dir (has manifest.pt).")
    p.add_argument("--dataset", default="HuggingFaceH4/MATH-500")
    p.add_argument("--split", default="test")
    p.add_argument("--problems", type=int, default=50, help="How many problems (from index 0) to roll out.")
    p.add_argument("--max-new-tokens", type=int, default=512, help="Master rollout length.")
    p.add_argument("--complete-tokens", type=int, default=0,
                   help="Cap on novice free-run length after a release (0 = auto: remaining+slack).")
    p.add_argument("--release-per-problem", type=int, default=4,
                   help="Flip release points per problem (each paired with a depth-matched control).")
    p.add_argument("--rank-fraction", type=float, default=None, help="Override the manifest rank.")
    p.add_argument("--device", default="auto")
    p.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    p.add_argument("--no-chat", action="store_true", help="Feed raw problem text (skip chat template).")
    p.add_argument("--think", action="store_true", help="Enable thinking mode if the template supports it.")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="reports/leverage", help="Output directory.")
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

    # ---- phase A.1: master rolls out; keep only problems it solves ---------- #
    print("Loading master (full rank) ...", flush=True)
    master = AutoModelForCausalLM.from_pretrained(
        model_name, trust_remote_code=True, dtype=dtype).to(device).eval()

    kept: list[dict] = []
    n_correct = 0
    for idx in indices:
        problem, gold = ds[idx]["problem"], ds[idx].get("answer")
        prompt = problem + BOX_INSTRUCTION if use_chat else problem
        input_ids, _ = build_input_ids(tokenizer, prompt, use_chat=use_chat, think=args.think)
        full_ids = greedy_generate(master, input_ids, max_new_tokens=args.max_new_tokens,
                                   pad_token_id=pad_id, device=device)
        prompt_len = input_ids.shape[1]
        gen_text = tokenizer.decode(full_ids[prompt_len:], skip_special_tokens=True)
        correct = answers_equal(extract_boxed(gen_text), gold)
        n_correct += correct
        print(f"[master {idx}] len={len(full_ids)-prompt_len} correct={correct}", flush=True)
        if correct:
            kept.append({"idx": idx, "gold": gold, "ids": full_ids.tolist(), "prompt_len": prompt_len})
    del master
    free_memory(device)
    print(f"master solved {n_correct}/{len(indices)}; using {len(kept)} rollouts for the leverage test", flush=True)
    if not kept:
        raise SystemExit("Master solved no problems — nothing to test. Try more problems or --max-new-tokens.")

    # ---- phase A.2: novice scores each kept rollout -> flip masks ----------- #
    print("Loading novice (baked, truncated) ...", flush=True)
    novice = NoviceModel.from_sharded_checkpoint(
        args.novice_dir, rank_fraction=rank_fraction, device=device, baked=True)
    novice.model.to(dtype=dtype).eval()

    token_rows: list[dict] = []
    for rec in kept:
        ids = rec["ids"]
        n_probs, _ = score_sequence(novice.model, torch.tensor(ids), device=device)
        rec["flips"] = flip_mask(n_probs, ids, rec["prompt_len"])
        for r in rec["flips"]:
            token_rows.append({"problem": rec["idx"], **{k: r[k] for k in
                              ("position", "depth", "is_flip", "confidence_gap")}})

    # ---- phase B: release-from-here at flips vs depth-matched controls ------ #
    trial_rows: list[dict] = []
    for rec in kept:
        ids, gold, T, plen = rec["ids"], rec["gold"], len(rec["ids"]), rec["prompt_len"]
        pairs = pick_release_positions(rec["flips"], args.release_per_problem, rng)
        for pi, (f, ctrl) in enumerate(pairs):
            pair_id = f"{rec['idx']}:{pi}"  # links a flip to its depth-matched control
            for arm, r in (("flip", f), ("control", ctrl)):
                pos = r["position"]
                budget = args.complete_tokens or ((T - pos) + 48)
                prefix = torch.tensor(ids[:pos]).unsqueeze(0)
                out = greedy_generate(novice.model, prefix, max_new_tokens=budget,
                                      pad_token_id=pad_id, device=device)
                # Grade the full released trajectory (master prefix + novice
                # continuation), not just the continuation: for a late release the
                # master's boxed answer already sits in the prefix, so grading the
                # continuation alone would wrongly score a correct trajectory as fail.
                traj = tokenizer.decode(out[plen:], skip_special_tokens=True)
                succ = answers_equal(extract_boxed(traj), gold)
                trial_rows.append({"problem": rec["idx"], "pair_id": pair_id, "arm": arm,
                                   "position": pos, "depth": r["depth"], "is_flip": r["is_flip"],
                                   "confidence_gap": r["confidence_gap"], "success": succ})
        print(f"[novice {rec['idx']}] released {len(pairs)} flip/control pairs", flush=True)
    del novice
    free_memory(device)

    # ---- persist + summarize ----------------------------------------------- #
    import pandas as pd
    tok_df = pd.DataFrame(token_rows)
    tri_df = pd.DataFrame(trial_rows)
    tok_df.to_parquet(os.path.join(args.out, "flip_tokens.parquet"))
    tri_df.to_parquet(os.path.join(args.out, "release_trials.parquet"))

    def rate(df):
        return float(df["success"].mean()) if len(df) else float("nan")

    flip_df = tri_df[tri_df.arm == "flip"]
    ctrl_df = tri_df[tri_df.arm == "control"]

    # Paired readout: each flip vs its OWN depth-matched control. This controls the
    # release-depth confound directly (late releases are trivially easier), which
    # the marginal flip/control rates cannot. paired_diff > 0 => flips more critical.
    pivot = tri_df.pivot_table(index="pair_id", columns="arm", values="success").dropna()
    paired_diff = float((pivot["control"] - pivot["flip"]).mean()) if len(pivot) else float("nan")
    ctrl_better = int(((pivot["control"] == 1) & (pivot["flip"] == 0)).sum()) if len(pivot) else 0
    flip_better = int(((pivot["flip"] == 1) & (pivot["control"] == 0)).sum()) if len(pivot) else 0
    concordant = int((pivot["control"] == pivot["flip"]).sum()) if len(pivot) else 0
    # depth-binned success (control for how far into the rollout the release is)
    bins = np.linspace(0, 1, 5)
    tri_df["depth_bin"] = np.clip(np.digitize(tri_df["depth"], bins) - 1, 0, len(bins) - 2)
    binned = {int(b): {"flip": rate(tri_df[(tri_df.arm == "flip") & (tri_df.depth_bin == b)]),
                       "control": rate(tri_df[(tri_df.arm == "control") & (tri_df.depth_bin == b)])}
              for b in sorted(tri_df["depth_bin"].unique())}

    summary = {
        "model_name": model_name, "rank_fraction": rank_fraction,
        "problems_seen": len(indices), "problems_master_correct": int(n_correct),
        "rollouts_used": len(kept), "release_pairs": int(len(flip_df)),
        "flip_token_fraction": float(tok_df["is_flip"].mean()) if len(tok_df) else None,
        "success_flip": rate(flip_df), "success_control": rate(ctrl_df),
        "leverage_gap": rate(ctrl_df) - rate(flip_df),  # marginal (depth-confounded)
        "paired_diff": paired_diff,  # depth-controlled effect size; the honest readout
        "pairs_scored": int(len(pivot)),
        "discordant_control_better": ctrl_better,  # flip failed, matched control succeeded
        "discordant_flip_better": flip_better,      # (should be rarer if flips are critical)
        "concordant_pairs": concordant,
        "success_by_depth_bin": binned,
    }
    with open(os.path.join(args.out, "leverage_summary.json"), "w") as fh:
        json.dump(summary, fh, indent=2)

    # ---- plot --------------------------------------------------------------- #
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))
    ax1.bar(["flip", "control"], [summary["success_flip"], summary["success_control"]],
            color=["#c0392b", "#2980b9"])
    ax1.set_ylim(0, 1)
    ax1.set_ylabel("novice success after release")
    ax1.set_title(f"Release-from-here success\n(gap={summary['leverage_gap']:+.3f}, "
                  f"n={summary['release_pairs']} pairs)")
    xs = sorted(binned)
    ax2.plot(xs, [binned[b]["flip"] for b in xs], "o-", color="#c0392b", label="flip")
    ax2.plot(xs, [binned[b]["control"] for b in xs], "o-", color="#2980b9", label="control")
    ax2.set_ylim(0, 1)
    ax2.set_xlabel("release depth bin (0=early → later)")
    ax2.set_ylabel("success")
    ax2.set_title("Success vs release depth")
    ax2.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(args.out, "leverage.png"), dpi=130)

    print("\n=== leverage summary ===", flush=True)
    print(json.dumps(summary, indent=2), flush=True)
    print(f"\nInterpretation: leverage_gap > 0 means releasing the novice at a flip "
          f"fails more often than at a depth-matched non-flip — i.e. flip steps carry "
          f"more outcome leverage and are the ones worth upweighting in RL.", flush=True)
    print(f"Wrote artifacts to {args.out}/", flush=True)


if __name__ == "__main__":
    main()
