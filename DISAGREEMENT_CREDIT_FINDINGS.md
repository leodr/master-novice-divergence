# Master/Novice disagreement as an RL credit signal — findings

**Question.** In the Master/Novice framework, can the positions where the truncated
**Novice** disagrees with the full-rank **Master** be used to decide *which steps to
weight more in reinforcement learning* — on the premise that the Master performs
better, so its disagreements mark the pivotal, high-leverage steps?

**Short answer.** No — not as a per-token or per-step credit signal, at least in this
setting. Across three progressively sharper tests, master/novice **top-1 disagreement
does not mark the steps whose corruption actually changes the final answer.** The only
signal that ever appeared was a modest *local* fragility at disagreement tokens, which
does not propagate to the outcome and is largely captured by the Master's own entropy.

---

## Setup

- **Models.** Master = `Qwen/Qwen3.5-4B` (full rank). Novice = the disentangled
  checkpoint `leodriesch/novice-qwen3.5-4b-rf0.1`, baked at **rank_fraction 0.25**
  (see "Coherence cliff" below for why not 0.1).
- **Data.** MATH-500 (`HuggingFaceH4/MATH-500`), problems 0–49. The Master solved
  **23/50** with a `\boxed{}` instruction; all tests run on those 23 rollouts.
- **Divergence definition (settled with the user).** A *position-aligned top-1 flip*:
  the Master greedily rolls out a correct solution; the Novice is teacher-forced over
  the Master's exact tokens; a **flip** is a position where `novice_argmax != master_token`.
- **Decoding/grading.** Greedy. A pragmatic `\boxed{}` extractor + LaTeX-normalizing
  comparator (`src/divergence_leverage.py`).

### Preliminary findings that shaped the tests

- **Coherence cliff.** At the shipped **rank 0.1** the Novice is *incoherent* (repetition
  collapse: `the the the …`), so its "disagreements" are noise. It becomes fluent at
  **rank ≥ 0.25**. All experiments therefore use rank 0.25 — the sharpest coherent point.
  (The stored factors are full-rank, so any bake rank is available without re-download.)
- **Flip concentration.** Flip token fraction falls from **0.44 at rank 0.1** to
  **0.12 at rank 0.25** — the signal is sharp, not smeared.
- **Ability gap (not "too close").** On the Master's own trajectories, handing control
  to the Novice yields only ~26% correct completions vs the Master's 100% — the Novice
  is far weaker generatively, so a null is *not* attributable to the two models being
  near-equal.

---

## Experiment 1 — Release-from-here leverage (`src/divergence_leverage.py`)

**Idea.** At a flip vs a depth-matched non-flip, hand control to the Novice (Master
prefix `ids[:p]`, then Novice free-runs to the answer). If flips are pivotal, releasing
at a flip should derail the answer more often.

**Result (n = 115 depth-matched pairs).**

| | success rate |
|---|---|
| release at flip | 0.261 |
| release at matched control | 0.270 |
| **paired diff (depth-controlled)** | **+0.009** |

Discordant pairs: 4 control-better vs 3 flip-better; **108/115 concordant**. Outcome is
governed by *release depth* (late = easy, early = hard), not flip-ness.

**Interpretation.** Null — but the test is *blunt*: after the handoff a whole weak Novice
free-runs to the answer in both arms, so one flip token is swamped by the shared tail.
Motivated a single-step intervention.

---

## Experiment 2 — Single-token master perturbation (`src/divergence_criticality.py`)

**Idea.** Perturb *one* token and let the strong **Master** continue, isolating a step's
causal effect. Three arms: `flip_novice` (splice the Novice's token), `flip_rank2`
(Master's 2nd choice, at a flip), `control_rank2` (Master's 2nd choice, at a matched
non-flip). Metrics: **local disruption** (drop in log-prob of the Master's own next-8
tokens, nats/tok), **answer_logprob_drop** (drop in log-prob of the Master's *answer*
tokens, reasoning teacher-forced — outcome-tied), **master entropy** (covariate), and a
capped **free-run binary** outcome.

**Result (n = 345; 115/arm).**

- **Local disruption:** flip 2.55 vs control 1.60 nats/tok (raw gap **+0.94**). Survives
  entropy control: **`is_flip` t = +2.35** (coef +0.62; entropy t = +2.69), and flip >
  control in **all three entropy terciles**. → *A real but modest position signal.*
- **Answer-tied drop:** raw gap **−0.0005**, **`is_flip` t = −0.14** (null); magnitudes
  ~0.001 for everyone. → *Does not reach the outcome.*
- **Novice's specific token vs generic 2nd choice** (same position, so entropy-matched):
  mean diff ≈ 0; Novice larger only **18%** (local) / **11%** (answer) of the time. → *No
  token-specific signal.*
- **Free-run binary** (~30/arm, noisy): flip_novice 0.24, flip_rank2 0.10, control_rank2
  0.19 — no clean order.

**Interpretation.** The only positive signal (local disruption) is *local*, does not
propagate to the answer, and the Novice's actual choice adds nothing over a generic
perturbation. The near-zero answer-drop for *everyone* suggested single tokens are the
wrong granularity → span-level test.

---

## Experiment 3 — Span (reasoning-step) perturbation (`src/divergence_span_criticality.py`)

**Idea.** Perturb a whole **reasoning step** (newline-delimited token span) and let the
Master free-run to the answer, so the intervention is large enough to move the outcome
and the downstream reasoning is free to adapt. Steps ranked by **flip density**. Arms:
`span_rank2_flip` / `span_rank2_control` (generic 2nd-choice corruption of a flip-dense
vs depth-matched low-density step) and `span_novice_flip` (replace the step with the
**Novice's** version of it). Primary metric: **free-run binary outcome**.

**Result (n = 207; ~50/arm on the free-run outcome).**

| corrupted step | answer-break rate (free-run) |
|---|---|
| flip-dense step | **0.151** (n=53) |
| low-disagreement step | **0.176** (n=51) |

- **No gap** (if anything control breaks slightly *more*).
- Graded answer-drop, entropy-controlled: **`is_flip` t = −0.79** (null).
- **Novice's version of a step vs generic corruption of the same step:** mean diff ≈ 0,
  Novice larger only **10%** of the time; free-run outcome identical (0.152 vs 0.151).

**Interpretation.** Null on the primary, reasoning-free-to-change metric — the strongest
test in the series, and it still finds nothing.

---

## Consolidated verdict

| Test | Metric | Flip effect |
|---|---|---|
| Release-from-here | free-run answer | null (paired +0.009) |
| Token perturbation | local disruption (next-8) | **+0.94 nats/tok, survives entropy (t=2.35)** |
| Token perturbation | answer-tied drop | null (t=−0.14) |
| Token perturbation | Novice's specific token | null |
| **Span perturbation** | **free-run outcome** | **null (0.151 vs 0.176)** |
| Span perturbation | answer drop, entropy-controlled | null (t=−0.79) |
| Span perturbation | Novice's version of the step | null |

**Master/novice top-1 disagreement is not a useful per-token or per-step credit signal
here.** Disagreement positions are marginally more *locally* fragile, but that fragility
does not reach the outcome at token or step granularity, and the Novice's specific
choices carry no more outcome leverage than a generic perturbation. Upweighting these
positions in RL would not concentrate gradient where the answer is actually decided.

### Two findings worth keeping (independent of the negative)

- **The Master is strikingly robust.** Corrupting an *entire* reasoning step still lets it
  recover the correct answer **~83%** of the time — only ~1 in 6 steps is load-bearing,
  and disagreement does not predict which.
- **Entropy, not the Novice, was the consistently predictive quantity.** In the token
  test the Master's own entropy was a significant predictor of disruption (t=+2.69). If
  the goal is per-step credit weighting, Master entropy is the cheaper, at-least-as-
  informative signal — consistent with the RLVR literature on high-entropy "forking" tokens.

### Caveats

One Novice (rank 0.25 of one disentangled Qwen3.5-4B), one dataset (MATH-500, 23 solved),
n ~50/arm on the binary outcomes. The null is consistent and appears in the outcome-tied
metric, so it is fairly robust — but it is not a claim across all ranks/tasks, and every
intervention (`rank2`, novice-substitution, step corruption) is one operationalization of
"corruption." Open reframes not tested: whether *aggregate* divergence mass per trajectory
predicts reward (a distributional, not pointwise, question), and whether a different rank
or a non-greedy Novice sharpens the flips.

---

## Reproduction

```bash
# Exp 1 — release-from-here leverage
uv run python src/divergence_leverage.py --novice-dir novice-qwen3.5-4b-rf0.1_shards \
    --rank-fraction 0.25 --problems 50 --release-per-problem 5 --out reports/leverage_rf025_n50

# Exp 2 — single-token master perturbation (entropy covariate + answer-tied metric)
uv run python src/divergence_criticality.py --novice-dir novice-qwen3.5-4b-rf0.1_shards \
    --rank-fraction 0.25 --problems 50 --pairs-per-problem 5 --free-run-cap 90 --out reports/criticality_n50

# Exp 3 — span (reasoning-step) perturbation
uv run python src/divergence_span_criticality.py --novice-dir novice-qwen3.5-4b-rf0.1_shards \
    --rank-fraction 0.25 --problems 50 --spans-per-problem 3 --free-run-cap 150 --out reports/span_criticality_n50
```

Summaries land as `*_summary.json` (+ per-trial `.parquet`) under each `--out` directory.
