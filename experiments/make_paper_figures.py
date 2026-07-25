"""Regenerate the paper's figures as LaTeX-native PGF.

Output goes to ``paper/figures/*.pgf`` and is pulled into the paper with
``\\input{figures/<name>.pgf}`` (the paper preamble loads ``pgf``). Text is
typeset by LaTeX itself at save time, so fonts and math match the document and
the result is vector rather than raster.

Usage::

    uv run --no-sync python experiments/make_paper_figures.py

Figures
-------
``lr_sweep``
    The learning-rate sweep, regenerated from ``experiments/lr/lr_sweep.csv``.

``e1_release_depth``
    E1 release-from-here. The aggregate arms are the reported 30/115 and
    31/115; the per-depth-bin curves come from ``leverage_summary.json`` when
    one is supplied via ``--leverage-summary``, and otherwise fall back to
    values read off the original figure (see ``E1_DEPTH_FALLBACK``).
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from collections import defaultdict

import matplotlib

matplotlib.use("pgf")
matplotlib.rcParams.update({
    "pgf.texsystem": "pdflatex",
    "font.family": "serif",
    "text.usetex": True,
    "pgf.rcfonts": False,
    # Match the 9pt document; sizes are in LaTeX points because usetex is on.
    "font.size": 8,
    "axes.titlesize": 8.5,
    "axes.labelsize": 8,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "legend.fontsize": 6.5,
    "axes.linewidth": 0.6,
    "grid.linewidth": 0.4,
    "lines.linewidth": 1.1,
})

import matplotlib.pyplot as plt  # noqa: E402  (must follow the backend switch)

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(REPO, "paper", "figures")
LR_CSV = os.path.join(REPO, "experiments", "lr", "lr_sweep.csv")
MMLU_CSV = os.path.join(REPO, "experiments", "mmlu_rank_sweep.csv")
# Per-step tuning trace of the published rf=0.1 checkpoint; mirror of
# novice-qwen3.5-4b-rf0.1_tuning.csv from the HF repo leodriesch/novice-qwen3.5-4b-rf0.1.
TUNING_CSV = os.path.join(REPO, "experiments", "qwen3.5-4b_rf0.1_tuning.csv")

# Warm (high lr) to cool (low lr), following the original figure.
LR_COLORS = {
    0.05: "#e8404a",
    0.02: "#f28c28",
    0.01: "#f0b429",
    0.007: "#8cc63f",
    0.005: "#22b573",
    0.001: "#2f7fe0",
    0.0005: "#8e5fd3",
}

FLIP_COLOR = "#c0392b"
CONTROL_COLOR = "#2980b9"

# E1 aggregate results as reported in the paper (Table 1) and
# DISAGREEMENT_CREDIT_FINDINGS.md.
E1_SUCCESS_FLIP = 30 / 115
E1_SUCCESS_CONTROL = 31 / 115
E1_PAIRS = 115

# Per-depth-bin success rates digitized from experiments/e1_release_depth.png.
# The run's leverage_summary.json was not kept, so these are accurate to about
# +/-0.01 rather than exact. Pass --leverage-summary to use real counts.
E1_DEPTH_FALLBACK = {
    0: {"flip": 0.034, "control": 0.000},
    1: {"flip": 0.090, "control": 0.118},
    2: {"flip": 0.105, "control": 0.113},
    3: {"flip": 0.680, "control": 0.714},
}


def load_lr_sweep(path: str) -> dict[float, dict[str, list[float]]]:
    """Group the sweep CSV by learning rate, preserving step order."""
    by_lr: dict[float, dict[str, list[float]]] = defaultdict(
        lambda: {"step": [], "cost": [], "effective_rank": []}
    )
    with open(path, newline="") as fh:
        for row in csv.DictReader(fh):
            series = by_lr[float(row["lr"])]
            series["step"].append(float(row["step"]))
            series["cost"].append(float(row["cost"]))
            series["effective_rank"].append(float(row["effective_rank"]))
    return dict(by_lr)


def diverged(cost: list[float]) -> bool:
    """A run diverged if it never worked its cost meaningfully below the start."""
    return cost[-1] > 0.5 * cost[0]


def figure_lr_sweep(out_path: str) -> None:
    by_lr = load_lr_sweep(LR_CSV)
    fig, (ax_cost, ax_rank) = plt.subplots(
        2, 1, figsize=(6.3, 4.3), sharex=True, height_ratios=[1.15, 1]
    )

    for lr in sorted(by_lr, reverse=True):
        series = by_lr[lr]
        color = LR_COLORS.get(lr, "#888888")
        style = "--" if diverged(series["cost"]) else "-"

        best_i = min(range(len(series["cost"])), key=lambda i: series["cost"][i])
        best_cost = series["cost"][best_i]
        best_step = int(series["step"][best_i])
        label = (
            rf"$\eta={lr:g}$ $\rightarrow$ best "
            rf"{best_cost:.2e}".replace("e+0", r"e{+}") + rf" @ step {best_step}"
        )

        ax_cost.plot(series["step"], series["cost"], style, color=color, label=label)
        ax_cost.plot([best_step], [best_cost], "o", color=color, markersize=3)
        ax_rank.plot(
            series["step"], series["effective_rank"], style, color=color,
            label=rf"$\eta={lr:g}$",
        )

    ax_cost.set_yscale("log")
    ax_cost.set_ylabel("Truncation cost (log)")
    ax_cost.set_title("Cost --- lower is better (dashed = diverges)")
    ax_cost.legend(loc="upper right", ncols=1, framealpha=0.9)

    ax_rank.set_ylabel("Effective rank of $A(QW)$")
    ax_rank.set_xlabel("Optimization step")
    ax_rank.set_title("Effective rank --- lower is better")
    ax_rank.legend(loc="upper right", ncols=2, framealpha=0.9)

    for ax in (ax_cost, ax_rank):
        ax.grid(True, alpha=0.25)
        ax.spines[["top", "right"]].set_visible(False)

    fig.tight_layout(pad=0.4)
    fig.savefig(out_path)
    plt.close(fig)


def figure_e1_release_depth(out_path: str, summary_path: str | None) -> None:
    success_flip, success_control = E1_SUCCESS_FLIP, E1_SUCCESS_CONTROL
    pairs = E1_PAIRS
    binned = E1_DEPTH_FALLBACK

    if summary_path:
        with open(summary_path) as fh:
            summary = json.load(fh)
        success_flip = summary["success_flip"]
        success_control = summary["success_control"]
        pairs = summary["release_pairs"]
        binned = {int(k): v for k, v in summary["success_by_depth_bin"].items()}

    # Single-column figure: narrow bar panel + depth curve, titles left to the
    # caption. leverage_gap is control - flip (src/divergence_leverage.py:345):
    # positive means releasing at a flip did worse, as H1 predicts.
    fig, (ax_bar, ax_depth) = plt.subplots(
        1, 2, figsize=(3.05, 1.95), width_ratios=[1, 1.9]
    )

    ax_bar.bar(
        ["flip", "control"], [success_flip, success_control],
        color=[FLIP_COLOR, CONTROL_COLOR], width=0.6,
    )
    for i, v in enumerate([success_flip, success_control]):
        ax_bar.text(i, v + 0.03, f"{v:.2f}", ha="center", va="bottom", fontsize=7)
    ax_bar.set_ylim(0, 1)
    ax_bar.set_ylabel("Success after release")

    bins = sorted(binned)
    ax_depth.plot(bins, [binned[b]["flip"] for b in bins], "o-",
                  color=FLIP_COLOR, markersize=3, label="flip")
    ax_depth.plot(bins, [binned[b]["control"] for b in bins], "o-",
                  color=CONTROL_COLOR, markersize=3, label="control")
    ax_depth.set_ylim(0, 1)
    ax_depth.set_xticks(bins)
    ax_depth.set_xlabel(r"Depth bin (0 = early)")
    ax_depth.legend(loc="upper left", framealpha=0.9)

    for ax in (ax_bar, ax_depth):
        ax.grid(True, axis="y", alpha=0.25)
        ax.spines[["top", "right"]].set_visible(False)
        ax.set_axisbelow(True)

    fig.tight_layout(pad=0.4)
    fig.savefig(out_path)
    plt.close(fig)


def figure_mmlu_rank_sweep(out_path: str) -> None:
    with open(MMLU_CSV, newline="") as fh:
        rows = sorted(
            (
                (float(r["rank_fraction"]), int(r["correct"]), int(r["total"]))
                for r in csv.DictReader(fh)
            ),
            reverse=True,
        )
    ranks = [r for r, _, _ in rows]
    acc = [c / t for _, c, t in rows]
    base = next(c / t for r, c, t in rows if r == 1.0)

    fig, ax = plt.subplots(figsize=(3.05, 2.15))
    ax.axhline(base, ls="--", lw=0.8, color="#7f7f7f",
               label=rf"base model ($\rho=1.0$: {base:.3f})")
    ax.axhline(0.25, ls=":", lw=0.8, color="#c0392b", label="random (0.25)")
    ax.plot(ranks, acc, "o-", color="#2980b9", markersize=3)

    ax.set_xlim(0.05, 1.05)  # conventional ascending axis: weak Novice left
    ax.set_ylim(0.2, 0.8)
    ax.set_xlabel(r"Rank fraction $\rho$")
    ax.set_ylabel("MMLU accuracy (5-shot)")
    ax.grid(True, alpha=0.25)
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_axisbelow(True)
    ax.legend(loc="lower right", framealpha=0.9)

    fig.tight_layout(pad=0.4)
    fig.savefig(out_path)
    plt.close(fig)


def figure_disentanglement_loss(out_path: str) -> None:
    """Cost trajectories of all 96 matrices of the published rf=0.1 run."""
    by_layer: dict[str, dict[str, list[float]]] = defaultdict(
        lambda: {"step": [], "cost": []}
    )
    with open(TUNING_CSV, newline="") as fh:
        for row in csv.DictReader(fh):
            series = by_layer[row["layer"]]
            series["step"].append(float(row["step"]))
            series["cost"].append(float(row["cost"]))

    proj_colors = {
        "gate_proj": "#2980b9",
        "up_proj": "#22b573",
        "down_proj": "#c0392b",
    }

    # Sized for the appendix's one-column layout (matches lr_sweep width).
    fig, ax = plt.subplots(figsize=(6.3, 2.8))
    grouped: dict[str, list[list[float]]] = defaultdict(list)
    for layer, series in by_layer.items():
        proj = layer.rsplit(".", 1)[-1]
        ax.plot(series["step"], series["cost"], "-", color=proj_colors[proj],
                lw=0.45, alpha=0.30)
        grouped[proj].append(series["cost"])

    steps = sorted({s for series in by_layer.values() for s in series["step"]})
    for proj, costs in grouped.items():
        median = [sorted(c[i] for c in costs)[len(costs) // 2]
                  for i in range(len(steps))]
        ax.plot(steps, median, "-", color=proj_colors[proj], lw=1.4,
                label=r"\texttt{%s}" % proj.replace("_", r"\_"))

    ax.set_yscale("log")
    ax.set_xlabel("Optimization step")
    ax.set_ylabel("Truncation cost (log)")
    ax.set_title(r"Disentanglement cost --- 96 matrices")
    ax.grid(True, alpha=0.25)
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_axisbelow(True)
    ax.legend(loc="lower left", framealpha=0.9)

    fig.tight_layout(pad=0.4)
    fig.savefig(out_path)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--leverage-summary",
        help="leverage_summary.json from src/divergence_leverage.py; without it "
             "the per-depth-bin curves use the digitized fallback values.",
    )
    parser.add_argument("--out", default=OUT_DIR, help="output directory")
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)

    lr_path = os.path.join(args.out, "lr_sweep.pgf")
    figure_lr_sweep(lr_path)
    print(f"wrote {lr_path}")

    mmlu_path = os.path.join(args.out, "mmlu_rank_sweep.pgf")
    figure_mmlu_rank_sweep(mmlu_path)
    print(f"wrote {mmlu_path}")

    tuning_path = os.path.join(args.out, "disentanglement_loss.pgf")
    figure_disentanglement_loss(tuning_path)
    print(f"wrote {tuning_path}")

    e1_path = os.path.join(args.out, "e1_release_depth.pgf")
    figure_e1_release_depth(e1_path, args.leverage_summary)
    print(f"wrote {e1_path}")
    if not args.leverage_summary:
        print("  note: depth bins use digitized fallback values")


if __name__ == "__main__":
    main()
