"""Recover the MMLU-vs-rank-fraction sweep values from the original figure.

The sweep itself (``src/mmlu_eval.py`` at a series of bake ranks) was never
checked in as data --- only the rendered plot, added in e757ead. This script
digitizes that PNG so the figure can be regenerated from numbers instead of
being re-embedded as a bitmap.

Method
------
1. Locate the axes box and the tick marks, and fit pixel -> data on each axis.
   The fit is cross-checked against the two reference lines drawn in the plot
   (base model 0.737 and random 0.25), which it reproduces to ~0.001.
2. Find the series markers as columns carrying more blue pixels than a mere
   line stroke. This deliberately avoids erosion/blob detection: the marker at
   rank 1.0 is bisected by the dashed base-model line, which splits it into two
   caps that erosion discards.
3. Snap each accuracy to the nearest ``k / 285``. The sweep evaluated MMLU with
   5 questions per subtask over 57 subtasks, so every attainable accuracy is
   such a fraction. Every marker lands within 0.21 of an integer ``k``, which
   is what makes the recovered values exact rather than approximate --- the
   script asserts this, so a bad digitization fails loudly instead of silently
   producing plausible numbers.

Usage::

    uv run --no-sync python experiments/extract_mmlu_sweep.py
"""

from __future__ import annotations

import argparse
import csv
import os

import numpy as np
from PIL import Image

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_PNG = os.path.join(REPO, "experiments", "mmlu_rank_sweep.png")
DEFAULT_CSV = os.path.join(REPO, "experiments", "mmlu_rank_sweep.csv")

# lm-eval: MMLU, 5-shot, --limit 5 per subtask, 57 subtasks.
QUESTIONS_PER_SUBTASK = 5
N_SUBTASKS = 57
TOTAL_QUESTIONS = QUESTIONS_PER_SUBTASK * N_SUBTASKS

# Axis tick labels, read off the figure.
X_TICK_VALUES = [1.0, 0.8, 0.6, 0.4, 0.2]
Y_TICK_VALUES = [0.7, 0.6, 0.5, 0.4, 0.3]

# Reference lines drawn in the plot, used to validate the calibration.
BASE_ACC = 0.737
RANDOM_ACC = 0.25

SERIES_RGB = (31, 119, 180)  # matplotlib C0
MAX_SNAP_RESIDUAL = 0.35     # in units of questions


def _groups(indices: list[int], gap: int = 2) -> list[list[int]]:
    """Split a sorted index list into runs separated by more than ``gap``."""
    out: list[list[int]] = []
    run = [indices[0]]
    for i in indices[1:]:
        if i - run[-1] <= gap:
            run.append(i)
        else:
            out.append(run)
            run = [i]
    out.append(run)
    return out


def extract(png_path: str) -> list[dict[str, float]]:
    im = np.array(Image.open(png_path).convert("RGB")).astype(int)
    dark = im.sum(2) < 200

    # --- axes box: the two longest dark rows/columns are the spines --------- #
    row_counts, col_counts = dark.sum(1), dark.sum(0)
    top, bottom = sorted(np.argsort(-row_counts)[:2])
    left, right = sorted(np.argsort(-col_counts)[:2])

    # --- ticks just outside the spines ------------------------------------- #
    faint = im.sum(2) < 250
    xt = np.where(faint[bottom + 2:bottom + 7, :].sum(0) >= 3)[0]
    yt = np.where(faint[:, left - 6:left - 1].sum(1) >= 3)[0]
    x_px = [float(np.mean(g)) for g in _groups(xt.tolist())]
    y_px = [float(np.mean(g)) for g in _groups(yt.tolist())]
    if len(x_px) != len(X_TICK_VALUES) or len(y_px) != len(Y_TICK_VALUES):
        raise RuntimeError(f"expected {len(X_TICK_VALUES)}/{len(Y_TICK_VALUES)} "
                           f"ticks, found {len(x_px)}/{len(y_px)}")

    ax, bx = np.polyfit(x_px, X_TICK_VALUES, 1)
    ay, by = np.polyfit(y_px, Y_TICK_VALUES, 1)

    # --- validate against the reference lines ------------------------------ #
    r, g, b = im[:, :, 0], im[:, :, 1], im[:, :, 2]
    gray = (abs(r - g) < 6) & (abs(g - b) < 6) & (r > 120) & (r < 210)
    red = (r > 200) & (g < 200) & (b < 200) & (abs(g - b) < 25)
    base_row = int(np.argmax(gray.sum(1)))
    random_row = int(np.argmax(red.sum(1)))
    for row, expected, name in ((base_row, BASE_ACC, "base model"),
                                (random_row, RANDOM_ACC, "random")):
        got = ay * row + by
        if abs(got - expected) > 0.005:
            raise RuntimeError(f"calibration off on the {name} line: "
                               f"got {got:.4f}, expected {expected}")

    # --- series markers ----------------------------------------------------- #
    blue = ((abs(r - SERIES_RGB[0]) < 50) & (abs(g - SERIES_RGB[1]) < 50)
            & (abs(b - SERIES_RGB[2]) < 50))
    blue[:, :left + 1] = False       # legend and anything outside the axes
    blue[:, right:] = False
    blue[bottom + 1:, :] = False

    counts = blue.sum(0)
    marker_cols = np.where(counts >= 7)[0]   # a bare line stroke is ~2-3 px
    rows: list[dict[str, float]] = []
    for group in _groups(marker_cols.tolist()):
        ys, xs = np.where(blue[:, group[0]:group[-1] + 1])
        rank = ax * (group[0] + xs.mean()) + bx
        acc = ay * ys.mean() + by

        k = acc * TOTAL_QUESTIONS
        correct = round(k)
        if abs(k - correct) > MAX_SNAP_RESIDUAL:
            raise RuntimeError(
                f"rank {rank:.3f}: accuracy {acc:.4f} is {abs(k - correct):.2f} "
                f"questions from an integer count; digitization is unreliable"
            )
        rows.append({
            "rank_fraction": round(rank, 1),
            "correct": correct,
            "total": TOTAL_QUESTIONS,
            "accuracy": correct / TOTAL_QUESTIONS,
            "digitized_accuracy": acc,
        })

    rows.sort(key=lambda d: -d["rank_fraction"])
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--png", default=DEFAULT_PNG)
    parser.add_argument("--out", default=DEFAULT_CSV)
    args = parser.parse_args()

    rows = extract(args.png)
    with open(args.out, "w", newline="") as fh:
        writer = csv.DictWriter(
            fh, fieldnames=["rank_fraction", "correct", "total", "accuracy"]
        )
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row[k] for k in writer.fieldnames})

    print(f"wrote {args.out}")
    print(f"{'rank':>6} {'correct':>8} {'accuracy':>9}  {'digitized':>9}")
    for row in rows:
        print(f"{row['rank_fraction']:6.1f} {row['correct']:8d} "
              f"{row['accuracy']:9.4f}  {row['digitized_accuracy']:9.4f}")


if __name__ == "__main__":
    main()
