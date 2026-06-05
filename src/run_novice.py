"""Disentangle all MLP linears of the base model and save the Novice factors."""

from __future__ import annotations

import argparse
import time

import torch
from transformers import AutoModelForCausalLM

try:
    from .disentangled_linear import DisentangledLinear
    from .prepare_qwen import MODEL_NAME, replace_mlp_linears_with_disentangled
    from .novice_io import save_disentangled
except ImportError:
    from disentangled_linear import DisentangledLinear
    from prepare_qwen import MODEL_NAME, replace_mlp_linears_with_disentangled
    from novice_io import save_disentangled


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--rank-fraction", type=float, default=0.1)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--lr", type=float, default=0.05)
    args = parser.parse_args()

    print(f"Loading {MODEL_NAME} (float32) ...", flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, trust_remote_code=True, dtype=torch.float32
    )
    n_replaced = replace_mlp_linears_with_disentangled(model, rank_fraction=args.rank_fraction)
    print(f"Replaced {n_replaced} MLP linear layers.", flush=True)

    targets = [
        (name, module)
        for name, module in model.named_modules()
        if isinstance(module, DisentangledLinear)
    ]
    total = len(targets)
    start = time.perf_counter()
    for index, (name, module) in enumerate(targets, start=1):
        layer_start = time.perf_counter()
        module.disentangle(steps=args.steps, learning_rate=args.lr, device=args.device)
        elapsed = time.perf_counter() - layer_start
        done = time.perf_counter() - start
        eta = done / index * (total - index)
        print(
            f"[{index:2d}/{total}] {name:40s} {elapsed:5.1f}s  "
            f"(elapsed {done/60:4.1f}m, eta {eta/60:4.1f}m)",
            flush=True,
        )

    n_saved = save_disentangled(
        model, args.out, model_name=MODEL_NAME, rank_fraction=args.rank_fraction
    )
    total_time = time.perf_counter() - start
    print(f"Saved {n_saved} disentangled modules to {args.out} in {total_time/60:.1f} min total.", flush=True)


if __name__ == "__main__":
    main()
