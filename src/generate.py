"""Generate tokens from a disentangled Novice model at a chosen inference rank.

Examples
--------
    python src/generate.py --prompt "The capital of France is" --rank-fraction 0.5
    python src/generate.py -p "Once upon a time" -r 0.3 --max-new-tokens 64 --sample

The rank fraction is applied at inference time by re-truncating each
DisentangledLinear's stored factors and baking the result into dense weights, so
one checkpoint covers every rank without re-running disentanglement.
"""

from __future__ import annotations

import argparse
import time

import torch
from tqdm import tqdm
from transformers import StoppingCriteria, StoppingCriteriaList

try:
    from .novice_model import NoviceModel
except ImportError:
    from novice_model import NoviceModel


C_DIM, C_BOLD, C_CYAN, C_GREEN, C_RESET = "\033[2m", "\033[1m", "\033[36m", "\033[32m", "\033[0m"


def stage(label: str) -> None:
    print(f"{C_CYAN}▸{C_RESET} {C_BOLD}{label}{C_RESET}", flush=True)


def kv(key: str, value: str) -> None:
    print(f"  {C_DIM}{key:<14}{C_RESET} {value}", flush=True)


class TokenProgress(StoppingCriteria):
    """Ticks a tqdm bar once per generated token (never stops generation)."""

    def __init__(self, total: int) -> None:
        self.bar = tqdm(total=total, desc="generating", unit="tok", leave=True, dynamic_ncols=True)

    def __call__(self, input_ids, scores, **kwargs) -> bool:
        self.bar.update(1)
        return False

    def close(self) -> None:
        self.bar.close()


def default_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("-p", "--prompt", required=True, help="Prompt text to continue.")
    parser.add_argument("-r", "--rank-fraction", type=float, required=True,
                        help="Inference rank fraction in (0, 1]; lower = weaker Novice.")
    parser.add_argument("-c", "--checkpoint", default="novice_qwen_rf0.1.pt", help="Path to the saved factors.")
    parser.add_argument("-n", "--max-new-tokens", type=int, default=64)
    parser.add_argument("--sample", action="store_true", help="Sample instead of greedy decoding.")
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default=default_device(), help="Compute device (default: cuda > mps > cpu).")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not (0.0 < args.rank_fraction <= 1.0):
        raise SystemExit(f"--rank-fraction must be in (0, 1], got {args.rank_fraction}")
    torch.manual_seed(args.seed)
    t0 = time.perf_counter()

    print(f"\n{C_BOLD}Novice token generation{C_RESET}  {C_DIM}(disentangled Qwen3.5){C_RESET}\n")

    # 1. Load factors directly onto the compute device, so the reconstruction
    #    matmuls in step 2 run there (baking on CPU is ~100x slower).
    stage("Loading disentangled factors")
    kv("checkpoint", args.checkpoint)
    kv("device", args.device)
    novice = NoviceModel.from_checkpoint(args.checkpoint, device=args.device)
    n_layers = novice.num_factored_layers
    kv("base model", novice.model_name)
    kv("disentangled", f"{n_layers} MLP layers  (trained at rank {novice.rank_fraction})")

    # 2. Bake the chosen inference rank into dense weights (one matmul/layer on device).
    keep, full = novice.rank_at(args.rank_fraction)
    stage(f"Baking rank_fraction={args.rank_fraction}")
    kv("kept rank", f"{keep}/{full}  ({keep/full:.0%} kept, {1-keep/full:.0%} truncated)")
    with tqdm(total=n_layers, desc="materializing", unit="layer", leave=True, dynamic_ncols=True) as bar:
        model = novice.bake(args.rank_fraction, in_place=True, progress_callback=bar.update)
    model = model.eval()

    # 3. Tokenize.
    stage("Tokenizing prompt")
    tokenizer = novice.tokenizer
    inputs = tokenizer(args.prompt, return_tensors="pt").to(args.device)
    n_prompt = inputs["input_ids"].shape[1]
    kv("prompt", f'"{args.prompt}"')
    kv("prompt tokens", str(n_prompt))
    kv("decoding", "sampling" if args.sample else "greedy"
       + (f"  (T={args.temperature}, top_p={args.top_p})" if args.sample else ""))

    # 4. Generate with a per-token progress bar.
    stage(f"Generating up to {args.max_new_tokens} tokens")
    gen_kwargs: dict = {"max_new_tokens": args.max_new_tokens, "pad_token_id": tokenizer.eos_token_id}
    gen_kwargs.update(
        {"do_sample": True, "temperature": args.temperature, "top_p": args.top_p}
        if args.sample else {"do_sample": False}
    )
    progress = TokenProgress(args.max_new_tokens)
    t_gen = time.perf_counter()
    with torch.no_grad():
        output = model.generate(**inputs, stopping_criteria=StoppingCriteriaList([progress]), **gen_kwargs)
    progress.close()

    new_ids = output[0, n_prompt:]
    continuation = tokenizer.decode(new_ids, skip_special_tokens=True)
    dt_gen = time.perf_counter() - t_gen
    rate = len(new_ids) / dt_gen if dt_gen > 0 else 0.0

    print(f"\n{C_GREEN}{'─'*64}{C_RESET}")
    print(f"{C_DIM}{args.prompt}{C_RESET}{C_BOLD}{continuation}{C_RESET}")
    print(f"{C_GREEN}{'─'*64}{C_RESET}")
    kv("generated", f"{len(new_ids)} tokens in {dt_gen:.1f}s  ({rate:.1f} tok/s)")
    kv("total time", f"{time.perf_counter()-t0:.1f}s")


if __name__ == "__main__":
    main()
