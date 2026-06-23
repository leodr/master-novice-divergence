"""Evaluate a disentangled Novice model on (a subset of) MMLU.

Loads the Novice factors, bakes them to a chosen inference rank, and runs MMLU
through ``lm-eval``. The subset size is configurable via ``--limit`` so you can
smoke-test on a handful of questions per task before committing to a full run.

Examples
--------
    # 5 questions per MMLU subtask at rank 0.5 (quick smoke test)
    python src/mmlu_eval.py --rank-fraction 0.5 --limit 5

    # full MMLU at full rank, write the raw lm-eval results to JSON
    python src/mmlu_eval.py -r 1.0 --output results_rf1.0.json

    # 10% of each subtask, 0-shot
    python src/mmlu_eval.py -r 0.3 --limit 0.1 --num-fewshot 0

``--limit`` accepts an integer (questions per subtask) or a float in (0, 1]
(fraction of each subtask). Omit it to run the full benchmark.
"""

from __future__ import annotations

import argparse
import json
import os
import time

import torch

try:
    from .novice_model import MANIFEST_NAME, MODEL_NAME, NoviceModel
except ImportError:
    from novice_model import MANIFEST_NAME, MODEL_NAME, NoviceModel


C_DIM, C_BOLD, C_CYAN, C_GREEN, C_RESET = "\033[2m", "\033[1m", "\033[36m", "\033[32m", "\033[0m"


def stage(label: str) -> None:
    print(f"{C_CYAN}▸{C_RESET} {C_BOLD}{label}{C_RESET}", flush=True)


def kv(key: str, value: str) -> None:
    print(f"  {C_DIM}{key:<14}{C_RESET} {value}", flush=True)


def default_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def parse_limit(value: str) -> int | float:
    """``--limit`` is a per-subtask question count (int) or a fraction (float in (0, 1])."""
    number = float(value)
    if number.is_integer() and number >= 1:
        return int(number)
    if 0.0 < number < 1.0:
        return number
    raise argparse.ArgumentTypeError("--limit must be an integer >= 1 or a fraction in (0, 1)")


def parse_dtype(value: str) -> torch.dtype:
    dtypes = {
        "float16": torch.float16,
        "fp16": torch.float16,
        "float32": torch.float32,
        "fp32": torch.float32,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
    }
    try:
        return dtypes[value.lower()]
    except KeyError as exc:
        choices = ", ".join(sorted(dtypes))
        raise argparse.ArgumentTypeError(f"--dtype must be one of: {choices}") from exc


def dtype_name(dtype: torch.dtype) -> str:
    return str(dtype).removeprefix("torch.")


def model_dtype(model: torch.nn.Module) -> torch.dtype:
    return next(model.parameters()).dtype


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("-c", "--checkpoint", default="novice_qwen_rf0.1.pt",
                        help="Path to the saved Novice factors.")
    parser.add_argument("-r", "--rank-fraction", type=float, default=None,
                        help="Inference rank fraction in (0, 1]; lower = weaker Novice. "
                             "Required unless --baseline.")
    parser.add_argument("--baseline", action="store_true",
                        help="Evaluate the untouched base model (no disentanglement); "
                             "ignores --checkpoint/--rank-fraction.")
    parser.add_argument("--model", default=MODEL_NAME,
                        help="Base model for --baseline (default: the Novice's base model).")
    parser.add_argument("--limit", type=parse_limit, default=None,
                        help="Subset size: int questions/subtask or float fraction in (0,1). "
                             "Omit for the full benchmark.")
    parser.add_argument("--tasks", nargs="+", default=["mmlu"],
                        help="lm-eval task names (default: mmlu).")
    parser.add_argument("--num-fewshot", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--dtype", type=parse_dtype, default=torch.float16,
                        help="Model dtype for eval: float16, bfloat16, or float32 (default: float16).")
    parser.add_argument("--output", help="Optional path to write the raw lm-eval results as JSON.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default=default_device(),
                        help="Compute device (default: cuda > mps > cpu).")
    return parser.parse_args()


def first_metric(metrics: dict, names: list[str]) -> float | None:
    for name in names:
        if name in metrics and isinstance(metrics[name], (int, float)):
            return float(metrics[name])
    return None


def extract_scores(results: dict, primary_task: str) -> dict:
    """Pull acc / acc_norm out of an lm-eval results dict, aggregating subtasks if needed."""
    result_dict = results["results"]
    if primary_task in result_dict:
        metrics = result_dict[primary_task]
        return {
            "acc": first_metric(metrics, ["acc,none", "acc"]),
            "acc_stderr": first_metric(metrics, ["acc_stderr,none", "acc_stderr"]),
            "acc_norm": first_metric(metrics, ["acc_norm,none", "acc_norm"]),
            "acc_norm_stderr": first_metric(metrics, ["acc_norm_stderr,none", "acc_norm_stderr"]),
            "num_tasks": 1,
        }

    acc_values, acc_norm_values = [], []
    for metrics in result_dict.values():
        if not isinstance(metrics, dict):
            continue
        acc = first_metric(metrics, ["acc,none", "acc"])
        acc_norm = first_metric(metrics, ["acc_norm,none", "acc_norm"])
        if acc is not None:
            acc_values.append(acc)
        if acc_norm is not None:
            acc_norm_values.append(acc_norm)
    return {
        "acc": sum(acc_values) / len(acc_values) if acc_values else None,
        "acc_stderr": None,
        "acc_norm": sum(acc_norm_values) / len(acc_norm_values) if acc_norm_values else None,
        "acc_norm_stderr": None,
        "num_tasks": len(result_dict),
    }


def fmt(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.4f}"


def main() -> None:
    args = parse_args()
    if not args.baseline and args.rank_fraction is None:
        raise SystemExit("--rank-fraction is required unless --baseline is set.")
    if args.rank_fraction is not None and not (0.0 < args.rank_fraction <= 1.0):
        raise SystemExit(f"--rank-fraction must be in (0, 1], got {args.rank_fraction}")
    torch.manual_seed(args.seed)
    t0 = time.perf_counter()

    print(f"\n{C_BOLD}Novice MMLU evaluation{C_RESET}  {C_DIM}(disentangled Qwen3.5){C_RESET}\n")

    # Import lm-eval lazily so --help works without the (heavy) eval stack installed.
    from lm_eval import evaluator
    from lm_eval.models.huggingface import HFLM

    if args.baseline:
        # Untouched base model — the reference point for the disentangled runs.
        from transformers import AutoModelForCausalLM, AutoTokenizer

        stage("Loading base model (no disentanglement)")
        kv("model", args.model)
        kv("device", args.device)
        kv("dtype", dtype_name(args.dtype))
        hf_model = AutoModelForCausalLM.from_pretrained(
            args.model, trust_remote_code=True, dtype=args.dtype
        ).to(device=args.device, dtype=args.dtype).eval()
        tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
    elif os.path.isdir(args.checkpoint) or os.path.exists(
        os.path.join(os.path.dirname(args.checkpoint), MANIFEST_NAME)
    ):
        # Sharded checkpoint (shard dir + manifest): bake the chosen rank one layer
        # at a time so we never hold the full factor set in memory.
        ckpt_dir = args.checkpoint if os.path.isdir(args.checkpoint) else os.path.dirname(args.checkpoint)
        stage("Loading sharded factors (baking layer-by-layer)")
        kv("checkpoint", ckpt_dir)
        kv("device", args.device)
        kv("dtype", dtype_name(args.dtype))
        stage(f"Baking rank_fraction={args.rank_fraction}")
        baked = NoviceModel.from_sharded_checkpoint(
            ckpt_dir, rank_fraction=args.rank_fraction, device=args.device, baked=True
        ).eval()
        kv("base model", baked.model_name)
        hf_model, tokenizer = baked.model, baked.tokenizer
        hf_model.to(device=args.device, dtype=args.dtype)
    else:
        # Load factors onto the compute device and bake the chosen inference rank into
        # dense weights — MMLU runs many forward passes, so the factored forward is too slow.
        stage("Loading disentangled factors")
        kv("checkpoint", args.checkpoint)
        kv("device", args.device)
        kv("dtype", dtype_name(args.dtype))
        novice = NoviceModel.from_checkpoint(args.checkpoint, device=args.device)
        kv("base model", novice.model_name)
        kv("disentangled", f"{novice.num_factored_layers} MLP layers  (trained at rank {novice.rank_fraction})")

        keep, full = novice.rank_at(args.rank_fraction)
        stage(f"Baking rank_fraction={args.rank_fraction}")
        kv("kept rank", f"{keep}/{full}  ({keep/full:.0%} kept, {1-keep/full:.0%} truncated)")
        baked = novice.bake(args.rank_fraction, in_place=True).eval()
        hf_model, tokenizer = baked.model, baked.tokenizer
        hf_model.to(device=args.device, dtype=args.dtype)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    kv("actual dtype", dtype_name(model_dtype(hf_model)))

    # Wrap the HF model for lm-eval and run the (optionally subset) benchmark.
    stage("Evaluating")
    kv("tasks", ", ".join(args.tasks))
    kv("num_fewshot", str(args.num_fewshot))
    kv("limit", "full" if args.limit is None else str(args.limit))
    kv("batch size", str(args.batch_size))

    lm = HFLM(
        pretrained=hf_model,
        tokenizer=tokenizer,
        batch_size=args.batch_size,
        max_batch_size=args.batch_size,
        max_length=args.max_length,
        device=args.device,
        trust_remote_code=True,
    )
    t_eval = time.perf_counter()
    with torch.inference_mode():
        results = evaluator.simple_evaluate(
            model=lm,
            tasks=args.tasks,
            num_fewshot=args.num_fewshot,
            limit=args.limit,
            log_samples=False,
            verbosity="ERROR",
        )
    dt_eval = time.perf_counter() - t_eval

    scores = extract_scores(results, args.tasks[0])

    print(f"\n{C_GREEN}{'─'*64}{C_RESET}")
    print(f"{C_BOLD}{args.tasks[0]}{C_RESET}  "
          f"acc={fmt(scores['acc'])}  acc_norm={fmt(scores['acc_norm'])}  "
          f"{C_DIM}(over {scores['num_tasks']} task(s)){C_RESET}")
    print(f"{C_GREEN}{'─'*64}{C_RESET}")
    kv("acc", f"{fmt(scores['acc'])}  (± {fmt(scores['acc_stderr'])})")
    kv("acc_norm", f"{fmt(scores['acc_norm'])}  (± {fmt(scores['acc_norm_stderr'])})")
    kv("eval time", f"{dt_eval/60:.1f} min")
    kv("total time", f"{(time.perf_counter()-t0)/60:.1f} min")

    if args.output:
        with open(args.output, "w") as handle:
            json.dump(results, handle, indent=2, default=str)
        kv("results", args.output)


if __name__ == "__main__":
    main()
