"""Disentangle a base model and publish the Novice factors + tuning logs to the Hub.

End-to-end: load a base LM, disentangle its MLP linears, then push three things to a
Hugging Face model repo — the Novice checkpoint, the per-step tuning CSV, and a short
generated ``README.md``.

Examples
--------
    # disentangle Qwen at rank 0.1 and push to <your-user>/novice-qwen-rf0.1
    python src/disentangle_and_upload.py --repo-id leodriesch/novice-qwen-rf0.1

    # heavier run, private repo
    python src/disentangle_and_upload.py -r 0.25 --steps 200 \
        --repo-id leodriesch/novice-qwen-rf0.25 --private

Authentication uses the standard ``HF_TOKEN`` environment variable (or a token cached
by ``huggingface-cli login``). Pass ``--token`` to override.
"""

from __future__ import annotations

import argparse
import os
import time

import torch

try:
    from .novice_model import MODEL_NAME, NoviceModel
except ImportError:
    from novice_model import MODEL_NAME, NoviceModel


def build_readme(
    *, repo_id: str, model_name: str, rank_fraction: float, steps: int,
    learning_rate: float, num_layers: int, log_name: str,
) -> str:
    """A short model card describing how the artifact was produced and how to load it."""
    return f"""---
license: apache-2.0
base_model: {model_name}
tags:
  - novice
  - disentanglement
  - low-rank
library_name: pytorch
---

# {repo_id}

A **Novice** checkpoint produced with the Master/Novice framework: the base model's
MLP weights are disentangled into truncatable factors (Q/U/s/Vh) so a weaker model can
be reconstructed at any chosen rank.

## How it was made

| | |
|---|---|
| Base model | [`{model_name}`](https://huggingface.co/{model_name}) |
| Rank fraction | `{rank_fraction}` |
| Disentangle steps | `{steps}` |
| Learning rate | `{learning_rate}` |
| Factored MLP layers | `{num_layers}` |

## Files

- **`*.factors.pt` + `manifest.pt`** — the disentangled factors, stored one shard per
  MLP linear plus a manifest. This sharded layout lets the loader materialize one
  layer at a time, so memory stays near the base model + a single layer (the full
  factor set is several × the model and would otherwise OOM).
- **`{log_name}`** — per-step tuning trace (one row per step per matrix: cost, entropy,
  effective rank) plus the run's hyperparameters, for plotting/analysis.

## Usage

```python
from huggingface_hub import snapshot_download
from novice_model import NoviceModel

# download just the factor shards + manifest
ckpt_dir = snapshot_download("{repo_id}", allow_patterns=["*.pt"])

# loads layer-by-layer, truncating to the chosen rank — memory-flat
fast = NoviceModel.from_sharded_checkpoint(ckpt_dir, rank_fraction={rank_fraction}, device="cuda")
out = fast.generate(**fast.tokenizer("Hello", return_tensors="pt").to(fast.device))
print(fast.tokenizer.decode(out[0]))
```
"""


def _main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--repo-id", required=True, help="Target HF repo, e.g. user/novice-qwen-rf0.1.")
    parser.add_argument("--model", default=MODEL_NAME, help="Base model to disentangle.")
    parser.add_argument("-r", "--rank-fraction", type=float, default=0.1)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--workers", type=int, default=1, help="Parallel layer workers.")
    parser.add_argument("--out", default=None, help="Local checkpoint path (default derived from repo-id).")
    parser.add_argument("--log-csv", default=None, help="Local tuning-log path (default derived from repo-id).")
    parser.add_argument(
        "--checkpoint-dir", default=None,
        help="Dir for streaming per-layer shards (default derived from repo-id). Enables resume.",
    )
    parser.add_argument(
        "--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"],
        help="Storage dtype for model + factors (optimization always runs in float32).",
    )
    parser.add_argument("--private", action="store_true", help="Create the repo as private.")
    parser.add_argument("--token", default=None, help="HF token (else HF_TOKEN / cached login).")
    args = parser.parse_args()

    from huggingface_hub import get_token

    token = (
        args.token
        or os.environ.get("HF_TOKEN")
        or os.environ.get("HUGGING_FACE_HUB_TOKEN")
        or get_token()  # cached `huggingface-cli login` / OAuth credential
    )
    if not token:
        raise SystemExit(
            "No Hugging Face token found. Set HF_TOKEN, run `huggingface-cli login`, "
            "or pass --token."
        )

    slug = args.repo_id.split("/")[-1]
    log_path = args.log_csv or f"{slug}_tuning.csv"
    checkpoint_dir = args.checkpoint_dir or f"{slug}_shards"
    dtype = getattr(torch, args.dtype)

    print(f"Loading {args.model} ({args.dtype}) ...", flush=True)
    novice = NoviceModel.from_pretrained(
        args.model, rank_fraction=args.rank_fraction, dtype=dtype
    )
    print(f"Replaced {novice.num_factored_layers} MLP linear layers.", flush=True)

    start = time.perf_counter()
    # Streaming + resumable: each layer's factors are written to checkpoint_dir and
    # freed as it completes, so RAM stays flat and a killed run resumes in place.
    novice.disentangle(
        steps=args.steps, learning_rate=args.lr, device=args.device,
        workers=args.workers, log_csv=log_path,
        checkpoint_dir=checkpoint_dir, resume=True,
    )
    n_saved = novice.finalize_checkpoint(checkpoint_dir)
    elapsed = (time.perf_counter() - start) / 60
    print(f"Wrote manifest for {n_saved} layers to {checkpoint_dir}/ in {elapsed:.1f} min.", flush=True)

    # Upload: create the repo, write a README, push the shard dir (factors +
    # manifest) and the tuning log. upload_folder streams shards from disk, so we
    # never hold the full factor set in RAM.
    from huggingface_hub import HfApi

    api = HfApi(token=token)
    api.create_repo(args.repo_id, repo_type="model", private=args.private, exist_ok=True)
    print(f"Uploading to https://huggingface.co/{args.repo_id} ...", flush=True)

    readme = build_readme(
        repo_id=args.repo_id, model_name=args.model, rank_fraction=args.rank_fraction,
        steps=args.steps, learning_rate=args.lr, num_layers=n_saved,
        log_name=os.path.basename(log_path),
    )
    api.upload_file(
        path_or_fileobj=readme.encode(), path_in_repo="README.md",
        repo_id=args.repo_id, repo_type="model",
    )
    api.upload_folder(
        folder_path=checkpoint_dir, repo_id=args.repo_id, repo_type="model",
    )
    api.upload_file(
        path_or_fileobj=log_path, path_in_repo=os.path.basename(log_path),
        repo_id=args.repo_id, repo_type="model",
    )

    print(f"Done. https://huggingface.co/{args.repo_id}", flush=True)


if __name__ == "__main__":
    _main()
