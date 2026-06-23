"""``NoviceModel``: a transformers-compatible wrapper around a disentangled LM.

Disentanglement is leaf-level surgery — individual ``nn.Linear`` modules inside the
decoder MLPs are swapped for :class:`DisentangledLinear`. The top-level object stays
a genuine ``PreTrainedModel`` (e.g. ``Qwen…ForCausalLM``), so ``.generate()``,
KV-cache, and attention keep working unchanged.

This class does not subclass ``PreTrainedModel``; it *owns* one and delegates
generation/forward to it. What it adds is the Novice lifecycle in one place:

    disentangle → save / load → bake (truncate to a chosen rank) → generate

Typical use::

    novice = NoviceModel.from_pretrained("Qwen/Qwen3.5-0.8B-Base", rank_fraction=0.1)
    novice.disentangle(steps=100, device="cuda")
    novice.save("novice_qwen_rf0.1.pt")

    novice = NoviceModel.from_checkpoint("novice_qwen_rf0.1.pt", device="cuda")
    fast = novice.bake(rank_fraction=0.5)          # non-destructive: returns a copy
    out = fast.generate(**tokenizer("Hello", return_tensors="pt").to(fast.device))

Run as a script to disentangle a base model and save its factors::

    python src/novice_model.py --out novice_qwen_rf0.1.pt --rank-fraction 0.1 --steps 100
"""

from __future__ import annotations

import contextlib
import copy
import csv
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer

try:
    from .disentangled_linear import DisentangledLinear
except ImportError:
    from disentangled_linear import DisentangledLinear


MODEL_NAME = "Qwen/Qwen3.5-0.8B-Base"
DEFAULT_RANK_FRACTION = 0.1
CHECKPOINT_FORMAT = "novice_disentangled_v1"


class NoviceModel:
    """Wraps a causal-LM whose MLP linears are (or will be) disentangled."""

    def __init__(
        self,
        model: nn.Module,
        tokenizer=None,
        *,
        model_name: str,
        rank_fraction: float,
        baked: bool = False,
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.model_name = model_name
        self.rank_fraction = rank_fraction
        self._baked = baked

    # ------------------------------------------------------------------ #
    # construction
    # ------------------------------------------------------------------ #
    @classmethod
    def from_pretrained(
        cls,
        model_name: str = MODEL_NAME,
        *,
        rank_fraction: float = DEFAULT_RANK_FRACTION,
        device: torch.device | str = "cpu",
        dtype: torch.dtype | None = torch.float32,
        trust_remote_code: bool = True,
    ) -> "NoviceModel":
        """Load a base model and swap its MLP linears for (not-yet-disentangled)
        ``DisentangledLinear`` layers. Call :meth:`disentangle` next."""
        model = AutoModelForCausalLM.from_pretrained(
            model_name, trust_remote_code=trust_remote_code, dtype=dtype
        )
        cls._swap_mlp_linears(model, rank_fraction)
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=trust_remote_code)
        return cls(
            model.to(device), tokenizer, model_name=model_name, rank_fraction=rank_fraction
        )

    @classmethod
    def from_checkpoint(
        cls,
        path: str,
        *,
        device: torch.device | str = "cpu",
        trust_remote_code: bool = True,
    ) -> "NoviceModel":
        """Rebuild a disentangled Novice from a :meth:`save` checkpoint.

        Only the factors were saved; the untouched (non-MLP) weights are restored
        by re-loading the base model, then each MLP linear is swapped back to a
        ``DisentangledLinear`` and re-populated from the stored Q/U/s/Vh factors.

        Both the base weights (via ``device_map``) and the factors (via
        ``map_location``) load straight onto ``device``, avoiding a CPU→device
        staging copy of the multi-GB factor set.
        """
        device = str(device)
        checkpoint = torch.load(path, map_location=device, weights_only=False)
        if checkpoint.get("format") != CHECKPOINT_FORMAT:
            raise ValueError(f"Unexpected checkpoint format: {checkpoint.get('format')!r}")
        model_name = checkpoint["model_name"]
        rank_fraction = checkpoint["rank_fraction"]

        model = AutoModelForCausalLM.from_pretrained(
            model_name, trust_remote_code=trust_remote_code, device_map=device
        )
        cls._swap_mlp_linears(model, rank_fraction)

        modules_by_name = dict(model.named_modules())
        for name, data in checkpoint["modules"].items():
            module = modules_by_name[name]
            if not isinstance(module, DisentangledLinear):
                raise RuntimeError(f"Expected DisentangledLinear at {name!r}, got {type(module)}")
            module.load_factors(data)
            module.rank_fraction = rank_fraction

        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=trust_remote_code)
        return cls(model, tokenizer, model_name=model_name, rank_fraction=rank_fraction)

    @staticmethod
    def _swap_mlp_linears(model: nn.Module, rank_fraction: float) -> int:
        """Replace every ``nn.Linear`` below a decoder ``mlp`` block with a
        ``DisentangledLinear``. Returns the number of layers swapped."""

        def replace_descendants(module: nn.Module) -> int:
            replaced = 0
            for child_name, child in list(module.named_children()):
                if isinstance(child, DisentangledLinear):
                    continue
                if isinstance(child, nn.Linear):
                    setattr(
                        module,
                        child_name,
                        DisentangledLinear.from_linear(child, rank_fraction=rank_fraction),
                    )
                    replaced += 1
                else:
                    replaced += replace_descendants(child)
            return replaced

        replaced = 0
        for module_name, module in list(model.named_modules()):
            if module_name == "mlp" or module_name.endswith(".mlp"):
                replaced += replace_descendants(module)
        return replaced

    # ------------------------------------------------------------------ #
    # lifecycle
    # ------------------------------------------------------------------ #
    def disentangle(
        self,
        *,
        steps: int = 100,
        learning_rate: float = 0.001,
        device: str | torch.device | None = None,
        verbose: bool = True,
        workers: int = 1,
        log_csv: str | None = None,
        checkpoint_dir: str | None = None,
        resume: bool = True,
    ) -> "NoviceModel":
        """Disentangle every MLP ``DisentangledLinear`` in place.

        ``workers`` > 1 processes that many layers concurrently on separate CUDA
        streams. Only useful when the per-layer VRAM footprint leaves headroom —
        each large layer (gate/up-proj) uses ~1.4 GB of working memory.

        The per-step optimization trace of every matrix (cost / entropy /
        effective_rank) is always written as CSV, one row per step per matrix,
        along with the run's hyperparameters — handy for later plots. ``log_csv``
        chooses the path; when omitted it defaults to a timestamped file in the
        working directory.

        ``checkpoint_dir`` turns on **streaming, resumable** disentanglement: each
        layer's factors are written to a per-layer shard as soon as they're
        computed and then dropped from memory, so RAM stays flat regardless of
        model size (the factors are several times larger than the weights they
        replace). With ``resume`` (default), layers whose shard already exists are
        skipped — so a run killed partway through picks up where it left off when
        re-invoked. Assemble the shards into a single checkpoint afterwards with
        :meth:`finalize_checkpoint`.
        """
        if self._baked:
            raise RuntimeError("Cannot disentangle a baked model (factors are gone).")
        targets = self.factored_layers
        total = len(targets)
        start = time.perf_counter()

        if checkpoint_dir is not None:
            os.makedirs(checkpoint_dir, exist_ok=True)

        if log_csv is None:
            log_csv = f"disentangle_tuning_{time.strftime('%Y%m%d_%H%M%S')}.csv"
        # Append (don't clobber) when resuming into an existing log.
        append = bool(checkpoint_dir) and resume and os.path.exists(log_csv)
        log = self._open_step_log(log_csv, steps=steps, learning_rate=learning_rate, append=append)
        if verbose:
            print(f"Logging per-step tuning to {log_csv}", flush=True)

        def _persist(name: str, module: "DisentangledLinear") -> None:
            """Stream one layer's factors to its shard and free them from RAM."""
            if checkpoint_dir is None:
                return
            torch.save(module.export_factors(), self._shard_path(checkpoint_dir, name))
            module.discard_factors()

        def _is_done(name: str) -> bool:
            return (
                checkpoint_dir is not None
                and resume
                and os.path.exists(self._shard_path(checkpoint_dir, name))
            )

        if workers <= 1:
            done = 0
            for index, (name, module) in enumerate(targets, start=1):
                if _is_done(name):
                    if verbose:
                        print(f"[{index:2d}/{total}] {name:40s} resumed (shard exists)", flush=True)
                    continue
                layer_start = time.perf_counter()
                module.disentangle(steps=steps, learning_rate=learning_rate, device=device)
                if log is not None:
                    log(name, module)
                _persist(name, module)
                done += 1
                if verbose:
                    elapsed = time.perf_counter() - start
                    eta = elapsed / done * (total - index)
                    print(
                        f"[{index:2d}/{total}] {name:40s} "
                        f"{time.perf_counter() - layer_start:5.1f}s  "
                        f"(elapsed {elapsed / 60:4.1f}m, eta {eta / 60:4.1f}m)",
                        flush=True,
                    )
            return self

        use_streams = device is not None and "cuda" in str(device)
        lock = threading.Lock()
        done_count = [0]

        def _run(name_module: tuple[str, "DisentangledLinear"]) -> None:
            name, module = name_module
            if _is_done(name):
                if verbose:
                    with lock:
                        print(f"[--/{total}] {name:40s} resumed (shard exists)", flush=True)
                return
            layer_start = time.perf_counter()
            stream = torch.cuda.Stream(device=device) if use_streams else None
            ctx = torch.cuda.stream(stream) if stream else contextlib.nullcontext()
            with ctx:
                module.disentangle(steps=steps, learning_rate=learning_rate, device=device)
            if stream:
                stream.synchronize()
            with lock:
                if log is not None:
                    log(name, module)
                _persist(name, module)
                done_count[0] += 1
                elapsed = time.perf_counter() - start
                eta = elapsed / done_count[0] * (total - done_count[0])
                if verbose:
                    print(
                        f"[{done_count[0]:2d}/{total}] {name:40s} "
                        f"{time.perf_counter() - layer_start:5.1f}s  "
                        f"(elapsed {elapsed / 60:4.1f}m, eta {eta / 60:4.1f}m)",
                        flush=True,
                    )

        with ThreadPoolExecutor(max_workers=workers) as pool:
            list(pool.map(_run, targets))

        return self

    @staticmethod
    def _shard_path(checkpoint_dir: str, layer_name: str) -> str:
        return os.path.join(checkpoint_dir, f"{layer_name}.factors.pt")

    def finalize_checkpoint(self, checkpoint_dir: str, path: str) -> int:
        """Assemble per-layer shards written during streaming :meth:`disentangle`
        into a single checkpoint in the same format as :meth:`save`. Returns the
        number of layers written.
        """
        modules: dict[str, Any] = {}
        for name, _ in self.factored_layers:
            shard = self._shard_path(checkpoint_dir, name)
            if not os.path.exists(shard):
                raise FileNotFoundError(f"Missing factor shard for {name!r}: {shard}")
            modules[name] = torch.load(shard, map_location="cpu", weights_only=False)
        torch.save(
            {
                "format": CHECKPOINT_FORMAT,
                "model_name": self.model_name,
                "rank_fraction": self.rank_fraction,
                "modules": modules,
            },
            path,
        )
        return len(modules)

    def _open_step_log(
        self, path: str | None, *, steps: int, learning_rate: float, append: bool = False
    ) -> Callable[[str, "DisentangledLinear"], None] | None:
        """Return a callback that appends a disentangled layer's per-step history to
        ``path`` as CSV (one row per step), or ``None`` when ``path`` is falsy.

        The returned callback is not thread-safe; callers serialize it under a lock.
        """
        if not path:
            return None

        fields = [
            "model_name", "rank_fraction", "learning_rate", "steps",
            "layer", "keep_rank", "max_rank",
            "step", "cost", "entropy", "effective_rank",
        ]
        handle = open(path, "a" if append else "w", newline="")
        writer = csv.DictWriter(handle, fieldnames=fields)
        if not append:
            writer.writeheader()

        def log(name: str, module: "DisentangledLinear") -> None:
            for record in module.history or []:
                writer.writerow({
                    "model_name": self.model_name,
                    "rank_fraction": self.rank_fraction,
                    "learning_rate": learning_rate,
                    "steps": steps,
                    "layer": name,
                    "keep_rank": module.keep_rank,
                    "max_rank": module.max_rank,
                    "step": record["step"],
                    "cost": record["cost"],
                    "entropy": record["entropy"],
                    "effective_rank": record["effective_rank"],
                })
            handle.flush()

        return log

    def save(self, path: str) -> int:
        """Persist the disentangled factors only. Returns the number of layers written.

        A disentangled ``DisentangledLinear`` stores its factors (Q, U, s, Vh) as
        parameters but keeps ``shape_4d`` as a plain Python attribute and swaps its
        ``weight`` for ``None`` — so a vanilla ``state_dict`` round-trip cannot
        reconstruct it. We persist exactly what :meth:`from_checkpoint` needs; the
        base weights are restored from ``model_name`` on load, keeping the file small.
        """
        modules: dict[str, Any] = {
            name: module.export_factors()
            for name, module in self.model.named_modules()
            if isinstance(module, DisentangledLinear) and module.is_disentangled
        }
        torch.save(
            {
                "format": CHECKPOINT_FORMAT,
                "model_name": self.model_name,
                "rank_fraction": self.rank_fraction,
                "modules": modules,
            },
            path,
        )
        return len(modules)

    def set_rank_fraction(self, rank_fraction: float) -> "NoviceModel":
        """Update the inference rank on every factored layer (non-destructive).

        Generation works in this factored mode too — just slower, since each
        forward recomputes the reconstruction. Use :meth:`bake` for fast inference.
        """
        self.rank_fraction = rank_fraction
        for _, module in self.factored_layers:
            module.rank_fraction = rank_fraction
        return self

    def bake(
        self,
        rank_fraction: float | None = None,
        *,
        in_place: bool = False,
        progress_callback: Callable[[], None] | None = None,
    ) -> "NoviceModel":
        """Truncate to ``rank_fraction`` and replace factors with dense ``nn.Linear``
        for fast generation.

        Disentangling is for live experimentation, but its ``forward`` recomputes
        the full ``Qᵀ·A⁻¹(U s Vh)`` reconstruction on *every* call — far too slow
        for token-by-token generation. Baking the chosen rank into a static weight
        makes inference a normal matmul.

        By default this is **non-destructive**: the factored model is left intact and
        a freshly baked copy is returned, so you can bake any number of ranks from one
        Novice. Pass ``in_place=True`` to bake this model and free the factors.
        """
        if self._baked:
            raise RuntimeError("Model is already baked.")
        rf = rank_fraction if rank_fraction is not None else self.rank_fraction
        target = self.model if in_place else copy.deepcopy(self.model)

        # Replace each disentangled layer with a plain Linear holding its truncated weight.
        replacements: list[tuple[nn.Module, str, nn.Linear]] = []
        for name, module in target.named_modules():
            if isinstance(module, DisentangledLinear) and module.is_disentangled:
                module.rank_fraction = rf
                parent_name, _, attr = name.rpartition(".")
                parent = target.get_submodule(parent_name) if parent_name else target
                replacements.append((parent, attr, module.to_dense_linear(truncate=True)))
                if progress_callback is not None:
                    progress_callback()
        for parent, attr, dense in replacements:
            setattr(parent, attr, dense)

        if in_place:
            self.rank_fraction = rf
            self._baked = True
            return self
        return NoviceModel(
            target, self.tokenizer, model_name=self.model_name, rank_fraction=rf, baked=True
        )

    # ------------------------------------------------------------------ #
    # transformers-compatible passthrough
    # ------------------------------------------------------------------ #
    def generate(self, *args, **kwargs):
        return self.model.generate(*args, **kwargs)

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)

    __call__ = forward

    def to(self, *args, **kwargs) -> "NoviceModel":
        self.model.to(*args, **kwargs)
        return self

    def eval(self) -> "NoviceModel":
        self.model.eval()
        return self

    def train(self, mode: bool = True) -> "NoviceModel":
        self.model.train(mode)
        return self

    # ------------------------------------------------------------------ #
    # introspection
    # ------------------------------------------------------------------ #
    @property
    def config(self):
        return self.model.config

    @property
    def device(self) -> torch.device:
        return next(self.model.parameters()).device

    @property
    def is_baked(self) -> bool:
        return self._baked

    @property
    def factored_layers(self) -> list[tuple[str, DisentangledLinear]]:
        """(name, module) for every ``DisentangledLinear`` in the model."""
        return [
            (name, module)
            for name, module in self.model.named_modules()
            if isinstance(module, DisentangledLinear)
        ]

    @property
    def num_factored_layers(self) -> int:
        return len(self.factored_layers)

    @property
    def is_disentangled(self) -> bool:
        """True once at least one factored layer has been disentangled."""
        return any(m.is_disentangled for _, m in self.factored_layers)

    def rank_at(self, rank_fraction: float | None = None) -> tuple[int, int]:
        """(kept_rank, max_rank) for the first factored layer at ``rank_fraction``.

        Representative only — layers of different shapes keep different ranks.
        """
        layers = self.factored_layers
        if not layers:
            raise RuntimeError("No factored layers (model is baked or not disentangled).")
        layer = layers[0][1]
        if rank_fraction is not None:
            layer.rank_fraction = rank_fraction
        return layer.keep_rank, layer.max_rank

    def __getattr__(self, name: str):
        # Fall back to the wrapped model for anything not defined here.
        # Guard against recursion before ``model`` is set.
        try:
            model = object.__getattribute__(self, "model")
        except AttributeError:
            raise AttributeError(name) from None
        return getattr(model, name)

    def __repr__(self) -> str:
        state = "baked" if self._baked else ("disentangled" if self.is_disentangled else "dense")
        return (
            f"NoviceModel(model_name={self.model_name!r}, rank_fraction={self.rank_fraction}, "
            f"state={state}, layers={self.num_factored_layers})"
        )


def _main() -> None:
    """CLI: disentangle a base model's MLP linears and save the Novice factors."""
    import argparse

    parser = argparse.ArgumentParser(description=_main.__doc__)
    parser.add_argument("--out", required=True, help="Path to write the Novice factors.")
    parser.add_argument("--model", default=MODEL_NAME, help="Base model to disentangle.")
    parser.add_argument("--rank-fraction", type=float, default=DEFAULT_RANK_FRACTION)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--workers", type=int, default=1, help="Parallel layer workers.")
    args = parser.parse_args()

    print(f"Loading {args.model} (float32) ...", flush=True)
    novice = NoviceModel.from_pretrained(
        args.model, rank_fraction=args.rank_fraction, dtype=torch.float32
    )
    print(f"Replaced {novice.num_factored_layers} MLP linear layers.", flush=True)

    start = time.perf_counter()
    novice.disentangle(steps=args.steps, learning_rate=args.lr, device=args.device, workers=args.workers)
    n_saved = novice.save(args.out)
    print(
        f"Saved {n_saved} disentangled modules to {args.out} "
        f"in {(time.perf_counter() - start) / 60:.1f} min total.",
        flush=True,
    )


if __name__ == "__main__":
    _main()
