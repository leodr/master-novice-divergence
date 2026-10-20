# master-novice-divergence

Research on the Master/Novice framework: disentangling a base LM's MLP weights into
truncatable factors (Q/U/s/Vh) so a weaker "Novice" can be produced at any chosen rank.

## Key files

- `src/novice_model.py` — the `NoviceModel` wrapper, the main entry point. It owns the
  full lifecycle: disentangle → save/load → bake (truncate to a rank) → generate.

## Conventions

- Dependency management uses **uv** (`pyproject.toml` + `uv.lock`). Run things with
  `uv run …` rather than invoking the interpreter directly.
- `src/tensor_disentanglement_code/` is **vendored** — treat it as read-only and
  rarely touch it.
