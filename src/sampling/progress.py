"""A ``tqdm`` progress bar driven by ``model.generate()``'s per-token callback."""

from __future__ import annotations

from tqdm import tqdm
from transformers import StoppingCriteria


class TokenProgress(StoppingCriteria):
    """Ticks a tqdm bar once per generated token (never stops generation).

    ``generate()`` has no per-token progress hook, so we register as a
    ``StoppingCriteria`` and advance the bar from its callback, always returning
    ``False`` (keep going). Pass via ``stopping_criteria=StoppingCriteriaList([...])``.
    """

    def __init__(self, total: int, *, desc: str = "generating") -> None:
        self.bar = tqdm(
            total=total, desc=desc, unit="tok", leave=True, dynamic_ncols=True, mininterval=1.0
        )

    def __call__(self, input_ids, scores, **kwargs) -> bool:
        self.bar.update(1)
        return False

    def close(self) -> None:
        self.bar.close()
