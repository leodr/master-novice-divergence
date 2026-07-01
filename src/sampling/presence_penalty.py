"""OpenAI/vLLM-style presence penalty for Hugging Face generation."""

from __future__ import annotations

import math
from typing import Optional, Sequence, Union

import torch
from transformers import LogitsProcessor


class PresencePenaltyLogitsProcessor(LogitsProcessor):
    """OpenAI/vLLM-style presence penalty for generated tokens only.

    Every token that has appeared in the generated output gets its next-token
    logit shifted by ``-presence_penalty``. Repeated tokens are penalized once,
    not once per occurrence.
    """

    def __init__(
        self,
        presence_penalty: float,
        prompt_lengths: Optional[Union[int, Sequence[int], torch.LongTensor]] = None,
    ) -> None:
        if not math.isfinite(presence_penalty):
            raise ValueError("presence_penalty must be finite")
        self.presence_penalty = float(presence_penalty)
        self.prompt_lengths = prompt_lengths

    def _resolve_prompt_lengths(self, input_ids: torch.LongTensor) -> torch.LongTensor:
        batch_size = input_ids.shape[0]
        device = input_ids.device

        if self.prompt_lengths is None:
            return torch.zeros(batch_size, dtype=torch.long, device=device)

        if isinstance(self.prompt_lengths, int):
            return torch.full((batch_size,), self.prompt_lengths, dtype=torch.long, device=device)

        prompt_lengths = torch.as_tensor(self.prompt_lengths, dtype=torch.long, device=device)
        if prompt_lengths.numel() != batch_size:
            raise ValueError(
                f"prompt_lengths has length {prompt_lengths.numel()}, "
                f"but current batch size is {batch_size}. "
                "For num_return_sequences / beams, pass expanded prompt lengths."
            )
        return prompt_lengths

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        if self.presence_penalty == 0.0:
            return scores

        scores = scores.clone()
        prompt_lengths = self._resolve_prompt_lengths(input_ids)

        for batch_idx in range(input_ids.shape[0]):
            start = int(prompt_lengths[batch_idx].item())
            generated_ids = input_ids[batch_idx, start:]
            if generated_ids.numel() == 0:
                continue
            scores[batch_idx, generated_ids.unique()] -= self.presence_penalty

        return scores
