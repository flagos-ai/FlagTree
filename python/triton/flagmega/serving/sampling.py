# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Explicit greedy or seeded temperature/top-k/top-p sampling."""

from dataclasses import dataclass
import math


@dataclass(frozen=True)
class SamplingConfig:
    temperature: float = 0.0
    top_k: int = 0
    top_p: float = 1.0
    seed: int = 0

    def __post_init__(self):
        if not math.isfinite(self.temperature) or self.temperature < 0:
            raise ValueError("temperature must be finite and nonnegative")
        if isinstance(self.top_k, bool) or not isinstance(self.top_k, int) or self.top_k < 0:
            raise ValueError("top_k must be a nonnegative integer")
        if not math.isfinite(self.top_p) or not 0 < self.top_p <= 1:
            raise ValueError("top_p must be in (0, 1]")
        if isinstance(self.seed, bool) or not isinstance(self.seed, int):
            raise ValueError("seed must be an integer")


def sample_logits(logits, config, *, generator=None):
    import torch
    if config.temperature == 0:
        return logits.argmax(dim=-1)
    scores = logits.float() / config.temperature
    if config.top_k:
        cutoff = scores.topk(min(config.top_k, scores.shape[-1]), dim=-1).values[..., -1:]
        scores = scores.masked_fill(scores < cutoff, -float("inf"))
    if config.top_p < 1:
        ordered, indices = scores.sort(dim=-1, descending=True)
        probabilities = ordered.softmax(dim=-1)
        # Keep the token that crosses the requested mass as well as the first.
        excluded = probabilities.cumsum(dim=-1) - probabilities >= config.top_p
        ordered = ordered.masked_fill(excluded, -float("inf"))
        scores = torch.empty_like(scores).scatter(-1, indices, ordered)
    return torch.multinomial(scores.softmax(dim=-1), 1, generator=generator).squeeze(-1)
