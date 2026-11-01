# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: MIT
"""FP32 coefficients used to construct the native CPU rotary cache."""

from functools import lru_cache


@lru_cache(maxsize=None)
def native_rope_inverse_frequencies(head_dim: int, theta: float) -> tuple[float, ...]:
    import torch

    # vLLM RotaryEmbedding._compute_inv_freq computes the positive power
    # first, then its reciprocal in CPU FP32. Exponentiating a negative
    # logarithm on the device introduces different rounded coefficients.
    exponents = torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim
    return tuple((1.0 / (theta ** exponents)).tolist())
