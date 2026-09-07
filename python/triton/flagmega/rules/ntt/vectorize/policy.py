# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT
"""NTT AutoVectorize rule registration and orchestration.

The rule set is backend-level policy parameterized by target vector width; it
is intentionally independent of a GPU vendor or architecture generation.
"""

from __future__ import annotations

from triton.flagmega.rules.ntt.vectorize.base import VectorizeRuleRegistry
from triton.flagmega.rules.ntt.vectorize.binary import VectorizeBinary
from triton.flagmega.rules.ntt.vectorize.matmul import VectorizeMatMul
from triton.flagmega.rules.ntt.vectorize.norm import VectorizeRMSNorm
from triton.flagmega.rules.ntt.vectorize.norm_apply import VectorizeNormApply
from triton.flagmega.rules.ntt.vectorize.norm_stats import VectorizeNormStats
from triton.flagmega.rules.ntt.vectorize.propagation import propagation_rules
from triton.flagmega.rules.ntt.vectorize.qkv_rope_with_cache import (
    VectorizeQKVRoPEWithCache,
)
from triton.flagmega.rules.ntt.vectorize.unary import VectorizeUnary


class NttVectorizationPolicy:
    def __init__(
        self,
        *,
        lane_bytes: int,
        max_axes: int = 2,
    ) -> None:
        if lane_bytes <= 0:
            raise ValueError("NTT vector lane width must be positive.")
        if max_axes <= 0:
            raise ValueError("NTT vector max_axes must be positive.")
        self.lane_bytes = lane_bytes
        self.max_axes = max_axes

    @property
    def identity(self) -> str:
        """Readable rule-set identity, deliberately independent of a machine."""

        return (
            "ntt-auto-vectorize/v2"
            f"(lane_bytes={self.lane_bytes},max_axes={self.max_axes})"
        )

    def register_rules(self, registry: VectorizeRuleRegistry) -> None:
        registry.add(VectorizeMatMul(
            lane_bytes=self.lane_bytes,
            max_axes=self.max_axes,
        ))
        registry.add(VectorizeRMSNorm(lane_bytes=self.lane_bytes))
        registry.add(VectorizeNormStats(lane_bytes=self.lane_bytes))
        registry.add(VectorizeNormApply(lane_bytes=self.lane_bytes))
        registry.add(VectorizeQKVRoPEWithCache())
        registry.add(VectorizeBinary(
            lane_bytes=self.lane_bytes,
            max_axes=self.max_axes,
        ))
        registry.add(VectorizeUnary(
            lane_bytes=self.lane_bytes,
            max_axes=self.max_axes,
        ))

    def register_propagation_rules(self, registry: VectorizeRuleRegistry) -> None:
        for rule in propagation_rules():
            registry.add_propagation(rule)


__all__ = ["NttVectorizationPolicy"]
