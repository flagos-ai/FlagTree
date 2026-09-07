# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT
"""One physical buffer edge of a PrimFunction invocation."""

from __future__ import annotations

from dataclasses import dataclass

from triton.flagmega.errors import IRSchemaError
from triton.flagmega.ir.tir.base import TIRNode, tir_node


@tir_node("prim_call_binding")
@dataclass(frozen=True)
class PrimCallBinding(TIRNode):
    """Bind a callee formal buffer to a caller-owned logical buffer."""

    formal: str
    actual: str

    def __post_init__(self) -> None:
        if not self.formal or not self.actual:
            raise IRSchemaError("PrimCallBinding requires non-empty formal and actual names.")


__all__ = ["PrimCallBinding"]
