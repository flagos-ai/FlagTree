# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Named optional storage-reuse contract for a selected TIR kernel."""

from __future__ import annotations

from dataclasses import dataclass

from triton.flagmega.errors import IRSchemaError
from triton.flagmega.ir.tir.base import TIRNode, tir_node


@tir_node("inplace_alias_candidate")
@dataclass(frozen=True)
class InplaceAliasCandidate(TIRNode):
    """Allow one output ABI value to reuse one input ABI value.

    This is an opportunity, not a requirement.  Bufferization may select it
    only after proving liveness, writability, type/layout compatibility, and
    requested-memory-space compatibility.
    """

    output: str
    input: str

    def __post_init__(self) -> None:
        if not self.output or not self.input:
            raise IRSchemaError(
                "InplaceAliasCandidate requires named output and input ABI values."
            )


__all__ = ["InplaceAliasCandidate"]
