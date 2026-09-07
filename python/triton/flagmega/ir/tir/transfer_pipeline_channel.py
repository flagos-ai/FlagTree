# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT
"""One independently synchronized global-to-shared transfer channel."""

from __future__ import annotations

from dataclasses import dataclass

from triton.flagmega.errors import IRSchemaError
from triton.flagmega.ir.tir.base import TIRNode, tir_node


@tir_node("transfer_pipeline_channel")
@dataclass(frozen=True)
class TIRTransferPipelineChannel(TIRNode):
    name: str
    source_argument_indices: tuple[int, ...]
    shared_workspace_indices: tuple[int, ...]
    source_alignment_bytes: int = 1

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name.strip():
            raise IRSchemaError("Transfer-pipeline channel name must not be empty.")
        object.__setattr__(
            self,
            "source_argument_indices",
            _indices(self.source_argument_indices),
        )
        object.__setattr__(
            self,
            "shared_workspace_indices",
            _indices(self.shared_workspace_indices),
        )
        alignment = self.source_alignment_bytes
        if (
            isinstance(alignment, bool)
            or not isinstance(alignment, int)
            or alignment <= 0
            or alignment & (alignment - 1)
        ):
            raise IRSchemaError(
                "Transfer source alignment must be a positive power of two."
            )


def _indices(values) -> tuple[int, ...]:
    result = tuple(values)
    if (
        not result
        or any(isinstance(value, bool) or not isinstance(value, int) or value < 0 for value in result)
        or len(set(result)) != len(result)
    ):
        raise IRSchemaError(
            "Pipeline operand indexes must be non-empty, non-negative, and unique."
        )
    return result


__all__ = ["TIRTransferPipelineChannel"]
