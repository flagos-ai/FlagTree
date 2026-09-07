# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT
"""One-shot consumer-to-producer dependency edge."""

from __future__ import annotations

from dataclasses import dataclass

from triton.flagmega.errors import IRSchemaError
from triton.flagmega.ir.tir.base import TIRCost, TIRStmt, tir_node


@tir_node("pipeline_handoff")
@dataclass(frozen=True)
class PipelineHandoff(TIRStmt):
    handoff_id: str

    def __post_init__(self) -> None:
        if not isinstance(self.handoff_id, str) or not self.handoff_id.strip():
            raise IRSchemaError("Pipeline handoff ID must not be empty.")

    @property
    def local_cost(self) -> TIRCost:
        return TIRCost(synchronizations=1)


__all__ = ["PipelineHandoff"]
