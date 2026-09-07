# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Terminal rendezvous before cyclic pipeline storage is reused."""

from __future__ import annotations

from dataclasses import dataclass

from triton.flagmega.errors import IRSchemaError
from triton.flagmega.ir.tir.base import TIRCost, TIRStmt, tir_node


@tir_node("pipeline_drain")
@dataclass(frozen=True)
class PipelineDrain(TIRStmt):
    stage_id: str

    def __post_init__(self) -> None:
        if not isinstance(self.stage_id, str) or not self.stage_id.strip():
            raise IRSchemaError("Pipeline drain stage ID must not be empty.")

    @property
    def local_cost(self) -> TIRCost:
        return TIRCost(synchronizations=1)


__all__ = ["PipelineDrain"]
