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
    # A drain that protects shared-arena reuse must run on every execution of
    # its body, because the bytes are refilled every time.  A drain that only
    # waits out a finished stream may move with the `close` it depends on.
    # The two are indistinguishable from the stage identity alone, so the
    # pass that creates them records which kind this is.
    arena_reuse: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.stage_id, str) or not self.stage_id.strip():
            raise IRSchemaError("Pipeline drain stage ID must not be empty.")
        object.__setattr__(self, "arena_reuse", bool(self.arena_reuse))

    @property
    def local_cost(self) -> TIRCost:
        return TIRCost(synchronizations=1)


__all__ = ["PipelineDrain"]
