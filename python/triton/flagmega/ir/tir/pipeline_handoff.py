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
    # A handoff whose consumer side runs *after* its producer side in program
    # order carries a dependency across an execution boundary: the producer
    # of execution N+1 waits for the consumer of execution N.  The first
    # execution has no predecessor to wait for, so such a handoff is primed
    # with one commit before the loop.
    wrap_around: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.handoff_id, str) or not self.handoff_id.strip():
            raise IRSchemaError("Pipeline handoff ID must not be empty.")
        object.__setattr__(self, "wrap_around", bool(self.wrap_around))

    @property
    def local_cost(self) -> TIRCost:
        return TIRCost(synchronizations=1)


__all__ = ["PipelineHandoff"]
