# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT
"""One operation viewed in a producer or consumer pipeline task."""

from __future__ import annotations

from dataclasses import dataclass

from triton.flagmega.errors import IRSchemaError
from triton.flagmega.ir.tir.base import TIRStmt, tir_node
from triton.flagmega.ir.tir.kernel_dispatch import KernelDispatch
from triton.flagmega.ir.tir.prim_function_call import PrimFunctionCall
from triton.flagmega.ir.tir.kernel_invoke import KernelInvoke
from triton.flagmega.ir.tir.pipeline_readiness import PipelineReadiness


@tir_node("pipeline_stage")
@dataclass(frozen=True)
class PipelineStage(TIRStmt):
    """A selected kernel invocation owned by both sides of a pipeline region.

    The enclosing :class:`ProducerConsumerRegion` gives the two occurrences
    their producer and consumer roles.  Keeping the operation explicit avoids
    recovering task ownership from template names or memory effects.
    """

    stage_id: str
    operation: KernelDispatch | PrimFunctionCall | KernelInvoke
    source_readiness: PipelineReadiness | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.stage_id, str) or not self.stage_id.strip():
            raise IRSchemaError("Pipeline stage ID must not be empty.")
        if not isinstance(self.operation, (KernelDispatch, PrimFunctionCall, KernelInvoke)):
            raise IRSchemaError(
                "PipelineStage requires a KernelDispatch or PrimFunctionCall."
            )
        if self.source_readiness is not None and (
            not isinstance(self.source_readiness, PipelineReadiness)
            or not isinstance(self.operation, PrimFunctionCall)
        ):
            raise IRSchemaError("Imported source readiness requires a typed execution-function call.")


__all__ = ["PipelineStage"]
