# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: MIT
"""An imported, one-shot source-publication reader for a pipeline call."""

from dataclasses import dataclass

from triton.flagmega.errors import IRSchemaError
from .base import TIRNode, tir_node


@tir_node("pipeline_readiness")
@dataclass(frozen=True)
class PipelineReadiness(TIRNode):
    """Route an existing caller handoff to a callee's first mutable transfer.

    This is a control binding, not storage or a replacement for tensor effects.
    The module verifier proves the readonly prefix and the caller publication.
    Only straight-line, one-shot invocations are supported by this contract.
    """

    handoff_id: str
    before_call_id: str
    readonly_parameters: tuple[str, ...]

    def __post_init__(self):
        object.__setattr__(self, "readonly_parameters", tuple(self.readonly_parameters))
        if not self.handoff_id or not self.before_call_id:
            raise IRSchemaError("PipelineReadiness requires handoff and callee call identities.")
        if (not self.readonly_parameters or any(not x for x in self.readonly_parameters)
                or len(set(self.readonly_parameters)) != len(self.readonly_parameters)):
            raise IRSchemaError("PipelineReadiness requires unique readonly physical formals.")


__all__ = ["PipelineReadiness"]
