# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT
"""First-class bufferized orchestration function."""

from __future__ import annotations

from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Mapping

from triton.flagmega.errors import IRSchemaError
from triton.flagmega.ir.tir.base import TIRNode, tir_node
from triton.flagmega.ir.tir.sequential import Sequential


@tir_node("execution_function")
@dataclass(frozen=True)
class ExecutionFunction(TIRNode):
    """Executable call order for one graph function after Bufferize.

    Graph ``Function`` remains available for semantic edit/resume, while this
    object is the target-independent, physical orchestration IR consumed by
    synchronization, transfer-pipeline lowering, and codegen.
    """

    name: str
    parameters: tuple[str, ...]
    results: tuple[str, ...]
    body: Sequential
    attrs: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "parameters", tuple(self.parameters))
        object.__setattr__(self, "results", tuple(self.results))
        object.__setattr__(self, "attrs", MappingProxyType(dict(sorted(self.attrs.items()))))
        if not self.name:
            raise IRSchemaError("ExecutionFunction requires a non-empty name.")
        if len(set(self.parameters)) != len(self.parameters):
            raise IRSchemaError(
                f"ExecutionFunction @{self.name} has duplicate physical parameters."
            )
        if len(set(self.results)) != len(self.results):
            raise IRSchemaError(
                f"ExecutionFunction @{self.name} has duplicate physical results."
            )
        if not isinstance(self.body, Sequential):
            raise IRSchemaError("ExecutionFunction body must be Sequential.")


__all__ = ["ExecutionFunction"]
