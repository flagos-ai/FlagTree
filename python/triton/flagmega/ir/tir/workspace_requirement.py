# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Selected kernel workspace requirement."""

from dataclasses import dataclass
from enum import Enum

from triton.flagmega.errors import IRSchemaError
from triton.flagmega.ir.model import TensorType
from triton.flagmega.ir.tir.base import TIRNode, tir_node


class WorkspaceLifetime(str, Enum):
    """How long caller-owned scratch contents must remain unaliased."""

    INVOCATION = "invocation"
    FUNCTION = "function"


@tir_node("workspace_requirement")
@dataclass(frozen=True)
class WorkspaceRequirement(TIRNode):
    name: str
    type: TensorType
    memory_space: str = "workspace"
    alignment: int = 256
    lifetime: WorkspaceLifetime = WorkspaceLifetime.INVOCATION

    def __post_init__(self) -> None:
        if not self.name or not isinstance(self.type, TensorType) or not self.memory_space:
            raise IRSchemaError("WorkspaceRequirement requires a name, TensorType and memory space.")
        if (
            isinstance(self.alignment, bool)
            or self.alignment <= 0
            or self.alignment & (self.alignment - 1)
        ):
            raise IRSchemaError("WorkspaceRequirement alignment must be a positive power of two.")
        object.__setattr__(self, "lifetime", WorkspaceLifetime(self.lifetime))


__all__ = ["WorkspaceLifetime", "WorkspaceRequirement"]
