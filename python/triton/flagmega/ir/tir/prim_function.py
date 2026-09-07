# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT
"""First-class PrimFunction and explicit caller-allocated ABI."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from types import MappingProxyType
from typing import Mapping

from triton.flagmega.errors import IRSchemaError
from triton.flagmega.ir.model import IRType, NoneType, TupleType
from triton.flagmega.ir.tir.base import TIRNode, tir_node
from triton.flagmega.ir.tir.buffer import Buffer
from triton.flagmega.ir.tir.buffer_tuple import BufferTuple
from triton.flagmega.ir.tir.return_stmt import Return
from triton.flagmega.ir.tir.sequential import Sequential


class PrimParameterRole(str, Enum):
    INPUT = "input"
    INOUT = "inout"
    # A logical operand whose ParameterInfo declares MemoryEffect.NONE.  Its
    # type/shape remains part of editable TIR, but it has no physical buffer
    # in the executable ABI.
    METADATA = "metadata"
    OUTPUT = "output"
    WORKSPACE = "workspace"


@tir_node("prim_parameter")
@dataclass(frozen=True)
class PrimParameter(TIRNode):
    name: str
    type: IRType
    role: PrimParameterRole = PrimParameterRole.INPUT
    buffers: tuple[Buffer, ...] = ()
    memory_space: str | None = None

    def __post_init__(self) -> None:
        if not self.name:
            raise IRSchemaError("PrimParameter requires a non-empty name.")
        object.__setattr__(self, "role", PrimParameterRole(self.role))
        object.__setattr__(self, "buffers", tuple(self.buffers))
        if self.role is PrimParameterRole.WORKSPACE and not self.memory_space:
            raise IRSchemaError(f"Workspace parameter {self.name!r} requires a memory_space.")
        if self.role is not PrimParameterRole.WORKSPACE and self.memory_space is not None:
            raise IRSchemaError("Only workspace parameters may override memory_space.")
        if len({value.name for value in self.buffers}) != len(self.buffers):
            raise IRSchemaError(f"PrimParameter {self.name!r} has duplicate buffer names.")

    @property
    def buffer_value(self):
        if not self.buffers:
            return None
        # Keep the logical ABI type (which may intentionally erase physical
        # distribution) while exposing every backing Buffer view.
        return BufferTuple(self.buffers, self.type)


@tir_node("prim_function")
@dataclass(frozen=True)
class PrimFunction(TIRNode):
    name: str
    module_kind: str
    parameters: tuple[PrimParameter, ...]
    body: Sequential
    results: Return = Return()
    attrs: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "parameters", tuple(self.parameters))
        object.__setattr__(self, "attrs", MappingProxyType(dict(sorted(self.attrs.items()))))
        validate_callable_signature(self)

    @property
    def parameter_map(self) -> dict[str, PrimParameter]:
        return {value.name: value for value in self.parameters}

    @property
    def runtime_parameters(self) -> tuple[PrimParameter, ...]:
        return tuple(value for value in self.parameters if value.role in {
            PrimParameterRole.INPUT,
            PrimParameterRole.INOUT,
            PrimParameterRole.METADATA,
        })

    @property
    def output_parameters(self) -> tuple[PrimParameter, ...]:
        return tuple(value for value in self.parameters if value.role is PrimParameterRole.OUTPUT)

    @property
    def workspaces(self) -> tuple[PrimParameter, ...]:
        return tuple(value for value in self.parameters if value.role is PrimParameterRole.WORKSPACE)

    @property
    def runtime_parameter_types(self) -> tuple[IRType, ...]:
        return tuple(value.type for value in self.runtime_parameters)

    @property
    def runtime_return_type(self) -> IRType:
        types = tuple(value.type for value in self.results.values)
        if not types:
            return NoneType()
        return types[0] if len(types) == 1 else TupleType(types)


def validate_callable_signature(signature):
    """Shared type/return-storage contract for functions and kernel definitions."""
    if not signature.name or not signature.module_kind:
        raise IRSchemaError("PrimFunction requires non-empty name and module_kind.")
    if len({value.name for value in signature.parameters}) != len(signature.parameters):
        raise IRSchemaError(f"PrimFunction {signature.name!r} has duplicate parameter names.")
    phase = 0
    for parameter in signature.parameters:
        next_phase = {
            PrimParameterRole.INPUT: 0,
            PrimParameterRole.INOUT: 0,
            PrimParameterRole.METADATA: 0,
            PrimParameterRole.OUTPUT: 1,
            PrimParameterRole.WORKSPACE: 2,
        }[parameter.role]
        if next_phase < phase:
            raise IRSchemaError(
                f"PrimFunction {signature.name!r} ABI parameters must be ordered inputs, outputs, workspaces."
            )
        phase = next_phase
    parameter_map = signature.parameter_map
    for result in signature.results.values:
        storage = parameter_map.get(result.storage)
        if storage is None or storage.role not in {
            PrimParameterRole.INPUT,
            PrimParameterRole.INOUT,
            PrimParameterRole.OUTPUT,
        }:
            raise IRSchemaError(
                f"PrimFunction {signature.name!r} result storage {result.storage!r} is not an input/output ABI parameter."
            )
        if storage.type != result.type:
            raise IRSchemaError(
                f"PrimFunction {signature.name!r} result type does not match storage {result.storage!r}."
            )


__all__ = ["PrimFunction", "PrimParameter", "PrimParameterRole"]
