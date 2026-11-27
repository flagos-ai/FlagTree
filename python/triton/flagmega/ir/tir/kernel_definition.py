# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Interned kernel signature/implementation, not an executable function.

A definition has no region, calling convention, frame or pipeline lifetime.
Its invocations belong to the surrounding function's scheduling domain.
Codegen may emit a noinline op implementation without introducing IR calls.
"""

from dataclasses import dataclass, field, replace
from types import MappingProxyType
from typing import Mapping

from triton.flagmega.errors import IRSchemaError
from triton.flagmega.ir.model import NoneType, TupleType
from .base import TIRNode, tir_node
from .kernel_dispatch import KernelDispatch
from .prim_function import PrimFunction, PrimParameter, PrimParameterRole, validate_callable_signature
from .return_stmt import Return
from .sequential import Sequential


@tir_node("kernel_definition")
@dataclass(frozen=True)
class KernelDefinition(TIRNode):
    name: str
    module_kind: str
    parameters: tuple[PrimParameter, ...]
    dispatch: KernelDispatch
    results: Return = Return()
    attrs: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self):
        object.__setattr__(self, "parameters", tuple(self.parameters))
        object.__setattr__(self, "attrs", MappingProxyType(dict(self.attrs)))
        validate_callable_signature(self)
        if not isinstance(self.dispatch, KernelDispatch):
            raise IRSchemaError("KernelDefinition contains exactly one KernelDispatch, never a region.")
        if any(key in self.attrs for key in ("noinline", "calling_convention")):
            raise IRSchemaError("KernelDefinition cannot own a function calling convention.")

    @property
    def body(self):
        """A traversal view for the shared typed-buffer verifier, not IR state."""
        return Sequential((self.dispatch,))

    @property
    def parameter_map(self):
        return {value.name: value for value in self.parameters}

    @property
    def runtime_parameters(self):
        return tuple(value for value in self.parameters if value.role in {
            PrimParameterRole.INPUT, PrimParameterRole.INOUT, PrimParameterRole.METADATA,
        })

    @property
    def output_parameters(self):
        return tuple(value for value in self.parameters if value.role is PrimParameterRole.OUTPUT)

    @property
    def workspaces(self):
        return tuple(value for value in self.parameters if value.role is PrimParameterRole.WORKSPACE)

    @property
    def runtime_parameter_types(self):
        return tuple(value.type for value in self.runtime_parameters)

    @property
    def runtime_return_type(self):
        types = tuple(value.type for value in self.results.values)
        return NoneType() if not types else types[0] if len(types) == 1 else TupleType(types)


def replace_kernel_dispatch(definition, dispatch, **changes):
    """Rewrite a contract or an explicitly loaded legacy one-op function."""
    if isinstance(definition, KernelDefinition):
        return replace(definition, dispatch=dispatch, **changes)
    return replace(definition, body=Sequential((dispatch,)), **changes)


def replace_kernel_callables(module, definitions, **changes):
    definitions = tuple(definitions)
    return replace(module,
                   prim_functions=tuple(value for value in definitions if isinstance(value, PrimFunction)),
                   kernel_definitions=tuple(value for value in definitions if isinstance(value, KernelDefinition)),
                   **changes)


__all__ = ["KernelDefinition"]
