# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Function-scoped pass contracts aligned with nncase PassManager groups."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Callable, Protocol, runtime_checkable

from triton.flagmega.ir import Function, IRModule
from triton.flagmega.passes.context import RunPassContext


class FunctionTraversalOrder(str, Enum):
    MODULE = "module"
    CALLEE_FIRST = "callee_first"


@runtime_checkable
class FunctionPass(Protocol):
    name: str
    preserves: frozenset[str]
    traversal_order: FunctionTraversalOrder

    def run_function(
        self,
        function: Function,
        module: IRModule,
        context: RunPassContext,
    ) -> IRModule: ...


FunctionTransform = Callable[[Function, IRModule, RunPassContext], IRModule]


@dataclass(frozen=True)
class FunctionalFunctionPass:
    name: str
    transform: FunctionTransform
    traversal_order: FunctionTraversalOrder = FunctionTraversalOrder.MODULE
    preserves: frozenset[str] = frozenset()

    def run_function(
        self,
        function: Function,
        module: IRModule,
        context: RunPassContext,
    ) -> IRModule:
        return self.transform(function, module, context)


__all__ = [
    "FunctionPass",
    "FunctionTransform",
    "FunctionTraversalOrder",
    "FunctionalFunctionPass",
]
