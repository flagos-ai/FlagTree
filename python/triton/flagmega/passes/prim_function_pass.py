# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT
"""PrimFunction-scoped pass and nncase-style mutator fixed point."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Protocol, runtime_checkable

from triton.flagmega.errors import IRVerificationError
from triton.flagmega.ir import IRModule, PrimFunction, TIRRewriter
from triton.flagmega.passes.context import RunPassContext


@runtime_checkable
class PrimFunctionPass(Protocol):
    name: str
    preserves: frozenset[str]

    def run_prim_function(
        self,
        function: PrimFunction,
        module: IRModule,
        context: RunPassContext,
    ) -> PrimFunction: ...


PrimFunctionTransform = Callable[[PrimFunction, IRModule, RunPassContext], PrimFunction]
MutatorFactory = Callable[[], TIRRewriter]


@dataclass(frozen=True)
class FunctionalPrimFunctionPass:
    name: str
    transform: PrimFunctionTransform
    preserves: frozenset[str] = frozenset()

    def run_prim_function(
        self,
        function: PrimFunction,
        module: IRModule,
        context: RunPassContext,
    ) -> PrimFunction:
        return self.transform(function, module, context)


class PrimFuncPass:
    """Run ordered, freshly-created TIR rewriters until none mutates.

    Like nncase, one successful mutator restarts the ordered list so rules can
    expose earlier canonicalizations.  Fresh instances prevent rewriter memo
    and mutation state from leaking across iterations or PrimFunctions.
    """

    def __init__(
        self,
        name: str,
        *,
        max_iterations: int = 128,
        preserves: frozenset[str] = frozenset(),
    ) -> None:
        if not name or max_iterations <= 0:
            raise ValueError("PrimFuncPass requires a name and positive max_iterations.")
        self.name = name
        self.max_iterations = max_iterations
        self.preserves = preserves
        self._mutators: list[MutatorFactory] = []

    def add(self, factory: MutatorFactory) -> PrimFuncPass:
        if not callable(factory):
            raise TypeError("PrimFuncPass.add requires a TIRRewriter factory.")
        self._mutators.append(factory)
        return self

    def run_prim_function(
        self,
        function: PrimFunction,
        module: IRModule,
        context: RunPassContext,
    ) -> PrimFunction:
        current = function
        for _ in range(self.max_iterations):
            for factory in self._mutators:
                mutator = factory()
                if not isinstance(mutator, TIRRewriter):
                    raise TypeError("PrimFuncPass mutator factories must return TIRRewriter instances.")
                rewritten = mutator.rewrite(current)
                if not isinstance(rewritten, PrimFunction):
                    raise IRVerificationError(
                        f"PrimFuncPass {self.name!r} mutator changed the root into {type(rewritten).__name__}."
                    )
                if mutator.is_mutated:
                    current = rewritten
                    break
            else:
                return current
        raise IRVerificationError(
            f"PrimFuncPass {self.name!r} did not converge for @{function.name} "
            f"after {self.max_iterations} mutations."
        )


__all__ = [
    "FunctionalPrimFunctionPass",
    "MutatorFactory",
    "PrimFuncPass",
    "PrimFunctionPass",
    "PrimFunctionTransform",
]
