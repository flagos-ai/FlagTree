# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT
"""RunPassContext scoped to one concrete pass execution."""

from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Iterator, Mapping

from triton.flagmega.diagnostics.dump import Dumper
from triton.flagmega.ir import IRModule
from triton.flagmega.passes.analysis import AnalysisManager


@dataclass(frozen=True)
class RunPassContext:
    manager_name: str
    pass_name: str
    pass_index: int
    module: IRModule
    analyses: AnalysisManager = field(compare=False, repr=False)
    dumper: Dumper | None = field(default=None, compare=False, repr=False)
    function: str | None = None
    properties: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "properties",
            MappingProxyType({str(key): value for key, value in self.properties.items()}),
        )

    def require_analysis(self, name: str) -> object:
        return self.analyses.require(name, self.module, function=self.function)


_CURRENT: ContextVar[RunPassContext | None] = ContextVar(
    "flagmega_run_pass_context",
    default=None,
)


def current_pass_context() -> RunPassContext:
    context = _CURRENT.get()
    if context is None:
        raise RuntimeError("No FlagMega pass is currently running.")
    return context


@contextmanager
def pass_context(context: RunPassContext) -> Iterator[RunPassContext]:
    token = _CURRENT.set(context)
    try:
        yield context
    finally:
        _CURRENT.reset(token)


__all__ = ["RunPassContext", "current_pass_context", "pass_context"]
