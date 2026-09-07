# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT
"""An op invocation in its containing function's physical schedule."""

from dataclasses import dataclass

from .base import TIRStmt, tir_node
from .buffer import Buffer
from .prim_call_binding import PrimCallBinding
from .prim_function_call import validate_physical_bindings


@tir_node("kernel_invoke")
@dataclass(frozen=True)
class KernelInvoke(TIRStmt):
    call_id: str
    kernel: str
    arguments: tuple[PrimCallBinding, ...] = ()
    results: tuple[PrimCallBinding, ...] = ()
    workspaces: tuple[PrimCallBinding, ...] = ()
    shared_workspace_buffers: tuple[Buffer, ...] = ()
    transfer_sources: tuple[str, ...] = ()
    reads: tuple[str, ...] = ()
    writes: tuple[str, ...] = ()
    effect_kind: str = "pure"
    effect_resource: str | None = None
    dependencies: tuple[str, ...] = ()

    def __post_init__(self):
        # Both node kinds share the typed physical binding contract, but only
        # PrimFunctionCall can create an interprocedural memory-pool frame.
        validate_physical_bindings(self)

    @property
    def callee(self):
        """Symbol lookup protocol shared by physical-binding analyses."""
        return self.kernel

    @property
    def memory_pools(self):
        return ()


__all__ = ["KernelInvoke"]
