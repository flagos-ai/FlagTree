# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT
"""A first-class call in a bufferized execution schedule."""

from __future__ import annotations

from dataclasses import dataclass

from triton.flagmega.errors import IRSchemaError
from triton.flagmega.ir.tir.base import TIRStmt, tir_node
from triton.flagmega.ir.tir.buffer import Buffer
from triton.flagmega.ir.tir.memory_pool_frame import MemoryPoolFrame
from triton.flagmega.ir.tir.prim_call_binding import PrimCallBinding


@tir_node("prim_function_call")
@dataclass(frozen=True, init=False)
class PrimFunctionCall(TIRStmt):
    """Invoke a kernel or another execution function with physical bindings.

    ``arguments/results/workspaces`` name BufferPlan objects rather than SSA
    expressions.  The call is therefore editable and executable after graph
    lowering without asking codegen to rediscover the ABI from a model graph.
    Shared buffers are caller-arena views and make interprocedural pipeline
    alias analysis independent of template or model names.
    """

    call_id: str
    callee: str
    arguments: tuple[PrimCallBinding, ...] = ()
    results: tuple[PrimCallBinding, ...] = ()
    workspaces: tuple[PrimCallBinding, ...] = ()
    shared_workspace_buffers: tuple[Buffer, ...] = ()
    transfer_sources: tuple[str, ...] = ()
    reads: tuple[str, ...] = ()
    writes: tuple[str, ...] = ()
    effect_kind: str = "pure"
    effect_resource: str | None = None
    memory_pools: tuple[MemoryPoolFrame, ...] = ()
    dependencies: tuple[str, ...] = ()

    def __init__(
        self,
        call_id: str,
        callee: str,
        arguments: tuple[PrimCallBinding, ...] = (),
        results: tuple[PrimCallBinding, ...] = (),
        workspaces: tuple[PrimCallBinding, ...] = (),
        shared_workspace_buffers: tuple[Buffer, ...] = (),
        transfer_sources: tuple[str, ...] = (),
        reads: tuple[str, ...] = (),
        writes: tuple[str, ...] = (),
        effect_kind: str = "pure",
        effect_resource: str | None = None,
        workspace_offset: int = 0,
        workspace_nbytes: int = 0,
        dependencies: tuple[str, ...] = (),
        *,
        memory_pools: tuple[MemoryPoolFrame, ...] | None = None,
    ) -> None:
        if memory_pools is not None and (workspace_offset or workspace_nbytes):
            raise IRSchemaError(
                "PrimFunctionCall memory_pools cannot be combined with legacy "
                "workspace fields."
            )
        pools = (
            tuple(memory_pools)
            if memory_pools is not None
            else (
                MemoryPoolFrame(
                    "workspace", None, workspace_offset, workspace_nbytes
                ),
            )
            if workspace_offset or workspace_nbytes
            else ()
        )
        object.__setattr__(self, "call_id", call_id)
        object.__setattr__(self, "callee", callee)
        object.__setattr__(self, "arguments", tuple(arguments))
        object.__setattr__(self, "results", tuple(results))
        object.__setattr__(self, "workspaces", tuple(workspaces))
        object.__setattr__(
            self, "shared_workspace_buffers", tuple(shared_workspace_buffers)
        )
        object.__setattr__(self, "transfer_sources", tuple(transfer_sources))
        object.__setattr__(self, "reads", tuple(reads))
        object.__setattr__(self, "writes", tuple(writes))
        object.__setattr__(self, "effect_kind", effect_kind)
        object.__setattr__(self, "effect_resource", effect_resource)
        object.__setattr__(self, "memory_pools", pools)
        object.__setattr__(self, "dependencies", tuple(dependencies))
        self.__post_init__()

    def __post_init__(self) -> None:
        validate_physical_bindings(self)

    @property
    def memory_pool_map(self) -> dict[str, MemoryPoolFrame]:
        return {value.memory_space: value for value in self.memory_pools}

    @property
    def workspace_pool(self) -> MemoryPoolFrame | None:
        return self.memory_pool_map.get("workspace") or (
            self.memory_pools[0] if len(self.memory_pools) == 1 else None
        )

    @property
    def workspace_offset(self) -> int:
        pool = self.workspace_pool
        return 0 if pool is None else pool.offset

    @property
    def workspace_nbytes(self) -> int:
        pool = self.workspace_pool
        return 0 if pool is None else pool.nbytes


def validate_physical_bindings(operation):
    """Shared actual-buffer/effect contract; only function calls own frames."""
    object.__setattr__(operation, "arguments", tuple(operation.arguments))
    object.__setattr__(operation, "results", tuple(operation.results))
    object.__setattr__(operation, "workspaces", tuple(operation.workspaces))
    object.__setattr__(
        operation, "shared_workspace_buffers", tuple(operation.shared_workspace_buffers)
    )
    object.__setattr__(operation, "transfer_sources", tuple(operation.transfer_sources))
    object.__setattr__(operation, "reads", tuple(operation.reads))
    object.__setattr__(operation, "writes", tuple(operation.writes))
    object.__setattr__(operation, "dependencies", tuple(operation.dependencies))
    if not operation.call_id or not operation.callee:
        raise IRSchemaError("PrimFunctionCall requires non-empty call and callee names.")
    for name, bindings in (
        ("argument", operation.arguments),
        ("result", operation.results),
        ("workspace", operation.workspaces),
    ):
        if any(not isinstance(value, PrimCallBinding) for value in bindings):
            raise IRSchemaError(
                f"PrimFunctionCall {name} bindings must be PrimCallBinding values."
            )
        formals = tuple(value.formal for value in bindings)
        if len(set(formals)) != len(formals):
            raise IRSchemaError(
                f"PrimFunctionCall {operation.call_id!r} has duplicate {name} formals."
            )
    if any(
        not isinstance(value, Buffer) for value in operation.shared_workspace_buffers
    ):
        raise IRSchemaError(
            "PrimFunctionCall shared workspaces must be typed TIR Buffers."
        )
    actuals = {
        value.actual
        for value in (*operation.arguments, *operation.results, *operation.workspaces)
    }
    if not set(operation.transfer_sources).issubset(actuals):
        raise IRSchemaError(
            "PrimFunctionCall transfer sources must reference physical call bindings."
        )
    if not set(operation.reads).issubset(actuals) or not set(operation.writes).issubset(actuals):
        raise IRSchemaError(
            "PrimFunctionCall effects must reference physical call bindings."
        )
    if not operation.effect_kind:
        raise IRSchemaError("PrimFunctionCall requires a non-empty effect kind.")
    if any(not isinstance(value, MemoryPoolFrame) for value in operation.memory_pools):
        raise IRSchemaError(
            "PrimFunctionCall memory pools must be MemoryPoolFrame values."
        )
    if len({value.memory_space for value in operation.memory_pools}) != len(
        operation.memory_pools
    ):
        raise IRSchemaError(
            f"PrimFunctionCall {operation.call_id!r} has duplicate memory-pool frames."
        )
    if (
        any(not value for value in operation.dependencies)
        or len(set(operation.dependencies)) != len(operation.dependencies)
        or operation.call_id in operation.dependencies
    ):
        raise IRSchemaError(
            "PrimFunctionCall dependencies must be unique non-empty call ids "
            "and cannot contain the call itoperation."
        )


__all__ = ["PrimFunctionCall"]
