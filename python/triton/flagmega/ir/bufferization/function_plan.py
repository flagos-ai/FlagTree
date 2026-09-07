# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Physical ABI records for TIR functions and calls."""

from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property
from typing import Any, Mapping

from triton.flagmega.errors import IRSchemaError
from triton.flagmega.ir.bufferization.memory_pool import (
    CallMemoryPoolBinding,
    FunctionMemoryPool,
)


BufferBinding = tuple[tuple[str, tuple[str, ...]], ...]


def _bindings_to_data(values: BufferBinding) -> list[dict[str, object]]:
    return [{"node": node, "buffers": list(buffers)} for node, buffers in values]


def _bindings_from_data(values) -> BufferBinding:
    return tuple(
        (str(value["node"]), tuple(str(item) for item in value.get("buffers", ())))
        for value in values
    )


@dataclass(frozen=True, init=False)
class CallBufferBinding:
    call: str
    caller: str
    callee: str
    arguments: tuple[tuple[str, str], ...]
    results: tuple[tuple[str, str], ...]
    memory_pools: tuple[CallMemoryPoolBinding, ...]

    def __init__(
        self,
        call: str,
        caller: str,
        callee: str,
        arguments: tuple[tuple[str, str], ...],
        results: tuple[tuple[str, str], ...],
        workspace_allocation: str | None = None,
        workspace_offset: int = 0,
        workspace_bytes: int = 0,
        *,
        memory_pools: tuple[CallMemoryPoolBinding, ...] | None = None,
    ) -> None:
        if memory_pools is not None and (
            workspace_allocation is not None or workspace_offset or workspace_bytes
        ):
            raise IRSchemaError(
                "Call memory_pools cannot be combined with legacy workspace fields."
            )
        pools = (
            tuple(memory_pools)
            if memory_pools is not None
            else (
                CallMemoryPoolBinding(
                    "workspace",
                    workspace_allocation,
                    workspace_offset,
                    workspace_bytes,
                ),
            )
            if workspace_allocation is not None or workspace_offset or workspace_bytes
            else ()
        )
        if len({value.memory_space for value in pools}) != len(pools):
            raise IRSchemaError("Call memory-pool identities must be unique.")
        for name, value in (
            ("call", call), ("caller", caller), ("callee", callee)
        ):
            if not value:
                raise IRSchemaError(f"Call buffer binding requires non-empty {name}.")
        object.__setattr__(self, "call", call)
        object.__setattr__(self, "caller", caller)
        object.__setattr__(self, "callee", callee)
        object.__setattr__(self, "arguments", tuple(arguments))
        object.__setattr__(self, "results", tuple(results))
        object.__setattr__(self, "memory_pools", pools)

    @cached_property
    def memory_pool_map(self) -> Mapping[str, CallMemoryPoolBinding]:
        return {value.memory_space: value for value in self.memory_pools}

    @property
    def workspace_pool(self) -> CallMemoryPoolBinding | None:
        return self.memory_pool_map.get("workspace") or (
            self.memory_pools[0] if len(self.memory_pools) == 1 else None
        )

    @property
    def workspace_allocation(self) -> str | None:
        pool = self.workspace_pool
        return None if pool is None else pool.allocation

    @property
    def workspace_offset(self) -> int:
        pool = self.workspace_pool
        return 0 if pool is None else pool.offset

    @property
    def workspace_bytes(self) -> int:
        pool = self.workspace_pool
        return 0 if pool is None else pool.scope_bytes

    def to_data(self) -> dict[str, object]:
        return {
            "call": self.call,
            "caller": self.caller,
            "callee": self.callee,
            "arguments": [
                {"formal": formal, "actual": actual} for formal, actual in self.arguments
            ],
            "results": [
                {"formal": formal, "actual": actual} for formal, actual in self.results
            ],
            "memory_pools": [value.to_data() for value in self.memory_pools],
        }

    @classmethod
    def from_data(
        cls,
        data: Mapping[str, Any],
        *,
        legacy_memory_space: str = "workspace",
    ) -> CallBufferBinding:
        common = dict(
            call=str(data["call"]),
            caller=str(data["caller"]),
            callee=str(data["callee"]),
            arguments=tuple(
                (str(value["formal"]), str(value["actual"]))
                for value in data.get("arguments", ())
            ),
            results=tuple(
                (str(value["formal"]), str(value["actual"]))
                for value in data.get("results", ())
            ),
        )
        if "memory_pools" in data:
            return cls(
                **common,
                memory_pools=tuple(
                    CallMemoryPoolBinding.from_data(value)
                    for value in data.get("memory_pools", ())
                ),
            )
        return cls(
            **common,
            memory_pools=(
                CallMemoryPoolBinding(
                    legacy_memory_space,
                    (
                        None
                        if data.get("workspace_allocation") is None
                        else str(data["workspace_allocation"])
                    ),
                    int(data.get("workspace_offset", 0)),
                    int(data.get("workspace_bytes", 0)),
                ),
            )
            if (
                data.get("workspace_allocation") is not None
                or int(data.get("workspace_offset", 0))
                or int(data.get("workspace_bytes", 0))
            )
            else (),
        )


@dataclass(frozen=True)
class KernelCallBufferBinding:
    """Caller-owned scratch passed to one selected PrimFunction invocation."""

    call: str
    caller: str
    callee: str
    workspaces: tuple[tuple[str, str], ...]

    def to_data(self) -> dict[str, object]:
        return {
            "call": self.call,
            "caller": self.caller,
            "callee": self.callee,
            "workspaces": [
                {"formal": formal, "actual": actual}
                for formal, actual in self.workspaces
            ],
        }

    @classmethod
    def from_data(cls, data: Mapping[str, Any]) -> KernelCallBufferBinding:
        return cls(
            call=str(data["call"]),
            caller=str(data["caller"]),
            callee=str(data["callee"]),
            workspaces=tuple(
                (str(value["formal"]), str(value["actual"]))
                for value in data.get("workspaces", ())
            ),
        )


@dataclass(frozen=True, init=False)
class FunctionBufferPlan:
    name: str
    parameters: BufferBinding
    outputs: BufferBinding
    memory_pools: tuple[FunctionMemoryPool, ...]
    calls: tuple[CallBufferBinding, ...] = ()
    kernel_calls: tuple[KernelCallBufferBinding, ...] = ()
    result_aliases: tuple[tuple[str, str], ...] = ()
    values: BufferBinding = ()

    def __init__(
        self,
        name: str,
        parameters: BufferBinding,
        outputs: BufferBinding,
        workspace_bytes: int = 0,
        workspace_alignment: int = 1,
        allocations: tuple[str, ...] = (),
        calls: tuple[CallBufferBinding, ...] = (),
        kernel_calls: tuple[KernelCallBufferBinding, ...] = (),
        result_aliases: tuple[tuple[str, str], ...] = (),
        values: BufferBinding = (),
        *,
        memory_pools: tuple[FunctionMemoryPool, ...] | None = None,
    ) -> None:
        if memory_pools is not None and (
            workspace_bytes or workspace_alignment != 1 or allocations
        ):
            raise IRSchemaError(
                "Function memory_pools cannot be combined with legacy workspace fields."
            )
        pools = (
            tuple(memory_pools)
            if memory_pools is not None
            else (
                FunctionMemoryPool(
                    "workspace", workspace_bytes, workspace_alignment, tuple(allocations)
                ),
            )
        )
        if len({value.memory_space for value in pools}) != len(pools):
            raise IRSchemaError("Function memory-pool identities must be unique.")
        if not name:
            raise IRSchemaError("Function buffer plan requires a non-empty name.")
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "parameters", tuple(parameters))
        object.__setattr__(self, "outputs", tuple(outputs))
        object.__setattr__(self, "memory_pools", pools)
        object.__setattr__(self, "calls", tuple(calls))
        object.__setattr__(self, "kernel_calls", tuple(kernel_calls))
        object.__setattr__(self, "result_aliases", tuple(result_aliases))
        object.__setattr__(self, "values", tuple(values))

    @cached_property
    def memory_pool_map(self) -> Mapping[str, FunctionMemoryPool]:
        return {value.memory_space: value for value in self.memory_pools}

    @property
    def workspace_pool(self) -> FunctionMemoryPool:
        pool = self.memory_pool_map.get("workspace")
        if pool is not None:
            return pool
        if len(self.memory_pools) != 1:
            raise IRSchemaError(
                f"@{self.name} has multiple pools and no legacy 'workspace' pool."
            )
        return self.memory_pools[0]

    @property
    def workspace_bytes(self) -> int:
        return self.workspace_pool.scope_bytes

    @property
    def workspace_alignment(self) -> int:
        return self.workspace_pool.alignment

    @property
    def allocations(self) -> tuple[str, ...]:
        return tuple(
            allocation
            for pool in self.memory_pools
            for allocation in pool.allocations
        )

    def to_data(self) -> dict[str, object]:
        return {
            "name": self.name,
            "parameters": _bindings_to_data(self.parameters),
            "outputs": _bindings_to_data(self.outputs),
            "memory_pools": [value.to_data() for value in self.memory_pools],
            "calls": [value.to_data() for value in self.calls],
            "kernel_calls": [value.to_data() for value in self.kernel_calls],
            "result_aliases": [
                {"result": result, "parameter": parameter}
                for result, parameter in self.result_aliases
            ],
            "values": _bindings_to_data(self.values),
        }

    @classmethod
    def from_data(
        cls,
        data: Mapping[str, Any],
        *,
        legacy_memory_space: str = "workspace",
    ) -> FunctionBufferPlan:
        common = dict(
            name=str(data["name"]),
            parameters=_bindings_from_data(data.get("parameters", ())),
            outputs=_bindings_from_data(data.get("outputs", ())),
            calls=tuple(
                CallBufferBinding.from_data(
                    value, legacy_memory_space=legacy_memory_space
                )
                for value in data.get("calls", ())
            ),
            kernel_calls=tuple(
                KernelCallBufferBinding.from_data(value)
                for value in data.get("kernel_calls", ())
            ),
            result_aliases=tuple(
                (str(value["result"]), str(value["parameter"]))
                for value in data.get("result_aliases", ())
            ),
            values=_bindings_from_data(data.get("values", ())),
        )
        if "memory_pools" in data:
            return cls(
                **common,
                memory_pools=tuple(
                    FunctionMemoryPool.from_data(value)
                    for value in data.get("memory_pools", ())
                ),
            )
        return cls(
            **common,
            memory_pools=(
                FunctionMemoryPool(
                    legacy_memory_space,
                    int(data.get("workspace_bytes", 0)),
                    int(data.get("workspace_alignment", 1)),
                    tuple(str(value) for value in data.get("allocations", ())),
                ),
            ),
        )


__all__ = [
    "BufferBinding",
    "CallBufferBinding",
    "CallMemoryPoolBinding",
    "FunctionBufferPlan",
    "FunctionMemoryPool",
    "KernelCallBufferBinding",
]
