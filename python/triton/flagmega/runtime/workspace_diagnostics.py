# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Typed, model-independent views over bufferized runtime memory pools.

These views expose the *current* bytes in the SAT arena.  They are not
historical SSA snapshots: disjoint lifetimes and reusable call frames may
legally occupy the same range at different points in the entry schedule.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import prod

from triton.flagmega.errors import RuntimeContractError
from triton.flagmega.ir.bufferization import BufferPlan
from triton.flagmega.ir.distributed_storage import DistributedBufferStorageKind
from triton.flagmega.ir.distributed_type import placement_owner_count
from triton.flagmega.ir.types import DType, VectorType


@dataclass(frozen=True)
class WorkspaceViewSpec:
    """One logical buffer resolved to an entry-pool byte interval.

    The historical name is retained as a public compatibility surface; the
    record and its producers are memory-space neutral.
    """

    key: str
    function: str
    call_path: tuple[str, ...]
    buffer: str
    memory_space: str
    byte_offset: int
    byte_size: int
    scalar_dtype: DType
    shape: tuple[int, ...]
    strides: tuple[int, ...]
    live_start: int | None
    live_end: int | None


def workspace_view_specs(
    plan: BufferPlan,
    *,
    entry: str,
) -> tuple[WorkspaceViewSpec, ...]:
    """Resolve the plan's default workspace through nested call frames."""

    return memory_pool_view_specs(
        plan, entry=entry, memory_space=plan.default_workspace
    )


def memory_pool_view_specs(
    plan: BufferPlan,
    *,
    entry: str,
    memory_space: str,
) -> tuple[WorkspaceViewSpec, ...]:
    """Resolve one function-scoped SAT pool through nested call frames."""

    if entry not in plan.function_map:
        raise RuntimeContractError(
            f"Diagnostic workspace entry {entry!r} is not in the buffer plan."
        )
    entry_pool = plan.function_map[entry].memory_pool_map.get(memory_space)
    if entry_pool is None:
        return ()
    arena_bytes = entry_pool.scope_bytes
    result: list[WorkspaceViewSpec] = []

    def visit(
        function_name: str,
        call_path: tuple[str, ...],
        frame_offset: int,
        active_functions: tuple[str, ...],
    ) -> None:
        if function_name in active_functions:
            raise RuntimeContractError(
                "Diagnostic workspace cannot resolve a recursive bufferized "
                f"call graph: {(*active_functions, function_name)!r}."
            )
        function = plan.function_map[function_name]
        function_pool = function.memory_pool_map.get(memory_space)
        function_bytes = 0 if function_pool is None else function_pool.scope_bytes
        if frame_offset < 0 or frame_offset + function_bytes > arena_bytes:
            raise RuntimeContractError(
                f"Function @{function_name} {memory_space!r} frame "
                f"[{frame_offset}, {frame_offset + function_bytes}) "
                f"exceeds the entry arena of {arena_bytes} bytes."
            )
        for descriptor in plan.buffers:
            if (
                descriptor.function != function_name
                or descriptor.storage != memory_space
                or descriptor.mem_span.buffer.memory_space != memory_space
            ):
                continue
            scalar_dtype, lane_shape = _scalar_type(descriptor.dtype)
            lane_count = prod(lane_shape, start=1)
            shape = (*descriptor.component_shape, *lane_shape)
            strides = (
                *(int(value) * lane_count for value in descriptor.strides),
                *_dense_strides(lane_shape),
            )
            if (
                descriptor.distributed_storage_kind
                is DistributedBufferStorageKind.COMPACT_PER_OWNER
            ):
                if descriptor.distributed_type is None:
                    raise RuntimeContractError(
                        f"Compact-per-owner buffer {descriptor.id!r} has no DistributedType."
                    )
                owners = placement_owner_count(descriptor.distributed_type)
                owner_stride = prod(descriptor.component_shape, start=1) * lane_count
                shape = (owners, *shape)
                strides = (owner_stride, *strides)
            byte_size = _view_byte_size(shape, strides, scalar_dtype.itemsize)
            byte_offset = frame_offset + descriptor.offset
            if byte_offset < 0 or byte_offset + byte_size > arena_bytes:
                raise RuntimeContractError(
                    f"Diagnostic view {descriptor.id!r} resolves to "
                    f"[{byte_offset}, {byte_offset + byte_size}), outside the "
                    f"{arena_bytes}-byte {memory_space!r} entry arena."
                )
            result.append(WorkspaceViewSpec(
                key=(
                    descriptor.id
                    if not call_path
                    else f"{'/'.join(call_path)}::{descriptor.id}"
                ),
                function=function_name,
                call_path=call_path,
                buffer=descriptor.id,
                memory_space=memory_space,
                byte_offset=byte_offset,
                byte_size=byte_size,
                scalar_dtype=scalar_dtype,
                shape=shape,
                strides=strides,
                live_start=descriptor.live_start,
                live_end=descriptor.live_end,
            ))
        for call in function.calls:
            callee = plan.function_map[call.callee]
            callee_pool = callee.memory_pool_map.get(memory_space)
            if callee_pool is None or not callee_pool.scope_bytes:
                continue
            call_pool = call.memory_pool_map.get(memory_space)
            if call_pool is None:
                raise RuntimeContractError(
                    f"Call {call.call!r} has no {memory_space!r} frame for "
                    f"@{call.callee}."
                )
            visit(
                call.callee,
                (*call_path, call.call),
                frame_offset + call_pool.offset,
                (*active_functions, function_name),
            )

    visit(entry, (), 0, ())
    keys = tuple(value.key for value in result)
    if len(set(keys)) != len(keys):
        raise RuntimeContractError(
            "Diagnostic workspace call-path keys are not unique."
        )
    return tuple(result)


def materialize_workspace_views(
    workspace,
    specs: tuple[WorkspaceViewSpec, ...],
    *,
    clone: bool = False,
) -> dict[str, object]:
    """Compatibility wrapper for a single physical workspace scope."""

    return materialize_memory_pool_views(workspace, specs, clone=clone)


def materialize_memory_pool_views(
    pool,
    specs: tuple[WorkspaceViewSpec, ...],
    *,
    scope_count: int = 1,
    scope_nbytes: int | None = None,
    clone: bool = False,
) -> dict[str, object]:
    """Create typed views over every physical scope of one runtime pool."""

    try:
        import torch
    except ImportError as error:
        raise RuntimeContractError(
            "Workspace diagnostics require PyTorch tensor/storage APIs."
        ) from error
    if (
        not isinstance(pool, torch.Tensor)
        or pool.dtype != torch.uint8
        or pool.ndim != 1
        or not pool.is_contiguous()
    ):
        raise RuntimeContractError(
            "Memory-pool diagnostics require a contiguous one-dimensional uint8 tensor."
        )
    if scope_count <= 0:
        raise RuntimeContractError("Memory-pool scope count must be positive.")
    if scope_nbytes is None:
        if pool.numel() % scope_count:
            raise RuntimeContractError(
                "Memory-pool allocation is not divisible by its scope count."
            )
        scope_nbytes = pool.numel() // scope_count
    if scope_nbytes < 0 or scope_nbytes * scope_count > pool.numel():
        raise RuntimeContractError(
            "Memory-pool scope ABI exceeds its runtime allocation."
        )
    values: dict[str, object] = {}
    for spec in specs:
        dtype = _torch_dtype(torch, spec.scalar_dtype)
        final_end = (
            (scope_count - 1) * scope_nbytes
            + spec.byte_offset
            + spec.byte_size
        )
        if final_end > pool.numel():
            raise RuntimeContractError(
                f"Diagnostic view {spec.key!r} exceeds the runtime pool allocation."
            )
        if scope_nbytes % spec.scalar_dtype.itemsize:
            raise RuntimeContractError(
                f"Memory-pool scope stride for {spec.key!r} is not aligned to "
                "its scalar dtype."
            )
        raw = pool.narrow(0, spec.byte_offset, final_end - spec.byte_offset)
        typed = raw.view(dtype)
        shape = spec.shape
        strides = spec.strides
        if scope_count > 1:
            shape = (scope_count, *shape)
            strides = (scope_nbytes // spec.scalar_dtype.itemsize, *strides)
        value = torch.as_strided(typed, shape, strides)
        values[spec.key] = value.clone() if clone else value
    return values


def _scalar_type(dtype) -> tuple[DType, tuple[int, ...]]:
    if isinstance(dtype, DType):
        return dtype, ()
    if isinstance(dtype, VectorType):
        return dtype.elem_type, dtype.lanes
    raise RuntimeContractError(
        f"Workspace diagnostics cannot materialize {type(dtype).__name__} buffers."
    )


def _dense_strides(shape: tuple[int, ...]) -> tuple[int, ...]:
    result = []
    stride = 1
    for extent in reversed(shape):
        result.append(stride)
        stride *= extent
    return tuple(reversed(result))


def _view_byte_size(
    shape: tuple[int, ...],
    strides: tuple[int, ...],
    itemsize: int,
) -> int:
    if len(shape) != len(strides) or any(value < 0 for value in strides):
        raise RuntimeContractError("Diagnostic workspace view has invalid strides.")
    if any(value == 0 for value in shape):
        return 0
    elements = 1 + sum(
        (extent - 1) * stride
        for extent, stride in zip(shape, strides, strict=True)
    )
    return elements * itemsize


def _torch_dtype(torch, dtype: DType):
    try:
        return {
            DType.BOOL: torch.bool,
            DType.INT32: torch.int32,
            DType.INT64: torch.int64,
            DType.BFLOAT16: torch.bfloat16,
            DType.FLOAT16: torch.float16,
            DType.FLOAT32: torch.float32,
            DType.FLOAT8_E4M3FN: torch.float8_e4m3fn,
        }[dtype]
    except (AttributeError, KeyError) as error:
        raise RuntimeContractError(
            f"Workspace diagnostics cannot materialize dtype {dtype.value!r}."
        ) from error


__all__ = [
    "WorkspaceViewSpec",
    "materialize_memory_pool_views",
    "materialize_workspace_views",
    "memory_pool_view_specs",
    "workspace_view_specs",
]
