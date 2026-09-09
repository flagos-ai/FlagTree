# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Target-memory pool facts shared by call ABI and runtime binding."""

from __future__ import annotations

from collections.abc import Mapping

from triton.flagmega.errors import CodegenError
from triton.flagmega.ir import IRModule, MemorySharingScope


def memory_scope_count(module: IRModule, scope: MemorySharingScope) -> int:
    """Return physical instances of a target memory scope in one launch."""

    if scope is MemorySharingScope.CHIP:
        return 1
    if scope is MemorySharingScope.DIE:
        raise CodegenError(
            "The Triton runtime has no die-scoped pool-addressing primitive."
        )
    launch = module.metadata.get("launch_contract", {})
    mesh = launch.get("grid_mesh") if isinstance(launch, Mapping) else None
    if mesh is None:
        return 1
    if not isinstance(mesh, Mapping):
        raise CodegenError("TIR launch grid_mesh must be a mapping.")
    hierarchy = tuple(int(value) for value in mesh.get("hierarchy", ()))
    levels = str(mesh.get("hierarchy_levels", ""))
    if not hierarchy or len(hierarchy) != len(levels):
        raise CodegenError("TIR launch grid_mesh has an invalid hierarchy.")
    count = 1
    for extent, level in zip(hierarchy, levels, strict=True):
        if level == "b":
            count *= extent
    return count


def memory_scope_counts(module: IRModule, plan) -> dict[str, int]:
    used_spaces = {
        value.memory_space for value in plan.physical_buffers
    } | {
        pool.memory_space
        for function in plan.functions
        for pool in function.memory_pools
        if pool.scope_bytes
    }
    return {
        space.name: memory_scope_count(module, space.sharing_scope)
        for space in plan.memory_spaces
        if space.name in used_spaces
    }


def is_runtime_pool_space(space) -> bool:
    return space.allocation_scope.value != "external" and space.kind != "shared"


def emit_pool_scope_base(pool: Mapping[str, object], argument: str) -> str:
    """Select the current physical scope from one replicated byte pool."""

    scope_count = int(pool.get("scope_count", 1))
    if scope_count <= 1:
        return argument
    scope_nbytes = int(pool.get("scope_nbytes", -1))
    scope_index = pool.get("scope_index")
    if scope_nbytes < 0 or not isinstance(scope_index, str) or not scope_index:
        raise CodegenError("Replicated runtime pool has an incomplete scope ABI.")
    if pool.get("scope") != MemorySharingScope.BLOCK.value:
        raise CodegenError(
            "The Triton runtime can replicate function pools only per block."
        )
    if scope_index != "program_id_x":
        raise CodegenError(
            f"Unsupported block-pool scope-index ABI {scope_index!r}."
        )
    if scope_nbytes == 0:
        # Empty per-block frames have the same base and cannot be dereferenced.
        return argument
    # The inline address helper preserves the stride/base alignment proof at
    # the actual op call boundary. Index/value live ranges are bounded by op
    # execution functions, not an artificial pointer-returning device call.
    return f"_flagmega_block_scope_base({argument}, {scope_nbytes})"


__all__ = [
    "emit_pool_scope_base",
    "is_runtime_pool_space",
    "memory_scope_count",
    "memory_scope_counts",
]
