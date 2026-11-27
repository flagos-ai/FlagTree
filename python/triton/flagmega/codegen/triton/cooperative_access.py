# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Shared physical ABI proofs for kernels that read chip-peer components."""

from collections.abc import Mapping
from math import prod

from triton.flagmega.codegen.triton.physical_access import _bind_coordinate_expression, emit_buffer_pointer
from triton.flagmega.errors import CodegenError


def chip_peer_pointer(binding, coordinates, *, owner):
    abi = binding["abi"]
    if abi.get("storage_kind") in {"compact_per_chip", "compact_per_owner"}:
        expression = abi.get("owner_index_expression")
        if expression == "shard_index" and abi.get("storage_kind") == "compact_per_owner":
            # The default compact-owner ABI indexes every placement owner.
            # Bind that documented row-major map to the requested peer rather
            # than leaving shard_index tied to the executing CTA. A chip-only
            # allocation cannot use this map: its chip coordinates are absent.
            kind = abi.get("distributed_type")
            placement = kind.get("placement") if isinstance(kind, Mapping) else None
            hierarchy = placement.get("hierarchy") if isinstance(placement, Mapping) else None
            if (not isinstance(hierarchy, (tuple, list)) or len(hierarchy) != len(coordinates)
                    or not hierarchy or any(type(size) is not int or size <= 0 for size in hierarchy)
                    or abi.get("owner_count") != prod(hierarchy)
                    or abi.get("placement_owner_count") != prod(hierarchy)):
                raise CodegenError(f"{owner} require a proved full-placement compact owner map.")
            expression = "0"
            for axis, size in enumerate(hierarchy):
                expression = f"({expression} * {size} + shard_coord_{axis})"
        if not isinstance(expression, str) or expression == "shard_index":
            raise CodegenError(f"{owner} require an explicit compact owner map.")
        index = _bind_coordinate_expression(expression, (), coordinates)
        return emit_buffer_pointer(abi, binding["runtime_argument"], owner_index=index)
    return emit_buffer_pointer(abi, binding["runtime_argument"])


def require_chip_shared(binding, name, *, owner):
    abi = binding["abi"]
    if (
        abi.get("memory_sharing_scope") != "chip"
        or abi.get("pool_scope_stride_bytes", 0)
        or abi.get("distributed_backing_type") is not None
        or "view_byte_offset" in abi
        or abi.get("coordinate_space") not in {"canonical_global", "local"}
        or abi.get("coordinate_space") == "local"
        and abi.get("storage_kind") not in {"compact_per_chip", "compact_per_owner"}
    ):
        raise CodegenError(f"{owner} require proved chip-shared {name} storage.")


def require_chip_workspace(binding, shape, name, *, owner):
    require_chip_shared(binding, name, owner=owner)
    abi = binding["abi"]
    strides = tuple(prod(shape[i + 1 :]) for i in range(len(shape)))
    if (
        abi.get("coordinate_space") != "canonical_global"
        or abi.get("scalar_dtype") != "float32"
        or abi.get("scalar_lane_count") != 1
        or tuple(abi.get("local_capacity_shape", ())) != shape
        or tuple(abi.get("scalar_storage_strides", ())) != strides
    ):
        raise CodegenError(f"{owner} require contiguous FP32 {name} with shape {shape}.")


def require_owner_output(binding, name, *, owner):
    """An owner-local merge may publish into one private owner component.

    Peer inputs and cross-phase scratch still require chip sharing. A private
    result has its own complete local component, rather than a partial or a
    view into another owner's backing. The normal result writer predicate
    then populates every private replica instead of electing a shared writer.
    """
    abi = binding["abi"]
    if abi.get("memory_sharing_scope") != "block":
        require_chip_shared(binding, name, owner=owner)
        return
    _require_complete_owner_component(binding, name, owner=owner)


def require_owner_input(binding, name, *, owner):
    """Read an identical complete activation from the current CTA's replica."""
    from triton.flagmega.codegen.triton.tensor_transform_renderers import _tensor_type
    from triton.flagmega.ir import SBP, DistributedType

    if binding["abi"].get("memory_sharing_scope") != "block":
        require_chip_shared(binding, name, owner=owner)
        return
    _require_complete_owner_component(binding, name, owner=owner)
    kind = _tensor_type(binding["abi"])
    if isinstance(kind, DistributedType) and (
        kind.partial is not None or kind.exclusive is not None
        or any(policy != SBP.broadcast() for policy in kind.axis_policies)
    ):
        raise CodegenError(f"{owner} require a complete broadcast {name} replica.")


def _require_complete_owner_component(binding, name, *, owner):
    abi = binding["abi"]
    if (
        abi.get("coordinate_space") != "local"
        or abi.get("storage_kind") != "compact_local"
        or abi.get("distributed_backing_type") is not None
        or "view_byte_offset" in abi
        or "view_byte_offset_expression" in abi
        or int(abi.get("pool_scope_count", 1)) > 1 and (
            int(abi.get("pool_scope_stride_bytes", 0)) <= 0
            or abi.get("pool_scope_index") != "block_index"
        )
    ):
        raise CodegenError(f"{owner} require a complete owner-local {name} component.")
