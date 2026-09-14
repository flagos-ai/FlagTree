# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Transfer source proofs shared by ABI planning and TIR verification."""

from math import prod

from triton.flagmega.errors import IRVerificationError
from triton.flagmega.ir.distributed_type import SBPSplit
from triton.flagmega.ir.memory_effect import (
    MemoryAccessMode, MemoryEffectKind, MemoryOwnerAccess, expand_memory_effect,
)
from triton.flagmega.ir.model import DistributedType
from triton.flagmega.ir.types import VectorType


def verify_transfer_sources(function, dispatch, implementation, *, check_alignment=True, stage=None):
    pipeline = implementation.transfer_pipeline
    if pipeline is None:
        return

    def fail(message):
        raise IRVerificationError(f"@{function.name} {message}", stage=stage)

    def argument_at(index, kind="transfer source"):
        if index >= len(dispatch.arguments):
            fail(f"has invalid {kind} operand {index}.")
        name = dispatch.arguments[index]
        if name not in function.parameter_map:
            fail(f"has unknown transfer source {name!r}.")
        return name

    for index in pipeline.producer_read_argument_indices:
        name = argument_at(index, "producer read")
        if name not in dispatch.reads or name in dispatch.writes:
            fail(f"has non-read-only producer read operand {index} ({name!r}).")
    mutable_leaves = set()
    for channel in pipeline.channels:
        for index in channel.source_argument_indices:
            name = argument_at(index)
            partition = channel.inplace_partition
            if partition is None:
                if name not in dispatch.reads or name in dispatch.writes:
                    fail(f"has non-read-only source operand {index} ({name!r}).")
            else:
                leaf_index = _verify_inplace_partition(function, dispatch, implementation, channel, name, fail)
                key = (name, leaf_index)
                if key in mutable_leaves:
                    fail(f"has overlapping inplace transfer channels for {name!r}.")
                mutable_leaves.add(key)
            alignment = function.parameter_map[name].alignment_bytes
            if check_alignment and alignment is not None and channel.source_alignment_bytes > alignment:
                fail(f"implementation exceeds the declared alignment of {name!r}: "
                     f"{channel.source_alignment_bytes} bytes required, exceeding its {alignment}-byte storage contract.")


def _verify_inplace_partition(function, dispatch, implementation, channel, name, fail):
    partition = channel.inplace_partition
    parameter = function.parameter_map[name]
    leaf_index, source = partition.source_leaf(parameter.type)
    if name not in dispatch.reads or name not in dispatch.writes:
        fail("inplace transfer requires a read-write source.")
    effect = expand_memory_effect(parameter.type, dispatch.memory_effect_map[name])[leaf_index]
    if (effect.physical_mode != MemoryAccessMode.READ_WRITE
            or effect.kind is not MemoryEffectKind.DIRECT or effect.owner_access is not MemoryOwnerAccess.LOCAL):
        fail("inplace transfer requires direct, owner-local read-write access to its source field.")
    row_rank = partition.source_row_rank
    if row_rank >= source.rank or any(not value.is_fixed or value.fixed_value <= 0 for value in source.shape):
        fail("inplace transfer source must have fixed positive row and column extents.")
    rows = prod(value.fixed_value for value in source.shape[:row_rank])
    columns = prod(value.fixed_value for value in source.shape[row_rank:])
    dtype = source.dtype
    if isinstance(dtype, VectorType):
        columns *= dtype.lane_count
        dtype = dtype.elem_type
    if partition.output_index >= len(dispatch.outputs):
        fail("inplace transfer names an absent output.")
    output_name = dispatch.outputs[partition.output_index]
    if output_name not in function.parameter_map or output_name not in dispatch.writes:
        fail("inplace transfer must name a written output parameter.")
    output = function.parameter_map[output_name].type
    if (not isinstance(output, DistributedType) or output.partial is not None or output.exclusive is not None
            or isinstance(output.tensor.dtype, VectorType) or partition.output_axis >= output.tensor.rank):
        fail("inplace transfer needs a nonpartial, scalar distributed output.")
    axis = partition.output_axis
    if any(not value.is_fixed or value.fixed_value != (rows if index == axis else 1)
           for index, value in enumerate(output.tensor.shape)):
        fail("inplace transfer output must cover exactly the source rows along one axis.")
    policy = output.axis_policies[axis]
    required_axes = {index for index, extent in enumerate(output.placement.hierarchy) if extent > 1}
    if not isinstance(policy, SBPSplit) or not required_axes.issubset(policy.hierarchy_axes):
        fail("inplace transfer rows must have unique owners across the placement.")
    workspace = implementation.shared_workspaces[channel.shared_workspace_indices[0]].type
    if (workspace.dtype != dtype or workspace.rank != 3 or any(not value.is_fixed for value in workspace.shape)
            or tuple(value.fixed_value for value in workspace.shape[:2]) != (
                implementation.transfer_pipeline.capacity, partition.tile_rows)
            or workspace.shape[2].fixed_value < columns):
        fail("inplace transfer workspace must match stage count, tile rows and scalar column capacity/dtype.")
    if parameter.buffers:
        buffer = parameter.buffers[leaf_index]
        stride = 1
        for extent, actual in reversed(tuple(zip(source.shape, buffer.strides, strict=True))):
            if not actual.is_fixed or actual.fixed_value != stride:
                fail("inplace transfer source must have a dense row-major physical ABI.")
            stride *= extent.fixed_value
    return leaf_index


__all__ = ["verify_transfer_sources"]
