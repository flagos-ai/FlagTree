# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Lift target splits of rotary groups and paired halves into tensor layouts."""

from dataclasses import replace
from itertools import permutations, product
from math import prod

from triton.flagmega.ir import DistributedType, SBP, dim, is_distributable
from triton.flagmega.ir.distributed_type import ContiguousSplit, scale_split_units
from triton.flagmega.ir.ops.nn._rotary_distribution import has_remote_rotary_pairs
from triton.flagmega.passes.auto_distributed.split_candidates import DistributedSplitCandidateContext


def rotary_dimension_splits(context, tensor, axis, mesh_axes, rotary_dim):
    extent = tensor.shape[axis]
    if not extent.is_fixed:
        return ()
    lanes = getattr(tensor.dtype, "lane_count", 1)
    rotary = extent.fixed_value * lanes if rotary_dim is None else rotary_dim
    if rotary % lanes:
        return ()
    physical_rotary = rotary // lanes
    owners = prod(context.placement.hierarchy[index] for index in mesh_axes)
    candidates = list(context.split_candidates(tensor, axis, mesh_axes, purpose="output"))
    # Splitting whole rotary groups keeps partial-RoPE's prefix together.
    groups = (extent.fixed_value + physical_rotary - 1) // physical_rotary
    grouped = replace(tensor, shape=(*tensor.shape[:axis], dim(groups), *tensor.shape[axis + 1:]))
    group_context = DistributedSplitCandidateContext(
        grouped, axis, context.placement, mesh_axes, (groups + owners - 1) // owners, groups, "output",
    )
    for split in context.split_candidate_provider.get_candidates(group_context):
        scaled = scale_split_units(split, physical_rotary, 1)
        if scaled is not None:
            candidates.append(scaled)
    # A split of one half repeats at the second half. Contiguous half slices
    # are therefore block-cyclic slices in the original dimension.
    if physical_rotary % 2 == 0:
        half = physical_rotary // 2
        factored = replace(tensor, shape=(*tensor.shape[:axis], dim(half), *tensor.shape[axis + 1:]))
        for split in context.split_candidates(factored, axis, mesh_axes, purpose="output"):
            if len(split.stages) == 1 and isinstance(split.stages[0].distribution, ContiguousSplit):
                granularity = split.stages[0].distribution.granularity
                if granularity is not None and granularity.is_fixed and half == owners * granularity.fixed_value:
                    candidates.append(SBP.split_block_cyclic(mesh_axes, granularity.fixed_value))
            else:
                candidates.append(split)
    result = []
    for split in dict.fromkeys(candidates):
        policies = [SBP.broadcast()] * tensor.rank
        policies[axis] = split
        if not is_distributable(tensor, tuple(policies), context.placement):
            continue
        value = DistributedType(tensor, tuple(policies), context.placement)
        if not has_remote_rotary_pairs(value, rotary_dim, axis):
            result.append(split)
    return tuple(result)


def coupled_rotary_layouts(context, tensors, head_axis, dim_axis, rotary_dim):
    """Avoid a Cartesian product of unrelated Q/K/V dimension assignments."""
    for head_axes, dim_axes in _mesh_assignments(context):
        fields = [tuple(_layout_options(context, tensor, head_axis, dim_axis, rotary_dim, head_axes, dim_axes))
                  for tensor in tensors]
        for query in fields[0]:
            for key in fields[1]:
                for value in fields[2]:
                    if key.axis_policies == value.axis_policies:
                        yield (query, key, value)


def rotary_layouts(context, tensor, head_axis, dim_axis, rotary_dim):
    for head_axes, dim_axes in _mesh_assignments(context):
        yield from _layout_options(context, tensor, head_axis, dim_axis, rotary_dim, head_axes, dim_axes)


def _mesh_assignments(context):
    mesh = tuple(index for index, count in enumerate(context.placement.hierarchy) if count > 1)
    assignments = [((), (axis,)) for axis in mesh]
    assignments.extend(((head,), (dimension,)) for head, dimension in permutations(mesh, 2))
    if len(mesh) > 1:
        assignments.extend(((), axes) for axes in (mesh, tuple(reversed(mesh))))
    return assignments


def _layout_options(context, tensor, head_axis, dim_axis, rotary_dim, head_axes, dim_axes):
    heads = context.split_candidates(tensor, head_axis, head_axes, purpose="output") if head_axes else ()
    heads = heads or (SBP.broadcast(),)
    dims = rotary_dimension_splits(context, tensor, dim_axis, dim_axes, rotary_dim)
    for head, dimension in product(heads, dims):
        policies = [SBP.broadcast()] * tensor.rank
        policies[head_axis], policies[dim_axis] = head, dimension
        yield DistributedType(tensor, tuple(policies), context.placement)
