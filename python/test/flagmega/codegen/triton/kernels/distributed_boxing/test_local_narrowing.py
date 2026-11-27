# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Compact local copies must distinguish narrowing from cross-owner routing."""

from itertools import product
from types import SimpleNamespace

import pytest

from triton.flagmega import ir as fm
from triton.flagmega.codegen.triton.dimension_expression import emit_dimension
from triton.flagmega.errors import CodegenError
from .test_partial_reduce_local_abi import _abi, _prepare


def local_abi(value_type):
    coordinates = tuple(fm.dim(f"shard_coord_{axis}", 0, extent - 1)
                        for axis, extent in enumerate(value_type.placement.hierarchy))
    shard = fm.local_shard_descriptor(value_type, coordinates)
    lanes = value_type.tensor.dtype.lane_count if isinstance(value_type.tensor.dtype, fm.VectorType) else 1
    abi = _abi(tuple(d.fixed_value for d in value_type.tensor.shape),
               local_shape=tuple(d.fixed_value for d in shard.local_capacity_shape),
               storage_kind="compact_local", coordinate_space="local", lane_count=lanes,
               coordinates=tuple(emit_dimension(axis.map_local_to_global(f"local_coord_{i}"))
                                 for i, axis in enumerate(shard.axes)))
    return {**abi, "distributed_type": value_type.to_data(),
            "active_shape_expressions": tuple(emit_dimension(d) for d in shard.active_shape)}


@pytest.mark.parametrize("lanes", (1, 4))
@pytest.mark.parametrize("refined", (False, True))
def test_local_narrowing_inverts_source_map_without_changing_owner(lanes, refined):
    mesh = fm.Placement((2, 4), "yx", "bb")
    dtype = "bfloat16" if lanes == 1 else fm.vector_type("bfloat16", lanes)
    tensor = fm.tensor_type(dtype, (3, 40))
    source = fm.DistributedType(tensor, (fm.SBP.split_block_cyclic((0,), 1),
        fm.SBP.split_contiguous((1,), 12) if refined else fm.SBP.broadcast()), mesh)
    destination = fm.DistributedType(tensor, (source.axis_policies[0],
        fm.SBP.split_contiguous((1,), 12) if refined else fm.SBP.split_block_cyclic((1,), 3)), mesh)
    # A broadcast axis may be narrowed while a cyclic axis stays unchanged.
    if refined:
        source = fm.DistributedType(tensor, (fm.SBP.broadcast(), source.axis_policies[1]), mesh)
    leaf = _prepare((local_abi(source),), (local_abi(destination),), ("gather_reduce_scatter",))["leaves"][0]
    assert leaf["mode"] == "local_gather"
    assert leaf["writer_active"] == "True"
    for owner in product(range(2), range(4)):
        src = fm.local_shard_descriptor(source, owner)
        dst = fm.local_shard_descriptor(destination, owner)
        shape = tuple(d.fixed_value for d in dst.local_capacity_shape)
        for row in range(dst.active_shape[0].fixed_value):
            for column in range(dst.active_shape[1].fixed_value):
                global_row = dst.axes[0].map_local_to_global(row).fixed_value
                global_column = dst.axes[1].map_local_to_global(column).fixed_value
                src_row = next(i for i in range(src.active_shape[0].fixed_value)
                               if src.axes[0].map_local_to_global(i).fixed_value == global_row)
                src_column = next(i for i in range(src.active_shape[1].fixed_value)
                                  if src.axes[1].map_local_to_global(i).fixed_value == global_column)
                for lane in range(lanes):
                    scope = {"boxing_offsets": (row * shape[1] + column) * lanes + lane,
                             "shard_coord_0": owner[0], "shard_coord_1": owner[1],
                             "shard_y": owner[0], "shard_x": owner[1],
                             "tl": SimpleNamespace(minimum=min, maximum=max)}
                    offset = eval(leaf["source_offset"], {"__builtins__": {}}, scope)
                    assert offset == (src_row * src.local_capacity_shape[1].fixed_value + src_column) * lanes + lane


def test_sibling_compact_owners_are_not_mistaken_for_local_narrowing():
    mesh = fm.Placement((2, 4), "yx", "bb")
    tensor = fm.tensor_type("bfloat16", (40,))
    source = fm.DistributedType(tensor, (fm.SBP.split_contiguous((0,), 20),), mesh)
    destination = fm.DistributedType(tensor, (fm.SBP.split_contiguous((1,), 12),), mesh)
    with pytest.raises(CodegenError, match="explicit routed transfer"):
        _prepare((local_abi(source),), (local_abi(destination),), ("gather_reduce_scatter",))
