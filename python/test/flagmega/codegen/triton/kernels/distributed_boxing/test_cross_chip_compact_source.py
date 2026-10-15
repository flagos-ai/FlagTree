# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: MIT
"""A remote gather must invert both the source chip and compact owner table."""

from itertools import product
from math import prod
from types import SimpleNamespace

import pytest

from triton.flagmega import ir as fm
from triton.flagmega.codegen.triton import render_triton_package
from .test_cross_chip_runtime import cross_chip_module
from .test_local_narrowing import local_abi
from .test_partial_reduce_local_abi import _prepare


def test_fully_split_computed_value_renders_a_remote_gather(tmp_path):
    # The producer chooses compact_per_owner only when all mesh axes split it.
    # Less complete splits used by older runtime cases choose canonical storage.
    package = render_triton_package(cross_chip_module("split_cyx"), tmp_path)
    assert package["device_window_storage"] == "workspace"
    source = (tmp_path / "generated_kernels.py").read_text()
    assert 'space="device"' in source


@pytest.mark.parametrize("lanes", (1, 4))
@pytest.mark.parametrize("cyclic", (False, True))
@pytest.mark.parametrize("target_split", (False, True))
@pytest.mark.parametrize("storage_kind", ("compact_per_owner", "compact_per_chip"))
def test_remote_gather_reads_each_logical_element_from_its_compact_component(lanes, cyclic, target_split, storage_kind):
    mesh = fm.Placement((4, 2, 4), "cyx", "cbb")
    dtype = "bfloat16" if lanes == 1 else fm.vector_type("bfloat16", lanes)
    width = 130 if cyclic else 160
    tensor = fm.tensor_type(dtype, (3, width))
    policy = (fm.SBP.split_block_cyclic((0, 1, 2), 4) if cyclic
              else fm.SBP.split_contiguous((0, 1, 2), 5))
    source_type = fm.DistributedType(tensor, (fm.SBP.broadcast(), policy), mesh)
    target_type = fm.DistributedType(tensor, (fm.SBP.broadcast(),
        fm.SBP.split_contiguous((1,), width // 2) if target_split else fm.SBP.broadcast()), mesh)
    source = local_abi(source_type)
    # A padded component stride distinguishes table addressing from assuming
    # that compact owner components happen to follow canonical logical order.
    stride = prod(source["local_capacity_shape"]) * lanes + 8
    source.update(storage="workspace", memory_space="workspace",
                  storage_kind=storage_kind, component_stride_scalar_elements=stride)
    target = local_abi(target_type)
    target.update(storage="workspace", memory_space="workspace", storage_kind="canonical_global",
                  coordinate_space="canonical_global", scalar_storage_strides=(width * lanes, lanes))
    leaf = _prepare((source,), (target,), ("gather_reduce_scatter",))["leaves"][0]

    # Build the independent expected pool contents from each owner's IR shard.
    components = 32 if storage_kind == "compact_per_owner" else 8
    images = [[None] * (components * stride) for _ in range(4)]
    for owner in product(range(4), range(2), range(4)):
        shard = fm.local_shard_descriptor(source_type, owner)
        ordinal = (owner[0] * 2 + owner[1]) * 4 + owner[2]
        if storage_kind == "compact_per_chip":
            ordinal %= 8
        shape = tuple(d.fixed_value for d in shard.local_capacity_shape)
        for row, column in product(range(shard.active_shape[0].fixed_value),
                                   range(shard.active_shape[1].fixed_value)):
            global_column = shard.axes[1].map_local_to_global(column).fixed_value
            for lane in range(lanes):
                images[owner[0]][ordinal * stride + (row * shape[1] + column) * lanes + lane] = (
                    row * width + global_column) * lanes + lane

    assert leaf["remote_peers"] == 4
    assert leaf["capacity"] == prod(target["local_capacity_shape"]) * lanes
    for owner in product(range(4), range(2), range(4)):
        shard = fm.local_shard_descriptor(target_type, owner)
        columns = shard.local_capacity_shape[1].fixed_value
        for row, column, lane in product(range(3), range(shard.active_shape[1].fixed_value), range(lanes)):
            scope = {"boxing_offsets": (row * columns + column) * lanes + lane,
                     **{f"shard_coord_{i}": value for i, value in enumerate(owner)},
                     **{f"shard_coord{i}": value for i, value in enumerate(owner)},
                     "tl": SimpleNamespace(minimum=min, maximum=max)}
            peer = eval(leaf["remote_owner"], {"__builtins__": {}}, scope)
            offset = eval(leaf["remote_source_offset"], {"__builtins__": {}}, scope)
            destination = eval(leaf["result_offset"], {"__builtins__": {}}, scope)
            expected = (row * width + shard.axes[1].map_local_to_global(column).fixed_value) * lanes + lane
            assert images[peer][offset] == expected
            assert destination == expected
