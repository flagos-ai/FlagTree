# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Tensor-map tables over chip-local packed storage.

A tensor-map table already emits one rebased entry per owner, so it never needs
every chip to hold the whole canonical weight.  These cases pin the chip-local
contract: entries address the owners resident in the current chip's segment, at
offsets relative to that segment rather than to a canonical global base.
"""

from dataclasses import replace

import pytest

from triton.flagmega import ir as fm
from triton.flagmega.codegen.triton.call_abi import describe_local_buffer_abi
from triton.flagmega.codegen.triton.tensor_descriptor_planner import (
    packed_distributed_tensor_map_table_request,
)
from triton.flagmega.errors import CodegenError
from triton.flagmega.passes.tir.bufferize.planner import BufferizationOptions
from triton.flagmega.passes.tir.bufferize.policy import NttBufferizationPolicy


# A chip axis that splits K, and a block axis that splits N: the shape of the
# GLM attention projections that drive the 4-card capacity budget.
_PLACEMENT = fm.Placement((4, 8, 16), "cyx", "cbb")
_LOGICAL_SHAPE = (384, 256)
_CHIP_EXTENT = 4
_BLOCK_EXTENT = 16


def _chip_local_weight_abi():
    """Bufferize a chip-split packed weight and describe its buffer ABI."""

    distributed = fm.DistributedType(
        fm.tensor_type(fm.vector_type("bfloat16", (8, 2, 8)), _LOGICAL_SHAPE),
        (
            fm.SBP.split_contiguous((0,), _LOGICAL_SHAPE[0] // _CHIP_EXTENT),
            fm.SBP.split_contiguous((2,), _LOGICAL_SHAPE[1] // _BLOCK_EXTENT),
        ),
        _PLACEMENT,
    )
    builder = fm.IRBuilder(dialect="tir", stage="selected_tir")
    weight = builder.weight(
        "weight", distributed, source="memory", key="weight", id="weight"
    )
    builder.function("main", (), (weight,))
    module = builder.build(entry="main")
    module = replace(
        module,
        metadata={"launch_contract": {"grid_mesh": _PLACEMENT.to_data()}},
    )
    module = NttBufferizationPolicy(replace(
        BufferizationOptions.generic(alignment=128), rdata_chip_sharding=True
    )).bufferize(module)
    buffer = fm.verify_buffer_plan(module).buffer_map["weight"]
    return describe_local_buffer_abi(buffer)


def _request(abi, **overrides):
    arguments = {
        "parameter": "weight_descriptor",
        "source": "rdata",
        "offset_bytes": 0,
        "descriptor_shape": (96, 16, 2, 64),
        "descriptor_strides": (2048, 128, 64, 1),
        "block_shape": (8, 8, 2, 64),
    }
    arguments.update(overrides)
    return packed_distributed_tensor_map_table_request(abi, **arguments)


def test_chip_split_packed_weight_keeps_chip_local_storage():
    abi = _chip_local_weight_abi()

    assert abi["storage_kind"] == "compact_per_chip"
    assert abi["coordinate_space"] == "local"
    # One chip holds only the owners of its own K slice.
    assert abi["owner_count"] == _BLOCK_EXTENT
    assert abi["local_capacity_shape"] == (
        _LOGICAL_SHAPE[0] // _CHIP_EXTENT,
        _LOGICAL_SHAPE[1] // _BLOCK_EXTENT,
    )


def test_table_addresses_only_the_owners_resident_in_this_chip():
    """One entry per chip-resident owner, not per placement owner.

    Emitting all 512 placement owners would require the whole canonical weight
    to be addressable from this chip, which is exactly the replication that
    chip-local storage exists to avoid.
    """

    abi = _chip_local_weight_abi()
    request = _chip_local_tensor_map_table(abi)
    entries = request["entries"]
    component_stride = int(abi["component_stride_scalar_elements"])

    assert request["kind"] == "table"
    assert len(entries) == _BLOCK_EXTENT
    for owner, entry in enumerate(entries):
        # Chip-local storage keeps one dense component per resident owner, so
        # an entry is the component itself: local shape, local strides, and an
        # offset that advances by exactly one component.
        assert entry["shape"] == (96, 16, 2, 64)
        assert entry["strides"] == (2048, 128, 64, 1)
        assert entry["offset_bytes"] == owner * component_stride * 2


def test_chip_local_table_offsets_stay_inside_one_chip_segment():
    abi = _chip_local_weight_abi()
    request = _chip_local_tensor_map_table(abi)

    component_stride = int(abi["component_stride_scalar_elements"])
    limit = component_stride * len(request["entries"]) * 2
    for entry in request["entries"]:
        span = 1 + sum(
            (extent - 1) * stride
            for extent, stride in zip(entry["shape"], entry["strides"])
        )
        assert entry["offset_bytes"] + span * 2 <= limit


def test_canonical_table_still_addresses_every_placement_owner():
    """Canonical-global storage keeps its whole-tensor table.

    Chip-local addressing is an additional contract, not a replacement: a
    canonical weight has every owner resident and must still enumerate them.
    """

    distributed = fm.DistributedType(
        fm.tensor_type(fm.vector_type("bfloat16", (8, 2, 8)), _LOGICAL_SHAPE),
        (
            fm.SBP.split_contiguous((0,), 48),
            fm.SBP.split_contiguous((1,), 16),
        ),
        fm.Placement((8, 16), "yx", "bb"),
    )
    abi = {
        "coordinate_space": "canonical_global",
        "storage_kind": "canonical_global",
        "distributed_type": distributed.to_data(),
        "logical_shape": _LOGICAL_SHAPE,
        "local_capacity_shape": (48, 16),
        "scalar_dtype": "bfloat16",
    }
    request = _request(
        abi,
        descriptor_shape=(384, 256, 2, 64),
        descriptor_strides=(32768, 128, 64, 1),
    )

    assert len(request["entries"]) == 8 * 16


def test_table_rejects_a_chip_local_abi_without_an_owner_stride():
    abi = {**_chip_local_weight_abi(), "component_stride_scalar_elements": 0}

    with pytest.raises(CodegenError, match="owner stride"):
        _chip_local_tensor_map_table(abi)


def _chip_local_tensor_map_table(abi):
    return _request(abi)
