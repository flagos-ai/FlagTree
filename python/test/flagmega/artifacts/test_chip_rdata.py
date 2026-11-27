# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT

from dataclasses import replace
from itertools import product

import numpy as np
import pytest
import torch

from triton.flagmega import ir as fm
from triton.flagmega.artifacts import pack_rdata, verify_rdata
from triton.flagmega.codegen.triton.call_abi import describe_local_buffer_abi
from triton.flagmega.codegen.triton.distributed_abi import distributed_buffer_abi
from triton.flagmega.codegen.triton.physical_access import emit_owner_base
from triton.flagmega.importer import MemoryCheckpoint, TensorInfo
from triton.flagmega.passes.tir.bufferize.planner import BufferizationOptions
from triton.flagmega.passes.tir.bufferize.policy import NttBufferizationPolicy


def test_chip_image_preserves_each_chip_and_block_partition(tmp_path):
    placement = fm.Placement((2, 2), "cx", "cb")
    tensor = fm.tensor_type("float32", (8, 4))
    distributed = fm.DistributedType(
        tensor, (fm.SBP.split_contiguous((0,)), fm.SBP.split_contiguous((1,))), placement
    )
    builder = fm.IRBuilder(dialect="tir", stage="selected_tir")
    weight = builder.weight("weight", distributed, source="memory", key="weight", id="weight")
    builder.function("main", (), (weight,))
    module = builder.build(entry="main")
    module = replace(module, metadata={"launch_contract": {"grid_mesh": placement.to_data()}})
    module = NttBufferizationPolicy(replace(
        BufferizationOptions.generic(alignment=4), rdata_chip_sharding=True
    )).bufferize(module)
    source = torch.arange(32, dtype=torch.float32).reshape(8, 4)
    checkpoint = MemoryCheckpoint({}, {"weight": TensorInfo("weight", fm.DType.FLOAT32, (8, 4), "memory")},
                                  {"weight": source})
    index = pack_rdata(module, checkpoint, tmp_path)
    assert index["chip_count"] == 2
    assert index["segment_nbytes"] == 64
    assert index["nbytes"] == 128
    image = np.fromfile(tmp_path / "rdata.bin", dtype=np.float32).reshape(2, 2, 4, 2)
    for chip in range(2):
        for block in range(2):
            np.testing.assert_array_equal(image[chip, block], source[chip * 4:(chip + 1) * 4,
                                                                    block * 2:(block + 1) * 2].numpy())
    assert verify_rdata(tmp_path, module=module)["chip_count"] == 2


def test_chip_local_broadcast_blocks_use_one_component(tmp_path):
    placement = fm.Placement((4, 8, 16), "cyx", "cbb")
    tensor = fm.tensor_type("float32", (16, 8, 1))
    distributed = fm.DistributedType(
        tensor,
        (fm.SBP.split_contiguous((0,)), fm.SBP.broadcast(), fm.SBP.broadcast()),
        placement,
    )
    builder = fm.IRBuilder(dialect="tir", stage="selected_tir")
    weight = builder.weight("weight", distributed, source="memory", key="weight", id="weight")
    builder.function("main", (), (weight,))
    module = builder.build(entry="main")
    module = replace(module, metadata={"launch_contract": {"grid_mesh": placement.to_data()}})
    module = NttBufferizationPolicy(replace(
        BufferizationOptions.generic(alignment=4), rdata_chip_sharding=True
    )).bufferize(module)
    plan = module.metadata["buffer_plan"]
    planned = next(item for item in plan["buffers"] if item["id"] == "weight")
    assert planned["distributed_storage_kind"] == "compact_per_chip"
    component_nbytes = 4 * 8 * 1 * 4
    assert planned["mem_span"]["size"]["value"] == component_nbytes
    physical_id = planned["mem_span"]["buffer"]
    physical = next(item for item in plan["physical_buffers"] if item["id"] == physical_id)
    assert physical["size"]["value"] == component_nbytes

    source = torch.arange(16 * 8, dtype=torch.float32).reshape(16, 8, 1)
    checkpoint = MemoryCheckpoint(
        {},
        {"weight": TensorInfo("weight", fm.DType.FLOAT32, (16, 8, 1), "memory")},
        {"weight": source},
    )
    index = pack_rdata(module, checkpoint, tmp_path)
    assert index["chip_count"] == 4
    assert index["segment_nbytes"] == component_nbytes
    image = np.fromfile(tmp_path / "rdata.bin", dtype=np.float32).reshape(4, 4, 8, 1)
    for chip in range(4):
        np.testing.assert_array_equal(image[chip], source[chip * 4:(chip + 1) * 4].numpy())


@pytest.mark.parametrize("split_blocks", [(), (1,), (2,), (1, 2)])
@pytest.mark.parametrize("access", ["rendered", "bound"])
def test_chip_image_read_addresses_match_block_partition(tmp_path, split_blocks, access):
    placement = fm.Placement((4, 2, 2), "cyx", "cbb")
    distributed = fm.DistributedType(
        fm.tensor_type("float32", (8, 8, 4)),
        tuple(fm.SBP.split_contiguous((axis,)) if axis == 0 or axis in split_blocks
              else fm.SBP.broadcast() for axis in range(3)),
        placement,
    )
    builder = fm.IRBuilder(dialect="tir", stage="selected_tir")
    weight = builder.weight("weight", distributed, source="memory", key="weight", id="weight")
    builder.function("main", (), (weight,))
    module = replace(builder.build(entry="main"),
                     metadata={"launch_contract": {"grid_mesh": placement.to_data()}})
    module = NttBufferizationPolicy(replace(
        BufferizationOptions.generic(alignment=4), rdata_chip_sharding=True
    )).bufferize(module)
    source = torch.arange(256, dtype=torch.float32).reshape(8, 8, 4)
    checkpoint = MemoryCheckpoint(
        {}, {"weight": TensorInfo("weight", fm.DType.FLOAT32, (8, 8, 4), "memory")},
        {"weight": source},
    )
    index = pack_rdata(module, checkpoint, tmp_path)
    image = np.fromfile(tmp_path / "rdata.bin", dtype=np.float32).reshape(4, -1)
    buffer = fm.verify_buffer_plan(module).buffer_map["weight"]
    expression = emit_owner_base(describe_local_buffer_abi(buffer), "base")
    for chip, y, x in product(range(4), range(2), range(2)):
        # Broadcast axes retain their full extent regardless of block coordinate.
        expected = source[chip * 2:(chip + 1) * 2,
                          slice(y * 4, (y + 1) * 4) if 1 in split_blocks else slice(None),
                          slice(x * 2, (x + 1) * 2) if 2 in split_blocks else slice(None)]
        owner = chip * 4 + y * 2 + x
        if access == "rendered":
            offset = eval(expression, {"__builtins__": {}}, {
                "base": 0, "block_index": y * 2 + x, "shard_index": owner,
                "shard_coord0": chip, "shard_coord1": y, "shard_coord2": x,
            })
        else:
            offset = distributed_buffer_abi(buffer, owner).component_offset_elements
        assert 0 <= offset <= index["segment_nbytes"] // 4 - expected.numel()
        np.testing.assert_array_equal(image[chip, offset:offset + expected.numel()],
                                      expected.numpy().reshape(-1))
