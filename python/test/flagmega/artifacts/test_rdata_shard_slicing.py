# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Chip-sharded rdata packing must select owner elements without index lists.

The 45-layer GLM pack spent ~85 minutes inside ``_extract_local_shard`` because
every owner rebuilt a Python index list per axis and fancy-indexed the whole
global tensor, including for axes that are not split at all.  These tests pin
both halves of the contract: the image bytes stay exactly what the per-element
map produces, and an affine axis never materializes an index list.
"""

from dataclasses import replace

import numpy as np
import pytest
import torch

from triton.flagmega import ir as fm
from triton.flagmega.artifacts import pack_rdata, verify_rdata
from triton.flagmega.artifacts import rdata as rdata_module
from triton.flagmega.importer import MemoryCheckpoint, TensorInfo
from triton.flagmega.passes.tir.bufferize.planner import BufferizationOptions
from triton.flagmega.passes.tir.bufferize.policy import NttBufferizationPolicy


def _sharded_module(placement, tensor, policies):
    builder = fm.IRBuilder(dialect="tir", stage="selected_tir")
    distributed = fm.DistributedType(tensor, policies, placement)
    weight = builder.weight("weight", distributed, source="memory", key="weight", id="weight")
    builder.function("main", (), (weight, ))
    module = replace(builder.build(entry="main"),
                     metadata={"launch_contract": {"grid_mesh": placement.to_data()}})
    return NttBufferizationPolicy(
        replace(BufferizationOptions.generic(alignment=4), rdata_chip_sharding=True)
    ).bufferize(module)


def _pack(module, source, dtype, shape, tmp_path):
    checkpoint = MemoryCheckpoint({}, {"weight": TensorInfo("weight", dtype, shape, "memory")},
                                  {"weight": source})
    return pack_rdata(module, checkpoint, tmp_path)


@pytest.fixture
def counted_take(monkeypatch):
    """Count fancy-index gathers performed while packing."""

    calls: list[int] = []
    original = np.take

    def counting(array, indices, axis=None, **kwargs):
        calls.append(len(indices) if hasattr(indices, "__len__") else 1)
        return original(array, indices, axis=axis, **kwargs)

    monkeypatch.setattr(rdata_module.numpy if hasattr(rdata_module, "numpy") else np,
                        "take", counting, raising=False)
    monkeypatch.setattr(np, "take", counting)
    return calls


def test_an_affine_shard_never_builds_an_index_list(tmp_path, counted_take):
    """A contiguous split plus an unsplit axis must pack purely through views."""

    placement = fm.Placement((2, 4), "cx", "cb")
    module = _sharded_module(placement, fm.tensor_type("float32", (8, 6)),
                            (fm.SBP.split_contiguous((0, )), fm.SBP.broadcast()))
    source = torch.arange(48, dtype=torch.float32).reshape(8, 6)
    index = _pack(module, source, fm.DType.FLOAT32, (8, 6), tmp_path)

    assert counted_take == []
    # Blocks are broadcast, so one chip segment holds a single 4x6 component.
    image = np.fromfile(tmp_path / "rdata.bin", dtype=np.float32)
    image = image[:index["nbytes"] // 4].reshape(2, 4, 6)
    for chip in range(2):
        np.testing.assert_array_equal(image[chip], source[chip * 4:(chip + 1) * 4].numpy())


def test_block_cyclic_shards_match_the_per_element_map(tmp_path):
    """A cyclic axis is affine with stride > 1; the bytes must not move."""

    placement = fm.Placement((2, 4), "cx", "cb")
    module = _sharded_module(placement, fm.tensor_type("float32", (8, 4)),
                            (fm.SBP.split_contiguous((0, )),
                             fm.SBP.split_block_cyclic((1, ), 1)))
    source = torch.arange(32, dtype=torch.float32).reshape(8, 4)
    index = _pack(module, source, fm.DType.FLOAT32, (8, 4), tmp_path)

    image = np.fromfile(tmp_path / "rdata.bin", dtype=np.float32)
    image = image[:index["nbytes"] // 4].reshape(2, 4, 4, 1)
    for chip in range(2):
        for block in range(4):
            np.testing.assert_array_equal(image[chip, block, :, 0],
                                          source[chip * 4:(chip + 1) * 4, block].numpy())
    assert verify_rdata(tmp_path, module=module)["chip_count"] == 2
