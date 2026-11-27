# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Numeric readonly constants retain every compact block-owner component."""

from dataclasses import replace
from math import prod

import numpy as np
import pytest
import torch
from safetensors.numpy import save_file

from triton.flagmega import ir as fm
from triton.flagmega.artifacts import pack_rdata, verify_rdata
from triton.flagmega.artifacts.readonly_layout import relocate_readonly
from triton.flagmega.artifacts.source_layout import component_layouts
from triton.flagmega.errors import ArtifactError
from triton.flagmega.importer import DirectoryCheckpoint
from triton.flagmega.passes.constants import freeze_constant_islands
from triton.flagmega.passes.tir.bufferize.planner import BufferizationOptions
from triton.flagmega.passes.tir.bufferize.policy import NttBufferizationPolicy
from triton.flagmega.runtime.shared_readonly import SharedReadonlyPool, readonly_fingerprints
from triton.flagmega.runtime.source_rdata import load_source_rdata


def fixture(tmp_path, *, later_owner_delta=0, zero=False):
    root = tmp_path / "checkpoint"
    root.mkdir()
    (root / "config.json").write_text("{}")
    values = {"a": np.arange(12, dtype=np.float32).reshape(2, 6),
              "b": np.arange(24, dtype=np.float32).reshape(4, 6) + 100}
    for value in values.values():
        value[:, 3:] += later_owner_delta
        if zero:
            value.fill(0)
    save_file(values, root / "model.safetensors")
    checkpoint = DirectoryCheckpoint(root)
    placement = fm.Placement((2, 2), "cx", "cb")
    builder = fm.IRBuilder(dialect="tir", stage="selected_tir",
                           metadata={"launch_contract": {"grid_mesh": placement.to_data()}})
    weights = [builder.weight(key, fm.tensor_type("float32", value.shape),
                              source="checkpoint", key=key, id=key) for key, value in values.items()]
    tensor = fm.tensor_type("float32", (6, 6))
    joined = builder.call("tensors.concat", weights, tensor, attrs={"axis": 0}, id="joined")
    numeric = builder.call("math.add", (joined, joined), tensor, id="numeric")
    distributed = fm.DistributedType(tensor, (fm.SBP.split_contiguous((0,)),
                                             fm.SBP.split_contiguous((1,))), placement)
    outputs = [builder.call("distributed.sharded_view", (value,), distributed,
                            attrs={"new_type": distributed}, id=identity)
               for value, identity in ((joined, "virtual_output"), (numeric, "numeric_output"))]
    builder.function("main", (), outputs)
    module = freeze_constant_islands(builder.build(entry="main"))
    module = NttBufferizationPolicy(replace(BufferizationOptions.generic(alignment=4),
                                           rdata_chip_sharding=True)).bufferize(module)
    return replace(module, stage="bufferized_tir"), checkpoint, np.concatenate(tuple(values.values())) * 2


def assets(tmp_path, module, checkpoint):
    root = tmp_path / "assets"
    index = pack_rdata(module, checkpoint, root)
    verify_rdata(root, module=module)
    numeric = next(entry["buffer"] for entry in index["entries"]
                   if entry["buffer"] not in {p["buffer"] for p in index["programs"]})
    return root, index, fm.verify_buffer_plan(module).buffer_map[numeric]


@pytest.mark.parametrize("chip", [0, 1])
@pytest.mark.parametrize("device", ["cpu", "cuda:0"])
def test_shared_numeric_load_keeps_all_block_owners(tmp_path, chip, device):
    from triton.flagmega.artifacts.rdata import _extract_local_shard
    if device.startswith("cuda") and not torch.cuda.is_available():
        pytest.skip("CUDA required")
    module, checkpoint, source = fixture(tmp_path)
    root, index, numeric = assets(tmp_path, module, checkpoint)
    pool = SharedReadonlyPool(index["segment_nbytes"], device=device)
    actual = pool._bind_verified(root, index, module, chip=chip, device=device,
                                 expected_nbytes=index["segment_nbytes"]).cpu()
    reference = load_source_rdata(root, index, module, chip=chip, device="cpu")
    for owner, (offset, shape, _) in enumerate(component_layouts(numeric, chip)):
        descriptor = fm.local_shard_descriptor(numeric.distributed_type, (chip, owner))
        expected = _extract_local_shard(source, descriptor, numeric.component_shape)
        np.testing.assert_array_equal(actual[offset:offset + prod(shape)].numpy().view(np.float32)
                                      .reshape(expected.shape), expected)
    torch.testing.assert_close(actual, reference, rtol=0, atol=0)


@pytest.mark.parametrize("zero", [False, True])
@pytest.mark.parametrize("device", ["cpu", "cuda:0"])
def test_selected_numeric_load_preserves_unselected_bytes(tmp_path, zero, device):
    if device.startswith("cuda") and not torch.cuda.is_available():
        pytest.skip("CUDA required")
    module, checkpoint, _ = fixture(tmp_path, zero=zero)
    root, index, numeric = assets(tmp_path, module, checkpoint)
    size = index["segment_nbytes"]
    actual = torch.full((size,), 197, dtype=torch.uint8, device=device)
    reference = load_source_rdata(root, index, module, chip=1, device="cpu").to(device)
    selected = torch.zeros(size, dtype=torch.bool, device=device)
    for offset, shape, _ in component_layouts(numeric, 1):
        selected[offset:offset + prod(shape)] = True
    returned = load_source_rdata(root, index, module, chip=1, device=device, pool=actual,
                                 buffer_ids=(numeric.id,))
    assert returned.data_ptr() == actual.data_ptr()
    torch.testing.assert_close(actual[selected], reference[selected], rtol=0, atol=0)
    assert torch.all(actual[~selected] == 197)


def test_numeric_proof_includes_later_owner_bytes(tmp_path):
    a, b = tmp_path / "a", tmp_path / "b"
    a.mkdir(); b.mkdir()
    first, checkpoint, _ = fixture(a)
    second, checkpoint2, _ = fixture(b, later_owner_delta=7)
    root, index, numeric = assets(a, first, checkpoint)
    root2, index2, numeric2 = assets(b, second, checkpoint2)
    before = load_source_rdata(root, index, first, chip=0, device="cpu")
    after = load_source_rdata(root2, index2, second, chip=0, device="cpu")
    layouts = list(component_layouts(numeric, 0))
    assert len(layouts) == 2
    offset, shape, _ = layouts[0]
    assert torch.equal(before[offset:offset + prod(shape)], after[offset:offset + prod(shape)])
    offset, shape, _ = layouts[1]
    assert not torch.equal(before[offset:offset + prod(shape)], after[offset:offset + prod(shape)])
    assert readonly_fingerprints(root, index, first)[numeric.id] != readonly_fingerprints(root2, index2, second)[numeric2.id]


def test_overlap_with_later_owner_rejected_before_copy(tmp_path):
    a, b = tmp_path / "a", tmp_path / "b"
    a.mkdir(); b.mkdir()
    first, checkpoint, _ = fixture(a)
    second, checkpoint2, _ = fixture(b, later_owner_delta=7)
    root, index, numeric = assets(a, first, checkpoint)
    size = index["segment_nbytes"] * 4
    first = relocate_readonly(first, {v.id: v.offset for v in fm.verify_buffer_plan(first).buffers
                                     if v.storage == "rdata"}, nbytes=size)
    offsets = {v.id: v.offset + index["segment_nbytes"] * 2
               for v in fm.verify_buffer_plan(second).buffers if v.storage == "rdata"}
    offsets[numeric.id] = numeric.offset + numeric.component_stride_bytes
    second = relocate_readonly(second, offsets, nbytes=size)
    root, index, _ = assets(a, first, checkpoint)
    root2, index2, _ = assets(b, second, checkpoint2)
    pool = SharedReadonlyPool(size, device="cpu")
    actual = pool._bind_verified(root, index, first, chip=0, device="cpu", expected_nbytes=size)
    expected = actual.clone()
    with pytest.raises(ArtifactError, match="byte proofs"):
        pool._bind_verified(root2, index2, second, chip=0, device="cpu", expected_nbytes=size)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
