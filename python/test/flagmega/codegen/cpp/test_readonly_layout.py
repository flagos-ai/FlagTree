# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Device weight layouts preserve every canonical byte and reject bad ranges."""
from dataclasses import replace

import pytest

from triton.flagmega import ir as fm
from triton.flagmega.codegen.cpp.readonly_layout import (
    READONLY_LAYOUT_SCHEMA, apply_readonly_layout, select_readonly_layouts,
)
from triton.flagmega.errors import ArtifactError, CodegenError
from triton.flagmega.importer import DirectoryCheckpoint
from triton.flagmega.passes.constants import freeze_constant_islands
from triton.flagmega.passes.tir.bufferize.planner import BufferizationOptions
from triton.flagmega.passes.tir.bufferize.policy import NttBufferizationPolicy


def _module(shape=(4, 6, 16), owners=2):
    builder = fm.IRBuilder(dialect="tir", stage="selected_tir")
    tensor = fm.tensor_type("bfloat16", shape)
    weight = builder.weight("w", tensor, source="checkpoint", key="w", id="w")
    placement = fm.Placement((owners,), "x", "b")
    distributed = fm.DistributedType(tensor,
        tuple(fm.SBP.split_contiguous((0,)) if axis == 1 else fm.SBP.broadcast()
              for axis in range(len(shape))), placement)
    output = builder.call("distributed.sharded_view", (weight,), distributed,
                          attrs={"new_type": distributed}, id="output")
    builder.function("main", (), (output,))
    module = freeze_constant_islands(builder.build(entry="main"))
    module = NttBufferizationPolicy(BufferizationOptions.generic(alignment=4)).bufferize(module)
    return replace(module, stage="bufferized_tir")


def _selection_case(slots=4, padding=0, external=False):
    module = _module((128, 768, 2, 64), owners=16)
    buffer = next(b for b in fm.verify_buffer_plan(module).buffers if b.storage == "rdata")
    abi = {"storage_kind": "canonical_global", "scalar_dtype": "bfloat16",
           "logical_shape": [128, 768, 2, 64], "local_capacity_shape": [128, 48, 2, 64],
           "scalar_lane_count": 1, "owner_count": 16}
    call = {"family": "dense_matmul_glu", "variant": "packed_k_major_gemv",
            "weight_k_lane": 16, "weight_group_stride": 98304,
            "weight_block_n": 8, "weight_block_k": 16,
            "local_k_capacity": 2048, "local_n_capacity": 384,
            "inputs": [{"formal": formal, "buffers": [{"actual": buffer.id,
                        "abi": {**abi, "storage_kind": "compact_local" if external else "canonical_global"}}]}
                       for formal in ("gate_weight", "up_weight")]}
    descriptor = {"render_calls": [call], "block_threads": 1024, "warp_size": 64,
                  "sme_slots": slots, "sme_pad": padding}
    return module, descriptor


@pytest.mark.parametrize("slots,padding,external", [(4, 0, False), (8, 0, False),
                                                   (4, 8, False), (4, 0, True)])
def test_layout_selection_requires_sme_and_readonly_geometry(slots, padding, external):
    module, descriptor = _selection_case(slots, padding, external)
    call = descriptor["render_calls"][0]
    selected = select_readonly_layouts(module, descriptor)
    expected = slots == 4 and padding == 0 and not external
    assert bool(selected["buffers"]) == expected
    assert bool(call.get("cpp_shard_major_weight")) == expected
    assert call["weight_group_stride"] == (6144 if expected else 98304)


def test_layout_rejects_another_canonical_reader_of_the_same_weight():
    module, descriptor = _selection_case()
    weight = descriptor["render_calls"][0]["inputs"][0]["buffers"]
    descriptor["render_calls"].append({"family": "elementwise", "inputs": [
        {"formal": "source", "buffers": weight}]})
    with pytest.raises(CodegenError, match="consumer without its device-address contract"):
        select_readonly_layouts(module, descriptor)


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_shard_major_bytes_preserve_payload_and_neighbors(device):
    torch = pytest.importorskip("torch")
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA is required")
    module = _module()
    buffer = next(b for b in fm.verify_buffer_plan(module).buffers if b.storage == "rdata")
    shape = [4, 2, 48, 2]
    pool = (torch.arange(buffer.offset + buffer.nbytes + 17, dtype=torch.int64, device=device)
            % 251).to(torch.uint8)
    before = pool.clone()
    record = {"buffer": buffer.id, "offset": buffer.offset, "nbytes": buffer.nbytes,
              "shape": shape, "axes": [1, 0, 2, 3]}
    # Validate the entire contract before touching any earlier valid range.
    damaged = {**record, "nbytes": buffer.nbytes - 2}
    with pytest.raises(ArtifactError, match="canonical buffer byte range"):
        apply_readonly_layout(pool, {"schema": READONLY_LAYOUT_SCHEMA,
                                    "buffers": [record, damaged]}, module)
    assert torch.equal(pool, before)
    apply_readonly_layout(pool, {"schema": READONLY_LAYOUT_SCHEMA, "buffers": [record]}, module)
    expected = before.narrow(0, buffer.offset, buffer.nbytes).view(*shape).permute(1, 0, 2, 3).contiguous()
    assert torch.equal(pool.narrow(0, buffer.offset, buffer.nbytes), expected.reshape(-1))
    assert torch.equal(pool[:buffer.offset], before[:buffer.offset])
    assert torch.equal(pool[buffer.offset + buffer.nbytes:], before[buffer.offset + buffer.nbytes:])
    damaged = {**record, "nbytes": buffer.nbytes - 2}
    with pytest.raises(ArtifactError, match="canonical buffer byte range"):
        apply_readonly_layout(pool, {"schema": READONLY_LAYOUT_SCHEMA, "buffers": [damaged]}, module)
    # Equal byte coverage alone cannot prove the K/owner address algebra.
    damaged = {**record, "shape": [2, 4, 48, 2]}
    with pytest.raises(ArtifactError, match="canonical buffer byte range"):
        apply_readonly_layout(pool, {"schema": READONLY_LAYOUT_SCHEMA, "buffers": [damaged]}, module)


def test_source_program_and_artifact_bytes_stay_canonical(tmp_path):
    np = pytest.importorskip("numpy")
    torch = pytest.importorskip("torch")
    from triton.flagmega.artifacts import pack_rdata, verify_rdata
    from triton.flagmega.runtime.source_rdata import load_source_rdata
    root = tmp_path / "checkpoint"
    root.mkdir()
    (root / "config.json").write_text("{}")
    # Distinct raw BF16 bit patterns reveal any byte/pair permutation error.
    value = np.arange(4 * 6 * 16, dtype=np.uint16).reshape(4, 6, 16)
    from safetensors.torch import save_file as save_torch
    save_torch({"w": torch.from_numpy(value).view(torch.bfloat16)}, root / "model.safetensors")
    module = _module()
    index = pack_rdata(module, DirectoryCheckpoint(root), tmp_path / "assets")
    assert index["programs"]
    buffer = next(b for b in fm.verify_buffer_plan(module).buffers if b.storage == "rdata")
    canonical = load_source_rdata(tmp_path / "assets", index, module, chip=0, device="cpu")
    before = canonical.clone()
    record = {"buffer": buffer.id, "offset": buffer.offset, "nbytes": buffer.nbytes,
              "shape": [4, 2, 48, 2], "axes": [1, 0, 2, 3]}
    apply_readonly_layout(canonical, {"schema": READONLY_LAYOUT_SCHEMA, "buffers": [record]}, module)
    reloaded = load_source_rdata(tmp_path / "assets", index, module, chip=0, device="cpu")
    assert torch.equal(reloaded, before)
    verify_rdata(tmp_path / "assets", module=module)
