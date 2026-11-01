# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Readonly transport changes device addresses without changing artifact bytes."""

from dataclasses import replace
from types import SimpleNamespace

import pytest

from triton.flagmega import ir as fm
from triton.flagmega.passes.constants import freeze_constant_islands
from triton.flagmega.passes.tir.bufferize.planner import BufferizationOptions
from triton.flagmega.passes.tir.bufferize.policy import NttBufferizationPolicy


def _module(shape=(4, 6, 16), owners=2, *, typed=False, split_axes=None,
            placement=None, cyclic=False, external=False):
    builder = fm.IRBuilder(dialect="tir", stage="selected_tir")
    dtype = fm.VectorType(fm.DType.BFLOAT16, (8, 2, 8)) if typed else "bfloat16"
    tensor = fm.tensor_type(dtype, shape)
    weight = (builder.var("w", tensor, id="w") if external else
              builder.weight("w", tensor, source="checkpoint", key="w", id="w"))
    placement = placement or fm.Placement((owners,), "x", "b")
    axes = tuple(range(len(placement.hierarchy))) if split_axes is None else split_axes
    split = fm.SBP.split_block_cyclic(axes, 1) if cyclic else fm.SBP.split_contiguous(axes)
    distributed = fm.DistributedType(tensor, tuple(
        split if axis == 1 else fm.SBP.broadcast()
        for axis in range(len(shape))), placement)
    output = builder.call("distributed.sharded_view", (weight,), distributed,
                          attrs={"new_type": distributed}, id="output")
    builder.function("main", (weight,) if external else (), (output,))
    module = freeze_constant_islands(builder.build(entry="main"))
    module = NttBufferizationPolicy(BufferizationOptions.generic(alignment=256)).bufferize(module)
    return replace(module, stage="bufferized_tir")


def _record(module):
    buffer = next(b for b in fm.verify_buffer_plan(module).buffers if b.storage == "rdata")
    return {"buffer": buffer.id, "offset": buffer.offset, "nbytes": buffer.nbytes,
            "shape": [4, 2, 48, 2], "axes": [1, 0, 2, 3]}


def _selection_case(*, typed=False, owners=16, **geometry):
    from triton.flagmega.codegen.triton.call_abi import describe_local_buffer_abi
    n, k = 128, 512
    shape = (k // 16, n * owners // 8) + (() if typed else (2, 64))
    module = _module(shape, owners, typed=typed, **geometry)
    buffer = next(b for b in fm.verify_buffer_plan(module).buffers
                  if b.storage == ("input" if geometry.get("external") else "rdata"))
    binding = {"formal": "rhs", "actual": buffer.id, "runtime_argument": "RDATA",
               "abi": describe_local_buffer_abi(buffer)}
    call = {"family": "dense_matmul", "variant": "packed_k_major_gemv_norm_stats",
            "local_n_capacity": n, "local_k_capacity": k,
            "inputs": [{"formal": "rhs", "buffers": [binding]}],
            "corex_flat_sme_down": {"source": "X", "weight": "canonical",
                                    "weight_k_group_stride": n * owners * 16}}
    return module, call


def _glu_selection_case(*, typed=False, owners=16, external_up=False):
    from triton.flagmega.codegen.triton.call_abi import describe_local_buffer_abi
    n, k = 384, 2048
    shape = (k // 16, n * owners // 8) + (() if typed else (2, 64))
    dtype = fm.VectorType(fm.DType.BFLOAT16, (8, 2, 8)) if typed else "bfloat16"
    tensor = fm.tensor_type(dtype, shape)
    placement = fm.Placement((owners,), "x", "b")
    distributed = fm.DistributedType(tensor, tuple(
        fm.SBP.split_contiguous((0,)) if axis == 1 else fm.SBP.broadcast()
        for axis in range(len(shape))), placement)
    builder = fm.IRBuilder(dialect="tir", stage="selected_tir")
    gate = builder.weight("gate", tensor, source="checkpoint", key="gate", id="gate")
    up = (builder.var("up", tensor, id="up") if external_up else
          builder.weight("up", tensor, source="checkpoint", key="up", id="up"))
    views = tuple(builder.call("distributed.sharded_view", (w,), distributed,
                               attrs={"new_type": distributed}, id=name)
                  for w, name in ((gate, "gate_view"), (up, "up_view")))
    builder.function("main", (up,) if external_up else (), views)
    module = freeze_constant_islands(builder.build(entry="main"))
    module = NttBufferizationPolicy(BufferizationOptions.generic(alignment=256)).bufferize(module)
    module = replace(module, stage="bufferized_tir")
    buffers = [b for b in fm.verify_buffer_plan(module).buffers if b.storage == "rdata"]
    if external_up:
        assert len(buffers) == 1
        buffers += [next(b for b in fm.verify_buffer_plan(module).buffers if b.storage == "input")]
    else:
        assert len(buffers) == 2
    bindings = [dict(formal=role, actual=buffer.id, runtime_argument="RDATA",
                     abi=describe_local_buffer_abi(buffer))
                for role, buffer in zip(("gate_weight", "up_weight"), buffers)]
    call = dict(family="dense_matmul_glu", variant="packed_k_major_gemv",
                local_n_capacity=n, local_k_capacity=k,
                inputs=[dict(formal=item["formal"], buffers=[item]) for item in bindings],
                corex_flat_sme_glu=dict(source="X", gate_weight="canonical_gate",
                    up_weight="canonical_up", result="Y", weight_k_group_stride=n * owners * 16))
    return module, call


@pytest.mark.parametrize("typed", (False, True))
@pytest.mark.parametrize("owners", (2, 16))
def test_glu_owner_contiguous_k_walk_proves_both_weight_ranges(typed, owners):
    from triton.flagmega.codegen.triton.corex_readonly_layout import select_readonly_layouts
    module, call = _glu_selection_case(typed=typed, owners=owners)
    contract = select_readonly_layouts(module, [call])
    assert len(contract["buffers"]) == 2
    for record in contract["buffers"]:
        assert record["shape"] == [128, owners, 6144, 2]
        assert record["nbytes"] == owners * 384 * 2048 * 2
    sme = call["corex_flat_sme_glu"]
    assert sme["weight_k_group_stride"] == 6144
    for role in ("gate_weight", "up_weight"):
        assert "(shard_index) * 786432" in sme[role]
        assert "tl.pointer_type(tl.bfloat16)" in sme[role]


@pytest.mark.parametrize("role", ("gate_weight", "up_weight"))
@pytest.mark.parametrize("change", ("storage", "coordinates", "owners", "k_split"))
def test_glu_pair_keeps_canonical_addresses_if_either_weight_is_unproved(role, change):
    import copy
    from triton.flagmega.codegen.triton.corex_readonly_layout import select_readonly_layouts
    module, call = _glu_selection_case()
    abi = next(a for a in call["inputs"] if a["formal"] == role)["buffers"][0]["abi"]
    if change == "storage":
        abi["storage_kind"] = "compact_per_owner"
    elif change == "coordinates":
        abi["coordinate_space"] = "local"
    elif change == "owners":
        abi["owner_count"] = 8
    else:
        abi["local_capacity_shape"] = (64, *abi["local_capacity_shape"][1:])
    before = copy.deepcopy(call)
    assert select_readonly_layouts(module, [call])["buffers"] == []
    assert call == before


def test_open_glu_weight_does_not_change_its_closed_partner():
    import copy
    from triton.flagmega.codegen.triton.corex_readonly_layout import select_readonly_layouts
    module, call = _glu_selection_case(external_up=True)
    before = copy.deepcopy(call)
    assert select_readonly_layouts(module, [call])["buffers"] == []
    assert call == before


@pytest.mark.parametrize("role", ("gate_weight", "up_weight"))
def test_glu_canonical_reader_is_rejected_before_either_address_changes(role):
    import copy
    from triton.flagmega.codegen.triton.corex_readonly_layout import select_readonly_layouts
    from triton.flagmega.errors import CodegenError
    module, call = _glu_selection_case()
    argument = next(a for a in call["inputs"] if a["formal"] == role)
    reader = dict(family="elementwise", inputs=[dict(formal="source", buffers=argument["buffers"])])
    before = copy.deepcopy(call)
    with pytest.raises(CodegenError, match="consumer without its device-address contract"):
        select_readonly_layouts(module, [call, reader])
    assert call == before


@pytest.mark.parametrize("typed", (False, True))
@pytest.mark.parametrize("owners", (2, 16))
def test_owner_contiguous_k_walk_uses_exact_packed_range(typed, owners):
    from triton.flagmega.codegen.triton.corex_readonly_layout import select_readonly_layouts
    module, call = _selection_case(typed=typed, owners=owners)
    contract = select_readonly_layouts(module, [call])
    assert len(contract["buffers"]) == 1
    record = contract["buffers"][0]
    assert record["shape"] == [32, owners, 2048, 2]
    assert record["nbytes"] == owners * 128 * 512 * 2
    sme = call["corex_flat_sme_down"]
    assert sme["weight_k_group_stride"] == 2048
    assert "(shard_index) * 65536" in sme["weight"]
    assert "tl.pointer_type(tl.bfloat16)" in sme["weight"]


@pytest.mark.parametrize("change", ("storage", "coordinates", "dtype", "owners", "stride", "k_split", "sme"))
def test_unproven_geometry_never_changes_the_weight_pool(change):
    from triton.flagmega.codegen.triton.corex_readonly_layout import select_readonly_layouts
    module, call = _selection_case()
    abi = call["inputs"][0]["buffers"][0]["abi"]
    if change == "storage":
        abi["storage_kind"] = "compact_per_owner"
    elif change == "coordinates":
        abi["coordinate_space"] = "local"
    elif change == "dtype":
        abi["scalar_dtype"] = "float16"
    elif change == "owners":
        abi["owner_count"] = 8
    elif change == "stride":
        call["corex_flat_sme_down"]["weight_k_group_stride"] += 128
    elif change == "k_split":
        abi["local_capacity_shape"] = (abi["local_capacity_shape"][0] // 2,
                                       *abi["local_capacity_shape"][1:])
    else:
        del call["corex_flat_sme_down"]
    import copy
    before = copy.deepcopy(call)
    assert select_readonly_layouts(module, [call])["buffers"] == []
    assert call == before


def test_all_canonical_consumers_are_proved_before_rewriting_addresses():
    from triton.flagmega.codegen.triton.corex_readonly_layout import select_readonly_layouts
    from triton.flagmega.errors import CodegenError
    module, call = _selection_case()
    reader = {"family": "elementwise", "inputs": [
        {"formal": "source", "buffers": call["inputs"][0]["buffers"]}]}
    with pytest.raises(CodegenError, match="consumer without its device-address contract"):
        select_readonly_layouts(module, [call, reader])
    assert call["corex_flat_sme_down"]["weight"] == "canonical"
    assert call["corex_flat_sme_down"]["weight_k_group_stride"] == 32768


@pytest.mark.parametrize("geometry", ("cyclic", "subset", "permuted", "external"))
def test_noncanonical_owner_maps_and_open_weight_roots_are_not_selected(geometry):
    from triton.flagmega.codegen.triton.corex_readonly_layout import select_readonly_layouts
    mesh = fm.Placement((4, 4), "yx", "bb")
    options = {"placement": mesh}
    if geometry == "cyclic":
        options["cyclic"] = True
    elif geometry == "subset":
        options["split_axes"] = (1,)
    elif geometry == "permuted":
        options["split_axes"] = (1, 0)
    else:
        options["external"] = True
    module, call = _selection_case(**options)
    assert select_readonly_layouts(module, [call])["buffers"] == []
    assert call["corex_flat_sme_down"]["weight"] == "canonical"


@pytest.mark.parametrize("device", ("cpu", "cuda"))
def test_byte_permutation_validates_all_records_before_mutating(device):
    torch = pytest.importorskip("torch")
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("requires CUDA")
    from triton.flagmega.codegen.readonly_layout import READONLY_DEVICE_LAYOUT_SCHEMA, apply_readonly_layout
    from triton.flagmega.errors import ArtifactError
    module = _module()
    record = _record(module)
    pool = (torch.arange(record["offset"] + record["nbytes"] + 17, device=device) % 251).to(torch.uint8)
    before = pool.clone()
    contract = {"schema": READONLY_DEVICE_LAYOUT_SCHEMA, "buffers": [record]}
    for damaged in ({**record, "nbytes": record["nbytes"] - 2},
                    {**record, "shape": [2, 4, 48, 2]},
                    {**record, "axes": [0, 1, 2, 3]}):
        with pytest.raises(ArtifactError, match="canonical buffer byte range"):
            apply_readonly_layout(pool, {**contract, "buffers": [record, damaged]}, module)
        assert torch.equal(pool, before)
    apply_readonly_layout(pool, contract, module)
    expected = before.narrow(0, record["offset"], record["nbytes"]).view(4, 2, 48, 2)
    assert torch.equal(pool.narrow(0, record["offset"], record["nbytes"]),
                       expected.permute(1, 0, 2, 3).contiguous().reshape(-1))
    assert torch.equal(pool[:record["offset"]], before[:record["offset"]])
    assert torch.equal(pool[record["offset"] + record["nbytes"]:], before[record["offset"] + record["nbytes"]:])


def test_runtime_applies_layout_once_after_loading_canonical_source(tmp_path):
    torch = pytest.importorskip("torch")
    from safetensors.torch import save_file
    from triton.flagmega.artifacts import pack_rdata, verify_rdata
    from triton.flagmega.importer import DirectoryCheckpoint
    from triton.flagmega.runtime.module import GeneratedTirCallGraphModule
    from triton.flagmega.runtime.source_rdata import load_source_rdata

    root = tmp_path / "checkpoint"
    root.mkdir()
    (root / "config.json").write_text("{}")
    # Distinct two-byte elements expose swapped owners, lanes and BF16 bytes.
    weight = torch.arange(4 * 6 * 16, dtype=torch.int16).reshape(4, 6, 16)
    save_file({"w": weight.view(torch.bfloat16)}, root / "model.safetensors")
    module = _module()
    assets = tmp_path / "artifact" / "assets"
    index = pack_rdata(module, DirectoryCheckpoint(root), assets)
    canonical = load_source_rdata(assets, index, module, chip=0, device="cpu")
    record = _record(module)
    expected = canonical.clone()
    expected.narrow(0, record["offset"], record["nbytes"]).copy_(
        canonical.narrow(0, record["offset"], record["nbytes"])
        .view(*record["shape"]).permute(*record["axes"]).contiguous().reshape(-1))
    runtime = SimpleNamespace(
        artifact=assets.parent, ir_module=module, manifest={"rdata": index},
        codegen={"triton_readonly_device_layout": {
            "schema": "flagmega.readonly-device-layout/v1", "buffers": [record]}})
    # Reloading constructs a fresh canonical pool; no cached pool is permuted twice.
    for _ in range(2):
        actual = GeneratedTirCallGraphModule._load_rdata(runtime, torch, "cpu", index["segment_nbytes"])
        assert torch.equal(actual, expected)
    assert torch.equal(load_source_rdata(assets, index, module, chip=0, device="cpu"), canonical)
    verify_rdata(assets, module=module)
