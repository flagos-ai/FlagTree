# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Reshard chip-shared owner tables without canonical staging copies."""

from itertools import product
from math import prod
from types import SimpleNamespace

import pytest
from triton.flagmega import ir as fm
from triton.flagmega.artifacts import write_artifact
from triton.flagmega.compiler import Compiler
from triton.flagmega.errors import CodegenError
from triton.flagmega.options import CompileOptions

from .test_local_narrowing import local_abi
from .test_partial_reduce_local_abi import _prepare


@pytest.mark.parametrize("lanes", [1, 4])
@pytest.mark.parametrize("cyclic", [False, True])
def test_chip_gather_inverts_owner_and_local_coordinates_with_padded_stride(lanes, cyclic):
    mesh = fm.Placement((2, 4), "yx", "bb")
    dtype = "bfloat16" if lanes == 1 else fm.vector_type("bfloat16", (lanes,))
    tensor = fm.tensor_type(dtype, (2, 20))
    row = fm.SBP.split_block_cyclic((1,), 3) if cyclic else fm.SBP.split_contiguous((1,), 6)
    source = fm.DistributedType(tensor, (fm.SBP.split_contiguous((0,), 1), row), mesh)
    target = fm.DistributedType(tensor, (source.axis_policies[0], fm.SBP.broadcast()), mesh)
    source_abi = local_abi(source)
    stride = prod(source_abi["local_capacity_shape"]) * lanes + 16
    source_abi.update(
        storage_kind="compact_per_owner",
        memory_sharing_scope="chip",
        owner_count=8,
        placement_owner_count=8,
        owner_index_expression="shard_index",
        component_stride_scalar_elements=stride,
        pool_byte_offset=32,
    )
    leaf = _prepare((source_abi,), (local_abi(target),), ("gather_reduce_scatter",))["leaves"][0]
    assert leaf["mode"] == "chip_gather"
    for owner in product(range(2), range(4)):
        dst = fm.local_shard_descriptor(target, owner)
        for column in range(20):
            global_row = dst.axes[0].map_local_to_global(0).fixed_value
            hits = []
            for source_owner in product(range(2), range(4)):
                src = fm.local_shard_descriptor(source, source_owner)
                if src.axes[0].map_local_to_global(0).fixed_value != global_row:
                    continue
                for local in range(src.active_shape[1].fixed_value):
                    if src.axes[1].map_local_to_global(local).fixed_value == column:
                        hits.append((source_owner[0] * 4 + source_owner[1], local))
            assert len(hits) == 1
            source_owner, local = hits[0]
            for lane in range(lanes):
                env = {
                    "boxing_offsets": column * lanes + lane,
                    "source_0": 0,
                    "shard_y": owner[0],
                    "shard_x": owner[1],
                    "shard_coord_0": owner[0],
                    "shard_coord_1": owner[1],
                    "tl": SimpleNamespace(minimum=min, maximum=max),
                }
                actual = eval(leaf["source"], {"__builtins__": {}}, env) + eval(
                    leaf["source_offset"], {"__builtins__": {}}, env
                )
                assert actual == 16 + source_owner * stride + local * lanes + lane


def graph():
    mesh = fm.Placement((2, 4), "yx", "bb")
    tensor = fm.tensor_type(fm.vector_type("bfloat16", (4,)), (1, 2, 16))
    b = fm.SBP.broadcast()
    source = fm.DistributedType(tensor, (b, fm.SBP.split_contiguous((0,), 1), fm.SBP.split_contiguous((1,), 4)), mesh)
    target = fm.DistributedType(tensor, (b, source.axis_policies[1], b), mesh)

    class Graph(fm.Module):
        def forward(self):
            x = self.input("x", source, id="x")
            produced = fm.F.math.add(x, x, name="produced")
            gathered = fm.F.distributed.boxing(produced, target, name="gathered")
            result = fm.F.math.reduce_sum(gathered, axes=(2,), keep_dims=False, name="summed")
            self.function("main", (x,), (result,))

    return Graph(
        dialect="distributed",
        stage="frozen_constants",
        entry="main",
        metadata={"auto_distribution": {"placement": mesh.to_data()}},
    ).build(), mesh


def compile_graph():
    value, mesh = graph()
    return Compiler(CompileOptions(placement=mesh)).compile(value).module


@pytest.mark.parametrize("cause", ["private", "scope", "stride", "map", "backing"])
def test_chip_gather_rejects_unproved_shared_component_table(cause):
    module, _ = graph()
    source = local_abi(module.node_map["x"].type)
    source.update(
        storage_kind="compact_per_owner",
        memory_sharing_scope="chip",
        owner_count=8,
        placement_owner_count=8,
        owner_index_expression="shard_index",
        component_stride_scalar_elements=16,
    )
    if cause == "private":
        source["storage_kind"] = "compact_local"
    elif cause == "scope":
        source["memory_sharing_scope"] = "block"
    elif cause == "stride":
        source["pool_scope_stride_bytes"] = 2048
    elif cause == "map":
        source["owner_count"] = 1
    else:
        source["distributed_backing_type"] = module.node_map["x"].type.to_data()
    with pytest.raises(CodegenError):
        _prepare((source,), (local_abi(module.node_map["gathered"].type),), ("gather_reduce_scatter",))


def test_compact_owner_reshard_compiles_with_chip_read_effect_and_publication(tmp_path):
    from triton.flagmega.ir import verify_buffer_plan

    module = compile_graph()
    plan = verify_buffer_plan(module)
    from triton.flagmega.passes.tir.bufferize.synchronization import (
        _node_accesses,
        _resolved_scope,
        plan_memory_synchronization,
    )

    accesses = _node_accesses(module, plan, "main", module.node_map["gathered"], dict(plan.function_map["main"].values))
    assert any(a.mode == "read" and _resolved_scope(a).value == "chip" for a in accesses)
    assert any(e.before == "gathered" and e.scope == "grid" for e in plan_memory_synchronization(module, plan).events)
    write_artifact(module, tmp_path / "artifact", target="nvidia-sm90", emit_executable=True)


def test_device_compact_owner_reshard_preserves_all_lanes(tmp_path):
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (9, 0):
        pytest.skip("requires SM90")
    from triton.flagmega.runtime import load
    from triton.flagmega.runtime.module import GeneratedTirCallGraphModule as Run

    rt = load(
        write_artifact(compile_graph(), tmp_path / "artifact", target="nvidia-sm90", emit_executable=True),
        device="cuda:0",
    )
    x = torch.empty((1, 2, 16, 4), dtype=torch.bfloat16, device="cuda")
    storage = torch.full((40,), 128.0, dtype=torch.bfloat16, device="cuda")
    out = storage[16:-16].reshape(1, 2, 4)
    values = {"x": x, "summed": out}
    actuals = tuple(values[s["value"]] for s in rt.external_arguments)
    try:
        Run.prepare(rt, *actuals)
        for mutation in range(4):
            x.copy_((torch.arange(128, device="cuda").reshape_as(x).float() / 8 + mutation / 16).bfloat16())
            out.fill_(float("nan"))
            Run.run_into(rt, *actuals)
            torch.cuda.synchronize()
            torch.testing.assert_close(out, (x.double() * 2).sum(2).bfloat16(), rtol=0, atol=0)
            assert bool((storage[:16] == 128).all() & (storage[-16:] == 128).all())
        assert rt.resource_report["spill_bytes"] == 0
    finally:
        rt.close()
