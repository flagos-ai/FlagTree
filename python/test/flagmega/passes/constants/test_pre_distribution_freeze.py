# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: MIT

from dataclasses import replace

import pytest
import torch
from triton.flagmega import ir as fm
from triton.flagmega.compiler import Compiler
from triton.flagmega.evaluator import (
    DictWeightResolver,
    iter_numpy_materialized_constant_assets,
    materialize_constant_assets,
)
from triton.flagmega.importer import MemoryCheckpoint, TensorInfo
from triton.flagmega.passes.auto_distributed import AutoDistributedPass
from triton.flagmega.passes.constants import ConstnessAnalysis, freeze_constant_islands, thaw_constant_islands
from triton.flagmega.stages import next_stage
from triton.flagmega.targets import NvidiaSm90Target


def _islands(*, depth=4, weight=True):
    class Graph(fm.Module):
        def forward(self):
            value_type = fm.tensor_type("float32", (8, 16))
            x = self.input("x", value_type, id="x")
            c = (self.weight("c", value_type, source="memory", key="c", id="c") if weight
                 else fm.F.builtin.splat_const(value_type, 2.0, name="c"))
            for index in range(depth):
                left = fm.F.tensors.slice(c, starts=(0,), ends=(8,), axes=(1,), name=f"left{index}")
                right = fm.F.tensors.slice(c, starts=(8,), ends=(16,), axes=(1,), name=f"right{index}")
                c = fm.F.tensors.concat(right, left, axis=1, name=f"concat{index}")
            output = fm.F.math.add(x, c, name="output")
            self.function("main", (x,), (output,))

    return Graph(dialect="ntt", stage="unused_functions_removed", entry="main").build()


@pytest.mark.parametrize("weight", [False, True])
def test_pre_distribution_freeze_is_default_and_resumable(tmp_path, weight):
    compiler = Compiler()
    source = _islands(weight=weight)
    frozen = compiler.compile(source, stop_after="pre-distribution-freeze").module
    assert frozen.stage == "distribution_constants_frozen"
    assert frozen.metadata["constant_phase"] == "frozen"
    assert len(frozen.nodes) == 3
    assert frozen.node_map["concat3"].op == "builtin.const_asset"
    assert next_stage(frozen.stage).name == "propose-distribution"
    restored = fm.load_module(fm.emit_module(frozen, tmp_path / "frozen.py"))
    assert compiler.compile(restored, stop_after="pre-distribution-freeze").module == frozen
    proposed = compiler.compile(restored, stop_after="propose-distribution").module
    assert len(proposed.nodes) == 3
    assert all("concat" not in p.id for p in proposed.selection_points)


def test_search_graph_does_not_scale_with_offline_slice_concat_depth():
    target = NvidiaSm90Target()
    counts = []
    for depth in (1, 6):
        module = Compiler().run_stage(_islands(depth=depth), "pre-distribution-freeze").module
        graph = AutoDistributedPass._build_graph(module, target)
        counts.append((len(graph.buckets), sum(len(b.candidates) for b in graph.buckets), len(graph.reshard_sites)))
        assert not any(graph.module.node_map[b.node_id].op in {"tensors.slice", "tensors.concat"}
                       for b in graph.buckets)
    assert counts[0] == counts[1]


@pytest.mark.parametrize("direct", [False, True])
def test_distribution_reopens_recipes_and_final_freeze_absorbs_adapters(direct):
    compiler = Compiler()
    source = _islands(depth=1)
    result = (compiler.run_stage(source, "auto-distributed") if direct
              else compiler.compile(source, stop_after="auto-distributed")).module
    assert result.constant_recipes
    opened = compiler.compile(result, stop_after="post-distribution-thaw").module
    assert opened.stage == "distribution_constants_open"
    assert opened.metadata["constant_phase"] == "open"
    assert not opened.constant_recipes
    for node in result.nodes:
        if node.op != "builtin.const_asset":
            assert opened.node_map[node.id] == node
    final = compiler.compile(opened, stop_after="freeze-constants").module
    constants = ConstnessAnalysis.analyze(final).constants
    assert all(n.op == "builtin.const_asset" for n in final.nodes if n.id in constants)
    assert not any(n.op in {"tensors.slice", "tensors.concat"} for n in final.nodes)
    weight = torch.arange(128, dtype=torch.float32).reshape(8, 16)
    weights = DictWeightResolver({"c": weight})
    values = materialize_constant_assets(final, weights)
    assert values
    expected = torch.cat((weight[:, 8:], weight[:, :8]), dim=1)
    for value in values.values():
        torch.testing.assert_close(value, expected)
    checkpoint = MemoryCheckpoint({}, {"c": TensorInfo("c", fm.DType.FLOAT32, (8, 16), "memory")}, {"c": weight})
    for name, value in iter_numpy_materialized_constant_assets(final, checkpoint):
        assert value.tobytes() == values[name].contiguous().numpy().tobytes()


def test_thaw_preserves_asset_physical_type_and_renamed_boundary():
    source = _islands(depth=1)
    frozen = freeze_constant_islands(source)
    tensor = source.node_map["concat0"].type
    placement = fm.Placement((2, 2), "yx", "bb")
    distributed = fm.DistributedType(tensor, (fm.SBP.broadcast(), fm.SBP.split_contiguous((0, 1))), placement)
    asset = replace(frozen.node_map["concat0"], id="renamed", type=distributed)
    frozen = fm.verify_module(replace(frozen, nodes=(asset,), functions=(fm.Function("main", (), ("renamed",)),)))
    opened = thaw_constant_islands(frozen)
    assert opened.node_map["renamed"].type == distributed
    assert opened.node_map["renamed"].op == "distributed.boxing"
    final = freeze_constant_islands(opened)
    assert final.node_map["renamed"].type == distributed
    weight = torch.arange(128, dtype=torch.float32).reshape(8, 16)
    torch.testing.assert_close(materialize_constant_assets(final, DictWeightResolver({"c": weight}))["renamed"],
                               torch.cat((weight[:, 8:], weight[:, :8]), dim=1))


def test_thaw_does_not_collide_with_main_graph_ids():
    frozen = freeze_constant_islands(_islands(depth=1))
    collision = fm.Node("left0", "builtin.var", (), fm.tensor_type("float32", (8, 8)), attrs={"name": "left0"})
    frozen = fm.verify_module(replace(frozen, nodes=(collision, *frozen.nodes)))
    opened = thaw_constant_islands(frozen)
    assert opened.node_map["left0"] == collision
    assert len({n.id for n in opened.nodes}) == len(opened.nodes)
    assert any(n.op == "tensors.slice" for n in opened.nodes)


def test_thaw_shared_outputs_preserves_internal_types_and_sharing():
    builder = fm.IRBuilder(dialect="ntt", stage="unused_functions_removed")
    tensor = fm.tensor_type("float32", (8,))
    weight = builder.weight("w", tensor, source="memory", key="w", id="w")
    square = builder.call("math.mul", (weight, weight), tensor, id="square")
    builder.function("main", (), (weight, square))
    frozen = freeze_constant_islands(builder.build(entry="main"))
    placement = fm.Placement((2, 2), "yx", "bb")
    distributed = fm.DistributedType(tensor, (fm.SBP.split_contiguous((0, 1)),), placement)
    frozen = fm.verify_module(replace(frozen, nodes=tuple(
        replace(n, type=distributed) if n.id == "w" else n for n in frozen.nodes)))
    opened = thaw_constant_islands(frozen)
    assert opened.node_map["w"].type == distributed
    left, right = opened.node_map["square"].inputs
    assert left == right != "w"
    assert opened.node_map[left].type == tensor
    assert len([n for n in opened.nodes if n.op == "builtin.weight"]) == 1


def test_post_distribution_constants_still_receive_cse_and_parameter_lifting():
    from triton.flagmega.passes.functions import function_nodes

    from python.test.flagmega.passes.functions.constant_parameters.helpers import module

    source = replace(module(nested=True), stage="unused_functions_removed")
    final = Compiler().compile(source, stop_after="freeze-constants").module
    assert not any(n.op == "tensors.cast" for n in function_nodes(final, final.function_map["worker"]))
    for index in range(2):
        actual = final.node_map[final.node_map[f"call{index}"].inputs[-1]]
        assert actual.op == "builtin.const_asset"
    assert len(final.constant_recipes) == 2


def test_pre_distribution_freeze_keeps_dynamic_expressions_in_search():
    source = _islands(depth=2)
    source = replace(source, nodes=tuple(
        replace(n, op="builtin.var", attrs={"name": "c"}) if n.id == "c" else n for n in source.nodes),
        functions=(replace(source.functions[0], parameters=("x", "c")),))
    frozen = Compiler().run_stage(source, "pre-distribution-freeze").module
    assert frozen.nodes == source.nodes
    assert not frozen.constant_recipes
    graph = AutoDistributedPass._build_graph(frozen, NvidiaSm90Target())
    assert graph.bucket_map["concat1"].executable


def test_vector_contract_lowering_preserves_offline_physical_expressions():
    from triton.flagmega.passes.auto_distributed.policy import lower_vectorization_contracts

    class Graph(fm.Module):
        def forward(self):
            weight = self.weight("w", fm.tensor_type("float32", (8, 16)), source="memory", key="w", id="w")
            packed = fm.F.tensors.pack(weight, lanes=(4,), axes=(1,), name="packed", metadata={
                "vectorization_internal": True, "vectorization_role": "pack", "vectorization_root": "retired",
            })
            concat = fm.F.tensors.concat(packed, packed, axis=0, name="concat", metadata={
                "vectorization_internal": True, "vectorization_role": "propagated-concat",
                "vectorization_root": "retired", "vectorization_semantic_id": "logical",
                "vectorization_candidate": "vectorization.propagated", "vector_axes": (1,), "vector_lanes": (4,),
                "vectorization_attrs": {"axis": 0}, "vectorized_from": "tensors.concat",
            })
            output = fm.F.tensors.unpack(concat, axes=(1,), name="logical")
            self.function("main", (), (output,))

    source = Graph(dialect="ntt", stage="add_norm_stats_lowered", entry="main").build()
    restored = thaw_constant_islands(freeze_constant_islands(source))
    lowered = fm.verify_module(lower_vectorization_contracts(restored))
    assert lowered.nodes == source.nodes
    weight = torch.arange(128, dtype=torch.float32).reshape(8, 16)
    assets = materialize_constant_assets(freeze_constant_islands(lowered), DictWeightResolver({"w": weight}))
    torch.testing.assert_close(assets["logical"], torch.cat((weight, weight), dim=0))


def test_qwen35_mixed_decoder_pre_freeze_preserves_all_live_edges():
    from triton.flagmega.importer import Qwen35MoeImporter

    from python.test.flagmega.importer.qwen3_5_moe.helpers import checkpoint

    source = Qwen35MoeImporter(checkpoint(with_values=False), fused_qkvg_projection=False).import_module()
    compiler = Compiler()
    packed = compiler.compile(source, stop_after="remove-unused-functions").module
    frozen = compiler.compile(packed, stop_after="pre-distribution-freeze").module
    assert len(frozen.nodes) < len(packed.nodes)
    constants = ConstnessAnalysis.analyze(packed).constants
    assert all(frozen.node_map[n.id] == n for n in packed.nodes if n.id not in constants)
    restored = thaw_constant_islands(frozen)
    assert {n.id: (n.op, n.inputs, n.type) for n in restored.nodes} == {
        n.id: (n.op, n.inputs, n.type) for n in packed.nodes
    }


def test_pre_distribution_freeze_executes_readonly_asset_on_gpu(tmp_path):
    from triton.flagmega.artifacts import write_artifact
    from triton.flagmega.runtime import load

    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (9, 0):
        pytest.skip("SM90 CUDA required")
    weight = torch.arange(128, dtype=torch.float32).reshape(8, 16)
    checkpoint = MemoryCheckpoint({}, {"c": TensorInfo("c", fm.DType.FLOAT32, (8, 16), "memory")}, {"c": weight})
    compiled = Compiler().compile(_islands(depth=1)).module
    artifact = write_artifact(compiled, tmp_path / "artifact", target="nvidia-sm90", checkpoint=checkpoint,
                              emit_executable=True)
    runtime = load(artifact, device="cuda:0")
    x = torch.ones(8, 16, device="cuda")
    runtime.prepare(x)
    actual = runtime.run(x)
    expected = x + torch.cat((weight[:, 8:], weight[:, :8]), dim=1).cuda()
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_canonical_inplace_consumer_masks_redundant_owners(tmp_path):
    from triton.flagmega.codegen.triton import render_triton_package
    from triton.flagmega.codegen.triton.kernel_call_renderers import prepare_kernel_calls

    compiler = Compiler()
    proposed = compiler.compile(_islands(depth=1), stop_after="propose-distribution").module
    point = next(point for point in proposed.selection_points if point.owner == "output")
    candidate = next(candidate for candidate in point.candidates
                     if candidate.parameters["return_type"].endswith(";B,S(C(1)@[1]))"))
    selected = replace(proposed, selections=tuple(
        replace(record, candidate_id=candidate.id, origin="agent", policy="test-redundant-owners/v1",
                rationale="Exercise canonical in-place writes with redundant y owners.")
        if record.point_id == point.id else record for record in proposed.selections))
    compiled = compiler.compile(selected).module
    package = render_triton_package(compiled, tmp_path / "package")
    calls = prepare_kernel_calls(package["runtime_binding"]["call_abi"]["kernel_calls"], function_name="main")
    call = next(call for call in calls if call["family"] == "elementwise")
    assert call["lhs"] == call["result"]
    # This selected B_y/S_x layout shares canonical in-place storage across y.
    assert "(shard_y == 0)" in call["active"]
