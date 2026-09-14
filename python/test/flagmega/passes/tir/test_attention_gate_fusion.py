# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Attention output gating must preserve types, owners and private boundaries."""

from dataclasses import replace

import pytest
import torch

from triton.flagmega import ir as fm
from triton.flagmega.codegen.triton.fusion import fusion_rules
from triton.flagmega.compiler import Compiler
from triton.flagmega.evaluator import DictWeightResolver, TorchEvaluator
from triton.flagmega.passes.tir.fuse_distributed_ops import fuse_distributed_ops


def graph(*, lanes=1, shared=False, swapped=False, distributed=False, dimension=16):
    class Graph(fm.Module):
        def __init__(self):
            super().__init__(dialect="high_level", stage="frozen_constants", entry="main")

        def forward(self):
            dtype = "bfloat16" if lanes == 1 else fm.vector_type("bfloat16", (lanes,))
            result_type = fm.tensor_type(dtype, (2, 3, dimension // lanes))
            stats_type = fm.tensor_type("float32", (2, 3, 1))
            acc_type = fm.tensor_type("float32", (2, 3, dimension))
            output_type = result_type
            if distributed:
                mesh = fm.Placement((2, 3), "yx", "bb")
                policies = (fm.SBP.broadcast(), fm.SBP.split_contiguous((1,), 1), fm.SBP.broadcast())
                result_type = fm.DistributedType(result_type, policies, mesh)
                output_type = fm.DistributedType(result_type.tensor, (*policies[:2], fm.SBP.split_contiguous((0,), dimension // lanes // 2)), mesh)
                max_type = fm.DistributedType(stats_type, policies, mesh, fm.SBP.partial((0,), fm.ReduceOp.MAX))
                sum_type = fm.DistributedType(stats_type, policies, mesh, fm.SBP.partial((0,), fm.ReduceOp.SUM))
                acc_type = fm.DistributedType(acc_type, policies, mesh, fm.SBP.partial((0,), fm.ReduceOp.SUM))
            else:
                max_type = sum_type = stats_type
            maximum = self.input("maximum", max_type)
            total = self.input("total", sum_type)
            accumulator = self.input("accumulator", acc_type)
            gate = self.input("gate", output_type)
            combined = fm.F.ntt.paged_attention_combine(
                maximum, total, accumulator, layout=("seq", "head", "dim"), hidden_size=3 * dimension,
                output_data_type=dtype, output_type=result_type, split_hierarchy_axis=0, split_count=2,
                name="combined")
            value = fm.F.distributed.sharded_view(combined, output_type) if distributed else combined
            sigmoid = (fm.F.math.sigmoid(gate) if lanes == 1 else
                       fm.F.math.vectorized_unary(gate, unary_op="sigmoid"))
            operands = (sigmoid, value) if swapped else (value, sigmoid)
            result = (fm.F.math.mul(*operands, name="result") if lanes == 1 else
                      fm.F.math.vectorized_binary(*operands, binary_op="mul", name="result"))
            self.function("main", (maximum, total, accumulator, gate), (result, combined) if shared else (result,))

    return Graph().build()


@pytest.mark.parametrize("lanes", [1, 8])
@pytest.mark.parametrize("swapped", [False, True])
@pytest.mark.parametrize("distributed", [False, True])
def test_fuses_combine_sigmoid_mul_with_exact_output_contract(lanes, swapped, distributed, tmp_path):
    original = graph(lanes=lanes, swapped=swapped, distributed=distributed)
    result = fuse_distributed_ops(original, fusion_rules=fusion_rules())
    assert result.node_map["result"].op == "ntt.paged_attention_gated_combine"
    assert result.node_map["result"].type == original.node_map["result"].type
    assert not any(node.op == "ntt.paged_attention_combine" for node in result.nodes)
    assert result.node_map["result"].inputs == original.functions[0].parameters
    assert fuse_distributed_ops(result, fusion_rules=fusion_rules()) == result
    path = tmp_path / "fused.py"
    fm.emit_module(result, path)
    assert fm.load_module(path).semantic_hash == result.semantic_hash
    if distributed:
        return
    torch.manual_seed(17)
    values = {"maximum": torch.zeros(2, 3, 1), "total": torch.rand(2, 3, 1) + .1,
              "accumulator": torch.randn(2, 3, 16),
              "gate": torch.randn(2, 3, 16).to(torch.bfloat16)}
    if lanes != 1:
        values["gate"] = values["gate"].reshape(2, 3, 16 // lanes, lanes)
    evaluator = TorchEvaluator(DictWeightResolver({}))
    torch.testing.assert_close(evaluator.run(result, values)[0], evaluator.run(original, values)[0], rtol=0, atol=0)


def test_exported_attention_result_is_not_removed_or_recomputed():
    original = graph(shared=True)
    result = fuse_distributed_ops(original, fusion_rules=fusion_rules())
    assert result.node_map["combined"].op == "ntt.paged_attention_combine"
    assert not any(node.op == "ntt.paged_attention_gated_combine" for node in result.nodes)


def test_non_sigmoid_preops_are_not_reinterpreted_as_sigmoid():
    original = graph()
    nodes = tuple(replace(node, op="math.silu") if node.op == "math.sigmoid" else node for node in original.nodes)
    result = fuse_distributed_ops(fm.verify_module(replace(original, nodes=nodes)), fusion_rules=fusion_rules())
    assert not any(node.op == "ntt.paged_attention_gated_combine" for node in result.nodes)


def test_shared_attention_consumer_is_preserved():
    original = graph()
    source = original.node_map["combined"]
    shared = fm.Node("other_consumer", "math.silu", (source.id,), source.type)
    function = replace(original.functions[0], outputs=(*original.functions[0].outputs, shared.id))
    original = fm.verify_module(replace(original, nodes=(*original.nodes, shared), functions=(function,)))
    result = fuse_distributed_ops(original, fusion_rules=fusion_rules())
    assert not any(node.op == "ntt.paged_attention_gated_combine" for node in result.nodes)
    assert result.node_map[source.id] == source


def test_shared_gate_input_survives():
    original = graph()
    gate = original.node_map[original.functions[0].parameters[3]]
    shared = fm.Node("gate_user", "math.silu", (gate.id,), gate.type)
    function = replace(original.functions[0], outputs=(*original.functions[0].outputs, shared.id))
    original = fm.verify_module(replace(original, nodes=(*original.nodes, shared), functions=(function,)))
    result = fuse_distributed_ops(original, fusion_rules=fusion_rules())
    assert result.node_map["result"].op == "ntt.paged_attention_gated_combine"
    assert result.node_map[shared.id] == shared


def test_formation_invalidates_only_replaced_and_removed_selections():
    from triton.flagmega.passes.tir.fuse_attention_gate import fuse_attention_gate
    original = graph(distributed=True)
    points = tuple(fm.SelectionPoint(f"distribution.{node.id}", "distribution",
                   (fm.Candidate(f"chosen.{node.id}"),), f"chosen.{node.id}", node.id) for node in original.nodes)
    records = tuple(fm.SelectionRecord(point.id, point.default_candidate, "default-policy", "test/v1") for point in points)
    original = fm.verify_module(replace(original, selection_points=points, selections=records))
    result = fuse_attention_gate(original)
    assert {point.owner for point in result.selection_points} == set(original.functions[0].parameters)
    assert result.selections == tuple(record for point, record in zip(points, records)
                                     if point.owner in original.functions[0].parameters)
    assert result.node_map["result"].op == "ntt.paged_attention_gated_combine"


@pytest.mark.parametrize("lanes", [1, 8])
@pytest.mark.parametrize("swapped", [False, True])
@pytest.mark.parametrize("stop", ["AutoDistributedPass", "freeze-constants"])
def test_standard_post_distribution_pipeline_fuses_before_local_passes(lanes, swapped, stop):
    original = replace(graph(lanes=lanes, swapped=swapped, distributed=True), stage="distributed")
    result = Compiler().compile(original, stop_after=stop).module
    assert result.node_map["result"].op == "ntt.paged_attention_gated_combine"
    assert result.node_map["result"].type == original.node_map["result"].type
    assert not any(node.op == "ntt.paged_attention_combine" for node in result.nodes)


@pytest.mark.parametrize("stage", ["vector_contracts_lowered", "frozen_constants"])
@pytest.mark.parametrize("shared", [False, True])
def test_old_checkpoints_resume_attention_gate_fusion(stage, shared):
    original = replace(graph(lanes=8, distributed=True, shared=shared), stage=stage)
    result = Compiler().compile(original, stop_after="fuse-distributed-ops").module
    assert any(node.op == "ntt.paged_attention_gated_combine" for node in result.nodes) != shared
    if shared:
        assert result.node_map["combined"].type == original.node_map["combined"].type


@pytest.mark.parametrize("use_output_stage", [False, True])
def test_post_distribution_fusion_checkpoint_is_resumable(tmp_path, use_output_stage):
    original = replace(graph(lanes=8, distributed=True), stage="distributed")
    compiler = Compiler()
    checkpoint = compiler.compile(original, stop_after="AutoDistributedPass").module
    assert checkpoint.node_map["result"].op == "ntt.paged_attention_gated_combine"
    source = fm.load_module(fm.emit_module(checkpoint, tmp_path / "input.py"))
    result = compiler.compile(source, stop_after=source.stage if use_output_stage else "AutoDistributedPass")
    assert result.module == source
    assert result.reports == ()
    resumed = compiler.compile(source, stop_after="fuse-distributed-ops").module
    uninterrupted = compiler.compile(original, stop_after="fuse-distributed-ops").module
    assert resumed.semantic_hash == uninterrupted.semantic_hash
