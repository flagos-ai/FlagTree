# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Keep completed route rounding at the result store, never before owner sums."""

from dataclasses import replace

import pytest
import torch

from triton.flagmega import ir as fm
from triton.flagmega.errors import IRSchemaError
from triton.flagmega.evaluator import DictWeightResolver, TorchEvaluator
from triton.flagmega.ir.ops.nn.sparse_experts_combine import SparseExpertsWeightedSum
from triton.flagmega.passes.tir.lower_sparse_experts import lower_sparse_experts


def graph(lanes, distributed, *, partial=False, round_weighted=False):
    width = 128
    scalar = fm.DType.FLOAT32
    dtype = fm.vector_type(scalar, lanes) if lanes else scalar
    result_dtype = fm.vector_type("bfloat16", lanes) if lanes else fm.DType.BFLOAT16
    lane_count = dtype.lane_count if lanes else 1
    values = fm.tensor_type(dtype, (2, 5, width // lane_count))
    coefficients = fm.tensor_type("float32", (2, 5))
    b = fm.SBP.broadcast()
    placement = fm.Placement((2, 2), "yx", "bb")
    if distributed:
        values = fm.DistributedType(
            values, (b, fm.SBP.split_block_cyclic((0,), 2) if partial else b, fm.SBP.split_contiguous((1,))), placement
        )
        coefficients = fm.DistributedType(coefficients, values.axis_policies[:2], placement)

    class Graph(fm.Module):
        def forward(self):
            v = self.input("projections", values, id="projections")
            c = self.input("coefficients", coefficients, id="coefficients")
            y = fm.F.nn.sparse_experts_combine(
                v, c, output_dtype=result_dtype, round_weighted_output=round_weighted, name="combined"
            )
            self.function("main", (v, c), (y,))

    return Graph(dialect="tir", stage="distributed_tir", entry="main").build()


@pytest.mark.parametrize("lanes", [(), (8,), (2, 4)])
@pytest.mark.parametrize("distributed", [False, True])
@pytest.mark.parametrize("round_weighted", [False, True])
def test_complete_combine_stores_requested_dtype_without_adapter(lanes, distributed, round_weighted, tmp_path):
    module = graph(lanes, distributed, round_weighted=round_weighted)
    lowered = lower_sparse_experts(module)
    assert not any(n.op in {"tensors.cast", "ntt.vectorized_cast"} for n in lowered.nodes)
    owner = lowered.node_map["combined"]
    assert owner.op == "nn.sparse_experts_weighted_sum" and owner.attrs["cast_output"] is True
    assert owner.type == module.node_map["combined"].type
    assert lower_sparse_experts(lowered) == lowered
    assert fm.load_module(fm.emit_module(lowered, tmp_path / "roundtrip.py")) == lowered
    generator = torch.Generator().manual_seed(17)
    v = torch.randint(-127, 128, (2, 5, 128), generator=generator).float() / 128
    c = torch.randint(-127, 128, (2, 5), generator=generator).float() / 128
    # Physical lane suffix remains in scalar order; independent reference
    # keeps the declared per-route BF16 boundary and ordered FP32 accumulation.
    wanted = torch.zeros((2, 128), dtype=torch.float32)
    for route in range(5):
        term = v[:, route] * c[:, route, None]
        if round_weighted:
            term = term.bfloat16().float()
        wanted += term
    shape = (2, 5, 128 // (8 if lanes else 1), *lanes)
    result_shape = (2, 128 // (8 if lanes else 1), *lanes)
    evaluator = TorchEvaluator(DictWeightResolver({}))
    got = evaluator.run(lowered, {"projections": v.reshape(shape), "coefficients": c})[0]
    assert torch.equal(got, wanted.bfloat16().reshape(result_shape))


@pytest.mark.parametrize("lanes", [(), (8,), (2, 4)])
def test_cross_owner_sum_retains_fp32_until_materialized(lanes):
    module = graph(lanes, True, partial=True)
    lowered = lower_sparse_experts(module)
    partial = lowered.node_map["combined.weighted_sum"]
    assert partial.type.partial is not None
    dtype = partial.type.tensor.dtype
    assert getattr(dtype, "elem_type", dtype) is fm.DType.FLOAT32
    assert not partial.attrs["cast_output"]
    assert lowered.node_map["combined"].inputs == ("combined.owner_sum",)
    with pytest.raises(IRSchemaError, match="owner reduction"):
        SparseExpertsWeightedSum.prepare(
            tuple(module.node_map[key] for key in partial.inputs), {**partial.attrs, "cast_output": True}
        )


def test_legacy_weighted_sum_keeps_fp32_default():
    module = graph((2, 4), False)
    node = module.node_map["combined"]
    call = SparseExpertsWeightedSum.prepare(tuple(module.node_map[key] for key in node.inputs), node.attrs)
    assert call.result_type.dtype == fm.vector_type("float32", (2, 4))


def test_complete_combine_cost_has_no_separate_cast_materialization():
    from triton.flagmega.ir.ops.nn.sparse_experts_combine import SparseExpertsCombine

    module = graph((8,), False)
    node = module.node_map["combined"]
    inputs = tuple(module.node_map[key] for key in node.inputs)
    factors = SparseExpertsCombine.cost_factors(inputs, node.attrs, node.type)
    assert factors.block_local_memory_load_bytes == 2 * 5 * 128 * 4 + 2 * 5 * 4
    assert factors.block_local_memory_store_bytes == 2 * 128 * 2


def test_pattern_distinguishes_output_rounding_contract():
    from triton.flagmega import pattern_match as pm

    module = lower_sparse_experts(graph((2, 4), False))
    node = module.node_map["combined"]
    assert pm.try_match_root(node, pm.F.nn.is_sparse_experts_weighted_sum(cast_output=True), module) is not None
    assert pm.try_match_root(node, pm.F.nn.is_sparse_experts_weighted_sum(cast_output=False), module) is None


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("lanes", [(), (2, 4)])
@pytest.mark.parametrize("round_weighted", [False, True])
def test_device_complete_route_sum_rounds_once_and_preserves_guards(tmp_path, lanes, round_weighted):
    from triton.flagmega.artifacts import write_artifact
    from triton.flagmega.compiler import Compiler
    from triton.flagmega.runtime import load
    from triton.flagmega.runtime.module import GeneratedTirCallGraphModule as Runtime
    from triton.flagmega.rules.neutral._utility import make_node

    module = graph(lanes, True, round_weighted=round_weighted)
    result = module.node_map["combined"]
    published = make_node("distributed.boxing", "published", (result,), {"new_type": result.type.tensor}, {})
    placement = result.type.placement
    module = replace(
        module,
        nodes=(*module.nodes, published),
        functions=(replace(module.functions[0], outputs=(published.id,)),),
        dialect="nn",
        stage="frozen_constants",
        metadata={"auto_distribution": {"placement": placement.to_data()}},
    )
    compiled = Compiler().compile(module).module
    definitions = {d.name: d.dispatch for d in compiled.kernel_definitions}
    assert not any(d.semantic_op in {"tensors.cast", "ntt.vectorized_cast"} for d in definitions.values())
    sums = [d for d in definitions.values() if d.semantic_op == "nn.sparse_experts_weighted_sum"]
    assert len(sums) == 1 and sums[0].semantic_attrs["cast_output"] is True
    artifact = write_artifact(compiled, tmp_path / "artifact", target="nvidia-sm90", emit_executable=True)
    runtime = load(artifact, device="cuda:0")
    try:
        generator = torch.Generator().manual_seed(1793)
        v = torch.randint(-127, 128, (2, 5, 128), generator=generator).float() / 128
        c = torch.randint(-127, 128, (2, 5), generator=generator).float() / 128
        v[:, :3] = 0
        v[:, 0] = 1
        v[:, 1:3] = 1 / 256
        c[:, :3] = 1
        shape = (2, 5, 128 // (8 if lanes else 1), *lanes)
        result_shape = (2, 128 // (8 if lanes else 1), *lanes)
        guards = []
        inputs = {}
        for name, value in (("projections", v.reshape(shape)), ("coefficients", c)):
            backing = torch.full((value.numel() + 128,), -123.0, dtype=value.dtype, device="cuda")
            inputs[name] = backing[64:-64].reshape(value.shape)
            inputs[name].copy_(value)
            guards.extend((backing[:64], backing[-64:]))
        backing = torch.full((256 + 128,), -123.0, dtype=torch.bfloat16, device="cuda")
        output = backing[64:-64].reshape(result_shape)
        guards.extend((backing[:64], backing[-64:]))
        args = tuple(
            output if spec["role"] == "result" else inputs[spec["value"]] for spec in runtime.external_arguments
        )
        Runtime.prepare(runtime, *args)
        assert runtime.resource_report["spill_bytes"] == 0
        for mutation in range(3):
            inputs["projections"].add_(1 / 128)
            value = inputs["projections"].cpu().reshape(2, 5, 128)
            wanted = torch.zeros((2, 128), dtype=torch.float32)
            for route in range(5):
                term = value[:, route] * c[:, route, None]
                if round_weighted:
                    term = term.bfloat16().float()
                wanted += term
            output.fill_(float("nan"))
            Runtime.run_into(runtime, *args)
            torch.cuda.synchronize()
            assert torch.equal(output.cpu(), wanted.bfloat16().reshape(result_shape)), mutation
            assert all(bool((guard == -123).all()) for guard in guards)
    finally:
        runtime.close()
