# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT

from dataclasses import replace

import pytest
import torch

from triton.flagmega import ir as fm
from triton.flagmega.evaluator import DictWeightResolver, TorchEvaluator
from triton.flagmega.rules.ntt.vectorize import VectorizeRMSNorm


def _norm_module(make_op_module, shape, *, dtype="bfloat16", epsilon=1e-6, weight_bias=1.0):
    return make_op_module(
        "nn.rms_norm",
        (fm.tensor_type(dtype, shape), fm.tensor_type(dtype, (shape[-1],))),
        attrs={"epsilon": epsilon, "weight_bias": weight_bias},
    )


@pytest.mark.parametrize("dtype,lane", [("bfloat16", 8), ("float32", 4)])
def test_norm_rule_emits_reduction_axis_candidate(make_op_module, dtype, lane):
    module = _norm_module(make_op_module, (3, 10), dtype=dtype)
    candidates = VectorizeRMSNorm().candidates(module.node_map["root"], module)
    assert len(candidates) == 1
    candidate = candidates[0]
    assert candidate.id == "vectorization.norm.reduction_axis"
    assert candidate.parameters["value_axes"] == [1]
    assert candidate.parameters["weight_axes"] == [0]
    assert candidate.parameters["lanes"] == [lane]
    assert candidate.parameters["logical_extent"] == 10
    assert candidate.facts["value_padding"] == [0, (-10) % lane]
    assert candidate.facts["weight_padding"] == [(-10) % lane]


@pytest.mark.parametrize("shape", [(3, 16), (3, 10)])
@pytest.mark.parametrize("weight_bias", [0.0, 1.0])
def test_norm_rule_rewrite_preserves_reduction_semantics_and_padding(
    make_op_module, rewrite_candidate, shape, weight_bias,
):
    module = _norm_module(make_op_module, shape, weight_bias=weight_bias)
    candidate, result, rewritten = rewrite_candidate(
        module, VectorizeRMSNorm(), "vectorization.norm.reduction_axis",
    )
    compute = next(node for node in result.prefix_nodes if node.op == "nn.vectorized_rms_norm")
    assert compute.attrs["logical_extent"] == shape[-1]
    assert compute.attrs["weight_bias"] == weight_bias
    assert result.replacement.metadata["vectorization_candidate"] == candidate.id

    value = torch.randn(shape, dtype=torch.bfloat16)
    weight = torch.randn(shape[-1], dtype=torch.bfloat16)
    evaluator = TorchEvaluator(DictWeightResolver({}))
    torch.testing.assert_close(
        evaluator.run(rewritten, {"arg0": value, "arg1": weight})[0],
        evaluator.run(module, {"arg0": value, "arg1": weight})[0],
    )


def test_norm_rule_rejects_dynamic_extent_small_lane_and_effect(make_op_module):
    n = fm.dim("n", minimum=1, maximum=64)
    dynamic = _norm_module(make_op_module, (2, n))
    assert VectorizeRMSNorm().candidates(dynamic.node_map["root"], dynamic) == ()

    module = _norm_module(make_op_module, (2, 16))
    assert VectorizeRMSNorm(lane_bytes=2).candidates(module.node_map["root"], module) == ()
    effectful = replace(module.node_map["root"], effect=fm.effect("write", "state"))
    assert VectorizeRMSNorm().candidates(
        effectful, replace(module, nodes=(*module.nodes[:-1], effectful)),
    ) == ()
