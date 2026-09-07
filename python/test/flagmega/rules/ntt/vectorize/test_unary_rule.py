# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT

import pytest
import torch

from triton.flagmega import ir as fm
from triton.flagmega.evaluator import DictWeightResolver, TorchEvaluator
from triton.flagmega.rules.ntt.vectorize import VectorizeUnary


@pytest.mark.parametrize("dtype,lane", [("bfloat16", 8), ("float32", 4)])
def test_unary_rule_enumerates_axes_and_padding(make_op_module, dtype, lane):
    module = make_op_module("math.silu", (fm.tensor_type(dtype, (5, 10)),))
    by_id = {
        candidate.id: candidate
        for candidate in VectorizeUnary().candidates(module.node_map["root"], module)
    }
    assert set(by_id) == {"vectorization.axes_0", "vectorization.last_axis", "vectorization.axes_0_1"}
    assert by_id["vectorization.last_axis"].lanes == (lane,)
    assert by_id["vectorization.axes_0_1"].facts["padding"] == [(-5) % lane, (-10) % lane]


@pytest.mark.parametrize("shape", [(8, 16), (3, 10)])
def test_unary_rule_rewrite_is_equivalent_for_exact_and_padded_shapes(
    make_op_module, rewrite_candidate, shape,
):
    module = make_op_module("math.silu", (fm.tensor_type("bfloat16", shape),))
    candidate, result, rewritten = rewrite_candidate(module, VectorizeUnary(), "vectorization.last_axis")
    assert result.replacement.op == ("tensors.unpack" if shape[-1] % 8 == 0 else "tensors.slice_to_shape")
    assert "math.vectorized_unary" in {node.op for node in result.prefix_nodes}
    assert result.replacement.metadata["vectorization_candidate"] == candidate.id

    value = torch.randn(shape, dtype=torch.bfloat16)
    evaluator = TorchEvaluator(DictWeightResolver({}))
    torch.testing.assert_close(
        evaluator.run(rewritten, {"arg0": value})[0],
        evaluator.run(module, {"arg0": value})[0],
    )


def test_unary_rule_rejects_existing_vector_type(make_op_module):
    value_type = fm.tensor_type(fm.vector_type("bfloat16", (8,)), (2,))
    module = make_op_module("math.silu", (value_type,))
    assert VectorizeUnary().candidates(module.node_map["root"], module) == ()
