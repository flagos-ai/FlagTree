# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT

import torch

from triton.flagmega import ir as fm
from triton.flagmega.evaluator import DictWeightResolver, TorchEvaluator
from triton.flagmega.passes import DataflowPass
from triton.flagmega.passes.auto_distributed.policy import (
    lower_vectorization_contracts,
)
from triton.flagmega.rules.ntt.vectorize.propagation.rope import (
    rope_propagation_rules,
)
from triton.flagmega.targets import NvidiaSm90Target


def _module(*, head_dim: int = 16, pack_axis: int = 2):
    class Graph(fm.Module):
        def __init__(self):
            super().__init__(dialect="high_level", stage="vectorized", entry="main")

        def forward(self):
            value = self.input(
                "value", fm.tensor_type("bfloat16", (1, 8, head_dim)), id="value"
            )
            cos = self.input(
                "cos", fm.tensor_type("bfloat16", (1, 1, head_dim)), id="cos"
            )
            sin = self.input(
                "sin", fm.tensor_type("bfloat16", (1, 1, head_dim)), id="sin"
            )
            rope = fm.F.nn.rope(value, cos, sin, name="rope")
            output = fm.F.tensors.pack(
                rope, (8,), axes=(pack_axis,), name="output"
            )
            self.function("main", (value, cos, sin), (output,))

    return Graph().build()


def _run(module):
    return DataflowPass("RoPEPropagation", rope_propagation_rules()).run(module)


def test_pack_rope_propagates_last_axis_and_double_packs_rotary_tables():
    original = _module()
    result = _run(original)

    output = result.node_map["output"]
    assert output.op == "ntt.vectorized_rope"
    packed_value = result.node_map[output.inputs[0]]
    packed_cos = result.node_map[output.inputs[1]]
    packed_sin = result.node_map[output.inputs[2]]
    assert packed_value.attrs == {"lanes": (8,), "axes": (2,)}
    assert packed_cos.attrs == {"lanes": (2, 8), "axes": (2, 2)}
    assert packed_sin.attrs == {"lanes": (2, 8), "axes": (2, 2)}
    assert result.node_map[packed_cos.inputs[0]].op == "tensors.cast"
    assert result.node_map[packed_sin.inputs[0]].op == "tensors.cast"

    torch.manual_seed(31)
    values = {
        "value": torch.randn((1, 8, 16), dtype=torch.bfloat16),
        "cos": torch.randn((1, 1, 16), dtype=torch.bfloat16),
        "sin": torch.randn((1, 1, 16), dtype=torch.bfloat16),
    }
    evaluator = TorchEvaluator(DictWeightResolver({}))
    torch.testing.assert_close(
        evaluator.run(result, values)[0], evaluator.run(original, values)[0]
    )


def test_rope_propagation_rejects_non_rotary_axis_pack():
    original = _module(pack_axis=1)
    result = _run(original)

    assert result.node_map["output"].op == "tensors.pack"
    assert result.node_map["rope"].op == "nn.rope"


def test_rope_propagation_requires_rotary_extent_divisible_by_pair_times_lane():
    original = _module(head_dim=24)
    result = _run(original)

    assert result.node_map["output"].op == "tensors.pack"
    assert result.node_map["rope"].op == "nn.rope"


def test_auto_vectorize_registers_rope_rule_and_retains_native_vector_semantics():
    original = _module()
    target = NvidiaSm90Target()
    vectorized = target.apply_vectorization(original)
    assert vectorized.node_map["output"].op == "ntt.vectorized_rope"

    normalized = fm.verify_module(lower_vectorization_contracts(vectorized))
    output = normalized.node_map[normalized.function_map["main"].outputs[0]]
    assert output.op == "ntt.vectorized_rope"

    torch.manual_seed(37)
    values = {
        "value": torch.randn((1, 8, 16), dtype=torch.bfloat16),
        "cos": torch.randn((1, 1, 16), dtype=torch.bfloat16),
        "sin": torch.randn((1, 1, 16), dtype=torch.bfloat16),
    }
    evaluator = TorchEvaluator(DictWeightResolver({}))
    # Like nncase, explicit VectorizedRoPE remains physical typed-vector IR.
    expected = evaluator.run(original, values)[0]
    torch.testing.assert_close(evaluator.run(normalized, values)[0], expected)
