# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT

import pytest
import torch

from triton.flagmega import ir as fm
from triton.flagmega.evaluator import DictWeightResolver, TorchEvaluator
from triton.flagmega.passes import DataflowPass
from triton.flagmega.rules.ntt.vectorize.propagation.rope import rope_propagation_rules


@pytest.mark.parametrize("head,rotary", [(24, 16), (256, 64)])
@pytest.mark.parametrize("shared", [False, True])
def test_pack_crosses_partial_rope_and_retains_shared_scalar_user(head, rotary, shared):

    class Graph(fm.Module):

        def forward(self):
            value = self.input("value", fm.tensor_type("bfloat16", (1, 4, head)))
            cos = self.input("cos", fm.tensor_type("bfloat16", (1, 1, rotary)))
            sin = self.input("sin", cos.type)
            rope = fm.F.nn.rope(value, cos, sin, rotary_dim=rotary, name="rope")
            packed = fm.F.tensors.pack(rope, (8, ), axes=(2, ), name="packed")
            self.function("main", (value, cos, sin), (packed, rope) if shared else (packed, ))

    original = Graph(dialect="high_level", stage="vectorized", entry="main").build()
    result = DataflowPass("RoPEPropagation", rope_propagation_rules()).run(original)
    fm.verify_module(result)
    packed = result.node_map["packed"]
    assert packed.op == "ntt.vectorized_rope"
    assert packed.attrs == {"rotary_dim": rotary}
    assert result.node_map[packed.inputs[0]].type.shape[-1].fixed_value == head // 8
    assert result.node_map[packed.inputs[1]].type.shape[-1].fixed_value == rotary // 16
    assert not any(node.op in ("tensors.slice", "tensors.concat") for node in result.nodes)
    if shared:
        assert result.node_map["rope"].op == "nn.rope"
    generator = torch.Generator().manual_seed(66)
    values = {
        "value": torch.randn((1, 4, head), generator=generator).bfloat16(), "cos": torch.randn(
            (1, 1, rotary), generator=generator).bfloat16(), "sin": torch.randn((1, 1, rotary),
                                                                                generator=generator).bfloat16()
    }
    evaluator = TorchEvaluator(DictWeightResolver({}))
    for actual, expected in zip(evaluator.run(result, values), evaluator.run(original, values)):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_unaligned_rotary_pairs_do_not_satisfy_vector_table_contract():

    class Graph(fm.Module):

        def forward(self):
            value = self.input("value", fm.tensor_type("float32", (1, 2, 24)))
            table = self.input("table", fm.tensor_type("float32", (1, 1, 8)))
            rope = fm.F.nn.rope(value, table, table, rotary_dim=8)
            packed = fm.F.tensors.pack(rope, 8, axis=-1, name="packed")
            self.function("main", (value, table), (packed, ))

    module = Graph(dialect="high_level", stage="vectorized", entry="main").build()
    result = DataflowPass("RoPEPropagation", rope_propagation_rules()).run(module)
    assert result.node_map["packed"].op == "tensors.pack"
