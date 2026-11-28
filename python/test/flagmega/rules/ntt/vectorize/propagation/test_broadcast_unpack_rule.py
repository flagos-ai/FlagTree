# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT

import pytest
import torch

from triton.flagmega import ir as fm
from triton.flagmega.evaluator import DictWeightResolver, TorchEvaluator
from triton.flagmega.rules import DataflowRewriter
from triton.flagmega.rules.ntt.vectorize.propagation import propagation_rules


@pytest.mark.parametrize("axes,target,remaining,result_lanes", [
    ((1, ), (3, 8), (), None),
    ((-1, -1), (3, 16), (), None),
    ((1, ), (3, 8), (4, ), None),
    ((1, ), (3, 8), (4, ), (2, 4)),
    ((0, ), (3, 2, 4), (), None),
])
def test_broadcast_moves_before_unpack_with_independent_element_broadcast(axes, target, remaining, result_lanes):
    lanes = (2, ) * len(axes)

    class Graph(fm.Module):

        def forward(self):
            value = self.input("value", fm.tensor_type(fm.vector_type("bfloat16", (*lanes, *remaining)), (1, 4)),
                               id="value")
            unpacked = fm.F.tensors.unpack(value, axes=axes)
            result = fm.F.tensors.broadcast_to(unpacked, shape=target, output_lanes=result_lanes, name="result")
            self.function("main", (value, ), (result, unpacked))

    original = Graph(dialect="ntt", stage="packed", entry="main").build()
    rewritten = DataflowRewriter(propagation_rules()).rewrite(original)
    assert rewritten.node_map["result"].op == "tensors.unpack"
    assert not any(n.op == "tensors.pack" for n in rewritten.nodes)
    value = torch.randn(1, 4, *lanes, *remaining).bfloat16()
    evaluator = TorchEvaluator(DictWeightResolver({}))
    torch.testing.assert_close(evaluator.run(original, {"value": value}), evaluator.run(rewritten, {"value": value}),
                               rtol=0, atol=0)
