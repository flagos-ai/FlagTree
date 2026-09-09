# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT

import pytest
import torch

from triton.flagmega import ir as fm
from triton.flagmega.evaluator import DictWeightResolver, TorchEvaluator
from triton.flagmega.rules import DataflowRewriter
from triton.flagmega.rules.ntt.vectorize.propagation import propagation_rules


@pytest.mark.parametrize("dtype", ["bfloat16", "float32"])
@pytest.mark.parametrize("unpack", [False, True])
@pytest.mark.parametrize("split", [False, True])
def test_sigmoid_propagates_without_changing_fp32_compute_and_output_rounding(dtype, unpack, split):

    class Graph(fm.Module):

        def forward(self):
            typ = fm.tensor_type(fm.vector_type(dtype, (8, )) if unpack else dtype, (2, 4 if unpack else 32))
            if split:
                typ = fm.DistributedType(typ, (fm.SBP.split_contiguous((0, )), fm.SBP.broadcast()),
                                         fm.Placement((2, 2), "yx", "bb"))
            value = self.input("value", typ, id="value")
            source = fm.F.tensors.unpack(value, axes=(1, )) if unpack else value
            sigmoid = fm.F.math.sigmoid(source)
            result = sigmoid if unpack else fm.F.tensors.pack(sigmoid, (8, ), axes=(1, ))
            self.function("main", (value, ), (result, ))

    original = Graph(dialect="ntt", stage="packed", entry="main").build()
    rewritten = DataflowRewriter(propagation_rules()).rewrite(original)
    assert any(n.op == "math.vectorized_unary" and n.attrs["unary_op"] == "sigmoid" for n in rewritten.nodes)
    value = torch.linspace(-30, 30, 64).to(getattr(torch, dtype)).reshape((2, 4, 8) if unpack else (2, 32))
    evaluator = TorchEvaluator(DictWeightResolver({}))
    torch.testing.assert_close(evaluator.run(rewritten, {"value": value}), evaluator.run(original, {"value": value}),
                               rtol=0, atol=0)
