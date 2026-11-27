# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT

import pytest
import torch

from triton.flagmega import ir as fm
from triton.flagmega.evaluator import DictWeightResolver, TorchEvaluator
from triton.flagmega.rules import DataflowRewriter
from triton.flagmega.rules.ntt.vectorize.propagation import propagation_rules


@pytest.mark.parametrize("shape,target,axes,lanes", [
    ((2, 1), (2, 32), (1, ), (8, )),
    ((), (4, 32), (0, 1), (2, 4)),
    ((32, ), (4, 32), (0, 1), (2, 4)),
    ((32, ), (4, 32), (1, 0), (4, 2)),
    ((1, 32), (4, 32), (1, ), (8, )),
    ((2, 1), (2, 32), (-1, -1), (2, 4)),
])
@pytest.mark.parametrize("old_lanes", [(), (2, 2)])
def test_pack_broadcast_emits_vector_broadcast_and_only_packs_nonbroadcast_source_axes(
        shape, target, axes, lanes, old_lanes):

    class Graph(fm.Module):

        def forward(self):
            dtype = fm.vector_type("bfloat16", old_lanes) if old_lanes else "bfloat16"
            value = self.input("value", fm.tensor_type(dtype, shape), id="value")
            expanded = fm.F.tensors.broadcast_to(value, shape=target)
            result = fm.F.tensors.pack(expanded, lanes, axes=axes, name="result")
            self.function("main", (value, ), (result, expanded))

    original = Graph(dialect="ntt", stage="packed", entry="main").build()
    rewritten = DataflowRewriter(propagation_rules()).rewrite(original)
    result = rewritten.node_map["result"]
    assert result.op == "tensors.broadcast_to"
    assert result.type == original.node_map["result"].type
    value = torch.randn(*shape, *old_lanes).bfloat16() if shape or old_lanes else torch.tensor(0.3).bfloat16()
    evaluator = TorchEvaluator(DictWeightResolver({}))
    torch.testing.assert_close(evaluator.run(rewritten, {"value": value}), evaluator.run(original, {"value": value}),
                               rtol=0, atol=0)
