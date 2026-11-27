# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT

import pytest
import torch

from triton.flagmega import ir as fm
from triton.flagmega.evaluator import DictWeightResolver, TorchEvaluator
from triton.flagmega.rules import DataflowRewriter
from triton.flagmega.rules.ntt.vectorize.propagation import propagation_rules


def graph(start, end, step=1, lanes=(2, 4), old_lanes=(), unpack=False, split=False):

    class Graph(fm.Module):

        def forward(self):
            dtype = fm.vector_type("bfloat16", (*lanes, *old_lanes)) if unpack else (
                fm.vector_type("bfloat16", old_lanes) if old_lanes else "bfloat16")
            value_type = fm.tensor_type(dtype, (4, 8 if unpack else 64))
            if split:
                value_type = fm.DistributedType(value_type, (fm.SBP.split_contiguous((0, )), fm.SBP.broadcast()),
                                                fm.Placement((2, 2), "yx", "bb"))
            value = self.input("value", value_type, id="value")
            operand = fm.F.tensors.unpack(value, axes=(1, ) * len(lanes)) if unpack else value
            sliced = fm.F.tensors.slice(operand, starts=(start, ), ends=(end, ), steps=(step, ), axes=(-1, ),
                                        name="sliced")
            result = sliced if unpack else fm.F.tensors.pack(sliced, lanes, axes=(-1, ) * len(lanes), name="packed")
            self.function("main", (value, ), (result, operand))

    return Graph(dialect="ntt", stage="packed", entry="main").build()


@pytest.mark.parametrize("start,end", [(0, 16), (16, None), (-32, -8), (None, 999)])
@pytest.mark.parametrize("unpack", [False, True])
@pytest.mark.parametrize("old_lanes,split", [((), False), ((2, ), False), ((), True)])
def test_aligned_slice_propagates_both_directions(start, end, unpack, old_lanes, split):
    original = graph(start, end, unpack=unpack, old_lanes=old_lanes, split=split)
    rewritten = DataflowRewriter(propagation_rules()).rewrite(original)
    sliced, = (n for n in rewritten.nodes if n.op == "tensors.slice")
    dtype = sliced.type.tensor.dtype if split else sliced.type.dtype
    assert dtype == fm.vector_type("bfloat16", (2, 4, *old_lanes))
    value = torch.randn(4, 8, 2, 4, *old_lanes).bfloat16() if unpack else torch.randn(4, 64, *old_lanes).bfloat16()
    evaluator = TorchEvaluator(DictWeightResolver({}))
    torch.testing.assert_close(evaluator.run(rewritten, {"value": value}), evaluator.run(original, {"value": value}),
                               rtol=0, atol=0)


@pytest.mark.parametrize("start,end,step", [(1, 17, 1), (0, 32, 2), (None, None, -1)])
def test_slice_cutting_lanes_or_reversing_lane_order_is_not_commuted(start, end, step):
    original = graph(start, end, step)
    assert DataflowRewriter(propagation_rules()).rewrite(original).semantic_hash == original.semantic_hash


def test_slice_preserves_dynamic_unsliced_axis_and_other_axis_stride():

    class Graph(fm.Module):

        def forward(self):
            batch = fm.dim("batch", minimum=1, maximum=8)
            value = self.input("value", fm.tensor_type("float32", (batch, 6, 32)), id="value")
            sliced = fm.F.tensors.slice(value, starts=(1, -16), ends=(6, None), steps=(2, 1), axes=(1, -1))
            result = fm.F.tensors.pack(sliced, (4, ), axes=(-1, ), name="result")
            self.function("main", (value, ), (result, ))

    original = Graph(dialect="ntt", stage="packed", entry="main").build()
    rewritten = DataflowRewriter(propagation_rules()).rewrite(original)
    result = rewritten.node_map["result"]
    assert result.op == "tensors.slice" and result.type == original.node_map["result"].type
    assert result.attrs["starts"] == (1, 4) and result.attrs["ends"] == (6, 8)
    assert result.attrs["steps"] == (2, 1)
