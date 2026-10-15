# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT

from math import prod

import pytest
import torch

from triton.flagmega import ir as fm
from triton.flagmega.evaluator import DictWeightResolver, TorchEvaluator
from triton.flagmega.rules import DataflowRewriter
from triton.flagmega.rules.neutral.fold_pack_bitcast import fold_pack_bitcast_rule


class _PackBitcast(fm.Module):
    def __init__(self):
        super().__init__(dialect="ntt", stage="boundary_layout_propagated", entry="main")

    def forward(self):
        vector = fm.VectorType(fm.DType.FLOAT32, (4,))
        value = self.input("value", fm.tensor_type(vector, (2, 4)))
        logical = fm.F.tensors.bitcast(value, fm.DType.FLOAT32, name="logical")
        result = fm.F.tensors.pack(
            logical, (4,), axes=(1,), name="result"
        )
        self.function("main", (value,), (result,))


def test_pack_of_inverse_bitcast_folds_to_original_storage():
    rewritten = DataflowRewriter((fold_pack_bitcast_rule(),)).rewrite(
        _PackBitcast().build()
    )

    main = rewritten.function_map["main"]
    assert main.outputs == main.parameters
    assert not any(
        node.op in {"tensors.pack", "tensors.bitcast"}
        for node in rewritten.nodes
    )


@pytest.mark.parametrize("axes,lanes,folds", [((1,), (2,), True), ((-1,), (8,), True),
                                            ((1, 1), (2, 4), True), ((0,), (2,), False)])
def test_repacking_contiguous_final_axis_is_a_view_not_a_copy(axes, lanes, folds):
    class Graph(fm.Module):
        def forward(self):
            value = self.input("value", fm.tensor_type(fm.vector_type("float32", (4,)), (2, 4)))
            scalar = fm.F.tensors.bitcast(value, fm.DType.FLOAT32)
            output = fm.F.tensors.pack(scalar, lanes, axes=axes, name="output")
            self.function("main", (value,), (output,))

    source = Graph(dialect="ntt", stage="boundary_layout_propagated", entry="main").build()
    result = DataflowRewriter((fold_pack_bitcast_rule(),)).rewrite(source)
    assert result.node_map["output"].op == ("tensors.bitcast" if folds else "tensors.pack")
    evaluator = TorchEvaluator(DictWeightResolver({}))
    feeds = {"value": torch.arange(32).float().reshape(2, 4, 4)}
    torch.testing.assert_close(evaluator.run(result, feeds), evaluator.run(source, feeds), rtol=0, atol=0)


@pytest.mark.parametrize("view_lanes", [(2,), (2, 1)])
@pytest.mark.parametrize("axis", [1, -1, 0])
def test_pack_restores_a_partially_unpacked_vector_without_copy(view_lanes, axis):
    class Graph(fm.Module):
        def forward(self):
            value = self.input("value", fm.tensor_type(fm.vector_type("float32", (4, *view_lanes)), (2, 2)))
            view = fm.F.tensors.bitcast(value, fm.vector_type("float32", view_lanes))
            packed = fm.F.tensors.pack(view, (4,) if axis != 0 else (2,), axes=(axis,))
            self.function("main", (value,), (packed,))

    original = Graph(dialect="ntt", stage="packed", entry="main").build()
    rewritten = DataflowRewriter((fold_pack_bitcast_rule(),)).rewrite(original)
    if axis != 0:
        assert rewritten.functions[0].outputs == rewritten.functions[0].parameters
    else:
        assert any(node.op == "tensors.pack" for node in rewritten.nodes)
    evaluator = TorchEvaluator(DictWeightResolver({}))
    feeds = {"value": torch.arange(16 * prod(view_lanes)).float().reshape(2, 2, 4, *view_lanes)}
    torch.testing.assert_close(evaluator.run(rewritten, feeds), evaluator.run(original, feeds), rtol=0, atol=0)
