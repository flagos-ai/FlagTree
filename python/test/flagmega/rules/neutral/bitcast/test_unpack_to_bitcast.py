# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT

import torch

from triton.flagmega import ir as fm
from triton.flagmega.evaluator import DictWeightResolver, TorchEvaluator
from triton.flagmega.rules import DataflowRewriter
from triton.flagmega.rules.neutral.unpack_to_bitcast import unpack_to_bitcast_rule


class _UnpackLastAxis(fm.Module):
    def __init__(self):
        super().__init__(dialect="ntt", stage="boundary_layout_propagated", entry="main")

    def forward(self):
        vector = fm.VectorType(fm.DType.FLOAT32, (4,))
        value = self.input("value", fm.tensor_type(vector, (2, 4)))
        result = fm.F.tensors.unpack(value, axes=(1,), name="result")
        self.function("main", (value,), (result,))


def test_last_axis_unpack_becomes_storage_bitcast():
    original = _UnpackLastAxis().build()
    rewritten = DataflowRewriter((unpack_to_bitcast_rule(),)).rewrite(original)

    assert rewritten.node_map["result"].op == "tensors.bitcast"
    value = torch.randn(2, 4, 4)
    evaluator = TorchEvaluator(DictWeightResolver({}))
    torch.testing.assert_close(
        evaluator.run(rewritten, {"value": value})[0],
        evaluator.run(original, {"value": value})[0],
    )
