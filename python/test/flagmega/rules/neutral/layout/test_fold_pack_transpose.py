# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT

import torch

from triton.flagmega import ir as fm
from triton.flagmega.evaluator import DictWeightResolver, TorchEvaluator
from triton.flagmega.rules import DataflowRewriter
from triton.flagmega.rules.neutral import fold_pack_transpose_rule


class _PackTranspose(fm.Module):
    def __init__(self):
        super().__init__(dialect="ntt", stage="packed", entry="main")

    def forward(self):
        value = self.input("value", fm.tensor_type("bfloat16", (2, 4, 16)), id="value")
        transposed = fm.F.tensors.permute(value, (1, 0, 2), name="transposed")
        result = fm.F.tensors.pack(transposed, (8,), axes=(2,), name="result")
        self.function("main", (value,), (result,))


def test_fold_pack_transpose_moves_pack_to_inverse_input_axis():
    original = _PackTranspose().build()
    rewritten = DataflowRewriter((fold_pack_transpose_rule(),)).rewrite(original)

    packed = rewritten.node_map["result.fold_pack_transpose.pack"]
    assert packed.op == "tensors.pack"
    assert packed.inputs == ("value",)
    assert packed.attrs["axes"] == (2,)
    assert rewritten.node_map["result"].op == "tensors.permute"
    evaluator = TorchEvaluator(DictWeightResolver({}))
    inputs = {"value": torch.randn(2, 4, 16, dtype=torch.bfloat16)}
    torch.testing.assert_close(
        evaluator.run(rewritten, inputs)[0],
        evaluator.run(original, inputs)[0],
    )


def test_fold_pack_transpose_remaps_non_identity_axis():
    class Graph(fm.Module):
        def __init__(self):
            super().__init__(dialect="ntt", stage="packed", entry="main")

        def forward(self):
            value = self.input("value", fm.tensor_type("float32", (2, 16)), id="value")
            transposed = fm.F.tensors.permute(value, (1, 0), name="transposed")
            result = fm.F.tensors.pack(transposed, (8,), axes=(0,), name="result")
            self.function("main", (value,), (result,))

    rewritten = DataflowRewriter((fold_pack_transpose_rule(),)).rewrite(Graph().build())
    assert rewritten.node_map["result.fold_pack_transpose.pack"].attrs["axes"] == (1,)
