# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT

from triton.flagmega import ir as fm
from triton.flagmega.rules import DataflowRewriter
from triton.flagmega.rules.neutral.fold_bitcast_bitcast import (
    fold_bitcast_bitcast_rule,
)


class _NestedBitcast(fm.Module):
    def __init__(self):
        super().__init__(dialect="ntt", stage="boundary_layout_propagated", entry="main")

    def forward(self):
        source_dtype = fm.VectorType(fm.DType.FLOAT32, (2, 2))
        middle_dtype = fm.VectorType(fm.DType.FLOAT32, (4,))
        value = self.input("value", fm.tensor_type(source_dtype, (2, 4)))
        middle = fm.F.tensors.bitcast(value, middle_dtype, name="middle")
        result = fm.F.tensors.bitcast(
            middle, fm.DType.FLOAT32, name="result"
        )
        self.function("main", (value,), (result,))


def test_nested_bitcasts_fold_to_one_direct_bitcast():
    rewritten = DataflowRewriter((fold_bitcast_bitcast_rule(),)).rewrite(
        _NestedBitcast().build()
    )

    bitcasts = [node for node in rewritten.nodes if node.op == "tensors.bitcast"]
    assert len(bitcasts) == 1
    assert bitcasts[0].id == "result"
    assert bitcasts[0].inputs == rewritten.function_map["main"].parameters
