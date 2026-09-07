# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT

from triton.flagmega import ir as fm
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
