# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT

import torch

from triton.flagmega import ir as fm
from triton.flagmega.evaluator import DictWeightResolver, TorchEvaluator
from triton.flagmega.passes.functions import (
    post_function_boundary_pack_propagation,
)
from triton.flagmega.targets import NvidiaSm90Target


class _KDevectorizedMatMul(fm.Module):
    def __init__(self):
        super().__init__(
            dialect="ntt", stage="boundary_layout_propagated", entry="main"
        )

    def forward(self):
        vector = fm.vector_type("float32", (8,))
        lhs = self.input(
            "lhs", fm.tensor_type(vector, (3, 3)), id="lhs"
        )
        rhs = self.input(
            "rhs", fm.tensor_type(vector, (24, 3)), id="rhs"
        )
        scalar_lhs = fm.F.tensors.unpack(
            lhs, axes=(1,), name="lhs_k_unpack"
        )
        product = fm.F.math.vectorized_matmul(
            scalar_lhs,
            rhs,
            lhs_axes=(),
            rhs_axes=(1,),
            output_axes=(1,),
            output_lanes=(8,),
            name="product",
        )
        result = fm.F.tensors.unpack(
            product, axes=(1,), name="result"
        )
        self.function("main", (lhs, rhs), (result,))


def test_post_boundary_egraph_reinterprets_k_devectorized_matmul_input():
    original = _KDevectorizedMatMul().build()

    rewritten = post_function_boundary_pack_propagation(
        original, NvidiaSm90Target()
    )

    product = rewritten.node_map["product"]
    lhs_view = rewritten.node_map[product.inputs[0]]
    assert product.op == "math.vectorized_matmul"
    assert lhs_view.op == "tensors.bitcast"
    assert lhs_view.inputs == ("lhs",)
    assert lhs_view.type == original.node_map["lhs_k_unpack"].type

    generator = torch.Generator().manual_seed(20260904)
    values = {
        "lhs": torch.randn((3, 3, 8), generator=generator),
        "rhs": torch.randn((24, 3, 8), generator=generator),
    }
    evaluator = TorchEvaluator(DictWeightResolver({}))
    torch.testing.assert_close(
        evaluator.run(rewritten, values)[0],
        evaluator.run(original, values)[0],
    )
