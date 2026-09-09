# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT

import pytest
import torch

from triton.flagmega import ir as fm
from triton.flagmega.evaluator import DictWeightResolver, TorchEvaluator
from triton.flagmega.passes.functions import post_function_boundary_pack_propagation
from triton.flagmega.targets import NvidiaSm90Target


@pytest.mark.parametrize("op", ["sigmoid", "silu"])
@pytest.mark.parametrize("extent", [1, 3, 7])
@pytest.mark.parametrize("shared", [False, True])
def test_pointwise_crop_does_not_require_repacking_discarded_tail(op, extent, shared):

    class Graph(fm.Module):

        def forward(self):
            value = self.input("value", fm.tensor_type(fm.vector_type("bfloat16", (8, )), (1, 1)), id="value")
            scalar = fm.F.tensors.bitcast(value, "bfloat16")
            crop = fm.F.tensors.slice_to_shape(scalar, (1, extent))
            padded = fm.F.tensors.pad(crop, (0, 8 - extent), pad_value=3.)
            packed = fm.F.tensors.pack(padded, (8, ), axes=(1, ))
            unary = fm.F.math.vectorized_unary(packed, unary_op=op)
            result = fm.F.tensors.slice_to_shape(fm.F.tensors.bitcast(unary, "bfloat16"), (1, extent))
            self.function("main", (value, ), (result, packed) if shared else (result, ))

    original = Graph(dialect="ntt", stage="packed", entry="main").build()
    result = post_function_boundary_pack_propagation(original, NvidiaSm90Target())
    if not shared:
        assert not any(n.op in {"tensors.pack", "tensors.pad"} for n in result.nodes)
    else:
        # The public padded value still needs its tail, unlike the cropped
        # pointwise result. It must not be redirected to the unpadded input.
        assert any(n.op == "tensors.pad" for n in result.nodes)
    value = torch.randn(1, 1, 8).bfloat16()
    evaluator = TorchEvaluator(DictWeightResolver({}))
    torch.testing.assert_close(evaluator.run(original, {"value": value}), evaluator.run(result, {"value": value}),
                               rtol=0, atol=0)
