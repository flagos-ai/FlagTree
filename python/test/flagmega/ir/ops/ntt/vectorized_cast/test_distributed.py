# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT

import pytest
import torch

from triton.flagmega import ir as fm
from triton.flagmega.errors import IRSchemaError
from triton.flagmega.evaluator import DictWeightResolver, TorchEvaluator


def _module(input_type, output_dtype):
    class Graph(fm.Module):
        def __init__(self):
            super().__init__(dialect="ntt", stage="distributed", entry="main")

        def forward(self):
            value = self.input("value", input_type, id="value")
            output = fm.F.ntt.vectorized_cast(
                value,
                output_dtype,
                (1,),
                name="output",
            )
            self.function("main", (value,), (output,))

    return Graph().build()


def test_vectorized_cast_scales_split_units_and_preserves_partial_contract():
    placement = fm.Placement((2, 2), "yx", "bb")
    input_type = fm.DistributedType(
        fm.tensor_type(fm.vector_type("bfloat16", (8,)), (1, 8)),
        (fm.SBP.broadcast(), fm.SBP.split_block_cyclic((0,), 2)),
        placement,
        partial=fm.SBP.partial((1,)),
    )
    module = _module(input_type, fm.vector_type("float32", (4,)))

    output_type = module.node_map["output"].type
    assert output_type == fm.DistributedType(
        fm.tensor_type(fm.vector_type("float32", (4,)), (1, 16)),
        (fm.SBP.broadcast(), fm.SBP.split_block_cyclic((0,), 4)),
        placement,
        partial=fm.SBP.partial((1,)),
    )

    value = torch.randn(1, 8, 8, dtype=torch.bfloat16)
    output = TorchEvaluator(DictWeightResolver({})).run(
        module, {"value": value}
    )[0]
    assert output.shape == (1, 16, 4)
    torch.testing.assert_close(output.reshape(-1), value.float().reshape(-1))


def test_vectorized_cast_rejects_a_split_that_cuts_output_lane_groups():
    input_type = fm.DistributedType(
        fm.tensor_type(fm.vector_type("bfloat16", (8,)), (1, 8)),
        (fm.SBP.broadcast(), fm.SBP.split_block_cyclic((0,), 1)),
        fm.Placement((2,), "x", "b"),
    )

    with pytest.raises(IRSchemaError, match="cannot scale axis 1 split policy"):
        _module(input_type, fm.vector_type("float32", (16,)))


def test_repeated_axis_scales_by_complete_lane_product_not_intermediate_factors():
    from triton.flagmega.ir.ops.ntt.vectorized_cast import VectorizedCast

    source = fm.Node("value", "builtin.var", (), fm.DistributedType(
        fm.tensor_type(fm.vector_type("bfloat16", (2, 8)), (2,)),
        (fm.SBP.split_block_cyclic((0,), 1),), fm.Placement((2,), "x", "b")), attrs={"name": "value"})
    prepared = VectorizedCast.prepare((source,), {"new_type": fm.vector_type("float32", (8, 2)),
                                                 "vectorize_axes": (0, 0)})
    assert prepared.result_type.tensor.shape == source.type.tensor.shape
    assert prepared.result_type.axis_policies == source.type.axis_policies
