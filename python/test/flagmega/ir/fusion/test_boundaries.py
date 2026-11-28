# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: MIT

import pytest
import torch

from triton.flagmega import ir as fm
from triton.flagmega.evaluator import DictWeightResolver, TorchEvaluator
from triton.flagmega.errors import IRSchemaError


def test_post_ops_bind_independent_tuple_fields(tmp_path):
    value_type = fm.tensor_type("bfloat16", (2, 16))

    @fm.fusion(value_type)
    def widen(x):
        return fm.F.tensors.cast(x, "float32")

    class Graph(fm.Module):

        def forward(self):
            x = self.input("x", value_type)
            y = fm.F.with_ops(fm.F.builtin.tuple, x, x, post_ops=(widen, None))
            self.function("main", (x, ), (y, ))

    module = Graph(dialect="high_level", stage="imported", entry="main").build()
    value = torch.randn(2, 16).bfloat16()
    result, = TorchEvaluator(DictWeightResolver({})).run(module, {"x": value})
    torch.testing.assert_close(result, (value.float(), value), rtol=0, atol=0)
    assert fm.load_module(fm.emit_module(module, tmp_path / "tuple.py")).semantic_hash == module.semantic_hash


def test_partial_reduction_cannot_acquire_post_ops():
    placement = fm.Placement((2, 2), "xy", "bb")
    value_type = fm.DistributedType(fm.tensor_type("float32", (4, 16)), (fm.SBP.broadcast(), fm.SBP.broadcast()),
                                    placement, partial=fm.SBP.partial((0, )))

    @fm.fusion(value_type)
    def narrow(x):
        return fm.F.tensors.cast(x, "bfloat16")

    class Graph(fm.Module):

        def forward(self):
            x = self.input("x", value_type)
            fm.F.with_ops(fm.F.math.add, x, x, post_ops=(narrow, ))

    with pytest.raises(IRSchemaError, match="partial|materialized"):
        Graph(dialect="high_level", stage="imported", entry="main").build()


@pytest.mark.parametrize("distributed", [False, True])
def test_body_contains_real_binary_and_constant_expressions(distributed):

    @fm.fusion(fm.tensor_type("float32", (2, 16)))
    def square_plus_bias(x):
        return fm.F.math.add(fm.F.math.mul(x, x), fm.F.builtin.splat_const(x.type, 0.25))

    value_type = square_plus_bias.input_type
    if distributed:
        placement = fm.Placement((2, 2), "xy", "bb")
        value_type = fm.DistributedType(value_type, (fm.SBP.split_contiguous((0, )), fm.SBP.broadcast()), placement)

    class Graph(fm.Module):

        def forward(self):
            x = self.input("x", value_type)
            y = fm.F.with_ops(fm.F.math.sigmoid, x, pre_ops={"value": square_plus_bias})
            self.function("main", (x, ), (y, ))

    module = Graph(dialect="high_level", stage="imported", entry="main").build()
    value = torch.randn(2, 16)
    result, = TorchEvaluator(DictWeightResolver({})).run(module, {"x": value})
    torch.testing.assert_close(result, (value.square() + 0.25).sigmoid(), rtol=0, atol=0)


def test_post_count_and_shape_mismatch_are_diagnostics():

    @fm.fusion(fm.tensor_type("float32", (2, 16)))
    def identity(x):
        return x

    class Graph(fm.Module):

        def forward(self):
            x = self.input("x", identity.input_type)
            fm.F.with_ops(fm.F.math.sigmoid, x, post_ops=(identity, identity))

    with pytest.raises(IRSchemaError, match="count"):
        Graph(dialect="high_level", stage="imported", entry="main").build()
    with pytest.raises(IRSchemaError, match="input type"):
        identity.infer_type(fm.tensor_type("float32", (2, 32)))
