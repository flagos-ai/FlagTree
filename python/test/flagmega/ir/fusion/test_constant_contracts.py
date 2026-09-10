# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: MIT

import pytest
import torch

from triton.flagmega import ir as fm
from triton.flagmega.artifacts.rdata import _checkpoint_storage_key
from triton.flagmega.errors import CodegenError, NumpyMaterializationUnsupported
from triton.flagmega.evaluator import DictWeightResolver, TorchEvaluator
from triton.flagmega.passes.constants import FreezeConstantIslandsPass
from triton.flagmega.compiler import Compiler


def graph(weight=False):

    @fm.fusion(fm.tensor_type("float32", (4, 8)))
    def square(x):
        return fm.F.math.mul(x, x)

    class Graph(fm.Module):

        def forward(self):
            x = (self.weight("x", square.input_type, source="test", key="x", id="x") if weight else self.input(
                "x", square.input_type, id="x"))
            y = fm.F.with_ops(fm.F.tensors.reshape, x, shape=(4, 8), pre_ops={"value": square}, name="y")
            self.function("main", () if weight else (x, ), (y, ))

    return Graph(dialect="high_level", stage="imported" if weight else "frozen_constants", entry="main").build()


def test_fused_storage_view_is_not_a_checkpoint_byte_identity():
    source = graph(weight=True)
    frozen = FreezeConstantIslandsPass().run(source)
    assert _checkpoint_storage_key(frozen, "y") is None
    value = torch.randn(4, 8)
    output, = TorchEvaluator(DictWeightResolver({"x": value})).run(source, {})
    torch.testing.assert_close(output, value.square(), rtol=0, atol=0)


def test_unfused_storage_view_retains_checkpoint_byte_identity():
    from dataclasses import replace
    source = graph(weight=True)
    node = source.node_map["y"]
    node = replace(node, attrs={key: value for key, value in node.attrs.items() if key != "pre_ops"})
    source = replace(source, nodes=tuple(node if value.id == node.id else value for value in source.nodes))
    frozen = FreezeConstantIslandsPass().run(source)
    assert _checkpoint_storage_key(frozen, "y") == "x"


def test_byte_layout_materializer_cannot_ignore_numeric_pre_ops():
    source = graph(weight=True)
    with pytest.raises(NumpyMaterializationUnsupported, match="semantic constant evaluator"):
        fm.get_definition("tensors.reshape").materialize_numpy(source.node_map["y"], (object(), ), object())


def test_unsupported_fused_view_is_rejected_before_zero_copy_lowering():
    with pytest.raises(CodegenError, match="No PreOps/PostOps boundary emitter"):
        Compiler().compile(graph())
