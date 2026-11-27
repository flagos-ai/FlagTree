# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: MIT

import pytest
import torch

from triton.flagmega import ir as fm
from triton.flagmega.codegen.triton.fusion import fusion_rules
from triton.flagmega.evaluator import DictWeightResolver, TorchEvaluator
from triton.flagmega.passes.rewriter import DataflowPass
from triton.flagmega.ir.op_fusion import has_ops


def graph(*, shared=False, vector=False, split=False):

    class Graph(fm.Module):

        def forward(self):
            dtype = fm.vector_type("bfloat16", (8, )) if vector else "bfloat16"
            value_type = fm.tensor_type(dtype, (2, 4 if vector else 32))
            if split:
                value_type = fm.DistributedType(value_type, (fm.SBP.split_contiguous((0, )), fm.SBP.broadcast()),
                                                fm.Placement((2, 2), "xy", "bb"))
            x = self.input("x", value_type, id="x")
            wide = (fm.F.ntt.vectorized_cast(x, fm.vector_type("float32", (4, )), vectorize_axes=(1, ), name="wide")
                    if vector else fm.F.tensors.cast(x, "float32", name="wide"))
            sigmoid = (fm.F.math.vectorized_unary(wide, unary_op="sigmoid", name="sigmoid")
                       if vector else fm.F.math.sigmoid(wide, name="sigmoid"))
            y = fm.F.math.mul(sigmoid, wide, name="y")
            self.function("main", (x, ), (y, sigmoid) if shared else (y, ))

    return Graph(dialect="high_level", stage="frozen_constants", entry="main").build()


@pytest.mark.parametrize("vector", [False, True])
@pytest.mark.parametrize("split", [False, True])
def test_composes_pre_and_post_without_changing_visible_values(tmp_path, vector, split):
    original = graph(vector=vector, split=split)
    result = DataflowPass("Fuse", fusion_rules(), rewrite_constants=False).run(original)
    assert any(has_ops(node.attrs) for node in result.nodes)
    fm.verify_module(result)
    assert fm.load_module(fm.emit_module(result, tmp_path / "ir.py")).semantic_hash == result.semantic_hash
    if not split:
        value = torch.randn(2, 4, 8).bfloat16() if vector else torch.randn(2, 32).bfloat16()
        evaluator = TorchEvaluator(DictWeightResolver({}))
        torch.testing.assert_close(evaluator.run(result, {"x": value}), evaluator.run(original, {"x": value}), rtol=0,
                                   atol=0)


def test_shared_producer_is_not_duplicated():
    original = graph(shared=True)
    result = DataflowPass("Fuse", fusion_rules(), rewrite_constants=False).run(original)
    assert "sigmoid" in result.node_map
    assert result.functions[0].outputs[1] == "sigmoid"


def test_unsupported_base_and_partial_values_stay_explicit():
    from triton.flagmega.codegen.triton.fusion import can_fuse
    from dataclasses import replace
    source = graph()

    @fm.fusion(fm.tensor_type("float32", (2, 32)))
    def identity(x):
        return x

    unsupported = replace(source.node_map["sigmoid"], op="nn.rms_norm", attrs={"post_ops": (identity, )})
    assert not can_fuse(unsupported, source)


def test_fusion_body_can_be_rewritten_with_normal_rules():
    from triton.flagmega.rules.neutral.fold_cast import fold_cast_rule

    @fm.fusion(fm.tensor_type("float32", (2, 32)))
    def round_trip(x):
        return fm.F.tensors.cast(fm.F.tensors.cast(x, "bfloat16"), "float32")

    simplified = round_trip.rewrite((fold_cast_rule(), ))
    assert simplified.output == simplified.parameter.id
    assert len(simplified.nodes) == 1
