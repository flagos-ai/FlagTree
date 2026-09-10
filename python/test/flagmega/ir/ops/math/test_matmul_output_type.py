# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: MIT

import pytest
import torch

from triton.flagmega import ir as fm
from triton.flagmega.evaluator import DictWeightResolver, TorchEvaluator


@pytest.mark.parametrize("transpose_a,transpose_b", [(False, False), (False, True), (True, False), (True, True)])
def test_bf16_inputs_can_produce_unrounded_f32(transpose_a, transpose_b, tmp_path):

    class Graph(fm.Module):

        def forward(self):
            lhs = self.input("lhs", fm.tensor_type("bfloat16", (64, 2) if transpose_a else (2, 64)))
            rhs = self.input("rhs", fm.tensor_type("bfloat16", (32, 64) if transpose_b else (64, 32)))
            result = fm.F.math.matmul(lhs, rhs, transpose_a=transpose_a, transpose_b=transpose_b,
                                      output_data_type="float32", name="result")
            self.function("main", (lhs, rhs), (result, ))

    module = Graph(dialect="high_level", stage="imported", entry="main").build()
    assert module.node_map["result"].type == fm.tensor_type("float32", (2, 32))
    generator = torch.Generator().manual_seed(374)
    lhs = torch.randn((2, 64), generator=generator).bfloat16()
    rhs = torch.randn((64, 32), generator=generator).bfloat16()
    result, = TorchEvaluator(DictWeightResolver({})).run(
        module, {"lhs": lhs.T if transpose_a else lhs, "rhs": rhs.T if transpose_b else rhs})
    expected = lhs.float() @ rhs.float()
    torch.testing.assert_close(result, expected, rtol=0, atol=0)
    assert not torch.equal(result, expected.bfloat16().float())
    assert fm.load_module(fm.emit_module(module, tmp_path / "matmul.py")) == module


def test_f32_result_retains_split_k_partial_ownership():
    placement = fm.Placement((2, 2), "yx", "bb")
    lhs_type = fm.DistributedType(fm.tensor_type("bfloat16", (2, 64)),
                                  (fm.SBP.broadcast(), fm.SBP.split_contiguous((0, ))), placement)
    rhs_type = fm.DistributedType(fm.tensor_type("bfloat16", (64, 32)), (fm.SBP.split_contiguous(
        (0, )), fm.SBP.split_contiguous((1, ))), placement)
    definition = fm.get_definition("math.matmul")
    lhs = fm.Node("lhs", "builtin.var", (), lhs_type, attrs={"name": "lhs"})
    rhs = fm.Node("rhs", "builtin.var", (), rhs_type, attrs={"name": "rhs"})
    result = definition.prepare((lhs, rhs), {"output_data_type": "float32"}).result_type
    assert result.tensor == fm.tensor_type("float32", (2, 32))
    assert result.axis_policies == (fm.SBP.broadcast(), fm.SBP.split_contiguous((1, )))
    assert result.partial == fm.SBP.partial((0, ))
