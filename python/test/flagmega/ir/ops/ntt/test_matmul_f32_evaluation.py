# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: MIT

import pytest
import torch

from triton.flagmega import ir as fm
from triton.flagmega.evaluator import DictWeightResolver, TorchEvaluator


@pytest.mark.parametrize("with_stats", [False, True])
def test_packed_f32_is_not_bf16_result_widening(with_stats):

    class Graph(fm.Module):

        def forward(self):
            lhs = self.input("lhs", fm.tensor_type("bfloat16", (2, 64)))
            rhs = self.input("rhs", fm.tensor_type(fm.vector_type("bfloat16", (8, 2, 8)), (4, 4)))
            parameters = [lhs, rhs]
            if with_stats:
                residual = self.input("residual", fm.tensor_type(fm.vector_type("float32", (2, 4)), (2, 4)))
                parameters.append(residual)
                result = fm.F.ntt.matmul_norm_stats(lhs, rhs, residual, rhs_layout="k_major", axis=1, use_mean=False,
                                                    output_data_type="float32")
            else:
                none = fm.F.builtin.none()
                result = fm.F.ntt.packed_matmul(lhs, rhs, none, none, output_data_type="float32")
            self.function("main", parameters, (result, ))

    module = Graph(dialect="high_level", stage="imported", entry="main").build()
    generator = torch.Generator().manual_seed(374)
    lhs = torch.randn((2, 64), generator=generator).bfloat16()
    rhs = torch.randn((64, 32), generator=generator).bfloat16()
    packed = rhs.reshape(4, 2, 8, 4, 8).permute(0, 3, 4, 1, 2).contiguous()
    feeds = {"lhs": lhs, "rhs": packed}
    expected = (lhs.float() @ rhs.float()).reshape(2, 4, 2, 4)
    if with_stats:
        feeds["residual"] = torch.randn((2, 4, 2, 4), generator=generator)
        expected = expected + feeds["residual"]
    actual, = TorchEvaluator(DictWeightResolver({})).run(module, feeds)
    if with_stats:
        actual, stats = actual
        torch.testing.assert_close(stats,
                                   expected.reshape(2, 32).square().sum(-1, keepdim=True).unsqueeze(0), rtol=0, atol=0)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
