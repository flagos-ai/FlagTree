# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT

import pytest
import torch

from triton.flagmega import ir as fm
from triton.flagmega.errors import IRSchemaError
from triton.flagmega.evaluator import DictWeightResolver, TorchEvaluator


class PackedMatMulModule(fm.Module):
    def __init__(self):
        super().__init__(dialect="high_level", stage="imported", entry="main")

    def forward(self):
        lhs = self.input("lhs", fm.tensor_type("bfloat16", (2, 64)), id="lhs")
        rhs = self.input(
            "rhs",
            fm.tensor_type(fm.VectorType(fm.DType.BFLOAT16, (8, 2, 8)), (4, 4)),
            id="rhs",
        )
        none = fm.F.builtin.none(name="none")
        result = fm.F.ntt.packed_matmul(
            lhs,
            rhs,
            none,
            none,
            output_data_type=fm.DType.BFLOAT16,
            name="result",
        )
        self.function("main", (lhs, rhs), (result,))


def test_parameter_info_and_functional_surface_are_static_and_named():
    definition = fm.get_definition("ntt.packed_matmul")
    assert [parameter.name for parameter in definition.parameters] == [
        "lhs", "rhs", "scale", "addend", "fused_reduce",
        "output_data_type", "rhs_layout",
    ]
    assert fm.F.ntt.packed_matmul.__flagmega_op_definition__ is definition


def test_k_major_type_inference_retains_output_n_vector_lane():
    module = PackedMatMulModule().build()
    result = module.node_map["result"]
    assert result.type == fm.tensor_type(
        fm.VectorType(fm.DType.BFLOAT16, (8,)), (2, 4)
    )


def test_k_major_reference_evaluation_matches_logical_matmul():
    module = PackedMatMulModule().build()
    generator = torch.Generator().manual_seed(20260904)
    lhs = torch.randn((2, 64), generator=generator, dtype=torch.bfloat16)
    logical_rhs = torch.randn((64, 32), generator=generator, dtype=torch.bfloat16)
    packed_rhs = logical_rhs.reshape(4, 2, 8, 4, 8).permute(0, 3, 4, 1, 2).contiguous()
    actual = TorchEvaluator(DictWeightResolver({})).run(
        module, {"lhs": lhs, "rhs": packed_rhs}
    )[0]
    expected = (lhs @ logical_rhs).reshape(2, 4, 8)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_k_major_rejects_mismatched_logical_reduction_extent():
    class Invalid(fm.Module):
        def __init__(self):
            super().__init__(dialect="high_level", stage="imported", entry="main")

        def forward(self):
            lhs = self.input("lhs", fm.tensor_type("bfloat16", (1, 63)), id="lhs")
            rhs = self.input(
                "rhs",
                fm.tensor_type(
                    fm.VectorType(fm.DType.BFLOAT16, (8, 2, 8)), (4, 2)
                ),
                id="rhs",
            )
            none = fm.F.builtin.none(name="none")
            fm.F.ntt.packed_matmul(lhs, rhs, none, none, name="result")

    with pytest.raises(IRSchemaError, match="packed RHS K"):
        Invalid().build()
