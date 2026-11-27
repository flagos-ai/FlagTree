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
        "output_data_type", "rhs_layout", "reduction_order",
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


def _distributed_packed_operands(*, lhs_owner=None, rhs_owner=None, lhs_split=None, rhs_split=None,
                                 scale_type=None, addend_type=None):
    placement = fm.Placement((8, 16), "yx", "bb")
    lhs_type = fm.DistributedType(
        fm.tensor_type("bfloat16", (16, 6144)),
        (lhs_split or fm.SBP.broadcast(), fm.SBP.broadcast()), placement, exclusive=lhs_owner)
    rhs_type = fm.DistributedType(
        fm.tensor_type(fm.vector_type("bfloat16", (8, 2, 8)), (384, 256)),
        (fm.SBP.broadcast(), rhs_split or fm.SBP.broadcast()), placement, exclusive=rhs_owner)
    return tuple(fm.Node(f"arg{i}", "builtin.var", (), value) for i, value in enumerate(
        (lhs_type, rhs_type, scale_type or fm.NoneType(), addend_type or fm.NoneType())))


@pytest.mark.parametrize("fused_reduce", [False, True])
@pytest.mark.parametrize("sides", ["lhs", "rhs", "both"])
def test_owner_local_arithmetic_retains_the_exact_exclusive_owner(sides, fused_reduce):
    owner = fm.SBP.exclusive((0, 1), (2, 7))
    operands = _distributed_packed_operands(
        lhs_owner=owner if sides in {"lhs", "both"} else None,
        rhs_owner=owner if sides in {"rhs", "both"} else None)
    result = fm.get_definition("ntt.packed_matmul").prepare(
        operands, {"output_data_type": "bfloat16", "fused_reduce": fused_reduce}).result_type
    assert result.exclusive == owner
    assert result.partial is None
    assert result.axis_policies == (fm.SBP.broadcast(), fm.SBP.broadcast())
    assert result.tensor.dtype == fm.vector_type("bfloat16", (8,))
    assert fm.type_from_data(result.to_data()) == result


@pytest.mark.parametrize("other", ["different_owner", "n_split", "m_split"])
def test_exclusive_arithmetic_requires_coincident_complete_operands(other):
    owner = fm.SBP.exclusive((0,), (2,))
    if other == "different_owner":
        operands = _distributed_packed_operands(lhs_owner=owner, rhs_owner=fm.SBP.exclusive((0,), (3,)))
    elif other == "n_split":
        operands = _distributed_packed_operands(lhs_owner=owner, rhs_split=fm.SBP.split_contiguous((1,)))
    else:
        operands = _distributed_packed_operands(rhs_owner=owner, lhs_split=fm.SBP.split_contiguous((1,)))
    with pytest.raises(IRSchemaError, match="exclusive|Exclusive"):
        fm.get_definition("ntt.packed_matmul").prepare(operands, {})


@pytest.mark.parametrize("scale_policy", ["partial", "exclusive", "other_placement"])
def test_scalar_scale_cannot_publish_unavailable_values(scale_policy):
    placement = fm.Placement((8, 16), "yx", "bb")
    scale = fm.DistributedType(
        fm.tensor_type("bfloat16", ()), (),
        fm.Placement((4, 32), "yx", "bb") if scale_policy == "other_placement" else placement,
        partial=fm.SBP.partial((0,)) if scale_policy == "partial" else None,
        exclusive=fm.SBP.exclusive((0,), (2,)) if scale_policy == "exclusive" else None)
    with pytest.raises(IRSchemaError, match="scale|placement"):
        fm.get_definition("ntt.packed_matmul").prepare(_distributed_packed_operands(scale_type=scale), {})


def test_exclusive_scale_and_addend_are_available_to_the_result_owner():
    owner = fm.SBP.exclusive((0,), (2,))
    placement = fm.Placement((8, 16), "yx", "bb")
    scale = fm.DistributedType(fm.tensor_type("bfloat16", ()), (), placement, exclusive=owner)
    addend = fm.DistributedType(fm.tensor_type(fm.vector_type("bfloat16", (8,)), (16, 256)),
                               (fm.SBP.broadcast(), fm.SBP.broadcast()), placement, exclusive=owner)
    operands = _distributed_packed_operands(lhs_owner=owner, scale_type=scale, addend_type=addend)
    assert fm.get_definition("ntt.packed_matmul").prepare(
        operands, {"output_data_type": "bfloat16"}).result_type == addend
