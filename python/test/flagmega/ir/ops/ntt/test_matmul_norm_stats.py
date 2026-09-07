# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT

import pytest
import torch

from triton.flagmega import ir as fm
from triton.flagmega.evaluator import DictWeightResolver, TorchEvaluator
from triton.flagmega.errors import IRSchemaError
from triton.flagmega.ir.ops.ntt.matmul_norm_stats import MatMulNormStats


def _node(name, value_type):
    return fm.Node(name, "builtin.var", (), value_type, attrs={"name": name})


def test_inference_materializes_distributed_matmul_partial_before_stats():
    placement = fm.Placement((2, 4), "yx", "bb")
    lhs_tensor = fm.tensor_type("bfloat16", (1, 32))
    rhs_tensor = fm.tensor_type("bfloat16", (16, 32))
    output_tensor = fm.tensor_type("bfloat16", (1, 16))
    lhs = fm.DistributedType(
        lhs_tensor,
        (fm.SBP.broadcast(), fm.SBP.split_contiguous((0, 1))),
        placement,
    )
    rhs = fm.DistributedType(
        rhs_tensor,
        (fm.SBP.broadcast(), fm.SBP.split_contiguous((0, 1))),
        placement,
    )
    addend = fm.DistributedType(
        output_tensor,
        (fm.SBP.broadcast(), fm.SBP.broadcast()),
        placement,
    )

    result = MatMulNormStats.infer_type(
        (_node("lhs", lhs), _node("rhs", rhs), _node("addend", addend)),
        {
            "transpose_a": False,
            "transpose_b": True,
            "axis": -1,
            "use_mean": False,
        },
    )

    assert result.fields[0] == addend
    assert result.fields[1] == fm.DistributedType(
        fm.tensor_type("float32", (1, 1, 1)),
        (fm.SBP.broadcast(), fm.SBP.broadcast(), fm.SBP.broadcast()),
        placement,
    )


class _Module(fm.Module):
    def __init__(self):
        super().__init__(dialect="ntt", stage="distributed", entry="main")

    def forward(self):
        lhs = self.input("lhs", fm.tensor_type("float32", (2, 8)), id="lhs")
        rhs = self.input("rhs", fm.tensor_type("float32", (4, 8)), id="rhs")
        addend = self.input(
            "addend", fm.tensor_type("float32", (2, 4)), id="addend")
        result = fm.F.ntt.matmul_norm_stats(
            lhs,
            rhs,
            addend,
            transpose_b=True,
            axis=-1,
            use_mean=False,
            name="result",
        )
        value, stats = fm.F.tensors.get_items(result, 0, 1)
        self.function("main", (lhs, rhs, addend), (value, stats))


def test_evaluator_returns_matmul_residual_and_stats_without_side_channel():
    module = _Module().build()
    lhs = torch.randn(2, 8)
    rhs = torch.randn(4, 8)
    addend = torch.randn(2, 4)

    value, stats = TorchEvaluator(DictWeightResolver({})).run(
        module, {"lhs": lhs, "rhs": rhs, "addend": addend})
    expected = lhs @ rhs.T + addend

    torch.testing.assert_close(value, expected)
    torch.testing.assert_close(
        stats, expected.square().sum(dim=-1, keepdim=True).unsqueeze(0))


class _PackedModule(fm.Module):
    def __init__(self):
        super().__init__(dialect="ntt", stage="distributed", entry="main")

    def forward(self):
        lhs = self.input("lhs", fm.tensor_type("bfloat16", (2, 64)), id="lhs")
        rhs = self.input(
            "rhs",
            fm.tensor_type(
                fm.VectorType(fm.DType.BFLOAT16, (8, 2, 8)), (4, 4)
            ),
            id="rhs",
        )
        addend = self.input(
            "addend",
            fm.tensor_type(fm.VectorType(fm.DType.BFLOAT16, (8,)), (2, 4)),
            id="addend",
        )
        result = fm.F.ntt.matmul_norm_stats(
            lhs,
            rhs,
            addend,
            rhs_layout="k_major",
            axis=1,
            use_mean=False,
            name="result",
        )
        value, stats = fm.F.tensors.get_items(result, 0, 1)
        self.function("main", (lhs, rhs, addend), (value, stats))


def test_packed_evaluator_unpacks_vector_lane_for_norm_statistics():
    module = _PackedModule().build()
    generator = torch.Generator().manual_seed(20260904)
    lhs = torch.randn((2, 64), generator=generator, dtype=torch.bfloat16)
    logical_rhs = torch.randn(
        (64, 32), generator=generator, dtype=torch.bfloat16
    )
    rhs = logical_rhs.reshape(4, 2, 8, 4, 8).permute(0, 3, 4, 1, 2).contiguous()
    addend = torch.randn((2, 4, 8), generator=generator, dtype=torch.bfloat16)

    value, stats = TorchEvaluator(DictWeightResolver({})).run(
        module, {"lhs": lhs, "rhs": rhs, "addend": addend}
    )
    expected = ((lhs @ logical_rhs).reshape(2, 4, 8) + addend).to(
        torch.bfloat16
    )
    torch.testing.assert_close(value, expected, rtol=0, atol=0)
    torch.testing.assert_close(
        stats,
        expected.float().reshape(2, 32).square().sum(-1, keepdim=True).unsqueeze(0),
        rtol=0,
        atol=0,
    )


def test_packed_local_epilogue_rejects_implicit_split_to_broadcast_publication():
    placement = fm.Placement((2, 2), "yx", "bb")
    lhs = fm.DistributedType(
        fm.tensor_type("bfloat16", (1, 64)),
        (fm.SBP.broadcast(), fm.SBP.broadcast()), placement,
    )
    rhs = fm.DistributedType(
        fm.tensor_type(fm.vector_type("bfloat16", (8, 2, 8)), (4, 4)),
        (fm.SBP.broadcast(), fm.SBP.split_contiguous((0, 1))), placement,
    )
    addend = fm.DistributedType(
        fm.tensor_type(fm.vector_type("bfloat16", (8,)), (1, 4)),
        (fm.SBP.broadcast(), fm.SBP.broadcast()), placement,
    )

    with pytest.raises(IRSchemaError, match="matching projection and addend local shards"):
        MatMulNormStats.infer_type(
            (_node("lhs", lhs), _node("rhs", rhs), _node("addend", addend)),
            {"rhs_layout": "k_major", "axis": 1, "use_mean": False},
        )
