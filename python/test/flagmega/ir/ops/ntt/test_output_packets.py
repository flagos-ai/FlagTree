# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Wider scalar results retain outer coordinates and split inner vector packets."""

import pytest
import torch

from triton.flagmega import ir as fm
from triton.flagmega.evaluator import DictWeightResolver, TorchEvaluator


@pytest.mark.parametrize("dtype", ["float16", "bfloat16"])
def test_packed_projection_widens_into_two_f32_packets(dtype, tmp_path):

    class Graph(fm.Module):

        def forward(self):
            lhs = self.input("lhs", fm.tensor_type(dtype, (1, 16)))
            rhs = self.input("rhs", fm.tensor_type(fm.vector_type(dtype, (8, 2, 8)), (1, 2)))
            none = fm.F.builtin.none()
            result = fm.F.ntt.packed_matmul(lhs, rhs, none, none, output_data_type="float32", name="result")
            self.function("main", (lhs, rhs), (result, ))

    module = Graph(dialect="ntt", stage="packed", entry="main").build()
    assert module.node_map["result"].type == fm.tensor_type(fm.vector_type("float32", (2, 4)), (1, 2))
    assert fm.load_module(fm.emit_module(module, tmp_path / "packed.py")) == module
    generator = torch.Generator().manual_seed(11)
    # Dyadic inputs make FP32 sums exact, independent of BLAS stride-specific
    # reduction order, while a BF16 materialized result would lose low bits.
    lhs = torch.randint(-16, 16, (1, 16), generator=generator).to(getattr(torch, dtype)) / 8
    rhs = torch.randint(-16, 16, (16, 16), generator=generator).to(getattr(torch, dtype)) / 8
    packed = rhs.reshape(1, 2, 8, 2, 8).permute(0, 3, 4, 1, 2).contiguous()
    actual, = TorchEvaluator(DictWeightResolver({})).run(module, {"lhs": lhs, "rhs": packed})
    expected = (lhs.float() @ rhs.float()).reshape(1, 2, 2, 4)
    assert not torch.equal(expected, expected.bfloat16().float())
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.parametrize("distributed", [False, True])
def test_vectorized_cast_changes_packet_rank_without_changing_outer_split_units(distributed):
    value_type = fm.tensor_type(fm.vector_type("bfloat16", (8, )), (2, 8))
    if distributed:
        value_type = fm.DistributedType(value_type, (fm.SBP.broadcast(), fm.SBP.split_block_cyclic((0, ), 2)),
                                        fm.Placement((2, 2), "xy", "bb"))

    class Graph(fm.Module):

        def forward(self):
            x = self.input("x", value_type)
            y = fm.F.ntt.vectorized_cast(x, fm.vector_type("float32", (2, 4)), (1, ), name="y")
            self.function("main", (x, ), (y, ))

    module = Graph(dialect="ntt", stage="packed", entry="main").build()
    y = module.node_map["y"].type
    if distributed:
        assert y.axis_policies == value_type.axis_policies
        assert y.tensor.shape == value_type.tensor.shape
    else:
        assert y.shape == value_type.shape
    x = torch.arange(128).reshape(2, 8, 8).bfloat16()
    actual, = TorchEvaluator(DictWeightResolver({})).run(module, {"x": x})
    torch.testing.assert_close(actual, x.float().reshape(2, 8, 2, 4), rtol=0, atol=0)
