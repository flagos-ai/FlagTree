# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT
"""nncase RoPE computes in FP32 and rounds only its final result."""

import pytest
import torch

from triton.flagmega import ir as fm
from triton.flagmega.evaluator import DictWeightResolver, TorchEvaluator
from triton.flagmega.ir.ops.nn.rope import RoPE
from triton.flagmega.ir.ops.ntt.vectorized_rope import VectorizedRoPE
from python.test.flagmega.ir.ops.primitive_helpers import evaluate, primitive_module


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize("table_dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize("rotary", [16, 24])
def test_rope_promotes_mixed_operands_without_product_rounding(dtype, table_dtype, rotary):
    generator = torch.Generator().manual_seed(713)
    value = torch.randn((2, 3, 24), generator=generator).to(dtype)
    cos = torch.randn((2, 1, rotary), generator=generator).to(table_dtype)
    sin = torch.randn((2, 1, rotary), generator=generator).float()
    prefix = value[..., :rotary].float()
    rotated = torch.cat((-prefix[..., rotary // 2:], prefix[..., :rotary // 2]), dim=-1)
    expected = torch.cat(((prefix * cos.float() + rotated * sin.float()).to(dtype), value[..., rotary:]), dim=-1)
    actual = evaluate(RoPE, (value, cos, sin), rotary_dim=rotary)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.parametrize("table_dtype", ["bfloat16", "float32"])
def test_vectorized_rope_preserves_table_precision_independently(table_dtype):
    value_type = fm.tensor_type(fm.vector_type("bfloat16", (8, )), (1, 3, 3))
    cos_type = fm.tensor_type(fm.vector_type(table_dtype, (2, 8)), (1, 1, 1))
    sin_type = fm.tensor_type(fm.vector_type("float32", (2, 8)), (1, 1, 1))
    module = primitive_module(VectorizedRoPE, (value_type, cos_type, sin_type), rotary_dim=16)
    generator = torch.Generator().manual_seed(932)
    value = torch.randn((1, 3, 24), generator=generator).bfloat16()
    cos = torch.randn((1, 1, 16), generator=generator).to(getattr(torch, table_dtype))
    sin = torch.randn((1, 1, 16), generator=generator)
    prefix = value[..., :16].float()
    rotated = torch.cat((-prefix[..., 8:], prefix[..., :8]), dim=-1)
    expected = torch.cat(((prefix * cos.float() + rotated * sin).bfloat16(), value[..., 16:]), dim=-1)
    actual = TorchEvaluator(DictWeightResolver({})).run(
        module,
        {"input": value.reshape(1, 3, 3, 8), "cos": cos.reshape(1, 1, 1, 2, 8), "sin": sin.reshape(1, 1, 1, 2, 8)})[0]
    torch.testing.assert_close(actual.reshape_as(expected), expected, rtol=0, atol=0)
