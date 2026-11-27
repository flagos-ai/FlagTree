# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT

import pytest
import torch

from triton.flagmega import ir as fm
from triton.flagmega.errors import IRSchemaError
from triton.flagmega.ir.ops.math.div import Div
from triton.flagmega.ir.ops.math.sigmoid import Sigmoid
from python.test.flagmega.ir.ops.primitive_helpers import evaluate, primitive_module


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
def test_div_and_sigmoid_preserve_result_rounding(dtype):
    value = torch.linspace(-9, 9, 24).reshape(3, 8).to(dtype)
    denominator = torch.linspace(1, 4, 24).reshape(3, 8).to(dtype)
    torch.testing.assert_close(evaluate(Div, (value, denominator)), value / denominator, rtol=0, atol=0)
    torch.testing.assert_close(evaluate(Sigmoid, (value, )), value.float().sigmoid().to(dtype), rtol=0, atol=0)


def test_div_requires_explicit_broadcast():
    with pytest.raises(IRSchemaError, match="broadcast"):
        primitive_module(Div, (fm.tensor_type("float32", (2, 4)), fm.tensor_type("float32", (2, 1))))
