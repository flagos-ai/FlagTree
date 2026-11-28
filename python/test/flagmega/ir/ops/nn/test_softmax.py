# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT

import pytest
import torch

from triton.flagmega import ir as fm
from triton.flagmega.errors import IRSchemaError
from triton.flagmega.ir.ops.nn.softmax import Softmax
from python.test.flagmega.ir.ops.primitive_helpers import evaluate, primitive_module


@pytest.mark.parametrize("axis", [0, 1, -1])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
def test_softmax_stable_fp32_accumulation(axis, dtype):
    value = torch.tensor([[1e4, 1e4, -1e4], [-3, 2, 1]], dtype=dtype)
    torch.testing.assert_close(evaluate(Softmax, (value, ), axis=axis),
                               value.float().softmax(axis).to(dtype), rtol=0, atol=0)


def test_softmax_rejects_split_reduction_axis():
    source = fm.DistributedType(fm.tensor_type("float32", (2, 8)), (fm.SBP.broadcast(), fm.SBP.split_contiguous((0, ))),
                                fm.Placement((2, ), "x", "b"))
    with pytest.raises(IRSchemaError, match="reduction axis"):
        primitive_module(Softmax, (source, ))
