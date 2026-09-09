# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT

import pytest

from triton.flagmega import ir as fm
from triton.flagmega.errors import IRSchemaError
from triton.flagmega.ir.ops.nn.l2_normalization import L2Normalization
from python.test.flagmega.ir.ops.primitive_helpers import primitive_module


@pytest.mark.parametrize("attrs", (
    {"axes": ()}, {"axes": 0}, {"axes": (True,)}, {"axes": (1, -1)}, {"axes": (2,)},
    {"epsilon": 0}, {"epsilon": -1}, {"epsilon": True}, {"epsilon": float("inf")},
    {"epsilon_mode": "max_norm"}, {"division_mode": "rsqrt"},
))
def test_l2_rejects_ambiguous_or_invalid_numerical_contracts(attrs):
    with pytest.raises(IRSchemaError):
        primitive_module(L2Normalization, (fm.tensor_type("float32", (3, 8)),), **attrs)


@pytest.mark.parametrize("dtype", ("int32", fm.vector_type("bfloat16", (4,))))
def test_l2_rejects_non_scalar_float_elements(dtype):
    with pytest.raises(IRSchemaError, match="scalar BF16/FP32"):
        primitive_module(L2Normalization, (fm.tensor_type(dtype, (3, 8)),))


def test_l2_dynamic_rows_and_partial_or_reduction_split():
    b, s = fm.SBP.broadcast(), fm.SBP.split_contiguous((0,))
    tensor = fm.tensor_type("bfloat16", ("tokens", 8, 16))
    placement = fm.Placement((2,), "x", "b")
    legal = fm.DistributedType(tensor, (b, s, b), placement)
    assert primitive_module(L2Normalization, (legal,)).node_map["output"].type == legal
    for policies in ((b, b, s),):
        with pytest.raises(IRSchemaError, match="reduction axes"):
            primitive_module(L2Normalization, (fm.DistributedType(tensor, policies, placement),))
