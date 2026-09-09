# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT

import pytest
import torch

from triton.flagmega import ir as fm
from triton.flagmega.errors import IRSchemaError
from triton.flagmega.ir.ops.math.reduce_sum import ReduceSum
from python.test.flagmega.ir.ops.primitive_helpers import evaluate, primitive_module


@pytest.mark.parametrize("axes", [(), (-1, ), (0, 2), (0, 1, 2)])
@pytest.mark.parametrize("keep_dims", [True, False])
def test_reduce_sum_explicit_axes_and_fp32_accumulation(axes, keep_dims):
    value = torch.linspace(-3, 2, 24).reshape(2, 3, 4).bfloat16()
    result = evaluate(ReduceSum, (value, ), axes=axes, keep_dims=keep_dims)
    expected = value if not axes else value.float().sum(axes, keepdim=keep_dims).bfloat16()
    torch.testing.assert_close(result, expected, rtol=0, atol=0)


@pytest.mark.parametrize("keep_dims", [True, False])
def test_reduce_sum_marks_mesh_reduction_as_partial(keep_dims):
    tensor = fm.tensor_type("float32", (4, 8))
    source = fm.DistributedType(tensor, (fm.SBP.split_contiguous((0, )), fm.SBP.split_contiguous((1, ))),
                                fm.Placement((2, 2), "xy", "bb"))
    module = primitive_module(ReduceSum, (source, ), axes=(-1, ), keep_dims=keep_dims)
    result = module.node_map["output"].type
    assert result.partial == fm.SBP.partial((1, ))
    assert result.axis_policies[0] == source.axis_policies[0]


@pytest.mark.parametrize("axes", [(1, -1), (3, ), (True, )])
def test_reduce_sum_rejects_invalid_axes(axes):
    with pytest.raises(IRSchemaError):
        primitive_module(ReduceSum, (fm.tensor_type("float32", (2, 3)), ), axes=axes)
