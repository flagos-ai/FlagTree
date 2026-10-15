# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT

import pytest
import torch
from math import prod

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


@pytest.mark.parametrize("dtype", ["float32", "bfloat16"])
@pytest.mark.parametrize("lanes", [(8,), (2, 4), (1,)])
@pytest.mark.parametrize("axes,keep_dims", [((1,), False), ((1,), True), ((), False)])
def test_reduce_sum_output_lanes_preserve_scalar_rounding(dtype, lanes, axes, keep_dims):
    value = torch.linspace(-3, 2, 144).reshape(2, 3, 24).to(getattr(torch, dtype))
    expected = value.float().sum(axes, keepdim=keep_dims).to(value.dtype) if axes else value
    actual = evaluate(ReduceSum, (value,), axes=axes, keep_dims=keep_dims, output_lanes=lanes)
    assert actual.shape == (*expected.shape[:-1], expected.shape[-1] // prod(lanes), *lanes)
    torch.testing.assert_close(actual.reshape(expected.shape), expected, rtol=0, atol=0)


@pytest.mark.parametrize("lanes", [None, 8, (0,), (-1,), (True,), (2.0,)])
def test_reduce_sum_rejects_invalid_output_lanes(lanes):
    with pytest.raises(IRSchemaError, match="output_lanes"):
        primitive_module(ReduceSum, (fm.tensor_type("float32", (2, 24)),), output_lanes=lanes)


@pytest.mark.parametrize("axes,keep_dims", [((1,), False), ((0, 1), False), ((1,), True)])
def test_reduce_sum_rejects_nondivisible_or_rank_zero_output(axes, keep_dims):
    with pytest.raises(IRSchemaError, match="output dimension"):
        primitive_module(ReduceSum, (fm.tensor_type("float32", (3, 24)),),
                         axes=axes, keep_dims=keep_dims, output_lanes=(8,))


def test_reduce_sum_output_lane_units_preserve_partial_mesh_axes():
    placement = fm.Placement((2, 2), "xy", "bb")
    source = fm.DistributedType(fm.tensor_type("float32", (1, 4, 64)), (
        fm.SBP.broadcast(), fm.SBP.split_contiguous((0,)), fm.SBP.split_block_cyclic((1,), 16)), placement)
    module = primitive_module(ReduceSum, (source,), axes=(1,), keep_dims=False, output_lanes=(2, 4))
    result = module.node_map["output"].type
    assert result.tensor == fm.tensor_type(fm.vector_type("float32", (2, 4)), (1, 8))
    assert result.axis_policies == (fm.SBP.broadcast(), fm.SBP.split_block_cyclic((1,), 2))
    assert result.partial == fm.SBP.partial((0,))
    incompatible = fm.DistributedType(source.tensor, (*source.axis_policies[:-1], fm.SBP.split_block_cyclic((1,), 4)),
                                      placement)
    with pytest.raises(IRSchemaError, match="split must align"):
        primitive_module(ReduceSum, (incompatible,), axes=(1,), keep_dims=False, output_lanes=(8,))


def test_reduce_sum_default_output_preserves_legacy_attributes():
    module = primitive_module(ReduceSum, (fm.tensor_type("float32", (2, 8)),))
    assert "output_lanes" not in module.node_map["output"].attrs
