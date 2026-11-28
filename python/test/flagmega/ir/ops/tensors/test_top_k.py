# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT

import pytest
import torch

from triton.flagmega import ir as fm
from triton.flagmega.errors import IRSchemaError
from triton.flagmega.ir.ops.tensors.top_k import TopK
from python.test.flagmega.ir.ops.primitive_helpers import evaluate, primitive_module


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32, torch.int32, torch.int64])
@pytest.mark.parametrize("largest,expected", [(True, [1, 2, 4]), (False, [0, 3, 1])])
@pytest.mark.parametrize("index_dtype", ["int32", "int64"])
def test_top_k_ties_are_lower_index_first(dtype, largest, expected, index_dtype):
    value = torch.tensor([[1, 3, 3, 1, 3]], dtype=dtype)
    values, indices = evaluate(TopK, (value, ), k=3, largest=largest, index_dtype=index_dtype)
    assert indices.tolist() == [expected]
    assert indices.dtype == getattr(torch, index_dtype)
    torch.testing.assert_close(values, value[:, expected], rtol=0, atol=0)


@pytest.mark.parametrize("k", [0, 3])
def test_top_k_nonlast_axis_and_empty_selection(k):
    value = torch.tensor([[2., 1.], [3., 3.], [1., 2.]])
    values, indices = evaluate(TopK, (value, ), k=k, axis=0, sorted=False)
    assert values.shape == (k, 2)
    torch.testing.assert_close(values, value.gather(0, indices), rtol=0, atol=0)


@pytest.mark.parametrize(
    "attrs",
    [dict(k=-1),
     dict(k=5),
     dict(k=True),
     dict(k=1, axis=3),
     dict(k=1, largest=1),
     dict(k=1, index_dtype="float32")])
def test_top_k_rejects_invalid_contract(attrs):
    with pytest.raises(IRSchemaError):
        primitive_module(TopK, (fm.tensor_type("float32", (2, 4)), ), **attrs)


def test_top_k_distributed_output_preserves_token_ownership():
    tensor = fm.tensor_type("float32", (4, 8))
    placement = fm.Placement((2, ), "x", "b")
    distributed = fm.DistributedType(tensor, (fm.SBP.split_contiguous((0, )), fm.SBP.broadcast()), placement)
    module = primitive_module(TopK, (distributed, ), k=2)
    assert all(field.axis_policies == distributed.axis_policies for field in module.node_map["output"].type.fields)
    split_experts = fm.DistributedType(tensor, tuple(reversed(distributed.axis_policies)), placement)
    with pytest.raises(IRSchemaError, match="selection axis"):
        primitive_module(TopK, (split_experts, ), k=2)
