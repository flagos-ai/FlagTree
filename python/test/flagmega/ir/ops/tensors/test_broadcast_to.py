# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT

import pytest
import torch

from triton.flagmega import ir as fm
from triton.flagmega.errors import IRSchemaError
from triton.flagmega.ir.ops.tensors.broadcast_to import BroadcastTo
from python.test.flagmega.ir.ops.primitive_helpers import evaluate, primitive_module


@pytest.mark.parametrize("source,shape", [((), ()), ((), (2, 3)), ((2, 1), (2, 4)), ((3, ), (2, 3)), ((1, ), (0, ))])
def test_broadcast_shapes_including_scalar_and_empty(source, shape):
    value = torch.ones(source, dtype=torch.bfloat16)
    actual = evaluate(BroadcastTo, (value, ), shape=shape)
    assert actual.shape == shape
    torch.testing.assert_close(actual, value.expand(shape), rtol=0, atol=0)


@pytest.mark.parametrize("shape", [(2, ), (2, 4), (-1, 3), (True, 3)])
def test_broadcast_rejects_invalid_shape(shape):
    with pytest.raises(IRSchemaError):
        primitive_module(BroadcastTo, (fm.tensor_type("float32", (2, 3)), ), shape=shape)


def test_expanding_a_split_singleton_requires_boxing_not_local_replication():
    source = fm.DistributedType(fm.tensor_type("float32", (1,)), (fm.SBP.split_block_cyclic((0,), 1),),
                                fm.Placement((2,), "x", "b"))
    with pytest.raises(IRSchemaError, match="Boxing"):
        primitive_module(BroadcastTo, (source,), shape=(8,))
