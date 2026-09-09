# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from triton.flagmega import ir as fm
from triton.flagmega.errors import IRSchemaError
from triton.flagmega.ir.ops.tensors.slice import Slice
from python.test.flagmega.ir.ops.primitive_helpers import evaluate, primitive_module


@pytest.mark.parametrize("start,end,step", [(1, 4, 1), (-4, -1, 1), (None, None, -1), (4, 0, -2), (8, 9, 1), (0, 9, 2)])
def test_slice_numpy_and_torch_have_identical_index_semantics(start, end, step):
    raw = np.arange(30, dtype=np.int16).reshape(2, 5, 3)
    attrs = dict(starts=(start, ), ends=(end, ), axes=(1, ), steps=(step, ))
    expected = raw[:, slice(start, end, step), :].copy()
    actual = evaluate(Slice, (torch.tensor(raw.astype(np.float32)), ), **attrs)
    np.testing.assert_array_equal(actual.numpy(), expected)
    module = primitive_module(Slice, (fm.tensor_type("bfloat16", raw.shape), ), **attrs)
    node = module.node_map["output"]
    context = SimpleNamespace(types={item.id: item.type for item in module.nodes}, as_contiguous=np.ascontiguousarray)
    # Raw BF16 payloads must be sliced as bytes/words, without numeric conversion.
    materialized = Slice.materialize_numpy(node, (raw, ), context)
    np.testing.assert_array_equal(materialized, expected)
    assert materialized.dtype == np.int16 and materialized.flags.c_contiguous


@pytest.mark.parametrize("attrs", [
    dict(starts=(0, ), ends=(3, ), steps=(0, )),
    dict(starts=(0, 0), ends=(1, 1), axes=(0, 0)),
    dict(starts=(0, ), ends=(1, ), axes=(3, )),
    dict(starts=(True, ), ends=(1, ))
])
def test_slice_rejects_invalid_ranges(attrs):
    with pytest.raises(IRSchemaError):
        primitive_module(Slice, (fm.tensor_type("float32", (2, 4)), ), **attrs)


def test_slice_preserves_untouched_dynamic_and_vector_axes():
    tokens = fm.dim("tokens", minimum=1, maximum=8)
    module = primitive_module(Slice, (fm.tensor_type(fm.vector_type("bfloat16", 8), (tokens, 4, 2)), ), starts=(1, ),
                              ends=(3, ), axes=(1, ))
    assert module.node_map["output"].type == fm.tensor_type(fm.vector_type("bfloat16", 8), (tokens, 2, 2))


def test_slice_cannot_silently_slice_a_local_split_axis():
    tensor = fm.tensor_type("float32", (4, 8))
    distributed = fm.DistributedType(tensor, (fm.SBP.broadcast(), fm.SBP.split_contiguous((0, ))),
                                     fm.Placement((2, ), "x", "b"))
    with pytest.raises(IRSchemaError, match="split axis"):
        primitive_module(Slice, (distributed, ), starts=(0, ), ends=(4, ), axes=(1, ))
    module = primitive_module(Slice, (distributed, ), starts=(0, ), ends=(2, ), axes=(0, ))
    assert module.node_map["output"].type.axis_policies == distributed.axis_policies
