# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT

from math import prod
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from triton.flagmega import ir as fm
from triton.flagmega.errors import IRSchemaError
from triton.flagmega.evaluator import DictWeightResolver, TorchEvaluator
from triton.flagmega.ir.ops.tensors.broadcast_to import BroadcastTo
from python.test.flagmega.ir.ops.primitive_helpers import primitive_module


@pytest.mark.parametrize("shape,lanes,target,output_lanes", [
    ((), (), (2, 3), (2, 4)),
    ((2, 1), (), (2, 3), (8, )),
    ((3, ), (4, ), (2, 3), (2, 4)),
    ((3, ), (4, 1), (2, 3), (4, 2)),
    ((1, ), (1, 4), (0, ), (2, 4)),
    ((2, 1), (2, 4), (2, 3), None),
    ((1, 1), (2, 1, 4), (3, 5), (2, 8, 4)),
])
def test_tensor_and_element_dimensions_broadcast_independently(tmp_path, shape, lanes, target, output_lanes):
    dtype = fm.vector_type("bfloat16", lanes) if lanes else "bfloat16"
    module = primitive_module(BroadcastTo, (fm.tensor_type(dtype, shape), ), shape=target, output_lanes=output_lanes)
    node = module.node_map["output"]
    result_lanes = lanes if output_lanes is None else output_lanes
    raw = np.arange(prod((*shape, *lanes)), dtype=np.int16).reshape((*shape, *lanes))
    padded_shape = (1, ) * (len(target) - len(shape)) + shape + (1, ) * (len(result_lanes) - len(lanes)) + lanes
    expected = np.broadcast_to(raw.reshape(padded_shape), (*target, *result_lanes))
    context = SimpleNamespace(types={n.id: n.type for n in module.nodes}, as_contiguous=np.ascontiguousarray)
    actual = BroadcastTo.materialize_numpy(node, (raw, ), context)
    np.testing.assert_array_equal(actual, expected)
    assert actual.dtype == raw.dtype and actual.flags.c_contiguous
    tensor = torch.from_numpy(raw.astype(np.float32)).bfloat16()
    evaluated, = TorchEvaluator(DictWeightResolver({})).run(module, {"value": tensor})
    torch.testing.assert_close(evaluated, tensor.reshape(padded_shape).expand(*target, *result_lanes), rtol=0, atol=0)
    assert fm.load_module(fm.emit_module(module, tmp_path / "ir.py")) == module
    if output_lanes is None:
        assert "output_lanes" not in node.attrs


@pytest.mark.parametrize("lanes", [(), (4, ), (3, 4), (2, 3), (0, 4), (True, 4), (2., 4)])
def test_invalid_lane_broadcast_is_rejected(lanes):
    with pytest.raises(IRSchemaError, match="lanes"):
        primitive_module(BroadcastTo, (fm.tensor_type(fm.vector_type("float32", (2, 4)), (1, )), ), shape=(4, ),
                         output_lanes=lanes)
