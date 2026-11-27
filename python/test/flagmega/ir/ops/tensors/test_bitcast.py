# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT

import torch
import pytest

from triton.flagmega import ir as fm
from triton.flagmega.evaluator import DictWeightResolver, TorchEvaluator
from triton.flagmega.errors import IRSchemaError
from triton.flagmega.ir.ops.tensors.bitcast import Bitcast


class _BitcastModule(fm.Module):
    def __init__(self, source_type, target_dtype):
        super().__init__(dialect="ntt", stage="packed", entry="main")
        self.source_type = source_type
        self.target_dtype = target_dtype

    def forward(self):
        value = self.input("value", self.source_type)
        result = fm.F.tensors.bitcast(value, self.target_dtype, name="result")
        self.function("main", (value,), (result,))


def test_bitcast_scalar_storage_to_vector_dtype_preserves_bytes_and_shape():
    source_type = fm.tensor_type("float32", (2, 16))
    target_dtype = fm.VectorType(fm.DType.FLOAT32, (4,))
    module = _BitcastModule(source_type, target_dtype).build()

    result_type = module.node_map["result"].type
    assert result_type == fm.tensor_type(target_dtype, (2, 4))

    value = torch.randn(2, 16)
    actual = TorchEvaluator(DictWeightResolver({})).run(
        module, {"value": value}
    )[0]
    assert actual.shape == (2, 4, 4)
    assert torch.equal(actual.reshape(2, 16), value)


def test_bitcast_changes_last_extent_when_scalar_item_sizes_differ():
    source_type = fm.tensor_type("float32", (2, 8))
    module = _BitcastModule(source_type, fm.DType.INT64).build()

    assert module.node_map["result"].type == fm.tensor_type("int64", (2, 4))
    value = torch.randn(2, 8)
    actual = TorchEvaluator(DictWeightResolver({})).run(
        module, {"value": value}
    )[0]
    assert actual.shape == (2, 4)
    assert torch.equal(actual.view(torch.uint8), value.view(torch.uint8))


def _node(value_type):
    return fm.Node("value", "builtin.var", (), value_type, attrs={"name": "value"})


def test_bitcast_scales_only_the_final_distributed_split_unit():
    placement = fm.Placement((2, 4), "yx", "bb")
    source = fm.DistributedType(
        fm.tensor_type("float32", (8, 16)),
        (
            fm.SBP.split_contiguous((0,), 2),
            fm.SBP.split_block_cyclic((1,), 4),
        ),
        placement,
    )

    result = Bitcast.infer_type(
        (_node(source),), {"dtype": fm.DType.INT64.value}
    )

    assert result.tensor == fm.tensor_type("int64", (8, 8))
    assert result.axis_policies == (
        fm.SBP.split_contiguous((0,), 2),
        fm.SBP.split_block_cyclic((1,), 2),
    )


def test_bitcast_rejects_partial_distributed_storage():
    placement = fm.Placement((2,), "x", "b")
    source = fm.DistributedType(
        fm.tensor_type("float32", (8,)),
        (fm.SBP.broadcast(),),
        placement,
        partial=fm.SBP.partial((0,)),
    )

    with pytest.raises(IRSchemaError, match="partial distributed"):
        Bitcast.infer_type((_node(source),), {"dtype": fm.DType.INT64.value})
