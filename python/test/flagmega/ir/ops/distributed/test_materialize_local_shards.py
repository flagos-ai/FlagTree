# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT

import torch
import pytest

from triton.flagmega import ir as fm
from triton.flagmega import pattern_match as pm
from triton.flagmega.errors import IRSchemaError
from triton.flagmega.evaluator import DictWeightResolver, TorchEvaluator
from triton.flagmega.ir.ops.distributed.materialize_local_shards import (
    MaterializeLocalShards,
)


class MaterializeShardsModule(fm.Module):
    def __init__(self, value_type):
        super().__init__(dialect="high_level", stage="imported", entry="main")
        self.value_type = value_type

    def forward(self):
        value = self.input("value", self.value_type)
        shards = fm.F.distributed.materialize_local_shards(
            value,
            name="shards",
        )
        self.function("main", (value,), (shards,))


def _distributed_type(dtype="float32"):
    placement = fm.Placement((2, 2), "yx", "bb")
    return fm.DistributedType(
        fm.tensor_type(dtype, (4, 8)),
        (
            fm.SBP.split_contiguous((0,), 2),
            fm.SBP.split_block_cyclic((1,), 2),
        ),
        placement,
    )


def test_materialize_local_shards_uses_owner_major_staged_sbp_order():
    module = MaterializeShardsModule(_distributed_type()).build()
    value = torch.arange(32, dtype=torch.float32).reshape(4, 8)

    actual = TorchEvaluator(DictWeightResolver({})).run(module, {"value": value})[0]
    expected = torch.stack((
        value[:2, (0, 1, 4, 5)],
        value[:2, (2, 3, 6, 7)],
        value[2:, (0, 1, 4, 5)],
        value[2:, (2, 3, 6, 7)],
    ))
    torch.testing.assert_close(actual, expected)
    assert module.node_map["shards"].type == fm.tensor_type("float32", (4, 2, 4))


def test_materialize_local_shards_preserves_vector_lanes_and_round_trips(tmp_path):
    source = _distributed_type(fm.VectorType(fm.DType.BFLOAT16, (2,)))
    module = MaterializeShardsModule(source).build()
    value = torch.arange(64, dtype=torch.bfloat16).reshape(4, 8, 2)

    actual = TorchEvaluator(DictWeightResolver({})).run(module, {"value": value})[0]
    assert actual.shape == (4, 2, 4, 2)
    checkpoint = fm.emit_module(module, tmp_path / "materialize_shards.py")
    assert fm.load_module(checkpoint).semantic_hash == module.semantic_hash


def test_materialize_local_shards_has_parameter_addressable_pattern():
    module = MaterializeShardsModule(_distributed_type()).build()
    value = pm.wildcard("value")
    pattern = pm.F.distributed.is_materialize_local_shards(
        value,
        call_name="materialization",
    )

    match = pm.try_match_root(module.node_map["shards"], pattern, module)
    assert match is not None
    assert match["value"].id == module.function_map["main"].parameters[0]
    assert match["materialization"].id == "shards"


def test_materialize_local_shards_pads_only_the_short_block_cyclic_owner():
    value_type = fm.DistributedType(
        fm.tensor_type("float32", (10,)),
        (fm.SBP.split_block_cyclic((0,), 4),),
        fm.Placement((3,), "x", "b"),
    )
    module = MaterializeShardsModule(value_type).build()
    value = torch.arange(10, dtype=torch.float32)

    actual = TorchEvaluator(DictWeightResolver({})).run(
        module, {"value": value}
    )[0]
    torch.testing.assert_close(actual, torch.tensor((
        (0, 1, 2, 3),
        (4, 5, 6, 7),
        (8, 9, 0, 0),
    ), dtype=torch.float32))


def test_materialize_local_shards_applies_multiple_stages_on_one_axis():
    value_type = fm.DistributedType(
        fm.tensor_type("float32", (8,)),
        (fm.SBP.split(
            fm.SplitStage.contiguous((0,), 4),
            fm.SplitStage.block_cyclic((1,), 1),
        ),),
        fm.Placement((2, 2), "yx", "bb"),
    )
    module = MaterializeShardsModule(value_type).build()
    value = torch.arange(8, dtype=torch.float32)

    actual = TorchEvaluator(DictWeightResolver({})).run(
        module, {"value": value}
    )[0]
    torch.testing.assert_close(actual, torch.tensor((
        (0, 2),
        (1, 3),
        (4, 6),
        (5, 7),
    ), dtype=torch.float32))


def test_materialize_local_shards_rejects_partial_values():
    value_type = fm.DistributedType(
        fm.tensor_type("float32", (4,)),
        (fm.SBP.broadcast(),),
        fm.Placement((2,), "x", "b"),
        partial=fm.SBP.partial((0,)),
    )

    with pytest.raises(IRSchemaError, match="cannot materialize a partial"):
        MaterializeShardsModule(value_type).build()


def test_materialize_local_shards_does_not_broadcast_lift_a_broadcast_input():
    placement = fm.Placement((2, 4), "yx", "bb")
    source_type = fm.DistributedType(
        fm.tensor_type("bfloat16", (3, 8)),
        (fm.SBP.broadcast(), fm.SBP.broadcast()),
        placement,
    )
    source = fm.Node(
        "value", "builtin.var", (), source_type, attrs={"name": "value"}
    )

    prepared = MaterializeLocalShards.prepare((source,), {})

    assert prepared.result_type == fm.tensor_type("bfloat16", (8, 3, 8))
