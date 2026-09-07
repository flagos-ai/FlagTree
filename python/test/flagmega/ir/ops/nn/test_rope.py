# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT

import pytest
import torch

from triton.flagmega import ir as fm
from triton.flagmega.errors import IRSchemaError
from triton.flagmega.evaluator import DictWeightResolver, TorchEvaluator
from triton.flagmega import pattern_match as pm


class _RoPEGraph(fm.Module):
    def __init__(self):
        super().__init__(dialect="high_level", stage="imported", entry="main")

    def forward(self):
        value = self.input("value", fm.tensor_type("bfloat16", (1, 2, 4)))
        cos = self.input("cos", fm.tensor_type("float32", (1, 1, 4)))
        sin = self.input("sin", fm.tensor_type("float32", (1, 1, 4)))
        result = fm.F.nn.rope(value, cos, sin, name="rope")
        self.function("main", (value, cos, sin), (result,))


def test_rope_broadcasts_position_pair_and_rotates_half_dimensions():
    module = _RoPEGraph().build()
    value = torch.arange(8, dtype=torch.float32).reshape(1, 2, 4).to(torch.bfloat16)
    cos = torch.full((1, 1, 4), 0.5, dtype=torch.float32)
    sin = torch.full((1, 1, 4), 0.25, dtype=torch.float32)

    result = TorchEvaluator(DictWeightResolver({})).run(
        module, {"value": value, "cos": cos, "sin": sin})[0]

    rotated = torch.cat((-value[..., 2:], value[..., :2]), dim=-1)
    expected = value * cos.to(torch.bfloat16) + rotated * sin.to(torch.bfloat16)
    torch.testing.assert_close(result, expected)
    assert result.dtype == torch.bfloat16
    assert pm.try_match_root(
        module.node_map["rope"], pm.F.nn.is_rope(call_name="rope"), module
    ) is not None


def test_rope_preserves_head_sharding_with_replicated_rotary_tables():
    placement = fm.Placement((8, 16), "yx", "bb")
    value_type = fm.DistributedType(
        fm.tensor_type("bfloat16", (1, 16, 128)),
        (fm.SBP.broadcast(), fm.SBP.split_contiguous((1,), 1), fm.SBP.broadcast()),
        placement,
    )
    rotary_type = fm.DistributedType(
        fm.tensor_type("float32", (1, 1, 128)),
        (fm.SBP.broadcast(), fm.SBP.broadcast(), fm.SBP.broadcast()),
        placement,
    )

    result = fm.get_definition("nn.rope").infer_type(
        (_typed("value", value_type), _typed("cos", rotary_type), _typed("sin", rotary_type)),
        {},
    )

    assert result == value_type


def test_rope_rejects_sharding_that_cuts_the_rotated_pair_axis():
    placement = fm.Placement((8,), "b", "b")
    value_type = fm.DistributedType(
        fm.tensor_type("bfloat16", (1, 16, 128)),
        (fm.SBP.broadcast(), fm.SBP.broadcast(), fm.SBP.split_contiguous((0,), 16)),
        placement,
    )
    rotary_type = fm.DistributedType(
        fm.tensor_type("float32", (1, 1, 128)),
        (fm.SBP.broadcast(), fm.SBP.broadcast(), fm.SBP.broadcast()),
        placement,
    )

    with pytest.raises(IRSchemaError, match="rotated dimension"):
        fm.get_definition("nn.rope").infer_type(
            (_typed("value", value_type), _typed("cos", rotary_type), _typed("sin", rotary_type)),
            {},
        )


def _typed(name: str, value_type: fm.IRType) -> fm.Node:
    return fm.Node(name, "builtin.var", (), value_type, attrs={"name": name})
