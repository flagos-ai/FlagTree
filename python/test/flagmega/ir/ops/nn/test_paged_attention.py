# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT

import pytest
import torch

from triton.flagmega import ir as fm
from triton.flagmega.errors import IRSchemaError
from triton.flagmega.evaluator import (
    DictWeightResolver,
    PagedAttentionStateConfig,
    TorchEvaluator,
    create_paged_attention_state,
)
from triton.flagmega import pattern_match as pm


class _PagedAttentionGraph(fm.Module):
    def __init__(self):
        super().__init__(dialect="high_level", stage="imported", entry="main")

    def forward(self):
        query = self.input("query", fm.tensor_type("bfloat16", (1, 4, 8)))
        state = self.input(
            "state", PagedAttentionStateConfig(1, 2, 8, block_size=4, num_blocks=2).ref_type)
        layer = self.input("layer", fm.tensor_type("int32", ()))
        result = fm.F.nn.paged_attention(
            query,
            state,
            layer,
            scale=8 ** -0.5,
            layout=("seq", "head", "dim"),
            hidden_size=32,
            name="attention",
        )
        self.function("main", (query, state, layer), (result,))


def test_paged_attention_reads_latest_slot_and_expands_gqa_heads():
    module = _PagedAttentionGraph().build()
    state = create_paged_attention_state(
        PagedAttentionStateConfig(1, 2, 8, block_size=4, num_blocks=2))
    key = torch.arange(16, dtype=torch.float32).reshape(2, 8).to(torch.bfloat16)
    value = (torch.arange(16, dtype=torch.float32).reshape(2, 8) + 10).to(torch.bfloat16)
    state.append(key, value, layer_id=0, advance_sequence=True)
    query = torch.randn((1, 4, 8), generator=torch.Generator().manual_seed(9)).to(
        torch.bfloat16)

    result = TorchEvaluator(DictWeightResolver({})).run(
        module,
        {
            "query": query,
            "state": state,
            "layer": torch.tensor(0, dtype=torch.int32),
        },
    )[0]

    expected = value[[0, 0, 1, 1]].reshape(1, 4, 8)
    torch.testing.assert_close(result, expected)
    assert pm.try_match_root(
        module.node_map["attention"],
        pm.F.nn.is_paged_attention(
            hidden_size=32,
            layout=("seq", "head", "dim"),
            call_name="attention",
        ),
        module,
    ) is not None


def test_paged_attention_preserves_head_sharding():
    placement = fm.Placement((8,), "b", "b")
    query_type = fm.DistributedType(
        fm.tensor_type("bfloat16", (1, 16, 128)),
        (fm.SBP.broadcast(), fm.SBP.split_contiguous((0,), 2), fm.SBP.broadcast()),
        placement,
    )
    state_type = PagedAttentionStateConfig(
        1, 8, 128, block_size=4, num_blocks=2
    ).ref_type

    result = fm.get_definition("nn.paged_attention").infer_type(
        (
            _typed("query", query_type),
            _typed("state", state_type),
            _typed("layer", fm.DistributedType(fm.tensor_type("int32", ()), (), placement)),
        ),
        {"scale": 128**-0.5, "layout": ("seq", "head", "dim"), "hidden_size": 2048},
    )

    assert result == query_type


def test_paged_attention_rejects_a_split_head_dimension():
    placement = fm.Placement((8,), "b", "b")
    query_type = fm.DistributedType(
        fm.tensor_type("bfloat16", (1, 16, 128)),
        (fm.SBP.broadcast(), fm.SBP.broadcast(), fm.SBP.split_contiguous((0,), 16)),
        placement,
    )
    state_type = PagedAttentionStateConfig(
        1, 8, 128, block_size=4, num_blocks=2
    ).ref_type

    with pytest.raises(IRSchemaError, match="head-dimension axis"):
        fm.get_definition("nn.paged_attention").infer_type(
            (
                _typed("query", query_type),
                _typed("state", state_type),
                _typed("layer", fm.DistributedType(fm.tensor_type("int32", ()), (), placement)),
            ),
            {"scale": 128**-0.5, "layout": ("seq", "head", "dim"), "hidden_size": 2048},
        )


def _typed(name: str, value_type: fm.IRType) -> fm.Node:
    return fm.Node(name, "builtin.var", (), value_type, attrs={"name": name})
