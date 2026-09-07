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


class _UpdateGraph(fm.Module):
    def __init__(self):
        super().__init__(dialect="high_level", stage="imported", entry="main")

    def forward(self):
        slots = self.input("slots", fm.tensor_type("bfloat16", (2, 1, 8)))
        state = self.input(
            "state", PagedAttentionStateConfig(1, 2, 8, block_size=4, num_blocks=2).ref_type)
        layer = self.input("layer", fm.tensor_type("int32", ()))
        advance = self.input("advance", fm.tensor_type("bool", ()))
        result = fm.F.nn.update_paged_attention_kv_cache(
            slots,
            state,
            layer,
            advance,
            cache_kind="key",
            layout=("head", "seq", "dim"),
            name="update",
        )
        self.function("main", (slots, state, layer, advance), (result,))


def test_update_paged_attention_cache_obeys_layout_and_reference_identity():
    module = _UpdateGraph().build()
    state = create_paged_attention_state(
        PagedAttentionStateConfig(1, 2, 8, block_size=4, num_blocks=2))
    slots = torch.arange(16, dtype=torch.float32).reshape(2, 1, 8).to(torch.bfloat16)

    result = TorchEvaluator(DictWeightResolver({})).run(
        module,
        {
            "slots": slots,
            "state": state,
            "layer": torch.tensor(0, dtype=torch.int32),
            "advance": torch.tensor(False),
        },
    )[0]

    assert result is state
    assert state.sequence_length == 0
    assert int(state.slot_mapping[0]) == 0
    torch.testing.assert_close(
        state.kv_caches[0, 0, 0, 0].reshape(2, 8), slots[:, 0, :])
    definition = fm.get_definition("nn.update_paged_attention_kv_cache")
    assert [value.name for value in definition.input_parameters] == [
        "slots", "state", "layer_id", "advance_sequence",
    ]
    assert pm.try_match_root(
        module.node_map["update"],
        pm.F.nn.is_update_paged_attention_kv_cache(
            cache_kind="key",
            layout=("head", "seq", "dim"),
            call_name="update",
        ),
        module,
    ) is not None


def test_cache_update_accepts_head_sharded_slots_and_keeps_reference_identity():
    placement = fm.Placement((8,), "b", "b")
    slots_type = fm.DistributedType(
        fm.tensor_type("bfloat16", (1, 8, 128)),
        (fm.SBP.broadcast(), fm.SBP.split_contiguous((0,), 1), fm.SBP.broadcast()),
        placement,
    )
    state_type = PagedAttentionStateConfig(
        1, 8, 128, block_size=4, num_blocks=2
    ).ref_type

    result = fm.get_definition("nn.update_paged_attention_kv_cache").infer_type(
        (
            _typed("slots", slots_type),
            _typed("state", state_type),
            _typed("layer", fm.DistributedType(fm.tensor_type("int32", ()), (), placement)),
            _typed("advance", fm.DistributedType(fm.tensor_type("bool", ()), (), placement)),
        ),
        {"cache_kind": "key", "layout": ("seq", "head", "dim")},
    )

    assert result == state_type


def test_cache_update_rejects_a_split_head_dimension():
    placement = fm.Placement((8,), "b", "b")
    slots_type = fm.DistributedType(
        fm.tensor_type("bfloat16", (1, 8, 128)),
        (fm.SBP.broadcast(), fm.SBP.broadcast(), fm.SBP.split_contiguous((0,), 16)),
        placement,
    )
    state_type = PagedAttentionStateConfig(
        1, 8, 128, block_size=4, num_blocks=2
    ).ref_type

    with pytest.raises(IRSchemaError, match="head-dimension axis"):
        fm.get_definition("nn.update_paged_attention_kv_cache").infer_type(
            (
                _typed("slots", slots_type),
                _typed("state", state_type),
                _typed("layer", fm.DistributedType(fm.tensor_type("int32", ()), (), placement)),
                _typed("advance", fm.DistributedType(fm.tensor_type("bool", ()), (), placement)),
            ),
            {"cache_kind": "key", "layout": ("seq", "head", "dim")},
        )


def _typed(name: str, value_type: fm.IRType) -> fm.Node:
    return fm.Node(name, "builtin.var", (), value_type, attrs={"name": name})
