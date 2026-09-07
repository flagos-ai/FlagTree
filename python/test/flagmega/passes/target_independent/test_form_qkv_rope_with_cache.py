# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT

from dataclasses import replace

import pytest

from triton.flagmega import ir as fm
from triton.flagmega.passes.form_qkv_rope_with_cache import (
    form_qkv_rope_with_cache,
)
from triton.flagmega.ir.ops.nn._paged_attention_state import (
    PagedAttentionStateConfig,
)


class QKVRoPERegion(fm.Module):
    def __init__(self, *, extra_query_user: bool = False):
        super().__init__(
            dialect="high_level",
            stage="normalization_decomposed",
            entry="main",
        )
        self.extra_query_user = extra_query_user

    def forward(self):
        q_type = fm.tensor_type("bfloat16", (1, 2, 64))
        kv_type = fm.tensor_type("bfloat16", (1, 1, 64))
        parameter_type = fm.tensor_type("bfloat16", (64,))
        trig_type = fm.tensor_type("float32", (1, 1, 64))
        q = self.input("q", q_type)
        k = self.input("k", kv_type)
        v = self.input("v", kv_type)
        q_scale = self.input("q_scale", parameter_type)
        k_scale = self.input("k_scale", parameter_type)
        q_bias = self.input("q_bias", parameter_type)
        k_bias = self.input("k_bias", parameter_type)
        cos = self.input("cos", trig_type)
        sin = self.input("sin", trig_type)
        state = self.input(
            "state",
            PagedAttentionStateConfig(
                1, 1, 64, block_size=4, num_blocks=2, lanes=8
            ).ref_type,
        )
        layer_id = self.input("layer_id", fm.tensor_type("int32", ()))
        advance = self.input("advance", fm.tensor_type("bool", ()))
        q_stats = fm.F.nn.norm_stats(
            q, axis=-1, use_mean=False, name="q_stats"
        )
        q_norm = fm.F.nn.norm_apply(
            q,
            q_stats,
            q_scale,
            q_bias,
            axis=-1,
            epsilon=1e-6,
            use_mean=False,
            name="q_norm",
        )
        k_stats = fm.F.nn.norm_stats(
            k, axis=-1, use_mean=False, name="k_stats"
        )
        k_norm = fm.F.nn.norm_apply(
            k,
            k_stats,
            k_scale,
            k_bias,
            axis=-1,
            epsilon=1e-6,
            use_mean=False,
            name="k_norm",
        )
        q_rope = fm.F.nn.rope(q_norm, cos, sin, name="q_rope")
        k_rope = fm.F.nn.rope(k_norm, cos, sin, name="k_rope")
        q_view = fm.F.tensors.pack(
            q_rope, lanes=(8,), axes=(2,), name="q_cache_pack"
        )
        k_view = fm.F.tensors.pack(
            k_rope, lanes=(8,), axes=(2,), name="k_cache_pack"
        )
        v_view = fm.F.tensors.pack(
            v, lanes=(8,), axes=(2,), name="v_cache_pack"
        )
        no_advance = fm.F.builtin.scalar_const(
            fm.tensor_type("bool", ()), False, name="no_advance"
        )
        key_state = fm.F.nn.update_paged_attention_kv_cache(
            k_view,
            state,
            layer_id,
            no_advance,
            cache_kind="key",
            layout=("seq", "head", "dim"),
            name="key_state",
        )
        updated = fm.F.nn.update_paged_attention_kv_cache(
            v_view,
            key_state,
            layer_id,
            advance,
            cache_kind="value",
            layout=("seq", "head", "dim"),
            name="updated",
        )
        attention = fm.F.nn.paged_attention(
            q_view,
            updated,
            layer_id,
            scale=0.125,
            layout=("seq", "head", "dim"),
            hidden_size=128,
            name="attention",
        )
        outputs = [attention, updated]
        if self.extra_query_user:
            outputs.append(fm.F.math.add(q_rope, q_rope, name="extra_query"))
        self.function(
            "main",
            (
                q,
                k,
                v,
                q_scale,
                k_scale,
                q_bias,
                k_bias,
                cos,
                sin,
                state,
                layer_id,
                advance,
            ),
            tuple(outputs),
        )


def test_pass_forms_one_effectful_semantic_region_and_preserves_boundaries(tmp_path):
    source = QKVRoPERegion().build()

    result = form_qkv_rope_with_cache(source)

    fused = [node for node in result.nodes if node.op == "nn.qkv_rope_with_cache"]
    assert len(fused) == 1
    fused = fused[0]
    assert fused.effect == fm.effect("read_write", "paged_attention_kv_cache")
    assert fused.inputs[-3:] == (
        source.node_map["key_state"].inputs[1],
        source.node_map["attention"].inputs[2],
        source.node_map["updated"].inputs[3],
    )
    assert fused.attrs == {
        "q_axis": -1,
        "q_epsilon": 1e-6,
        "q_use_mean": False,
        "q_round_before_scale": False,
        "k_axis": -1,
        "k_epsilon": 1e-6,
        "k_use_mean": False,
        "k_round_before_scale": False,
        "qkv_layout": ("seq", "head", "dim"),
        "attention_layout": ("seq", "head", "dim"),
    }
    assert not [
        node
        for node in result.nodes
        if node.op == "nn.update_paged_attention_kv_cache"
    ]
    assert result.node_map["updated"].op == "builtin.get_item"
    assert result.node_map["updated"].inputs == (fused.id,)
    assert result.node_map["attention"].inputs[:2] == (
        f"{fused.id}.query",
        "updated",
    )
    assert result.node_map["attention"].type == source.node_map["attention"].type
    assert result.node_map["updated"].type == source.node_map["updated"].type
    path = fm.emit_module(result, tmp_path / "formed.py")
    assert fm.load_module(path) == result


def test_pass_rejects_region_when_query_rope_has_an_external_user():
    source = QKVRoPERegion(extra_query_user=True).build()

    result = form_qkv_rope_with_cache(source)

    assert not [node for node in result.nodes if node.op == "nn.qkv_rope_with_cache"]
    assert len([
        node
        for node in result.nodes
        if node.op == "nn.update_paged_attention_kv_cache"
    ]) == 2


def test_pass_allocates_fresh_ids_instead_of_looping_on_agent_name_collision():
    source = QKVRoPERegion().build()
    scalar = source.node_map["no_advance"]
    collisions = (
        replace(scalar, id="updated.qkv"),
        replace(scalar, id="updated.qkv_rope_with_cache"),
        replace(scalar, id="updated.qkv_rope_with_cache_1.query"),
    )
    edited = fm.verify_module(replace(source, nodes=source.nodes + collisions))

    result = form_qkv_rope_with_cache(edited)

    fused = tuple(
        node for node in result.nodes if node.op == "nn.qkv_rope_with_cache"
    )
    assert len(fused) == 1
    assert fused[0].id == "updated.qkv_rope_with_cache_1"
    assert result.node_map["attention"].inputs[0] == (
        "updated.qkv_rope_with_cache_1.query_1"
    )


@pytest.mark.parametrize("q_round,k_round", [(False, True), (True, False), (True, True)])
def test_qkv_fusion_preserves_independent_normalization_rounding(q_round, k_round, tmp_path):
    source = QKVRoPERegion().build()
    policies = {"q_norm": q_round, "k_norm": k_round}
    source = replace(source, nodes=tuple(
        replace(node, attrs={**node.attrs, "round_before_scale": policies[node.id]})
        if node.id in policies else node for node in source.nodes
    ))
    result = form_qkv_rope_with_cache(source)
    fused = next(node for node in result.nodes if node.op == "nn.qkv_rope_with_cache")
    assert fused.attrs["q_round_before_scale"] is q_round
    assert fused.attrs["k_round_before_scale"] is k_round
    assert fm.load_module(fm.emit_module(result, tmp_path / "qkv.py")) == result
