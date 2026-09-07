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
    def __init__(self, *, extra_query_user: bool = False, wide: bool = False):
        super().__init__(
            dialect="high_level",
            stage="normalization_decomposed",
            entry="main",
        )
        self.extra_query_user = extra_query_user
        self.wide = wide

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
        cos_input, sin_input = cos, sin
        state = self.input(
            "state",
            PagedAttentionStateConfig(
                1, 1, 64, block_size=4, num_blocks=2, lanes=8
            ).ref_type,
        )
        layer_id = self.input("layer_id", fm.tensor_type("int32", ()))
        advance = self.input("advance", fm.tensor_type("bool", ()))
        q_input, k_input = q, k
        if self.wide:
            q_input = fm.F.tensors.cast(q, dtype="float32", name="q_wide")
            k_input = fm.F.tensors.cast(k, dtype="float32", name="k_wide")
            cos = fm.F.tensors.cast(fm.F.tensors.cast(cos, dtype="bfloat16", name="cos_bf16"),
                                    dtype="float32", name="cos_wide")
            sin = fm.F.tensors.cast(fm.F.tensors.cast(sin, dtype="bfloat16", name="sin_bf16"),
                                    dtype="float32", name="sin_wide")
        q_stats = fm.F.nn.norm_stats(
            q_input, axis=-1, use_mean=False, name="q_stats"
        )
        q_norm = fm.F.nn.norm_apply(
            q_input,
            q_stats,
            q_scale,
            q_bias,
            axis=-1,
            epsilon=1e-6,
            use_mean=False,
            name="q_norm",
        )
        k_stats = fm.F.nn.norm_stats(
            k_input, axis=-1, use_mean=False, name="k_stats"
        )
        k_norm = fm.F.nn.norm_apply(
            k_input,
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
        if self.wide:
            q_rope = fm.F.tensors.cast(q_rope, dtype="bfloat16", name="q_rounded")
            k_rope = fm.F.tensors.cast(k_rope, dtype="bfloat16", name="k_rounded")
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
                cos_input,
                sin_input,
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


def test_wide_qkv_is_formed_by_the_regular_pass_and_round_trips(tmp_path):
    source = QKVRoPERegion(wide=True).build()
    result = form_qkv_rope_with_cache(source)
    fused = [node for node in result.nodes if node.op == "nn.qkv_rope_with_cache"]
    assert len(fused) == 1
    assert fused[0].attrs["round_qk_intermediates"] is False
    assert result.node_map[fused[0].inputs[0]].inputs == source.functions[0].parameters[:3]
    assert fused[0].inputs[5:7] == ("cos_bf16", "sin_bf16")
    assert not any(node.op == "nn.update_paged_attention_kv_cache" for node in result.nodes)
    assert fm.load_module(fm.emit_module(result, tmp_path / "wide.py")) == result
    assert form_qkv_rope_with_cache(result) == result


@pytest.mark.parametrize("wide", [False, True])
@pytest.mark.parametrize("boundary", ["q_norm", "q_rope", "q_cache_pack", "key_state"])
def test_qkv_pass_preserves_function_output_boundaries(boundary, wide):
    source = QKVRoPERegion(wide=wide).build()
    function = source.functions[0]
    source = fm.verify_module(replace(source, functions=(replace(function, outputs=(*function.outputs, boundary)),)))
    assert form_qkv_rope_with_cache(source) == source


@pytest.mark.parametrize("round_before_scale", [False, True])
def test_wide_qkv_preserves_query_and_all_cache_bytes(round_before_scale):
    import torch
    from triton.flagmega.evaluator import DictWeightResolver, TorchEvaluator, create_paged_attention_state

    source = QKVRoPERegion(wide=True).build()
    source = replace(source, nodes=tuple(
        replace(node, attrs={**node.attrs, "round_before_scale": round_before_scale})
        if node.op == "nn.norm_apply" else node for node in source.nodes))
    result = form_qkv_rope_with_cache(source)
    fused = next(node for node in result.nodes if node.op == "nn.qkv_rope_with_cache")
    evaluator = TorchEvaluator(DictWeightResolver({}))
    generator = torch.Generator().manual_seed(187)
    values = {}
    for node_id in source.functions[0].parameters:
        node = source.node_map[node_id]
        if isinstance(node.type, fm.TensorType) and node.type.dtype in {fm.DType.BFLOAT16, fm.DType.FLOAT32}:
            values[node.attrs["name"]] = torch.randn(
                tuple(dim.fixed_value for dim in node.type.shape), generator=generator
            ).to(torch.bfloat16 if node.type.dtype == fm.DType.BFLOAT16 else torch.float32)
    values.update(layer_id=torch.tensor(0, dtype=torch.int32), advance=torch.tensor(True))
    states = [create_paged_attention_state(PagedAttentionStateConfig(
        1, 1, 64, block_size=4, num_blocks=2, lanes=8)) for _ in range(2)]
    for state in states:
        state.block_table.copy_(torch.tensor([[1, 0]], dtype=torch.int32))
    original, before = evaluator.run_with_trace(source, {**values, "state": states[0]})
    optimized, after = evaluator.run_with_trace(result, {**values, "state": states[1]})
    torch.testing.assert_close(before["q_cache_pack"], after[f"{fused.id}.query"], rtol=0, atol=0)
    torch.testing.assert_close(original[0], optimized[0], rtol=0, atol=0)
    torch.testing.assert_close(states[0].kv_caches, states[1].kv_caches, rtol=0, atol=0)


def test_qkv_table_equivalence_never_equates_distinct_variables():
    from triton.flagmega.passes._qkv_head import same_table

    table = fm.Node("table", "builtin.var", (), fm.tensor_type("bfloat16", (1, 1, 64)), attrs={"name": "table"})
    assert same_table(table, table)
    assert not same_table(table, replace(table, id="other_table"))
