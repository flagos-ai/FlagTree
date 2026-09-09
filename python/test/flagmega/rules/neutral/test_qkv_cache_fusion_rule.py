# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT

from dataclasses import replace

import pytest

from triton.flagmega import ir as fm
from triton.flagmega.rules import DataflowRewriter
from triton.flagmega.rules.neutral.form_qkv_rope_with_cache import form_qkv_rope_with_cache_rule
from python.test.flagmega.passes.target_independent.test_form_qkv_rope_with_cache import QKVRoPERegion


@pytest.mark.parametrize("axis_attrs", [{"axes": (2, )}, {"axes": (-1, )}, {"axis": -1}, {"axis": 2}])
def test_pattern_accepts_equivalent_pack_axis_forms(axis_attrs):
    source = QKVRoPERegion(rotary_dim=32).build()
    source = replace(
        source, nodes=tuple(
            replace(n, attrs={"lanes": (8, ), **axis_attrs}) if n.op == "tensors.pack" else n for n in source.nodes))
    rule = form_qkv_rope_with_cache_rule()
    assert rule.pattern is not None and rule.matches is None
    result = DataflowRewriter((rule, )).rewrite(source)
    fused, = (node for node in result.nodes if node.op == "nn.qkv_rope_with_cache")
    assert fused.attrs["rotary_dim"] == 32


def test_pattern_rejects_shared_rotary_value():
    source = QKVRoPERegion(rotary_dim=32, extra_query_user=True).build()
    rule = form_qkv_rope_with_cache_rule()
    assert rule.apply(source.node_map["attention"], source) is None


def test_cache_fusion_may_not_cross_an_intervening_state_read():
    source = QKVRoPERegion(rotary_dim=32).build()
    definition = fm.get_definition("nn.rotary_embedding")
    attrs = definition.normalize_attrs({"head_dim": 32, "theta": 10000.0})
    q = next(n for n in source.nodes if n.op == "builtin.var" and n.attrs.get("name") == "q")
    state = source.node_map[source.node_map["key_state"].inputs[1]]
    reference = fm.Node("reference", "tensors.reshape", (q.id, ), fm.tensor_type("bfloat16", (1, 128)),
                        attrs={"shape": (1, 128)})
    inputs = (reference, state)
    observer = fm.Node("observer", definition.op_name, tuple(n.id for n in inputs),
                       definition.infer_call_type(inputs, attrs), definition.infer_effect(inputs, attrs),
                       definition.ir_attrs(attrs))
    index = next(i for i, n in enumerate(source.nodes) if n.id == "updated")
    source = fm.verify_module(replace(source,
                                      nodes=(*source.nodes[:index], reference, observer, *source.nodes[index:])))
    result = DataflowRewriter((form_qkv_rope_with_cache_rule(), )).rewrite(source)
    assert not any(n.op == "nn.qkv_rope_with_cache" for n in result.nodes)
    assert [n.id for n in result.nodes if not n.effect.is_pure] == ["key_state", "observer", "updated", "attention"]
