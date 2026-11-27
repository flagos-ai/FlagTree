# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT

from triton.flagmega import ir as fm
from triton.flagmega.passes import ConstantCSEPass, ConstnessAnalysis


def _duplicate_weight_module():
    builder = fm.IRBuilder(dialect="high_level", stage="tir_constants_ready")
    value_type = fm.tensor_type("float32", (2, 8))
    runtime = builder.var("runtime", value_type, id="runtime")
    first = builder.weight("shared", value_type, source="memory", key="shared", id="first")
    duplicate = builder.weight("shared", value_type, source="memory", key="shared", id="duplicate")
    first_silu = builder.call("math.silu", (first,), value_type, id="first_silu")
    duplicate_silu = builder.call("math.silu", (duplicate,), value_type, id="duplicate_silu")
    combined = builder.call("math.add", (first_silu, duplicate_silu), value_type, id="combined")
    output = builder.call("math.add", (runtime, combined), value_type, id="output")
    builder.function("main", (runtime,), (output,))
    return builder.build(entry="main")


def test_constant_cse_deduplicates_sources_and_entire_constant_expressions():
    result = ConstantCSEPass().run(_duplicate_weight_module())

    assert "duplicate" not in result.node_map
    assert "duplicate_silu" not in result.node_map
    assert result.node_map["combined"].inputs == ("first_silu", "first_silu")
    assert ConstnessAnalysis.analyze(result).is_constant("combined")


def test_constant_cse_is_stable_at_a_fixed_point():
    once = ConstantCSEPass().run(_duplicate_weight_module())
    twice = ConstantCSEPass().run(once)

    assert twice == once


def test_constant_cse_preserves_unchanged_node_identities():
    module = _duplicate_weight_module()
    result = ConstantCSEPass().run(module)
    for identity in ("runtime", "first", "first_silu", "output"):
        assert result.node_map[identity] is module.node_map[identity]
    assert result.node_map["combined"] is not module.node_map["combined"]


def test_constant_cse_encodes_shared_type_once(monkeypatch):
    from triton.flagmega.ir import model

    module = _duplicate_weight_module()
    value_type = module.node_map["first"].type
    model._SEMANTIC_COMPONENT_JSON.pop(id(value_type), None)
    original = fm.TensorType.to_data
    encoded = []

    def record(self):
        if self is value_type:
            encoded.append(self)
        return original(self)

    monkeypatch.setattr(fm.TensorType, "to_data", record)
    ConstantCSEPass().run(module)
    assert len(encoded) == 1
