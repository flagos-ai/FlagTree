# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT

from __future__ import annotations

import inspect
from pathlib import Path
from typing import get_type_hints

import pytest

from triton.flagmega import ir as fm
from triton.flagmega import pattern_match as pm
from triton.flagmega.errors import IRSchemaError, IRVerificationError
from triton.flagmega.rules import RewriteRule


def make_graph():
    class Graph(fm.Module):
        def __init__(self):
            super().__init__(dialect="high_level", stage="imported", entry="main")

        def forward(self):
            source = self.input("source", fm.tensor_type("bfloat16", [1, 16]), id="source")
            added = fm.F.math.add(source, source, name="added")
            output = fm.F.math.silu(added, name="output")
            self.function("main", [source], [output])

    return Graph().build()


def test_type_pattern_is_composable_and_reports_reason():
    rank2_bf16 = fm.is_tensor() & fm.has_rank(2) & fm.has_dtype("bfloat16")

    assert rank2_bf16.match_leaf(fm.tensor_type("bfloat16", [1, 16]))
    assert not rank2_bf16.match_leaf(fm.tensor_type("float32", [1, 16]))
    assert (fm.is_tuple() | fm.is_ref()).match_leaf(fm.TupleType(()))
    with pytest.raises(IRSchemaError, match="parameter.*requires.*rank = 2"):
        rank2_bf16.check(fm.tensor_type("bfloat16", [16]), "parameter")


def test_parameter_info_owns_name_type_pattern_kind_and_effect():
    add = fm.get_definition("math.add")
    recurrent = fm.get_definition("nn.gdn_recurrent_core")

    assert add.lhs.name == "lhs"
    assert add.lhs.index == 0
    assert add.lhs.input_index == 0
    assert add.lhs.type_pattern.reason == "is_tensor"
    assert add.lhs.pattern is add.lhs.type_pattern
    assert add.lhs.kind == fm.ParameterKind.INPUT
    assert recurrent.state.name == "state"
    assert recurrent.state.type_pattern.reason == "is_ref"
    assert recurrent.state.memory_effect == fm.MemoryEffect.READ_WRITE

    definitions = fm.definitions()
    assert all(
        parameter.name and isinstance(parameter.type_pattern, fm.TypePattern)
        for definition in definitions
        for parameter in definition.input_parameters
    )
    op_root = Path(__file__).parents[2] / "triton" / "flagmega" / "ir" / "ops"
    assert "input_parameter()" not in "".join(
        path.read_text(encoding="utf-8")
        for path in op_root.rglob("*.py")
        if path.name != "core.py"
    )


def test_functional_apis_and_emitted_python_are_canonical_snake_case():
    module = make_graph()
    source = fm.module_source(module)

    assert "F.math.add(" in source
    assert "F.math.silu(" in source
    assert not hasattr(fm.F, "Math")
    assert tuple(inspect.signature(fm.F.math.add).parameters) == ("lhs", "rhs", "name", "metadata")
    assert tuple(inspect.signature(pm.F.math.is_add).parameters) == (
        "lhs", "rhs", "target_name", "call_name", "condition",
    )
    pattern = pm.F.math.is_add()
    assert pattern[fm.get_definition("math.add").lhs].type_pattern.reason == "is_tensor"


def test_functional_apis_are_static_typed_source_and_cover_every_public_op():
    runtime_source = Path(inspect.getsourcefile(fm.F) or "").read_text(encoding="utf-8")
    pattern_source = Path(inspect.getsourcefile(pm.F) or "").read_text(encoding="utf-8")

    assert "def silu(" in runtime_source
    assert "def is_silu(" in pattern_source
    assert "_materialize_functionals" not in runtime_source + pattern_source
    assert "setattr(" not in runtime_source + pattern_source
    assert get_type_hints(fm.F.math.silu)["value"] is fm.Node
    assert get_type_hints(fm.F.nn.rms_norm)["epsilon"] is float

    for definition in fm.definitions():
        if definition.namespace is None:
            continue
        runtime = getattr(getattr(fm.F, definition.namespace), definition.functional_name)
        pattern = getattr(getattr(pm.F, definition.namespace), f"is_{definition.functional_name}")
        declared = tuple(parameter.name for parameter in definition.parameters)
        assert tuple(inspect.signature(runtime).parameters) == (*declared, "name", "metadata")
        assert tuple(inspect.signature(pattern).parameters) == (
            *declared, "target_name", "call_name", "condition",
        )
        assert runtime.__flagmega_op_definition__ is definition
        assert pattern.__flagmega_op_definition__ is definition


def test_handwritten_functional_convenience_variants_build_and_match():
    class Graph(fm.Module):
        def __init__(self):
            super().__init__(dialect="high_level", stage="imported", entry="main")

        def forward(self):
            source = self.input("source", fm.tensor_type("bfloat16", [1, 16]), id="source")
            gated = fm.F.math.silu_mul(
                source,
                source,
                silu_name="activated",
                name="gated",
            )
            pair = fm.F.tir.kernel(
                source,
                result_type=fm.TupleType((source.type, source.type)),
                semantic_op="test.pair",
                candidate="reference",
                parameters={},
                facts={},
                semantic_attrs={},
                name="pair",
            )
            first, second = fm.F.tensors.get_items(pair, 0, 1, name_prefix="field")
            self.function("main", [source], [gated, first, second])

    module = Graph().build()
    fm.verify_module(module)

    source = pm.wildcard("source")
    result = pm.try_match_root(
        module.node_map["gated"],
        pm.F.math.is_silu_mul(source, source, silu_name="activation", mul_name="gated"),
        module,
    )
    assert result is not None
    assert result["source"].id == "source"
    assert result["activation"].id == "activated"
    assert result["gated"].id == "gated"
    assert pm.try_match_root(
        module.node_map["field_0"],
        pm.F.tensors.is_get_item_at(0, call_name="first"),
        module,
    )["first"].id == "field_0"


def test_handwritten_tensor_recipe_and_sampling_patterns_match_attributes():
    class Graph(fm.Module):
        def __init__(self):
            super().__init__(dialect="high_level", stage="imported", entry="main")

        def forward(self):
            lhs = self.input("lhs", fm.tensor_type("bfloat16", [1, 4]), id="lhs")
            rhs = self.input("rhs", fm.tensor_type("bfloat16", [1, 4]), id="rhs")
            joined = fm.F.tensors.concat(lhs, rhs, axis=1, name="joined")
            reshaped = fm.F.tensors.reshape(joined, (2, 4), name="reshaped")
            permuted = fm.F.tensors.permute(reshaped, (1, 0), name="permuted")
            token = fm.F.nn.greedy_sample(joined, name="token")
            self.function("main", [lhs, rhs], [permuted, token])

    module = Graph().build()
    lhs = pm.wildcard("lhs")
    rhs = pm.wildcard("rhs")
    recipe = pm.F.tensors.is_permute(
        pm.F.tensors.is_reshape(
            pm.F.tensors.is_concat(lhs, rhs, axis=1, call_name="concat"),
            shape=(2, 4),
        ),
        axes=(1, 0),
    )

    match = pm.try_match_root(module.node_map["permuted"], recipe, module)
    assert match is not None
    assert match["lhs"].id == "lhs"
    assert match["rhs"].id == "rhs"
    assert match["concat"].id == "joined"
    assert pm.try_match_root(
        module.node_map["token"],
        pm.F.nn.is_greedy_sample(pm.wildcard("logits")),
        module,
    )["logits"].id == "joined"


def test_commutative_pattern_variant_matches_reversed_operands():
    builder = fm.IRBuilder(dialect="high_level", stage="imported")
    tensor = fm.tensor_type("bfloat16", [1, 8])
    lhs = builder.var("lhs", tensor, id="lhs")
    rhs = builder.var("rhs", tensor, id="rhs")
    added = builder.call("math.add", [rhs, lhs], tensor, id="added")
    builder.function("main", [lhs, rhs], [added])
    module = builder.build(entry="main")
    fm.verify_module(module)
    lhs_pattern = pm.wildcard("lhs", lambda node: node.id == "lhs")
    rhs_pattern = pm.wildcard("rhs", lambda node: node.id == "rhs")

    assert pm.try_match_root(
        added, pm.F.math.is_add(lhs_pattern, rhs_pattern), module) is None
    assert pm.try_match_root(
        added,
        pm.F.math.is_add_commutative(lhs_pattern, rhs_pattern, call_name="add"),
        module,
    )["add"].id == "added"


def test_parameter_type_contract_is_used_by_construction_and_verifier():
    class InvalidGraph(fm.Module):
        def __init__(self):
            super().__init__(dialect="high_level", stage="imported", entry="main")

        def forward(self):
            state = self.input("state", fm.RefType("state"), id="state")
            fm.F.math.silu(state, name="invalid")

    with pytest.raises(IRSchemaError, match="math.silu.value.*is_tensor"):
        InvalidGraph().build()

    builder = fm.IRBuilder(dialect="high_level", stage="imported")
    state = builder.var("state", fm.RefType("state"), id="state")
    invalid = builder.call("math.silu", [state], state.type, id="invalid")
    builder.function("main", [state], [invalid])
    module = builder.build(entry="main")
    with pytest.raises(IRVerificationError, match="math.silu.value.*is_tensor"):
        fm.verify_module(module)


def test_nested_call_pattern_uses_parameter_info_and_named_captures():
    module = make_graph()
    source = pm.wildcard("source", type_pattern=fm.is_tensor() & fm.has_rank(2))
    add = pm.F.math.is_add(source, source, target_name="add_op", call_name="add")
    pattern = pm.F.math.is_silu(add, call_name="root")

    result = pm.try_match_root(module.node_map["output"], pattern, module)

    assert result is not None
    assert result.root.id == "output"
    assert result["source"].id == "source"
    assert result["add"].id == "added"
    assert result["add_op"].op == "math.add"
    assert result["root"].id == "output"
    assert add[fm.get_definition("math.add").lhs] is source


def test_ordered_alternative_and_pattern_driven_rewrite_rule():
    module = make_graph()
    pattern = pm.is_alt(
        pm.F.math.is_mul(call_name="mul"),
        pm.F.math.is_silu(call_name="activation"),
    )
    result = pm.try_match(module, pattern)

    assert result is not None
    assert result["activation"].id == "output"

    rule = RewriteRule(
        "return_input",
        pm.F.math.is_silu(pm.wildcard("input")),
        lambda match, _: match["input"],
    )
    replacement = rule.apply(module.node_map["output"], module)
    assert replacement is not None
    assert replacement.id == "added"


def test_generated_call_pattern_rejects_invalid_operand_type_even_without_verifier():
    builder = fm.IRBuilder(dialect="high_level", stage="imported")
    state = builder.var("state", fm.RefType("state"), id="state")
    invalid = builder.call("math.silu", [state], state.type, id="invalid")
    builder.function("main", [state], [invalid])
    module = builder.build(entry="main")

    assert pm.try_match_root(invalid, pm.F.math.is_silu(), module) is None
