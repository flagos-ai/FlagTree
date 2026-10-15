# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: MIT
"""A predicate over dimensions is an expression, not stored data."""

import pytest
from triton.flagmega.codegen.triton.dimension_expression import emit_logical
from triton.flagmega.errors import IRSchemaError
from triton.flagmega.ir import (
    BoolConst,
    DimCompare,
    IRBuilder,
    LogicalExpr,
    dim_compare,
    dim_equal,
    logical_and,
    logical_const,
    logical_from_data,
    logical_not,
    logical_or,
    tensor_type,
    verify_module,
)
from triton.flagmega.ir.dim_expr import DimVar, dim


def iv(count=28):
    return DimVar("iv", 0, count - 1)


def test_comparison_is_boolean_and_evaluates_per_binding():
    predicate = dim_equal(iv(), 27)
    assert isinstance(predicate, DimCompare)
    assert not predicate.is_fixed
    assert predicate.evaluate({"iv": 27}) is True
    assert predicate.evaluate({"iv": 0}) is False


def test_provable_comparisons_fold_to_constants():
    assert logical_const(True).value is True
    assert dim_equal(5, 5).value is True
    assert dim_equal(5, 6).value is False
    # Two provably identical expressions compare equal without being constant.
    assert dim_equal(iv() * 2, iv() + iv()).value is True
    assert dim_compare(iv() * 2, iv() + iv(), "not_equal").value is False


def test_substitution_decides_the_predicate():
    predicate = dim_equal(iv(), 27)
    assert predicate.substitute({"iv": 27}).value is True
    assert predicate.substitute({"iv": 3}).value is False


@pytest.mark.parametrize("compare,expected", [
    ("equal", False), ("not_equal", True), ("less_than", True),
    ("less_or_equal", True), ("greater_than", False), ("greater_or_equal", False),
])
def test_every_comparison_operator_evaluates(compare, expected):
    assert dim_compare(iv(), 27, compare).evaluate({"iv": 3}) is expected


def test_connectives_short_circuit_on_known_operands():
    unknown = dim_equal(iv(), 27)
    assert logical_and(unknown, logical_const(False)).value is False
    assert logical_or(unknown, logical_const(True)).value is True
    assert logical_and(unknown, logical_const(True)).value is None
    assert logical_not(logical_const(True)).value is False
    assert (~unknown).evaluate({"iv": 27}) is False
    assert (unknown & dim_compare(iv(), 0, "greater_than")).evaluate({"iv": 27}) is True
    assert (unknown | dim_compare(iv(), 0, "equal")).evaluate({"iv": 0}) is True


def test_round_trips_through_plain_data():
    for predicate in (
        dim_equal(iv(), 27),
        logical_not(dim_compare(iv(), 4, "less_than")),
        logical_and(dim_equal(iv(), 1), dim_compare(iv(), 9, "less_or_equal")),
        logical_const(True),
    ):
        assert logical_from_data(predicate.to_data()) == predicate


def test_emits_a_scalar_triton_predicate():
    assert emit_logical(dim_equal(iv(), 27)) == "((iv) == (27))"
    assert emit_logical(logical_const(True)) == "True"
    assert emit_logical(logical_not(dim_equal(iv(), 0))) == "(not ((iv) == (0)))"
    joined = emit_logical(logical_and(dim_equal(iv(), 1), dim_equal(iv(), 2)))
    assert joined == "(((iv) == (1)) and ((iv) == (2)))"
    # A bound symbol renders under the name the generated loop uses.
    assert emit_logical(dim_equal(iv(), 27), symbols={"iv": "layer_iv"}) == "((layer_iv) == (27))"


def test_rejects_malformed_expressions():
    with pytest.raises(IRSchemaError, match="two operands"):
        type(logical_and(dim_equal(iv(), 1), dim_equal(iv(), 2)))("and", (dim_equal(iv(), 1),))
    with pytest.raises(IRSchemaError, match="boolean expressions"):
        type(logical_and(dim_equal(iv(), 1), dim_equal(iv(), 2)))("and", (dim(1), dim(2)))
    with pytest.raises(IRSchemaError, match="Unknown logical expression"):
        logical_from_data({"kind": "nope"})


def test_a_predicate_is_a_rank_zero_bool_value(tmp_path):
    from triton.flagmega.ir import F, emit_module, load_module
    from triton.flagmega.ir.ops.core import construction_scope

    builder = IRBuilder(dialect="high_level", stage="imported")
    index = builder.var("layer", tensor_type("int64", ()), id="layer")
    with construction_scope(builder):
        advance = F.shapes.dim_as_tensor(
            tensor_type("bool", ()), dim_equal(iv(), 27), name="advance")
        layer_id = F.shapes.dim_as_tensor(
            tensor_type("int32", ()), iv(), name="layer_id")
    builder.function("main", (index,), (advance, layer_id))
    module = verify_module(builder.build(entry="main"))

    assert module.node_map["advance"].type.dtype.value == "bool"
    assert module.node_map["layer_id"].type.dtype.value == "int32"
    restored = load_module(emit_module(module, tmp_path / "logical.py"))
    assert restored.semantic_hash == module.semantic_hash
    assert restored.node_map["advance"].attrs["expression"] == dim_equal(iv(), 27)


def test_a_logical_value_must_be_typed_bool():
    from triton.flagmega.ir import F
    from triton.flagmega.ir.ops.core import construction_scope

    builder = IRBuilder(dialect="high_level", stage="imported")
    with construction_scope(builder), pytest.raises(IRSchemaError, match="typed bool"):
        F.shapes.dim_as_tensor(tensor_type("int32", ()), dim_equal(iv(), 27), name="bad")


def test_base_class_contract_is_abstract():
    with pytest.raises(NotImplementedError):
        LogicalExpr().evaluate({})
    assert BoolConst(False).value is False
