import pytest

from triton.flagmega import ir as fm
from triton.flagmega.errors import IRVerificationError


@pytest.mark.parametrize(
    "value_type",
    (
        fm.AnyType(),
        fm.NoneType(),
        fm.CallableType(
            fm.tensor_type("float32", [1]),
            (fm.tensor_type("bfloat16", [2]), fm.NoneType()),
        ),
        fm.TupleType((fm.tensor_type("int32", []),), is_variadic=True),
    ),
)
def test_extended_ir_type_data_round_trip(value_type):
    assert fm.type_from_data(value_type.to_data()) == value_type


@pytest.mark.parametrize(
    "value_type",
    (
        fm.AnyType(),
        fm.NoneType(),
        fm.CallableType(fm.NoneType(), (fm.tensor_type("int64", []),)),
        fm.TupleType((fm.tensor_type("int32", []),), is_variadic=True),
    ),
)
def test_extended_ir_types_round_trip_through_editable_python(tmp_path, value_type):
    builder = fm.IRBuilder(dialect="high_level", stage="unit")
    value = builder.var("value", value_type, id="value")
    builder.function("main", [value], [value])
    module = fm.verify_module(builder.build(entry="main"))
    path = tmp_path / "module.py"

    fm.emit_module(module, path)
    restored = fm.load_module(path)

    assert restored.node_map["value"].type == value_type
    assert restored.semantic_hash == module.semantic_hash


def test_invalid_type_is_serializable_but_rejected_by_ir_verifier():
    invalid = fm.InvalidType("type inference failed")
    assert fm.type_from_data(invalid.to_data()) == invalid
    builder = fm.IRBuilder(dialect="high_level", stage="unit")
    value = builder.var("value", invalid, id="value")
    builder.function("main", [value], [value])

    with pytest.raises(IRVerificationError, match="type inference failed"):
        fm.verify_module(builder.build(entry="main"))


def test_type_patterns_cover_extended_nncase_type_family():
    assert fm.is_any().match_leaf(fm.AnyType())
    assert fm.is_none().match_leaf(fm.NoneType())
    assert fm.is_callable().match_leaf(fm.CallableType(fm.NoneType(), ()))
    assert fm.is_invalid().match_leaf(fm.InvalidType("broken"))
