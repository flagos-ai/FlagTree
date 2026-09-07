# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT

from triton.flagmega import ir as fm
import pytest

from triton.flagmega.errors import IRVerificationError
from triton.flagmega.passes.tir import materialize_kernel_prim_functions


@pytest.mark.parametrize("result_effect", [fm.MemoryEffect.WRITE, fm.MemoryEffect.CHIP_WRITE])
def test_typed_dispatch_effects_are_editable_python_ir(tmp_path, monkeypatch, result_effect):
    monkeypatch.setattr(fm.get_definition("math.add"), "result_memory_effects", (result_effect,))
    builder = fm.IRBuilder(dialect="semantic_tir", stage="selected_tir")
    value_type = fm.tensor_type("bfloat16", (1, 16))
    lhs = builder.var("lhs", value_type, id="lhs")
    rhs = builder.var("rhs", value_type, id="rhs")
    result = builder.call(
        "tir.kernel", (lhs, rhs), value_type, id="result",
        attrs={
            "semantic_op": "math.add",
            "candidate": "tir.add.unit",
            "parameters": {"family": "add", "variant": "unit"},
            "facts": {},
            "semantic_attrs": {},
        },
    )
    builder.function("main", (lhs, rhs), (result,))
    module = materialize_kernel_prim_functions(builder.build(entry="main"))

    path = fm.emit_module(module, tmp_path / "selected.py")
    source = path.read_text(encoding="utf-8")
    resumed = fm.load_module(path)
    dispatch = fm.kernel_dispatch_for_call(resumed, resumed.node_map["result"])

    assert "memory_effects=" in source
    assert "fm.MemoryEffect(" in source
    assert "fm.MemoryAccessMode.READ" in source
    assert dispatch.memory_effect_map == {
        "lhs": fm.MemoryEffect.READ,
        "rhs": fm.MemoryEffect.READ,
        "result": result_effect,
    }
    assert resumed.semantic_hash == module.semantic_hash


def test_declared_result_effect_arity_is_verified_before_materialization(monkeypatch):
    monkeypatch.setattr(fm.get_definition("math.add"), "result_memory_effects", (
        fm.MemoryEffect.WRITE, fm.MemoryEffect.WRITE,
    ))
    builder = fm.IRBuilder(dialect="semantic_tir", stage="selected_tir")
    value_type = fm.tensor_type("float32", (1,))
    lhs = builder.var("lhs", value_type)
    rhs = builder.var("rhs", value_type)
    result = builder.call("tir.kernel", (lhs, rhs), value_type, attrs={
        "semantic_op": "math.add", "candidate": "tir.add.unit",
        "parameters": {"family": "add", "variant": "unit"},
        "facts": {}, "semantic_attrs": {},
    })
    builder.function("main", (lhs, rhs), (result,))
    with pytest.raises(IRVerificationError, match="2 result memory effects for 1"):
        materialize_kernel_prim_functions(builder.build(entry="main"))


def test_qkv_effect_schema_names_partial_owner_and_partition_relations():
    definition = fm.get_definition("nn.qkv_rope_with_cache")

    assert definition.qkv.memory_effect == (
        fm.MemoryEffect.READ.across_partial_owners()
    )
    assert definition.state.memory_effect == (
        fm.MemoryEffect.CHIP_READ_WRITE.partitioned_by_argument(8)
    )
    assert definition.layer_id.memory_effect is fm.MemoryEffect.NONE


def test_none_effect_tensor_is_a_metadata_only_prim_function_parameter():
    builder = fm.IRBuilder(dialect="semantic_tir", stage="selected_tir")
    placement = fm.Placement((2, 4), "yx", "bb")
    reference_type = fm.DistributedType(
        fm.tensor_type("bfloat16", (1, 16)),
        (fm.SBP.broadcast(), fm.SBP.broadcast()),
        placement,
    )
    state_type = fm.RefType(
        "state",
        (("seq_lens", fm.tensor_type("int32", (1,))),),
    )
    result_type = fm.DistributedType(
        fm.tensor_type("float32", (1, 1, 8)),
        (fm.SBP.broadcast(), fm.SBP.broadcast(), fm.SBP.broadcast()),
        placement,
    )
    reference = builder.var("reference", reference_type, id="reference")
    state = builder.var("state", state_type, id="state")
    result = builder.call(
        "tir.kernel",
        (reference, state),
        fm.TupleType((result_type, result_type)),
        id="rotary",
        attrs={
            "semantic_op": "nn.rotary_embedding",
            "candidate": "tir.rotary_embedding.decode",
            "parameters": {
                "family": "rotary_embedding",
                "variant": "decode",
                "elements_per_program": 8,
            },
            "facts": {},
            "semantic_attrs": {
                "head_dim": 8,
                "theta": 10000.0,
                "attention_scaling": 1.0,
            },
        },
    )
    builder.function("main", (reference, state), (result,))

    module = materialize_kernel_prim_functions(builder.build(entry="main"))
    primitive = module.kernel_definitions[0]
    dispatch = fm.kernel_dispatch_for_call(module, module.node_map["rotary"])

    assert primitive.parameters[0].role is fm.PrimParameterRole.METADATA
    assert dispatch.memory_effect_map["reference"] is fm.MemoryEffect.NONE
