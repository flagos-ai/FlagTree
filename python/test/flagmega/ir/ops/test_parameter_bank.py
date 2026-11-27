# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Parameter banks keep layer rows indexed without copying the layer graph."""

import pytest
from triton.flagmega.ir import (
    BankField,
    BankFieldProgram,
    F,
    IRBuilder,
    ParameterBankSchema,
    Node,
    TensorType,
    emit_module,
    load_module,
    verify_module,
)
from triton.flagmega.ir.ops.core import construction_scope


def make_module():
    builder = IRBuilder(dialect="high_level", stage="imported")
    scalar = TensorType("int64", ())
    weight = TensorType("bfloat16", (4, 4))
    schema = ParameterBankSchema("decoder_dense", (BankField("weight", weight),))
    index = builder.var("layer", scalar, id="layer")
    with construction_scope(builder):
        bank = F.builtin.parameter_bank(schema=schema, source="checkpoint", rows=(("l0.weight",), ("l1.weight",)), name="weights")
        row = F.control.bank_get(bank, index, name="row")
        output = F.tensors.get_item(row, 0, name="weight")
    builder.function("main", (index,), (output,))
    return builder.build(entry="main")


def test_parameter_bank_is_schema_checked_and_round_trips(tmp_path):
    module = verify_module(make_module())
    assert sum(node.op == "builtin.parameter_bank" for node in module.nodes) == 1
    assert sum(node.op == "control.bank_get" for node in module.nodes) == 1
    path = emit_module(module, tmp_path / "bank.py")
    restored = load_module(path)
    assert restored.semantic_hash == module.semantic_hash


def test_parameter_bank_rejects_wrong_row_arity():
    builder = IRBuilder(dialect="high_level", stage="imported")
    schema = ParameterBankSchema("bad", (BankField("weight", TensorType("bfloat16", (4,))),))
    with construction_scope(builder), pytest.raises(Exception, match="rows"):
        F.builtin.parameter_bank(schema=schema, source="checkpoint", rows=((),), name="bad")


def test_parameter_bank_lazily_resolves_the_selected_row():
    torch = pytest.importorskip("torch")
    from triton.flagmega.evaluator import DictWeightResolver, TorchEvaluator

    module = verify_module(make_module())
    result, = TorchEvaluator(DictWeightResolver({
        "l0.weight": torch.full((4, 4), 1.0, dtype=torch.bfloat16),
        "l1.weight": torch.full((4, 4), 2.0, dtype=torch.bfloat16),
    })).run(module, {"layer": torch.tensor(1, dtype=torch.int64)})
    torch.testing.assert_close(result, torch.full((4, 4), 2.0, dtype=torch.bfloat16))


def transformed_module():
    """A bank whose rows carry a selected physical layout as offline work."""
    builder = IRBuilder(dialect="high_level", stage="imported")
    scalar = TensorType("int64", ())
    stored = TensorType("bfloat16", (4, 2))
    physical = TensorType("bfloat16", (2, 4))
    program = BankFieldProgram(
        "raw",
        (Node("transposed", "tensors.permute", ("raw",), physical, attrs={"axes": (1, 0)}),),
        "transposed",
    )
    schema = ParameterBankSchema(
        "decoder_physical",
        (BankField("weight", physical, "checkpoint", program, stored),),
    )
    index = builder.var("layer", scalar, id="layer")
    with construction_scope(builder):
        bank = F.builtin.parameter_bank(schema=schema, source="checkpoint",
                                        rows=(("l0.weight",), ("l1.weight",)), name="weights")
        row = F.control.bank_get(bank, index, name="row")
        output = F.tensors.get_item(row, 0, name="weight")
    builder.function("main", (index,), (output,))
    return builder.build(entry="main")


def test_bank_field_program_is_typed_and_round_trips(tmp_path):
    module = verify_module(transformed_module())
    field = module.node_map["weights"].attrs["schema"].fields[0]
    # Storage keeps the checkpoint layout; the row access exposes the physical one.
    assert field.stored_type == TensorType("bfloat16", (4, 2))
    assert field.type == TensorType("bfloat16", (2, 4))
    assert module.node_map["weight"].type == field.type
    restored = load_module(emit_module(module, tmp_path / "bank.py"))
    assert restored.semantic_hash == module.semantic_hash
    assert restored.node_map["weights"].attrs["schema"] == module.node_map["weights"].attrs["schema"]


def test_bank_field_program_applies_per_row_without_copying_the_graph():
    torch = pytest.importorskip("torch")
    from triton.flagmega.evaluator import DictWeightResolver, TorchEvaluator

    module = verify_module(transformed_module())
    # No transform op appears in the runtime graph: it belongs to the schema.
    assert not any(node.op == "tensors.permute" for node in module.nodes)
    weights = {
        "l0.weight": torch.arange(8, dtype=torch.bfloat16).reshape(4, 2),
        "l1.weight": torch.arange(8, 16, dtype=torch.bfloat16).reshape(4, 2),
    }
    for index, key in enumerate(("l0.weight", "l1.weight")):
        result, = TorchEvaluator(DictWeightResolver(weights)).run(
            module, {"layer": torch.tensor(index, dtype=torch.int64)})
        torch.testing.assert_close(result, weights[key].transpose(0, 1))


def test_bank_field_program_rejects_type_mismatch():
    physical = TensorType("bfloat16", (2, 4))
    program = BankFieldProgram(
        "raw",
        (Node("transposed", "tensors.permute", ("raw",), physical, attrs={"axes": (1, 0)}),),
        "transposed",
    )
    with pytest.raises(Exception, match="program output type"):
        BankField("weight", TensorType("bfloat16", (4, 2)), "checkpoint", program, physical)


def fused_module():
    """One field whose program fuses three stored tensors per row."""
    part = TensorType("bfloat16", (2, 4))
    fused = TensorType("bfloat16", (6, 4))
    program = BankFieldProgram(
        ("q", "k", "v"),
        (Node("qkv", "tensors.concat", ("q", "k", "v"), fused, attrs={"axis": 0}),),
        "qkv",
    )
    schema = ParameterBankSchema(
        "decoder_fused",
        (BankField("qkv", fused, "checkpoint", program, None, (part, part, part)),),
    )
    builder = IRBuilder(dialect="high_level", stage="imported")
    index = builder.var("layer", TensorType("int64", ()), id="layer")
    with construction_scope(builder):
        bank = F.builtin.parameter_bank(
            schema=schema, source="checkpoint",
            rows=((("l0.q", "l0.k", "l0.v"),), (("l1.q", "l1.k", "l1.v"),)), name="weights")
        row = F.control.bank_get(bank, index, name="row")
        output = F.tensors.get_item(row, 0, name="qkv")
    builder.function("main", (index,), (output,))
    return builder.build(entry="main")


def test_fused_bank_field_names_one_key_per_stored_source(tmp_path):
    module = verify_module(fused_module())
    field = module.node_map["weights"].attrs["schema"].fields[0]
    assert len(field.stored_types) == 3 and field.type == TensorType("bfloat16", (6, 4))
    assert module.node_map["weights"].attrs["rows"][0] == (("l0.q", "l0.k", "l0.v"),)
    restored = load_module(emit_module(module, tmp_path / "fused.py"))
    assert restored.semantic_hash == module.semantic_hash


def test_fused_bank_field_concatenates_its_row_offline():
    torch = pytest.importorskip("torch")
    from triton.flagmega.evaluator import DictWeightResolver, TorchEvaluator

    module = verify_module(fused_module())
    assert not any(node.op == "tensors.concat" for node in module.nodes)
    weights = {
        f"l{row}.{name}": torch.full((2, 4), float(row * 3 + part), dtype=torch.bfloat16)
        for row in range(2) for part, name in enumerate("qkv")
    }
    result, = TorchEvaluator(DictWeightResolver(weights)).run(
        module, {"layer": torch.tensor(1, dtype=torch.int64)})
    expected = torch.cat([weights[f"l1.{name}"] for name in "qkv"], dim=0)
    torch.testing.assert_close(result, expected)


def test_fused_bank_field_requires_one_key_per_source():
    module = fused_module()
    schema = module.node_map["weights"].attrs["schema"]
    with pytest.raises(Exception, match="rows"):
        with construction_scope(IRBuilder(dialect="high_level", stage="imported")):
            F.builtin.parameter_bank(schema=schema, source="checkpoint",
                                     rows=((("l0.q", "l0.k"),),), name="short")
