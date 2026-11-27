# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Bank-row constants stay offline, including multi-source arithmetic."""

from triton.flagmega import ir as fm
from triton.flagmega.ir.ops.core import construction_scope
from triton.flagmega.passes.functions.absorb_bank_transforms import absorb_bank_parameter_transforms
from triton.flagmega.passes.tir.fuse_bank_qkv_weights import fuse_bank_qkv_weights


def _module(*, dynamic_scale=False, shared=False, different_row=False):
    builder = fm.IRBuilder(dialect="high_level", stage="imported")
    index_type = fm.tensor_type("int64", ())
    weight_type = fm.tensor_type("float8_e4m3fn", (4, 8))
    scale_type = fm.tensor_type("float32", (2, 2))
    output_type = fm.tensor_type("bfloat16", (4, 8))
    schema = fm.ParameterBankSchema("weights", (
        fm.BankField("weight", weight_type, "checkpoint"), fm.BankField("scale", scale_type, "checkpoint"),
    ))
    iv = builder.var("iv", index_type, id="iv")
    carried = builder.var("carried", output_type, id="carried")
    captured = builder.var("captured", schema.type, id="captured")
    with construction_scope(builder):
        row = fm.F.control.bank_get(captured, iv, name="row")
        weight = fm.F.tensors.get_item(row, 0, name="weight")
        if different_row:
            zero = fm.F.builtin.scalar_const(index_type, 0)
            scale_row = fm.F.control.bank_get(captured, zero, name="scale_row")
        else:
            scale_row = row
        scale = fm.F.tensors.get_item(scale_row, 1, name="scale")
        wide = fm.F.tensors.cast(weight, "float32", name="wide")
        blocked = fm.F.tensors.reshape(wide, (2, 2, 2, 4), name="blocked")
        if dynamic_scale:
            # An activation-derived scale must never enter an offline program.
            expanded = fm.F.tensors.cast(carried, "float32", name="runtime_scale")
            expanded = fm.F.tensors.reshape(expanded, (2, 2, 2, 4), name="runtime_blocks")
        else:
            expanded = fm.F.tensors.reshape(scale, (2, 1, 2, 1), name="scale_axes")
            expanded = fm.F.tensors.broadcast_to(expanded, shape=(2, 2, 2, 4), name="expanded")
        product = fm.F.math.mul(blocked, expanded, name="decoded")
        rounded = fm.F.tensors.cast(product, "bfloat16", name="rounded")
        restored = fm.F.tensors.reshape(rounded, (4, 8), name="restored")
        output = fm.F.math.add(restored, carried, name="output")
        if shared:
            output = fm.F.math.add(output, fm.F.tensors.cast(wide, "bfloat16"), name="shared_output")
    body = builder.region("body", (iv, carried, captured), (output,), iter_count=1)
    value = builder.var("value", output_type, id="value")
    with construction_scope(builder):
        bank = fm.F.builtin.parameter_bank(schema=schema, source="checkpoint",
            rows=(("l0.w", "l0.s"), ("l1.w", "l1.s")), name="bank")
        lower = fm.F.builtin.scalar_const(index_type, 0)
        upper = fm.F.builtin.scalar_const(index_type, 2)
        step = fm.F.builtin.scalar_const(index_type, 1)
        loop = fm.F.control.for_loop(lower, upper, step, value, captures=(bank,), body=body, name="loop")
        result = fm.F.tensors.get_item(loop, 0, name="result")
    builder.function("main", (value,), (result,))
    return fm.verify_module(builder.build(entry="main"))


def test_multisource_weight_decode_is_offline_and_roundtrips(tmp_path):
    import torch
    from triton.flagmega.evaluator import DictWeightResolver, TorchEvaluator

    original = _module()
    transformed = fm.verify_module(fuse_bank_qkv_weights(original))
    assert "decoded" not in transformed.node_map
    assert "rounded" not in transformed.node_map
    field = transformed.node_map["bank"].attrs["schema"].fields[-1]
    assert len(field.program.sources) == 2
    assert field.type == fm.tensor_type("bfloat16", (4, 8))
    restored = fm.load_module(fm.emit_module(transformed, tmp_path / "offline.py"))
    assert restored.semantic_hash == transformed.semantic_hash
    weights = {}
    for row in range(2):
        weights[f"l{row}.w"] = ((torch.arange(32).reshape(4, 8) - 17 + row) / 8).to(torch.float8_e4m3fn)
        weights[f"l{row}.s"] = torch.tensor([[1.003, .019], [.513, 2.013]]) + row / 32
    inputs = {"value": torch.zeros((4, 8), dtype=torch.bfloat16)}
    before, = TorchEvaluator(DictWeightResolver(weights)).run(original, inputs)
    after, = TorchEvaluator(DictWeightResolver(weights)).run(restored, inputs)
    torch.testing.assert_close(before, after, rtol=0, atol=0)


def test_runtime_dependent_parent_does_not_hide_a_constant_subtree():
    transformed = fm.verify_module(fuse_bank_qkv_weights(_module(dynamic_scale=True)))
    assert "decoded" in transformed.node_map
    assert "blocked" not in transformed.node_map
    assert "runtime_scale" in transformed.node_map


def test_shared_intermediates_remain_live_without_duplicate_programs():
    transformed = fm.verify_module(fuse_bank_qkv_weights(_module(shared=True)))
    assert "wide" not in transformed.node_map
    assert "decoded" in transformed.node_map
    programs = [field.program for field in transformed.node_map["bank"].attrs["schema"].fields if field.program]
    assert sum(node.op == "tensors.cast" for program in programs for node in program.nodes) == 1


def test_later_layout_program_composes_with_multisource_decode():
    from dataclasses import replace

    from triton.flagmega.rules.neutral._utility import make_node

    module = fuse_bank_qkv_weights(_module())
    field_value = module.node_map["restored.bank_field"]
    reshape = make_node("tensors.reshape", "layout", (field_value,), {"shape": (8, 4)}, {})
    restore = make_node("tensors.reshape", "layout_restore", (reshape,), {"shape": (4, 8)}, {})
    nodes = []
    for node in module.nodes:
        nodes.append(node if node.id == field_value.id else replace(node, inputs=tuple(
            restore.id if value == field_value.id else value for value in node.inputs)))
        if node.id == field_value.id:
            nodes.extend((reshape, restore))
    module = fm.verify_module(replace(module, nodes=tuple(nodes)))
    result = fm.verify_module(absorb_bank_parameter_transforms(module))
    assert "layout" not in result.node_map and "layout_restore" not in result.node_map
    field = result.node_map["bank"].attrs["schema"].fields[-1]
    assert len(field.program.sources) == len(field.stored_types) == 2
    assert [node.op for node in field.program.nodes][-2:] == ["tensors.reshape", "tensors.reshape"]


def test_distinct_dynamic_rows_are_not_fused_into_one_field_program():
    result = fm.verify_module(fuse_bank_qkv_weights(_module(different_row=True)))
    assert "decoded" in result.node_map
    assert "row" in result.node_map and "scale_row" in result.node_map
    # Each independent branch may move offline, but multiplication still
    # combines two separately indexed rows at runtime.
    for node in result.nodes:
        if node.op == "builtin.get_item" and node.id.endswith(".bank_field"):
            expected_row = "scale_row" if node.id.startswith("expanded") else "row"
            assert node.inputs == (expected_row,)
