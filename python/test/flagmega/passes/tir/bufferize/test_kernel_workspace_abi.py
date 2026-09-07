# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT

from dataclasses import replace

import pytest

from triton.flagmega import ir as fm
from triton.flagmega.errors import IRVerificationError
from triton.flagmega.passes.tir import (
    bind_prim_function_buffers,
    materialize_kernel_prim_functions,
)
from triton.flagmega.ir.printer import script_source


def _selected_workspace_kernels() -> fm.IRModule:
    builder = fm.IRBuilder(dialect="semantic_tir", stage="selected_tir")
    value_type = fm.tensor_type("bfloat16", (1, 16))
    lhs = builder.var("lhs", value_type, id="lhs")
    attributes = {
        "semantic_op": "math.silu",
        "candidate": "tir.unary.workspace",
        "parameters": {
            "family": "unary",
            "variant": "workspace",
            "workspaces": ({
                "name": "partials",
                "type": fm.tensor_type("float32", (32,)),
                "memory_space": "workspace",
                "alignment": 512,
            },),
        },
        "facts": {},
        "semantic_attrs": {},
    }
    first = builder.call("tir.kernel", (lhs,), value_type, id="first", attrs=attributes)
    second = builder.call("tir.kernel", (lhs,), value_type, id="second", attrs=attributes)
    builder.function("main", (lhs,), (first, second))
    return materialize_kernel_prim_functions(builder.build(entry="main"))


def _bufferized_module():
    selected = _selected_workspace_kernels()
    plan = fm.make_buffer_plan(selected)
    bound = bind_prim_function_buffers(selected)
    module = replace(
        bound,
        stage="bufferized_tir",
        metadata={**bound.metadata, "buffer_plan": plan.to_data()},
    )
    return fm.verify_module(module), plan


def _attention_workspace_plan():
    builder = fm.IRBuilder(dialect="semantic_tir", stage="selected_tir")
    hidden_type = fm.tensor_type("bfloat16", (1, 2048))
    source = builder.var("source", hidden_type, id="source")
    workspaces = tuple({
        "name": name,
        "type": value_type,
        "memory_space": "workspace",
        "alignment": 256,
    } for name, value_type in (
        ("qkv_query", fm.tensor_type("bfloat16", (16, 128))),
        ("qkv_key", fm.tensor_type("bfloat16", (8, 128))),
        ("qkv_value", fm.tensor_type("bfloat16", (8, 128))),
    ))
    attention = builder.call(
        "tir.kernel",
        (source,),
        hidden_type,
        id="attention",
        attrs={
            "semantic_op": "nn.paged_attention",
            "candidate": "tir.paged_attention.decode",
            "parameters": {
                "family": "paged_attention",
                "variant": "decode",
                "workspaces": workspaces,
            },
            "facts": {},
            "semantic_attrs": {},
        },
    )
    builder.function("main", (source,), (attention,))
    selected = materialize_kernel_prim_functions(builder.build(entry="main"))
    return fm.make_buffer_plan(selected)


def _retained_workspace_plan():
    builder = fm.IRBuilder(dialect="semantic_tir", stage="selected_tir")
    value_type = fm.tensor_type("bfloat16", (1, 128))
    source = builder.var("source", value_type, id="source")
    first = builder.call(
        "tir.kernel",
        (source,),
        value_type,
        id="first",
        attrs={
            "semantic_op": "math.silu",
            "candidate": "tir.unary.retained_workspace",
            "parameters": {
                "family": "unary",
                "variant": "retained_workspace",
                "workspaces": ({
                    "name": "carry",
                    "type": fm.tensor_type("float32", (128,)),
                    "memory_space": "workspace",
                    "alignment": 256,
                    "lifetime": "function",
                },),
            },
            "facts": {},
            "semantic_attrs": {},
        },
    )
    second = builder.call(
        "math.silu", (first,), value_type, id="second"
    )
    builder.function("main", (source,), (second,))
    return fm.make_buffer_plan(
        materialize_kernel_prim_functions(builder.build(entry="main"))
    )


def test_kernel_workspaces_are_caller_owned_sat_allocations():
    module, plan = _bufferized_module()
    first = plan.resolve_kernel_workspace("first", "partials")
    second = plan.resolve_kernel_workspace("second", "partials")

    assert first.id != second.id
    assert first.mem_span.buffer.id != second.mem_span.buffer.id
    assert first.offset == second.offset
    assert first.alignment == second.alignment == 512
    assert first.role == second.role == "kernel_workspace"
    assert first.live_end < second.live_start
    assert plan.workspace_bytes == 256
    assert plan.alignment == 512
    assert tuple(value.call for value in plan.function_map["main"].kernel_calls) == (
        "first",
        "second",
    )
    assert module.kernel_definitions[0].workspaces[0].buffers[0].elem_type == fm.DType.FLOAT32


def test_simultaneously_live_attention_scratch_cannot_alias_each_other_or_result():
    plan = _attention_workspace_plan()
    scratch = tuple(
        plan.resolve_kernel_workspace("attention", name)
        for name in ("qkv_query", "qkv_key", "qkv_value")
    )
    result = plan.buffer_map[dict(plan.function_map["main"].outputs)["attention"][0]]

    for index, lhs in enumerate(scratch):
        assert not lhs.mem_span.may_alias(result.mem_span)
        for rhs in scratch[index + 1:]:
            assert not lhs.mem_span.may_alias(rhs.mem_span)


def test_function_lifetime_kernel_workspace_cannot_alias_later_values():
    plan = _retained_workspace_plan()
    carry = plan.resolve_kernel_workspace("first", "carry")
    second = plan.buffer_map[dict(plan.function_map["main"].outputs)["second"][0]]

    assert carry.live_end == 3
    assert not carry.mem_span.may_alias(second.mem_span)


def test_kernel_workspace_buffer_plan_round_trips_through_python(tmp_path):
    module, plan = _bufferized_module()
    loaded = fm.load_module(fm.emit_module(module, tmp_path / "bufferized.py"))

    loaded_plan = fm.verify_buffer_plan(loaded)
    assert loaded_plan.to_data() == plan.to_data()
    assert loaded_plan.resolve_kernel_workspace("second", "partials").offset == 0


def test_kernel_workspace_is_visible_in_readable_script_dump():
    module, _ = _bufferized_module()

    source = script_source(module)

    assert "%partials: f32[32]@workspace" in source
    assert "T.KernelCallWorkspace('first'" in source
    assert "%partials -> %first.workspace.partials" in source


def test_kernel_workspace_verifier_rejects_wrong_actual_type():
    module, _ = _bufferized_module()
    data = {
        **module.metadata["buffer_plan"],
        "buffers": [dict(value) for value in module.metadata["buffer_plan"]["buffers"]],
    }
    target = next(
        value for value in data["buffers"]
        if value["id"] == "first.workspace.partials"
    )
    target["dtype"] = "bfloat16"
    corrupted = replace(module, metadata={**module.metadata, "buffer_plan": data})

    with pytest.raises(IRVerificationError, match="does not match formal"):
        fm.verify_buffer_plan(corrupted)


def test_kernel_workspace_verifier_rejects_missing_call_record():
    module, _ = _bufferized_module()
    data = {
        **module.metadata["buffer_plan"],
        "functions": [dict(value) for value in module.metadata["buffer_plan"]["functions"]],
    }
    data["functions"][0]["kernel_calls"] = []
    corrupted = replace(module, metadata={**module.metadata, "buffer_plan": data})

    with pytest.raises(IRVerificationError, match="missing or out of order"):
        fm.verify_buffer_plan(corrupted)
