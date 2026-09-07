# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT

from dataclasses import replace

from triton.flagmega import ir as fm
from triton.flagmega.codegen.triton.call_abi import describe_function_call_abi
from triton.flagmega.codegen.triton.runtime_binding import (
    describe_function_runtime_binding,
)
from triton.flagmega.codegen.triton.function_call import (
    emit_function_call_arguments,
)
from triton.flagmega.passes.tir import (
    bind_prim_function_buffers,
    materialize_kernel_prim_functions,
)


def _attrs(candidate: str) -> dict[str, object]:
    return {
        "semantic_op": "math.add",
        "candidate": candidate,
        "parameters": {"family": "elementwise", "variant": "add"},
        "facts": {},
        "semantic_attrs": {},
    }


def _module() -> fm.IRModule:
    builder = fm.IRBuilder(dialect="semantic_tir", stage="selected_tir")
    value_type = fm.tensor_type("float32", (16,))
    lhs = builder.var("lhs", value_type, id="lhs")
    rhs = builder.weight(
        "rhs",
        value_type,
        source="memory",
        key="rhs",
        id="rhs",
    )
    temporary = builder.call(
        "tir.kernel",
        (lhs, rhs),
        value_type,
        id="temporary",
        attrs=_attrs("tir.add.temporary"),
    )
    output = builder.call(
        "tir.kernel",
        (temporary, rhs),
        value_type,
        id="output",
        attrs=_attrs("tir.add.output"),
    )
    builder.function("main", (lhs,), (output,))
    selected = materialize_kernel_prim_functions(builder.build(entry="main"))
    plan = fm.make_buffer_plan(selected)
    bound = bind_prim_function_buffers(selected)
    nodes = tuple(
        replace(
            node,
            op="tir.buffer",
            attrs={
                **dict(node.attrs),
                "storage": "rdata",
                "alignment": plan.alignment,
            },
            metadata={
                **dict(node.metadata),
                "bufferized_from": "builtin.weight",
            },
        )
        if node.op == "builtin.weight"
        else node
        for node in bound.nodes
    )
    return fm.verify_module(replace(
        bound,
        nodes=nodes,
        stage="bufferized_tir",
        dialect="bufferized_tir",
        metadata={**bound.metadata, "buffer_plan": plan.to_data()},
    ))


def _nested_rdata_module() -> fm.IRModule:
    builder = fm.IRBuilder(dialect="semantic_tir", stage="selected_tir")
    value_type = fm.tensor_type("float32", (16,))
    weight = builder.weight(
        "rhs",
        value_type,
        source="memory",
        key="rhs",
        id="rhs",
    )
    worker_input = builder.var(
        "worker_input", value_type, id="worker_input"
    )
    worker_weight = builder.var(
        "worker_weight", value_type, id="worker_weight"
    )
    scalar_type = fm.tensor_type("int32", ())
    worker_layer = builder.var(
        "worker_layer", scalar_type, id="worker_layer"
    )
    worker_result = builder.call(
        "tir.kernel",
        (worker_input, worker_weight),
        value_type,
        id="worker_result",
        attrs=_attrs("tir.add.worker"),
    )
    builder.function(
        "worker",
        (worker_input, worker_weight, worker_layer),
        (worker_result,),
        attrs={"noinline": True, "reusable": True},
    )
    source = builder.var("source", value_type, id="source")
    prepared = builder.call(
        "tir.kernel",
        (source, source),
        value_type,
        id="prepared",
        attrs=_attrs("tir.add.prepared"),
    )
    layer = builder.call(
        "tir.scalar_const",
        (),
        scalar_type,
        id="layer",
        attrs={"value": 7},
    )
    output = builder.call(
        "tir.call",
        (prepared, weight, layer),
        value_type,
        id="output",
        attrs={"callee": "worker"},
    )
    builder.function("main", (source,), (output,))
    selected = materialize_kernel_prim_functions(builder.build(entry="main"))
    plan = fm.make_buffer_plan(selected)
    bound = bind_prim_function_buffers(selected)
    nodes = tuple(
        replace(
            node,
            op="tir.buffer",
            attrs={
                **dict(node.attrs),
                "storage": "rdata",
                "alignment": plan.alignment,
            },
            metadata={
                **dict(node.metadata),
                "bufferized_from": "builtin.weight",
            },
        )
        if node.op == "builtin.weight"
        else node
        for node in bound.nodes
    )
    return fm.verify_module(replace(
        bound,
        nodes=nodes,
        stage="bufferized_tir",
        dialect="bufferized_tir",
        metadata={**bound.metadata, "buffer_plan": plan.to_data()},
    ))


def test_runtime_binding_uses_external_rdata_and_sat_workspace_roots():
    binding = describe_function_runtime_binding(_module())

    assert binding["schema"] == "flagmega.function-runtime-binding/v1"
    assert binding["signature"] == ["lhs", "output", "rdata", "workspace"]
    assert binding["pools"] == [
        {"name": "rdata", "storage": "rdata", "nbytes": 256},
        {"name": "workspace", "storage": "workspace", "nbytes": 256},
    ]
    calls = binding["call_abi"]["kernel_calls"]
    assert [call["call"] for call in calls] == ["temporary", "output"]
    assert [
        buffer["runtime_argument"]
        for parameter in calls[0]["inputs"]
        for buffer in parameter["buffers"]
    ] == ["lhs", "rdata"]
    assert calls[0]["outputs"][0]["buffers"][0][
        "runtime_argument"
    ] == "workspace"
    assert calls[1]["outputs"][0]["buffers"][0][
        "runtime_argument"
    ] == "output"


def test_runtime_binding_is_deterministic_and_does_not_mutate_call_abi():
    module = _module()

    first = describe_function_runtime_binding(module)
    second = describe_function_runtime_binding(module)

    assert first == second
    plain_call_abi = describe_function_call_abi(module)
    assert "runtime_argument" not in (
        plain_call_abi["kernel_calls"][0]["inputs"][0]["buffers"][0]
    )


def test_nested_call_keeps_callee_only_rdata_in_entry_signature():
    binding = describe_function_runtime_binding(_nested_rdata_module())

    assert binding["signature"] == [
        "source",
        "rdata",
        "workspace",
    ]
    nested = binding["call_abi"]["events"][1]
    assert nested["kind"] == "function_call"
    assert [
        edge["actual_runtime_argument"] for edge in nested["arguments"]
    ] == ["workspace", "rdata", "7"]
    assert nested["arguments"][2]["actual_runtime_value_kind"] == "immediate"
    assert nested["results"][0]["actual_runtime_argument"] == "workspace"
    assert binding["values"][-1] == {
        "role": "result",
        "value": "output",
        "buffer": "prepared",
        "storage": "workspace",
        "runtime_argument": "workspace",
        "alias": True,
    }


def test_nested_call_arguments_follow_callee_signature_and_storage_bases():
    module = _nested_rdata_module()
    caller = describe_function_runtime_binding(module)
    event = caller["call_abi"]["events"][1]

    arguments = emit_function_call_arguments(module, caller, event)

    assert arguments == (
        "workspace.to(tl.pointer_type(tl.float32))",
        "rdata.to(tl.pointer_type(tl.float32))",
        "tl.full((), 7, tl.int32)",
    )
