# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT

from dataclasses import replace
from importlib import import_module

from triton.flagmega import ir as fm
from triton.flagmega.codegen.triton.call_abi import describe_function_call_abi
from triton.flagmega.passes.tir import (
    bind_prim_function_buffers,
    materialize_kernel_prim_functions,
)


def _kernel_attrs(candidate: str) -> dict[str, object]:
    return {
        "semantic_op": "math.silu",
        "candidate": candidate,
        "parameters": {"family": "unary", "variant": "local"},
        "facts": {},
        "semantic_attrs": {},
    }


def _bufferized_nested_module() -> fm.IRModule:
    builder = fm.IRBuilder(dialect="semantic_tir", stage="selected_tir")
    value_type = fm.tensor_type("bfloat16", (1, 16))

    worker_input = builder.var("worker_input", value_type, id="worker_input")
    worker_result = builder.call(
        "tir.kernel",
        (worker_input,),
        value_type,
        id="worker_result",
        attrs=_kernel_attrs("tir.silu.worker"),
    )
    builder.function(
        "worker",
        (worker_input,),
        (worker_result,),
        attrs={
            "calling_convention": "device",
            "noinline": True,
            "reusable": True,
        },
    )

    source = builder.var("source", value_type, id="source")
    prepared = builder.call(
        "tir.kernel",
        (source,),
        value_type,
        id="prepared",
        attrs=_kernel_attrs("tir.silu.prepared"),
    )
    nested = builder.call(
        "tir.call",
        (prepared,),
        value_type,
        id="nested",
        attrs={"callee": "worker"},
    )
    output = builder.call(
        "tir.kernel",
        (nested,),
        value_type,
        id="output",
        attrs=_kernel_attrs("tir.silu.output"),
    )
    builder.function("main", (source,), (output,))

    selected = materialize_kernel_prim_functions(builder.build(entry="main"))
    plan = fm.make_buffer_plan(selected)
    bound = bind_prim_function_buffers(selected)
    return fm.verify_module(replace(
        bound,
        stage="bufferized_tir",
        dialect="bufferized_tir",
        metadata={**bound.metadata, "buffer_plan": plan.to_data()},
    ))


def test_function_call_abi_preserves_kernel_and_nested_call_order():
    described = describe_function_call_abi(_bufferized_nested_module())

    assert described["schema"] == "flagmega.function-call-abi/v2"
    assert [(event["kind"], event["call"]) for event in described["events"]] == [
        ("kernel_call", "prepared"),
        ("function_call", "nested"),
        ("kernel_call", "output"),
    ]
    nested = described["events"][1]
    assert nested["callee"] == "worker"
    assert nested["arguments"][0]["formal"] == "worker_input"
    assert nested["arguments"][0]["actual"] == "prepared"
    assert nested["results"][0]["formal"] == "worker_result"
    assert nested["results"][0]["actual"] == "nested"


def test_reusable_device_function_keeps_its_bufferized_local_call_contract():
    described = describe_function_call_abi(
        _bufferized_nested_module(), function_name="worker"
    )

    assert described["calling_convention"] == "device"
    assert described["noinline"] is True
    assert described["reusable"] is True
    assert [event["call"] for event in described["events"]] == ["worker_result"]
    assert described["parameters"][0]["buffers"][0]["abi"][
        "local_capacity_shape"
    ] == (1, 16)
    assert described["outputs"][0]["buffers"][0]["abi"][
        "local_capacity_shape"
    ] == (1, 16)


def test_repeated_call_analysis_reuses_bounded_immutable_buffer_abi_cache():
    call_abi_module = import_module("triton.flagmega.codegen.triton.call_abi")
    module = _bufferized_nested_module()
    call_abi_module._describe_local_buffer_abi_base.cache_clear()

    first = describe_function_call_abi(module)
    first_info = call_abi_module._describe_local_buffer_abi_base.cache_info()
    second = describe_function_call_abi(module)
    second_info = call_abi_module._describe_local_buffer_abi_base.cache_info()

    assert first == second
    assert first_info.maxsize == 2048
    assert second_info.hits > first_info.hits
    assert second_info.currsize <= second_info.maxsize
