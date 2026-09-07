# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT

from dataclasses import replace

from triton.flagmega import ir as fm
from triton.flagmega.passes.tir import (
    bind_prim_function_buffers,
    materialize_execution_functions,
    materialize_kernel_prim_functions,
)


def kernel_attrs(candidate: str):
    return {
        "semantic_op": "math.silu",
        "candidate": candidate,
        "parameters": {"family": "elementwise", "variant": "silu"},
        "facts": {},
        "semantic_attrs": {},
    }


def nested_allocated_module():
    builder = fm.IRBuilder(dialect="semantic_tir", stage="selected_tir")
    value_type = fm.tensor_type("bfloat16", (1, 16))

    worker_input = builder.var("worker_input", value_type, id="worker_input")
    worker_result = builder.call(
        "tir.kernel",
        (worker_input,),
        value_type,
        id="worker_result",
        attrs=kernel_attrs("tir.silu.worker"),
    )
    builder.function(
        "worker",
        (worker_input,),
        (worker_result,),
        attrs={"calling_convention": "device", "reusable": True},
    )

    source = builder.var("source", value_type, id="source")
    prepared = builder.call(
        "tir.kernel",
        (source,),
        value_type,
        id="prepared",
        attrs=kernel_attrs("tir.silu.prepared"),
    )
    prepared_view = builder.call(
        "tir.buffer_view",
        (prepared,),
        value_type,
        id="prepared_view",
        attrs={"alias_kind": "reshape"},
    )
    nested = builder.call(
        "tir.call",
        (prepared_view,),
        value_type,
        id="nested",
        attrs={"callee": "worker"},
    )
    output = builder.call(
        "tir.kernel",
        (nested,),
        value_type,
        id="output",
        attrs=kernel_attrs("tir.silu.output"),
    )
    builder.function("main", (source,), (output,))

    selected = materialize_kernel_prim_functions(builder.build(entry="main"))
    plan = fm.make_buffer_plan(selected)
    bound = bind_prim_function_buffers(selected, plan=plan)
    return replace(
        bound,
        stage="allocated_tir",
        dialect="bufferized_tir",
        metadata={**bound.metadata, "buffer_plan": plan.to_data()},
    )


def scheduled_nested_module():
    return materialize_execution_functions(nested_allocated_module())
