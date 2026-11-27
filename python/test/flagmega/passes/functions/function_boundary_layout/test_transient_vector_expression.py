# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT

from triton.flagmega import ir as fm
from triton.flagmega.compiler import Compiler
from triton.flagmega.passes.functions import propagate_function_boundary_layouts
from triton.flagmega.selection import override_plan


def _vectorized_reusable_matmul() -> fm.IRModule:
    builder = fm.IRBuilder(dialect="high_level", stage="imported")
    value_type = fm.tensor_type("bfloat16", (1, 128))
    weight_type = fm.tensor_type("bfloat16", (128, 128))

    worker_value = builder.var("worker_value", value_type, id="worker_value")
    worker_weight = builder.var("worker_weight", weight_type, id="worker_weight")
    projection = builder.call(
        "math.matmul",
        (worker_value, worker_weight),
        value_type,
        id="projection",
        attrs={"transpose_a": False, "transpose_b": False},
    )
    builder.function(
        "worker",
        (worker_value, worker_weight),
        (projection,),
        attrs={"calling_convention": "device", "noinline": True, "reusable": True},
    )

    value = builder.var("value", value_type, id="value")
    weight = builder.var("weight", weight_type, id="weight")
    result = builder.call(
        "builtin.call", (value, weight), value_type,
        id="result", attrs={"callee": "worker"},
    )
    builder.function("main", (value, weight), (result,))

    compiler = Compiler()
    proposed = compiler.compile(
        builder.build(entry="main"), stop_after="propose-vectorization"
    ).module
    return compiler.run_stage(
        proposed,
        "apply-vectorization",
        plan=override_plan(
            proposed,
            (("vectorization.projection", "vectorization.matmul.n"),),
        ),
    ).module


def test_schedule_only_vector_expression_is_not_moved_into_function_abi():
    vectorized = _vectorized_reusable_matmul()
    rewritten = propagate_function_boundary_layouts(vectorized)

    worker = rewritten.function_map["worker"]
    assert rewritten.node_map["worker_weight"].type == fm.tensor_type(
        "bfloat16", (128, 128)
    )
    assert worker.outputs == ("projection",)
    assert rewritten.node_map["projection"].op == "tensors.unpack"
    assert rewritten.node_map["projection.vectorized.pack1"].inputs == (
        "worker_weight",
    )
    assert rewritten.node_map["result"].inputs == ("value", "weight")
    assert "function_boundary_layout" not in worker.attrs
