# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT

from triton.flagmega import ir as fm
from triton.flagmega.ir.ops.ntt.matmul_norm_stats import MatMulNormStats
from triton.flagmega.passes.tir import materialize_kernel_prim_functions


def _module() -> fm.IRModule:
    builder = fm.IRBuilder(dialect="semantic_tir", stage="selected_tir")
    lhs_type = fm.tensor_type("bfloat16", (1, 4))
    rhs_type = fm.tensor_type("bfloat16", (4, 4))
    value_type = fm.tensor_type("bfloat16", (1, 4))
    lhs = builder.var("lhs", lhs_type, id="lhs")
    rhs = builder.var("rhs", rhs_type, id="rhs")
    addend = builder.var("addend", value_type, id="addend")
    prepared = MatMulNormStats.prepare(
        (lhs, rhs, addend),
        {
            "transpose_a": False,
            "transpose_b": False,
            "axis": -1,
            "use_mean": False,
        },
    )
    combined = builder.call(
        "tir.kernel",
        prepared.inputs,
        prepared.result_type,
        id="combined",
        attrs={
            "semantic_op": MatMulNormStats.op_name,
            "candidate": "tir.unit.matmul_norm_stats",
            "parameters": {
                "family": "unit",
                "variant": "matmul_norm_stats",
            },
            "facts": {},
            "semantic_attrs": prepared.attrs,
        },
    )
    value = builder.call(
        "builtin.get_item", (combined,), value_type,
        id="value", attrs={"index": 0},
    )
    stats_type = prepared.result_type.fields[1]
    stats = builder.call(
        "builtin.get_item", (combined,), stats_type,
        id="stats", attrs={"index": 1},
    )
    builder.function("decode", (lhs, rhs, addend), (value, stats))

    main_lhs = builder.var("main_lhs", lhs_type, id="main_lhs")
    main_rhs = builder.var("main_rhs", rhs_type, id="main_rhs")
    main_addend = builder.call(
        "math.add", (main_lhs, main_lhs), value_type, id="main_addend",
    )
    call = builder.call(
        "builtin.call",
        (main_lhs, main_rhs, main_addend),
        prepared.result_type,
        id="decode_call",
        attrs={"callee": "decode"},
    )
    builder.function("main", (main_lhs, main_rhs), (call,))
    return materialize_kernel_prim_functions(builder.build(entry="main"))


def test_tuple_value_result_reuses_named_addend_parameter_memspan():
    plan = fm.make_buffer_plan(_module())
    value = plan.buffer_map["combined.0"]
    addend = plan.buffer_map["addend"]
    stats = plan.buffer_map["combined.1"]

    assert value.alias_of == addend.id
    assert value.mem_span.must_alias(addend.mem_span)
    assert stats.alias_of is None
    assert not stats.mem_span.may_alias(value.mem_span)
