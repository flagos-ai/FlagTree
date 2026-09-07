# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT

from triton.flagmega import ir as fm
from triton.flagmega.ir.bufferization import AliasKind


def test_vector_reinterpret_view_preserves_the_exact_memspan():
    packed_type = fm.tensor_type(fm.vector_type("bfloat16", (8,)), (2, 4))
    unpacked_type = fm.tensor_type("bfloat16", (2, 32))
    builder = fm.IRBuilder(dialect="semantic_tir", stage="packaged_tir")
    value = builder.var("value", packed_type, id="value")
    view = builder.call(
        "tir.buffer_view",
        (value,),
        unpacked_type,
        id="view",
        attrs={"alias_kind": "vector_reinterpret"},
    )
    builder.function("main", (value,), (view,))
    module = builder.build(entry="main")

    plan = fm.make_buffer_plan(module)
    source = plan.buffer_map[dict(plan.entry_inputs)["value"][0]]
    result = plan.buffer_map[dict(plan.entry_outputs)["view"][0]]

    assert source.mem_span.must_alias(result.mem_span)
    assert source.nbytes == result.nbytes
    assert source.dtype == fm.vector_type("bfloat16", (8,))
    assert result.dtype == fm.DType.BFLOAT16
    assert result.alias.kind is AliasKind.VIEW
