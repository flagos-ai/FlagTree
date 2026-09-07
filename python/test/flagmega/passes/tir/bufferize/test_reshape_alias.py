# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT

from triton.flagmega import ir as fm
from triton.flagmega.ir.bufferization import AliasKind


def test_reshape_view_preserves_memspan_with_new_dense_shape():
    source_type = fm.tensor_type("bfloat16", (1, 16))
    result_type = fm.tensor_type("bfloat16", (1, 2, 8))
    builder = fm.IRBuilder(dialect="semantic_tir", stage="packaged_tir")
    source = builder.var("source", source_type, id="source")
    result = builder.call(
        "tir.buffer_view",
        (source,),
        result_type,
        id="result",
        attrs={"alias_kind": "reshape"},
    )
    builder.function("main", (source,), (result,))

    plan = fm.make_buffer_plan(builder.build(entry="main"))
    source_buffer = plan.buffer_map[dict(plan.entry_inputs)["source"][0]]
    result_buffer = plan.buffer_map[dict(plan.entry_outputs)["result"][0]]

    assert source_buffer.mem_span.must_alias(result_buffer.mem_span)
    assert result_buffer.shape == (1, 2, 8)
    assert result_buffer.alias.kind is AliasKind.VIEW
