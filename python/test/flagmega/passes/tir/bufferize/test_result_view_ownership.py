# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT

from dataclasses import replace

from triton.flagmega import ir as fm


def _view_result_module():
    tensor = fm.tensor_type("bfloat16", (1, 16))
    builder = fm.IRBuilder(dialect="semantic_tir", stage="packaged_tir")
    lhs = builder.var("lhs", tensor, id="lhs")
    rhs = builder.var("rhs", tensor, id="rhs")
    source = builder.call("math.add", (lhs, rhs), tensor, id="source")
    result = builder.call(
        "tir.buffer_view",
        (source,),
        tensor,
        id="result",
        attrs={"alias_kind": "reshape"},
    )
    builder.function("main", (lhs, rhs), (result,))
    return builder.build(entry="main")


def test_entry_result_view_promotes_the_complete_alias_group_to_external_storage():
    plan = fm.make_buffer_plan(_view_result_module())
    source = plan.buffer_map["source"]
    result = plan.buffer_map["result"]

    assert source.mem_span.must_alias(result.mem_span)
    assert source.storage == result.storage == "output"
    assert source.mem_span.buffer.memory_space == "external"
    assert result.physical_id.startswith("external:main:result:")
    assert plan.function_map["main"].workspace_bytes == 0


def test_runtime_binding_exposes_a_view_result_as_an_output_argument():
    from triton.flagmega.codegen.triton.runtime_binding import (
        describe_function_runtime_binding,
    )
    module = _view_result_module()
    plan = fm.make_buffer_plan(module)
    module = replace(
        module,
        metadata={**dict(module.metadata), "buffer_plan": plan.to_data()},
    )
    binding = describe_function_runtime_binding(module)

    result = next(value for value in binding["arguments"] if value["role"] == "result")
    assert result["value"] == "result"
    assert result["buffer"] == "result"
    assert "workspace" not in binding["signature"]
