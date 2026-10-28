# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: MIT
"""A reusable role must retain pool origins instead of a wide pointer ABI."""

import ast

import pytest
from triton.flagmega.codegen.triton.tir_package import describe_tir_package, render_tir_package
from python.test.flagmega.devices import is_nvidia_sm90


def test_reusable_function_passes_one_pool_origin_and_a_call_record(compile_pipeline_module):
    package = describe_tir_package(compile_pipeline_module(reusable=True, worker_depth=4))
    definition, = package["device_functions"]
    pointer_abi = definition.get("function_pool_abi")
    assert pointer_abi is not None, "Pool-derived pointers still cross the reusable function boundary independently"
    assert len(pointer_abi["bindings"]) >= 2
    tables = package["function_offset_tables"]
    assert len(tables) == 1
    assert len(tables[0]["values"]) == 2 * pointer_abi["record_width"]
    bound = {binding["formal"] for binding in pointer_abi["bindings"]}
    assert not bound.intersection(definition["signature_arguments"])
    tree = ast.parse(render_tir_package(package, "unit"))
    functions = {node.name: node for node in tree.body if isinstance(node, ast.FunctionDef)}
    source_tables = next(node.value for node in tree.body if isinstance(node, ast.Assign)
                         and any(isinstance(target, ast.Name)
                                 and target.id == "FLAGMEGA_FUNCTION_OFFSET_TABLES" for target in node.targets))
    assert ast.literal_eval(source_tables) == tables
    role = functions[definition["symbol"] + "__consumer"]
    assert any(isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
               and node.func.attr == "load" and any(keyword.arg == "volatile" for keyword in node.keywords)
               for node in ast.walk(role))
    # Each invocation has its own offset record while sharing the same body.
    caller = functions["flagmega_main__consumer"]
    calls = [node for node in ast.walk(caller) if isinstance(node, ast.Call)
             and isinstance(node.func, ast.Name) and node.func.id == role.name]
    assert len(calls) == 2
    assert ast.dump(calls[0].args[-1]) != ast.dump(calls[1].args[-1])


def test_function_pool_record_executes_distinct_weights_and_frames_on_sm90(
    tmp_path, compile_pipeline_module,
):
    torch = pytest.importorskip("torch")
    if not is_nvidia_sm90():
        pytest.skip("SM90 CUDA is required")
    from triton.flagmega.artifacts import write_artifact
    from triton.flagmega.runtime import load

    from python.test.flagmega.codegen.triton.pipelines.test_pipeline_runtime import (
        _assert_reused_device_roles,
        _checkpoint,
    )

    torch.manual_seed(109)
    weights = tuple((torch.eye(128) * (1 + index / 16) + torch.randn(128, 128) / 200)
                    .to(torch.bfloat16) for index in range(8))
    checkpoint = {f"{prefix}_weight_{index}": weights[group * 4 + index]
                  for group, prefix in enumerate(("first", "second")) for index in range(4)}
    artifact = write_artifact(
        compile_pipeline_module(reusable=True, worker_depth=4), tmp_path / "pool-call-records",
        target="nvidia-sm90", checkpoint=_checkpoint(checkpoint), emit_executable=True,
    )
    runtime = load(artifact, device="cuda:0")
    value = torch.randn(1, 128, dtype=torch.bfloat16, device="cuda")
    output = torch.empty_like(value)
    runtime.prepare(value, output=output)
    runtime.run_into(output, value)
    torch.cuda.synchronize()
    expected = value
    for weight in weights:
        expected = expected @ weight.to("cuda").T
    torch.testing.assert_close(output, expected, rtol=2e-2, atol=2e-2)
    assert runtime.resource_report["spill_bytes"] == 0
    assert len(runtime._function_offset_values) == 1
    _assert_reused_device_roles(runtime)
    runtime.close()


def test_function_offset_payload_must_match_generated_source():
    from triton.flagmega.errors import ArtifactError
    from triton.flagmega.runtime.function_offsets import (
        validate_function_offset_tables,
        verify_function_offset_source,
    )

    values = [{"name": "_flagmega_function_offsets", "dtype": "int64", "values": [2**33, 64]}]
    namespace = {"FLAGMEGA_FUNCTION_OFFSET_TABLES": values}
    exec("def kernel(): pass", namespace)
    specs = validate_function_offset_tables(values, existing_names=())
    verify_function_offset_source(namespace["kernel"], specs, existing_names=())
    changed = validate_function_offset_tables(
        [dict(values[0], values=[2**33, 128])], existing_names=(),
    )
    with pytest.raises(ArtifactError, match="verified generated source"):
        verify_function_offset_source(namespace["kernel"], changed, existing_names=())


@pytest.mark.parametrize("value", [True, 1.0, 2**63, -(2**63) - 1])
def test_function_offsets_reject_non_i64_values(value):
    from triton.flagmega.errors import ArtifactError
    from triton.flagmega.runtime.function_offsets import validate_function_offset_tables

    with pytest.raises(ArtifactError, match="signed-i64 payload"):
        validate_function_offset_tables(
            [{"name": "_flagmega_function_offsets", "dtype": "int64", "values": [value]}],
            existing_names=(),
        )


def _row_strided_plan(row_stride_bytes):
    return {"bindings": [{"formal": "w", "storage": "s0", "row_stride_bytes": row_stride_bytes}],
            "pools": [], "record_width": 0, "calls": []}


def test_non_loop_call_passes_the_bank_row_as_a_tl_value():
    from triton.flagmega.codegen.triton.function_pool_abi import function_pool_actuals
    caller = {"function": "caller", "pools": []}
    (row,) = function_pool_actuals(caller, None, _row_strided_plan(128), {}, has_tables=False)
    # The callee folds this into ``(row).to(tl.int64) * stride``; a bare Python
    # int literal has no ``.to`` and fails to compile.
    expression = ast.parse(row, mode="eval").body
    assert isinstance(expression, ast.Call) and ast.unparse(expression.func) == "tl.full"
    assert ast.literal_eval(expression.args[1]) == 0
    assert ast.unparse(expression.args[2]) == "tl.int64"


def test_loop_call_and_unstrided_plan_keep_their_row_actuals():
    from triton.flagmega.codegen.triton.function_pool_abi import function_pool_actuals
    caller = {"function": "caller", "pools": []}
    assert function_pool_actuals(caller, None, _row_strided_plan(128), {}, has_tables=False,
                                 row_index="layer") == ["layer"]
    assert function_pool_actuals(caller, None, _row_strided_plan(0), {}, has_tables=False) == []
