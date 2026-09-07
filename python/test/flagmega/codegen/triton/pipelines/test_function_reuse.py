# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT

import ast

import pytest

from triton.flagmega.codegen.triton.tir_package import describe_tir_package, render_tir_package


@pytest.mark.parametrize("depth", [1, 2])
@pytest.mark.parametrize("auxiliary", [False, True])
def test_execution_function_has_one_formal_role_body(compile_pipeline_module, depth, auxiliary):
    implementation = (
        "tir.dense_matmul.tensor_descriptor_smem_pipeline_aux_gemv"
        if auxiliary else "tir.dense_matmul.tensor_descriptor_smem_pipeline_gemv"
    )
    package = describe_tir_package(compile_pipeline_module(
        reusable=True, worker_depth=depth, implementation=implementation,
    ))
    schedule = package["pipeline_schedule"]
    definitions = schedule.get("device_functions", ())
    assert len(definitions) == 1, "Reusing leaf wrappers is not ExecutionFunction reuse"
    definition = definitions[0]
    assert definition["function"] == "worker"
    assert definition["symbol"] == "_flagmega_function_worker"
    assert len(definition["schedule"]["stages"]) == depth
    assert len(schedule["stages"]) == depth * 2

    tree = ast.parse(render_tir_package(package, "unit"))
    functions = {node.name: node for node in tree.body if isinstance(node, ast.FunctionDef)}
    roles = ("consumer", "producer", "auxiliary_consumer") if auxiliary else ("consumer", "producer")
    for role in roles:
        role_name = f"_flagmega_function_worker__{role}"
        function = functions[role_name]
        assert any(
            isinstance(decorator, ast.Call)
            and any(keyword.arg == "noinline" and isinstance(keyword.value, ast.Constant)
                    and keyword.value.value is True for keyword in decorator.keywords)
            for decorator in function.decorator_list
        )
        caller = functions[f"flagmega_main__{role}"]
        calls = [node for node in ast.walk(caller)
                 if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                 and node.func.id == role_name]
        assert len(calls) == 2
        assert ast.dump(calls[0].args[0]) != ast.dump(calls[1].args[0])
        assert not any(
            isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
            and node.func.id.startswith("_flagmega_worker_call_")
            for node in ast.walk(caller)
        ), "The entry role must call the worker, not inline its leaf schedule"


def test_formal_descriptors_are_instantiated_per_call(compile_pipeline_module):
    package = describe_tir_package(compile_pipeline_module(reusable=True))
    definition, = package["pipeline_schedule"].get("device_functions", ())
    formal, = definition["host_tensor_descriptor_specs"]
    assert formal["source"] in definition["signature_arguments"]
    instances = package["host_tensor_descriptor_specs"]
    assert len(instances) == 2
    assert instances[0]["name"] != instances[1]["name"]
    assert instances[0]["offset_bytes"] != instances[1]["offset_bytes"]
    events = [event for event in package["pipeline_schedule"]["producer_events"]
              if event["kind"] == "function_call"]
    assert len(events) == 2
    for event, descriptor in zip(events, instances, strict=True):
        assert descriptor["name"] in event["arguments"]
