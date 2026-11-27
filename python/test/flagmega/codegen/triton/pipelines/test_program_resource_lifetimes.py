# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: MIT
"""A complete transfer stage can own an independent kernel lifetime."""

import ast

import pytest

from triton.flagmega.codegen.triton.tir_package import describe_tir_package, render_tir_package


@pytest.mark.parametrize("wrapper_depth", [0, 1])
def test_complete_nested_pipeline_projects_both_roles(compile_pipeline_module, wrapper_depth):
    module = compile_pipeline_module(reusable=True, wrapper_depth=wrapper_depth, program_dag=True)
    package = describe_tir_package(module)
    source = render_tir_package(package, "unit")
    ast.parse(source)
    pipelines = [p for p in package["programs"] if p.get("pipeline_schedule")]
    assert len(pipelines) == 2
    for program in pipelines:
        schedule = program["pipeline_schedule"]
        assert len(schedule["stages"]) == 1
        assert schedule["consumer_endpoints"] and schedule["producer_endpoints"]
        consumer = next(e for e in schedule["consumer_events"] if e["kind"] == "function_call")
        producer = next(e for e in schedule["producer_events"] if e["kind"] == "function_call")
        assert consumer["call"] == producer["call"]
        assert set(consumer["endpoint_bindings"].values()) == set(schedule["consumer_endpoints"])
        assert set(producer["endpoint_bindings"].values()) == set(schedule["producer_endpoints"])
    assert source.count("_flagmega_shared_arena = tle.gpu.alloc(") == 2


def test_complete_root_pipeline_projects_its_resources(compile_pipeline_module):
    module = compile_pipeline_module(reusable=False, program_dag=True)
    package = describe_tir_package(module)
    source = render_tir_package(package, "unit")
    ast.parse(source)
    programs = package["programs"]
    pipelines = [p for p in programs if p.get("pipeline_schedule")]
    assert len(pipelines) == 1
    pipeline = pipelines[0]
    schedule = pipeline["pipeline_schedule"]
    assert len(schedule["stages"]) == 1
    assert schedule["stages"][0]["call"] in next(
        k["calls"] for k in package["kernels"] if k["symbol"] == pipeline["symbol"])
    assert "tle.gpu.warp_specialize(" in source
    assert source.count("_flagmega_shared_arena = tle.gpu.alloc(") == 1
    assert all(not p.get("pipeline_schedule") for p in programs if p is not pipeline)


@pytest.mark.parametrize("reusable,wrapper_depth", [(False, 0), (True, 0), (True, 1)])
def test_root_pipeline_program_runs_with_strict_resources(tmp_path, compile_pipeline_module, reusable, wrapper_depth):
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (9, 0):
        pytest.skip("requires SM90")
    from triton.flagmega.artifacts import write_artifact
    from triton.flagmega.importer import MemoryCheckpoint, TensorInfo
    from triton.flagmega.ir import DType
    from triton.flagmega.runtime import load
    module = compile_pipeline_module(reusable=reusable, wrapper_depth=wrapper_depth, program_dag=True)
    weight = ((torch.arange(128)[:, None] + torch.arange(128)[None, :]) % 7 - 3).bfloat16()
    weights = {"first_weight": weight, "second_weight": weight.roll(1, 0)} if reusable else {"weight": weight}
    checkpoint = MemoryCheckpoint({}, {name: TensorInfo(name, DType.BFLOAT16, (128, 128), "memory")
                                       for name in weights}, weights)
    artifact = write_artifact(module, tmp_path / "artifact", target="nvidia-sm90", checkpoint=checkpoint, emit_executable=True)
    runtime = load(artifact, device="cuda:0")
    try:
        value = ((torch.arange(128) % 11 - 5) / 128).reshape(1, 128).bfloat16().cuda()
        expected = (value.cpu().double() @ weight.double().T).bfloat16().cuda()
        if reusable:
            expected = (expected.cpu().double() @ weights["second_weight"].double().T).bfloat16().cuda()
        output = runtime.create_outputs()
        runtime.prepare(value, output=output)
        for _ in range(2):
            output.fill_(float("nan"))
            runtime.run_into(output, value)
            torch.cuda.synchronize()
            torch.testing.assert_close(output, expected, rtol=0, atol=0)
    finally:
        runtime.close()
