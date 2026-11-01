# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Physical tasks partition one typed owner while retaining its stored ABI."""
import ast
from dataclasses import replace
import importlib.util
from pathlib import Path

import pytest
from triton.flagmega import ir as fm
from triton.flagmega.artifacts import write_artifact
from triton.flagmega.codegen.triton import render_triton_package
from triton.flagmega.codegen.triton.tir_package import describe_tir_package
from triton.flagmega.errors import ArtifactError, CodegenError
from triton.flagmega.runtime import load

_spec = importlib.util.spec_from_file_location("_task_owner_down_graph",
    Path(__file__).with_name("kernels") / "sparse_experts/test_down_output_cooperation.py")
_graphs = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_graphs)
compile_graph, graph = _graphs.compile_graph, _graphs.graph


def partition(module, tasks=4, *, grid=512, tile=64):
    definitions = tuple(replace(d, dispatch=replace(d.dispatch, microkernel=replace(
        d.dispatch.microkernel, parameters={**d.dispatch.microkernel.parameters,
                                           "tasks_per_owner": tasks, "output_tile_n": tile})))
                        for d in module.kernel_definitions)
    calls = tuple(n.id for n in module.nodes if n.op == "tir.call")
    programs = [dict(symbol="down", calls=calls, num_warps=8, grid=[grid, 1, 1], tasks_per_owner=tasks)]
    return replace(module, kernel_definitions=definitions, metadata={**module.metadata,
        "kernel_program_partition": programs,
        "codegen_package_plan": {**module.metadata["codegen_package_plan"], "programs": programs}})


@pytest.fixture(scope="module")
def compiled():
    return compile_graph(graph(k=128, experts=8), 256)


def test_tasks_roundtrip_with_complete_nonoverlapping_writes(tmp_path, compiled):
    module = partition(compiled)
    fm.verify_module(module)
    reloaded = fm.load_module(fm.emit_module(module, tmp_path / "final.py"))
    package = render_triton_package(reloaded, tmp_path / "source")
    assert package["kernels"][0]["task_owner_partition"] == {
        "tasks_per_owner": 4, "owner_count": 128, "order": "owner-then-task"}
    assert package["kernels"][0]["grid"] == [512, 1, 1]
    text = (tmp_path / "source/generated_kernels.py").read_text()
    ast.parse(text)
    assert "('block_task', 4)" in text
    assert "block_index = block_index * 4 + _flagmega_task_index" in text
    assert "tl.range(_flagmega_task_index * 64, (_flagmega_task_index + 1) * 64, 64)" in text
    writes = [(physical // 4, (physical % 4) * 64 + column)
              for physical in range(512) for column in range(64)]
    assert len(writes) == len(set(writes)) == 128 * 256
    assert set(writes) == {(owner, column) for owner in range(128) for column in range(256)}
    old_binding = describe_tir_package(compiled)["runtime_binding"]
    for key in ("arguments", "pools", "signature"):
        assert package["runtime_binding"][key] == old_binding[key]


@pytest.mark.parametrize("tasks,grid,tile", [(3, 384, 64), (4, 128, 64), (4, 512, 128), (True, 512, 64), (0, 512, 64)])
def test_rejects_incomplete_or_mismatched_task_partition(compiled, tasks, grid, tile):
    with pytest.raises(CodegenError, match="[Tt]ask|[Dd]own"):
        describe_tir_package(partition(compiled, tasks, grid=grid, tile=tile))


def test_runtime_reconstructs_owner_partition(tmp_path, compiled):
    module = partition(compiled)
    artifact = write_artifact(module, tmp_path / "artifact", target="nvidia-sm90", emit_executable=True)
    runtime = load(artifact)
    runtime._kernel_launch_specs()
    runtime.codegen["kernels"][0]["task_owner_partition"]["owner_count"] = 64
    with pytest.raises(ArtifactError, match="[Pp]rogram|ownership"):
        runtime._kernel_launch_specs()


def test_rejects_callee_task_count_without_physical_tasks(compiled):
    module = partition(compiled)
    programs = [dict(p, tasks_per_owner=1, grid=[128, 1, 1]) for p in module.metadata["kernel_program_partition"]]
    module = replace(module, metadata={**module.metadata, "kernel_program_partition": programs,
        "codegen_package_plan": {**module.metadata["codegen_package_plan"], "programs": programs}})
    with pytest.raises(CodegenError, match="task count"):
        describe_tir_package(module)


def test_rejects_callee_tasks_without_a_program(compiled):
    module = partition(compiled)
    metadata = dict(module.metadata)
    metadata.pop("kernel_program_partition")
    plan = dict(metadata["codegen_package_plan"])
    plan.pop("programs")
    metadata["codegen_package_plan"] = plan
    with pytest.raises(CodegenError, match="explicit execution program"):
        describe_tir_package(replace(module, metadata=metadata))


def test_device_tasks_preserve_packed_output_and_guards(tmp_path, compiled):
    import torch
    from math import prod
    from triton.flagmega.runtime.module import GeneratedTirCallGraphModule as Run

    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (9, 0):
        pytest.skip("requires SM90")
    runtimes, values, outputs, guards = [], {}, [], []
    try:
        for index, module in enumerate((compiled, partition(compiled))):
            artifact = write_artifact(module, tmp_path / str(index), target="nvidia-sm90", emit_executable=True)
            runtime = load(artifact, device="cuda:0")
            runtimes.append(runtime)
            for spec in runtime.external_arguments:
                name = spec["value"]
                kind = fm.logical_type(runtime.ir_module.node_map[name].type)
                shape = tuple(d.fixed_value for d in kind.shape) + tuple(getattr(kind.dtype, "lanes", ()))
                dtype = getattr(torch, getattr(kind.dtype, "elem_type", kind.dtype).value)
                if spec["role"] == "result":
                    storage = torch.full((prod(shape) + 32,), 128, dtype=dtype, device="cuda")
                    guards.append(storage)
                    outputs.append(storage[16:-16].reshape(shape))
                elif name not in values:
                    values[name] = torch.empty(shape, dtype=dtype, device="cuda")
        generator = torch.Generator(device="cuda").manual_seed(20261020)
        for mutation in range(6):
            for name, value in values.items():
                if name == "router_expert_ids":
                    value.copy_(torch.randperm(8, generator=generator, device="cuda").reshape(1, 8))
                elif name == "down_weight":
                    value.copy_((torch.randn(value.shape, generator=generator, device="cuda") * .125).to(value.dtype))
                elif "scale" in name:
                    value.copy_(2. ** torch.randint(-3, 3, value.shape, generator=generator, device="cuda").float())
                else:
                    value.copy_(torch.randn(value.shape, generator=generator, device="cuda").to(value.dtype))
            if mutation == 4:
                values["router_expert_ids"].fill_(7)
            if mutation == 5:
                values["activations"].zero_()
            for runtime, output in zip(runtimes, outputs, strict=True):
                output.fill_(float("nan"))
                arguments = tuple(output if spec["role"] == "result" else values[spec["value"]]
                                  for spec in runtime.external_arguments)
                Run.prepare(runtime, *arguments)
                Run.run_into(runtime, *arguments)
            torch.cuda.synchronize()
            assert torch.equal(outputs[0].view(torch.uint8), outputs[1].view(torch.uint8))
            assert all(bool((s[:16] == 128).all()) and bool((s[-16:] == 128).all()) for s in guards)
        reports = runtimes[1].resource_report
        if isinstance(reports, dict):
            reports = [reports]
        assert len(reports) == 1 and reports[0]["spill_bytes"] == 0
        assert reports[0]["grid"] == [512, 1, 1]
    finally:
        for runtime in runtimes:
            runtime.close()
