# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Full-K chip tasks retain the original ordered FP8 numerical method."""

import re
from copy import deepcopy
from dataclasses import replace
from math import prod
from types import SimpleNamespace

import pytest
from triton.flagmega import ir as fm
from triton.flagmega.artifacts import write_artifact
from triton.flagmega.compiler import Compiler
from triton.flagmega.options import CompileOptions
from triton.flagmega.targets import NvidiaSm90Target
from triton.flagmega.targets.nvidia.sparse_experts import sparse_experts_complete_k_implementations

from python.test.flagmega.codegen.triton.kernels.sparse_experts.test_cooperative_wgmma import (
    TensorCheckpoint,
    compile_stage,
    stage_graph,
    stage_values,
)


def complete_target(task_n, stages=1, warps=8):
    target = NvidiaSm90Target()
    model = target.triton_implementation_model
    implementations = tuple(replace(value, parameters={**value.parameters, "num_stages": stages, "compute_num_warps": warps})
                            for value in sparse_experts_complete_k_implementations())
    target.triton_implementation_model = replace(model,
        implementations=(*model.implementations, *implementations),
        preferences={**model.preferences, "sparse_experts_gate_up":
                     (f"tir.sparse_experts_gate_up.complete_k_wgmma_n{task_n}",)})
    return target


def compile_complete(graph, mesh, task_n, stages=1, warps=8):
    compiler = Compiler(CompileOptions(placement=mesh))
    compiler.target = complete_target(task_n, stages, warps)
    return compiler.compile(graph).module


@pytest.fixture(scope="module")
def complete_raw():
    from triton.flagmega.codegen.triton.call_abi import describe_kernel_call_abis

    mesh = fm.Placement((4, 8, 16), "cyx", "cbb")
    module = compile_complete(stage_graph(placement=mesh, packed=True, readonly=True,
        rows=2048, reduction=512), mesh, 64)
    raw = next(call for call in describe_kernel_call_abis(module) if call["family"] == "sparse_experts_gate_up")
    for section in ("inputs", "outputs", "workspaces"):
        for parameter in raw[section]:
            for binding in parameter["buffers"]:
                binding.setdefault("runtime_argument", parameter["formal"])
    return raw


@pytest.mark.parametrize("chip", [0, 3])
@pytest.mark.parametrize("storage", ["compact_per_chip", "canonical_global"])
def test_complete_k_peer_output_retains_chip_route_row_and_lanes(complete_raw, chip, storage):
    import torch
    from triton.flagmega.codegen.triton.sparse_experts.gate_up import sparse_experts_gate_up_call

    raw = deepcopy(complete_raw)
    abi = raw["outputs"][0]["buffers"][0]["abi"]
    assert abi["storage_kind"] == "compact_per_chip"
    if storage == "canonical_global":
        abi.update(storage_kind=storage, coordinate_space="canonical_global",
                   storage_strides=(2048, 256, 1), scalar_storage_strides=(16384, 2048, 8),
                   component_stride_elements=0, component_stride_scalar_elements=0)
    call = sparse_experts_gate_up_call(raw)

    class Pointer(int):
        def to(self, dtype):
            return self

    env = {"tl": SimpleNamespace(pointer_type=lambda dtype: None, uint8=None, bfloat16=None),
           "shard_coord0": chip, "shard_coord1": 7, "shard_coord2": 9,
           "_fm_token": 0, "_fm_route_owner": 6, "_fm_local_route": 0}
    for parameter in complete_raw["outputs"]:
        for binding in parameter["buffers"]:
            env[binding["runtime_argument"]] = Pointer(0)
    for tile in (0, 7):
        n = tile * 64 + torch.arange(64, dtype=torch.int64)
        env.update(_fm_row_owner=n // 32, _fm_local_n=n % 32)
        actual = eval(call["result_peer_pointer"], {}, env) + eval(call["result_peer_offset"], {}, env)
        expected = 6 * 512 + n if storage == "compact_per_chip" else 6 * 2048 + chip * 512 + n
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.parametrize("chip", [0, 3])
def test_complete_k_internal_output_substitutes_full_owner_coordinates(chip):
    import torch
    from triton.flagmega.codegen.triton.call_abi import describe_kernel_call_abis
    from triton.flagmega.codegen.triton.sparse_experts.gate_up import sparse_experts_gate_up_call

    mesh = fm.Placement((4, 8, 16), "cyx", "cbb")
    module = compile_complete(stage_graph(placement=mesh, packed=True, readonly=True,
        rows=2048, reduction=256, publish_result=True), mesh, 32, stages=3)
    raw = next(call for call in describe_kernel_call_abis(module) if call["family"] == "sparse_experts_gate_up")
    for section in ("inputs", "outputs", "workspaces"):
        for parameter in raw[section]:
            for binding in parameter["buffers"]:
                binding.setdefault("runtime_argument", parameter["formal"])
    abi = raw["outputs"][0]["buffers"][0]["abi"]
    assert abi["storage_kind"] == "compact_per_owner"
    assert abi["owner_count"] == 512
    call = sparse_experts_gate_up_call(raw)

    class Pointer(int):
        def to(self, dtype):
            return self

    env = {"tl": SimpleNamespace(pointer_type=lambda dtype: None, uint8=None, bfloat16=None),
           "shard_coord0": chip, "shard_coord1": 7, "shard_coord2": 9,
           "_fm_token": 0, "_fm_route_owner": 6, "_fm_local_route": 0}
    for parameter in raw["outputs"]:
        for binding in parameter["buffers"]:
            env[binding["runtime_argument"]] = Pointer(0)
    for tile in (0, 15):
        n = tile * 32 + torch.arange(32, dtype=torch.int64)
        env.update(_fm_row_owner=n // 32, _fm_local_n=n % 32)
        actual = eval(call["result_peer_pointer"], {}, env) + eval(call["result_peer_offset"], {}, env)
        expected = int(abi["pool_byte_offset"]) // 2 + chip * 8 * 512 + 6 * 512 + n
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    raw["outputs"][0]["buffers"][0]["abi"]["owner_count"] = 128
    from triton.flagmega.errors import CodegenError
    with pytest.raises(CodegenError, match="compact owner map"):
        sparse_experts_gate_up_call(raw)


@pytest.mark.parametrize("cause", ["result_scope", "input_scope", "scratch", "task", "barrier", "owner_map"])
def test_complete_k_rejects_unproved_peer_contract(complete_raw, cause):
    from triton.flagmega.codegen.triton.sparse_experts.gate_up import sparse_experts_gate_up_call
    from triton.flagmega.errors import CodegenError

    raw = deepcopy(complete_raw)
    if cause in {"result_scope", "input_scope"}:
        section = "outputs" if cause == "result_scope" else "inputs"
        raw[section][0]["buffers"][0]["abi"]["memory_sharing_scope"] = "block"
    elif cause == "scratch":
        raw["workspaces"] = []
    elif cause == "task":
        raw["parameters"]["task_n"] = 16
    elif cause == "owner_map":
        raw["outputs"][0]["buffers"][0]["abi"]["owner_index_expression"] = "shard_index"
    else:
        raw["facts"]["internal_grid_barriers"] = 0
    with pytest.raises(CodegenError, match="[Ee]xperts"):
        sparse_experts_gate_up_call(raw)


@pytest.mark.parametrize("task_n", [32, 64])
def test_complete_k_contract_has_no_raw_projection_scratch(tmp_path, task_n):
    mesh = fm.Placement((8, 16), "yx", "bb")
    module = compile_complete(stage_graph(placement=mesh, packed=True, rows=640, reduction=1536), mesh, task_n)
    dispatch, = (f.dispatch for f in module.kernel_definitions if f.dispatch.semantic_op == "nn.sparse_experts_gate_up")
    assert dispatch.microkernel.variant == "complete_k_wgmma"
    assert {w.name for w in dispatch.workspaces} == {"quantized_activations", "activation_scales"}
    assert dispatch.microkernel.facts["internal_grid_barriers"] == 1
    assert dispatch.memory_effect_map["result"].scope.value == "chip"
    assert dispatch.memory_effect_map["dispatched"].scope.value == "chip"
    artifact = write_artifact(module, tmp_path / "artifact", target="nvidia-sm90", emit_executable=True)
    source = (artifact / "generated_kernels.py").read_text()
    assert "projection_partials" not in source and "__merge(" not in source
    assert "tl.fma(libdevice.mul_rn(" in source
    assert "tl.arange(0, 64)" in source
    assert fm.load_module(fm.emit_module(module, tmp_path / "selected.py")).semantic_hash == module.semantic_hash


@pytest.mark.parametrize("task_n,warps", [(32, 4), (32, 8), (64, 8)])
def test_complete_k_retains_mma_accumulator_through_k_loop(tmp_path, task_n, warps):
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (9, 0):
        pytest.skip("requires SM90")
    from triton.flagmega.runtime import load
    from triton.flagmega.runtime.module import GeneratedTirCallGraphModule as Runtime

    mesh = fm.Placement((8, 16), "yx", "bb")
    graph = stage_graph(placement=mesh, readonly=True, rows=512, reduction=4096)
    module = compile_complete(graph, mesh, task_n, stages=3, warps=warps)
    values = stage_values(torch, graph)
    runtime = load(write_artifact(module, tmp_path / "artifact", target="nvidia-sm90", emit_executable=True,
        checkpoint=TensorCheckpoint(values)), device="cuda:0")
    inputs = {name: value.cuda() for name, value in values.items() if not name.endswith("_weight")}
    results = [torch.empty((1, 8, 512), device="cuda", dtype=torch.bfloat16),
               torch.empty((128,), device="cuda", dtype=torch.float32)]
    result_specs = [spec for spec in runtime.external_arguments if spec["role"] == "result"]
    inputs.update((spec["value"], value) for spec, value in zip(result_specs, results, strict=True))
    try:
        Runtime.prepare(runtime, *(inputs[spec["value"]] for spec in runtime.external_arguments))
        assembly = runtime._prepared[0].compiled_kernel.asm["ttgir"]
        (tmp_path / "kernel.ttgir").write_text(assembly)
        # Keeping the matrix result until after K reduction avoids a shared
        # layout round trip on every K128 block. Inspect the innermost actual
        # dot loop rather than source spelling or total instruction counts.
        loops = []
        lines = assembly.splitlines()
        for index, line in enumerate(lines):
            if "scf.for " not in line:
                continue
            body = []
            depth = 1
            for current in lines[index + 1:]:
                depth += current.count("{") - current.count("}")
                if depth == 0:
                    break
                body.append(current)
            text = "\n".join(body)
            if "ttng.warp_group_dot " in text and "scf.for " not in text:
                loops.append((line, text))
        assert len(loops) == 1
        header, body = loops[0]
        assert re.search(r"tensor<\d+x16xf32, #mma", header), header
        conversions = [line for line in body.splitlines() if "ttg.convert_layout" in line]
        assert not any(re.search(r": tensor<[^>]+, #mma\w*> ->", line) for line in conversions), conversions
        assert any(re.search(r"ttg.async_copy_global_to_local .*-> <128x16xf8E4M3FN", line)
                   for line in body.splitlines()), "FP8 activation matrix must participate in the K prefetch pipeline"
        assert runtime.resource_report["spill_bytes"] == 0
    finally:
        runtime.close()


@pytest.mark.parametrize("task_n", [32, 64])
@pytest.mark.parametrize("stages", [1, 3])
@pytest.mark.parametrize("tokens,routes,rows,reduction,packed,distributed", [
    (1, 2, 72, 256, False, False),
    (1, 8, 512, 4096, False, True),
    (2, 16, 640, 1536, True, True),
])
def test_complete_k_matches_ordered_cooperative_bytes(tmp_path, task_n, stages, tokens, routes, rows, reduction, packed, distributed):
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (9, 0):
        pytest.skip("requires SM90")
    from triton.flagmega.runtime import load
    from triton.flagmega.runtime.module import GeneratedTirCallGraphModule as Runtime

    # Both implementations execute on an explicit physical mesh. The small
    # tail case retains one CTA and all 72 rows.
    mesh = fm.Placement((8, 16) if distributed else (1, 1), "yx", "bb")
    graph = stage_graph(tokens=tokens, routes=routes, rows=rows, reduction=reduction,
        packed=packed, placement=mesh, readonly=True, experts=max(17, routes + 1))
    values = stage_values(torch, graph)
    modules = [compile_stage(graph, mesh, True), compile_complete(graph, mesh, task_n, stages)]
    runtimes, arguments, outputs, guards = [], [], [], []
    inputs = {name: value.cuda() for name, value in values.items() if not name.endswith("_weight")}
    try:
        for label, module in zip(("reference", "complete"), modules, strict=True):
            runtime = load(write_artifact(module, tmp_path / label, target="nvidia-sm90", emit_executable=True,
                checkpoint=TensorCheckpoint(values)), device="cuda:0")
            runtimes.append(runtime)
            shapes = [(tokens, routes, rows // 8, 8) if packed else (tokens, routes, rows), (128,)]
            result = {}
            for spec in runtime.external_arguments:
                if spec["role"] != "result":
                    continue
                shape = shapes[len(result)]
                dtype = torch.bfloat16 if not result else torch.float32
                storage = torch.full((prod(shape) + 64,), -123., device="cuda", dtype=dtype)
                result[spec["value"]] = storage[32:-32].reshape(shape)
                guards.extend((storage[:32], storage[-32:]))
            outputs.append(tuple(result.values()))
            arguments.append(tuple({**inputs, **result}[spec["value"]] for spec in runtime.external_arguments))
        for mutation in range(3):
            if mutation:
                inputs["dispatched"].mul_(1.03125).add_(1 / 128)
                inputs["router_expert_ids"].add_(3).remainder_(values["gate_weight"].shape[0])
                inputs["gate_proj_scale"].mul_(1.03125)
                inputs["up_input_scale"].add_(1 / 32)
            for runtime, actuals, result in zip(runtimes, arguments, outputs, strict=True):
                for value in result:
                    value.fill_(float("nan"))
                Runtime.prepare(runtime, *actuals)
                Runtime.run_into(runtime, *actuals)
                torch.cuda.synchronize()
                assert runtime.resource_report["spill_bytes"] == 0
            for reference, actual in zip(*outputs, strict=True):
                assert torch.equal(reference.view(torch.uint8), actual.view(torch.uint8))
            assert all(bool((guard == -123.).all()) for guard in guards)
            torch.testing.assert_close(outputs[1][1], inputs["sentinel"] * 2, rtol=0, atol=0)
    finally:
        for runtime in runtimes:
            runtime.close()
