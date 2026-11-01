# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: MIT
"""The local cooperative MLA override reads complete private input replicas."""

from importlib import import_module
from pathlib import Path

import pytest
import torch
from triton.flagmega import ir as fm
from triton.flagmega.artifacts import write_artifact
from triton.flagmega.codegen.triton import render_triton_package
from triton.flagmega.compiler import Compiler
from triton.flagmega.options import CompileOptions
from triton.flagmega.runtime import load
from triton.flagmega.runtime.module import GeneratedTirCallGraphModule
from triton.flagmega.targets import NvidiaSm90Target

from python.test.flagmega.ir.ops.nn.test_mla_projection import graph


def _profile(monkeypatch):
    from triton.flagmega.codegen.triton.templates import TritonTemplateRegistry

    # Local template overrides must not leak into unrelated generic-target tests.
    monkeypatch.setattr(TritonTemplateRegistry, "render", TritonTemplateRegistry.render)
    tutorial = Path(__file__).parents[3] / "tutorials/flagmega/04-glm5.3-flash-fp8/nvidia-h800"
    monkeypatch.syspath_prepend(str(tutorial))
    profile = import_module("local_optimizations.native_tp4")
    override = import_module("local_optimizations.mla")
    monkeypatch.setattr(TritonTemplateRegistry, "render", override.override)
    return profile


def test_native_mla_worksharing_declares_shared_results_only_in_its_profile(monkeypatch):
    profile = _profile(monkeypatch)
    selected, = (i for i in profile.native_implementation_model().implementations if i.id == "tir.mla_projection.mma")
    assert selected.facts.get("result_memory_scope", "inferred") == "chip"
    ordinary, = (i for i in NvidiaSm90Target().triton_implementation_model.implementations if i.id == selected.id)
    assert ordinary.facts.get("result_memory_scope", "inferred") == "inferred"


@pytest.mark.parametrize("direction", ("absorb_query", "expand_value"))
@pytest.mark.parametrize("fp8", (False, True))
def test_native_worksharing_preserves_complete_head_owners(tmp_path, monkeypatch, direction, fp8):
    from python.test.flagmega.ir.ops.nn._mla_head_fixtures import graph as head_graph
    from triton.flagmega.ir.distributed_type import scale_split_units

    profile = _profile(monkeypatch)
    import_module("local_optimizations.mla")
    placement = fm.Placement((4, 8, 16), "cyx", "cbb")
    head = fm.SBP.split_contiguous((0,), 2)
    source = head_graph(direction=direction, fp8=fp8, mesh=placement,
                        head_policy=head, row_policy=scale_split_units(head, 32, 1))
    compiler = Compiler(CompileOptions(placement=placement))
    compiler.target = NvidiaSm90Target(triton_implementation_model=profile.native_implementation_model())
    compiled = compiler.compile(source).module
    fm.verify_buffer_plan(compiled)
    render_triton_package(compiled, tmp_path / "codegen")


@pytest.mark.parametrize("changed", ("coordinate_space", "local_capacity_shape", "distributed_backing_type"))
def test_implicit_broadcast_proof_rejects_incomplete_physical_tensors(monkeypatch, changed):
    from triton.flagmega.codegen.triton.call_abi import describe_kernel_call_abis
    from triton.flagmega.codegen.triton.mla_renderer import mla_projection_call
    from triton.flagmega.errors import CodegenError

    profile = _profile(monkeypatch)
    placement = fm.Placement((8, 16), "yx", "bb")
    compiler = Compiler(CompileOptions(placement=placement))
    compiler.target = NvidiaSm90Target(triton_implementation_model=profile.native_implementation_model())
    source = graph("absorb_query", (2, 4), input_lanes=(2, 4), tokens=1,
                   placement=placement, private_input=True)
    compiled = compiler.compile(source).module
    raw = next(call for call in describe_kernel_call_abis(compiled) if call["family"] == "mla_projection")
    for section in ("inputs", "outputs", "workspaces"):
        for argument in raw[section]:
            for binding in argument["buffers"]:
                binding.setdefault("runtime_argument", argument["formal"])
    weight = next(arg["buffers"][0]["abi"] for arg in raw["inputs"] if arg["formal"] == "weight")
    if changed == "coordinate_space":
        weight[changed] = "local"
    elif changed == "local_capacity_shape":
        weight[changed] = (1, 1)
    else:
        weight[changed] = raw["inputs"][0]["buffers"][0]["abi"]["distributed_type"]
    with pytest.raises(CodegenError, match="complete canonical"):
        mla_projection_call(raw)


@pytest.mark.parametrize("direction", ("absorb_query", "expand_value"))
def test_native_mla_codegen_accepts_complete_private_value_replicas(tmp_path, monkeypatch, direction):
    profile = _profile(monkeypatch)
    import_module("local_optimizations.mla")
    placement = fm.Placement((8, 16), "yx", "bb")
    source = graph(direction, (2, 4), input_lanes=(2, 4), tokens=1, placement=placement, private_input=True)
    compiler = Compiler(CompileOptions(placement=placement))
    compiler.target = NvidiaSm90Target(triton_implementation_model=profile.native_implementation_model())
    module = compiler.compile(source).module
    plan = fm.verify_buffer_plan(module)
    call, = (n for n in module.nodes if n.op == "tir.call"
             and (d := fm.kernel_dispatch_for_call(module, n)) is not None
             and d.microkernel.family == "mla_projection")
    assert plan.buffer_map[call.inputs[0]].storage == "block_local_data"
    render_triton_package(module, tmp_path / "codegen")


@pytest.mark.parametrize("direction", ("absorb_query", "expand_value"))
@pytest.mark.parametrize("placement", (
    fm.Placement((8, 16), "yx", "bb"),
    fm.Placement((4, 8, 16), "cyx", "cbb"),
    fm.Placement((8, 4, 16), "ycx", "bcb"),
))
def test_native_mla_internal_result_remains_shared(tmp_path, monkeypatch, direction, placement):
    profile = _profile(monkeypatch)
    import_module("local_optimizations.mla")
    source = graph(direction, (2, 4), input_lanes=(2, 4), tokens=1, placement=placement,
                   private_input=True, consume_output=True)
    compiler = Compiler(CompileOptions(placement=placement))
    compiler.target = NvidiaSm90Target(triton_implementation_model=profile.native_implementation_model())
    module = compiler.compile(source).module
    plan = fm.verify_buffer_plan(module)
    call, = (n for n in module.nodes if n.op == "tir.call"
             and (d := fm.kernel_dispatch_for_call(module, n)) is not None
             and d.microkernel.family == "mla_projection")
    dispatch = fm.kernel_dispatch_for_call(module, call)
    assert dispatch.memory_effect_map["result"].scope is fm.MemoryAccessScope.CHIP
    assert plan.buffer_map[call.inputs[0]].storage == "block_local_data"
    assert plan.buffer_map[call.id].storage == "workspace"
    render_triton_package(module, tmp_path / "codegen")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("direction", ("absorb_query", "expand_value"))
@pytest.mark.parametrize("fp8", (False, True))
def test_native_mla_device_reads_private_replicas(tmp_path, monkeypatch, direction, fp8):
    profile = _profile(monkeypatch)
    import_module("local_optimizations.mla")
    placement = fm.Placement((8, 16), "yx", "bb")
    source = graph(direction, (2, 4), input_lanes=(2, 4), tokens=1, placement=placement,
                   private_input=True, consume_output=True, fp8=fp8)
    compiler = Compiler(CompileOptions(placement=placement))
    compiler.target = NvidiaSm90Target(triton_implementation_model=profile.native_implementation_model())
    module = compiler.compile(source).module
    artifact = write_artifact(module, tmp_path / "artifact", target="nvidia-sm90", emit_executable=True)
    runtime = load(artifact, device="cuda:0")
    width = 16 if direction == "absorb_query" else 32
    output_width = 32 if direction == "absorb_query" else 8
    x = torch.empty((1, 2, width // 8, 2, 4), dtype=torch.float32, device="cuda")
    w = ((torch.arange(48 * 32, device="cuda") % 17) - 8).float().mul_(0.125).reshape(48, 32)
    values = {"x": x, "w": w.to(torch.float8_e4m3fn if fp8 else torch.bfloat16),
              "sentinel": torch.arange(128, dtype=torch.float32, device="cuda")}
    if fp8:
        values["scale"] = torch.full((6, 2), 0.5, device="cuda")
    output_backing = torch.full((64 + 2 * output_width + 64,), -91, dtype=torch.float32, device="cuda")
    output = output_backing[64:-64].reshape(1, 2, output_width)
    sentinel_backing = torch.full((64 + 128 + 64,), -93, dtype=torch.float32, device="cuda")
    sentinel_output = sentinel_backing[64:-64]
    result_values = iter((output, sentinel_output))
    values.update((spec["value"], next(result_values))
                  for spec in runtime.external_arguments if spec["role"] == "result")
    arguments = tuple(values[spec["value"]] for spec in runtime.external_arguments)
    try:
        GeneratedTirCallGraphModule.prepare(runtime, *arguments)
        for mutation in range(4):
            value = ((torch.arange(2 * width, device="cuda") + mutation) % 13 - 6).float()
            x.copy_((value * 0.125 + 1 / 2048).reshape(x.shape))
            if fp8:
                values["scale"].fill_(0.5 + mutation * 0.125)
            actual_weight = values["w"].float()
            if fp8:
                actual_weight = (actual_weight * values["scale"][0, 0]).bfloat16().float()
            selected = actual_weight.reshape(2, 24, 32)
            selected = selected[:, :16] if direction == "absorb_query" else selected[:, 16:].transpose(1, 2)
            rounded = x.reshape(1, 2, width).bfloat16().float()
            expected = torch.bmm(rounded.transpose(0, 1), selected).bfloat16().float().transpose(0, 1)
            output.fill_(float("nan"))
            sentinel_output.fill_(float("nan"))
            GeneratedTirCallGraphModule.run_into(runtime, *arguments)
            assert torch.equal(output.view(torch.uint8), expected.contiguous().view(torch.uint8))
            torch.testing.assert_close(sentinel_output, values["sentinel"] * 2, rtol=0, atol=0)
            assert bool((output_backing[:64] == -91).all() & (output_backing[-64:] == -91).all())
            assert bool((sentinel_backing[:64] == -93).all() & (sentinel_backing[-64:] == -93).all())
        assert runtime.resource_report["grid"] == [128, 1, 1]
        assert runtime.resource_report["spill_bytes"] == 0
    finally:
        runtime.close()
