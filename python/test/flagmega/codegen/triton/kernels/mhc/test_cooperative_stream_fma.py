# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Cooperative mHC keeps its arithmetic and declares every cross-CTA value."""

from copy import deepcopy
from dataclasses import replace

import pytest
from triton.flagmega import ir as fm
from triton.flagmega.artifacts import write_artifact
from triton.flagmega.compiler import Compiler
from triton.flagmega.errors import CodegenError
from triton.flagmega.options import CompileOptions
from triton.flagmega.targets import NvidiaSm90Target
from triton.flagmega.targets.nvidia.mhc import mhc_cooperative_stream_fma_implementations

from python.test.flagmega.ir.ops.nn.test_mhc_ordered_projection import module


def cooperative_target(compute_warps=8):
    target = NvidiaSm90Target()
    catalog = target.triton_implementation_model
    implementations = tuple(replace(value, parameters={**value.parameters, "compute_num_warps": compute_warps})
                            for value in mhc_cooperative_stream_fma_implementations())
    target.triton_implementation_model = replace(
        catalog,
        implementations=(*catalog.implementations, *implementations),
        preferences={**catalog.preferences,
                     "mhc_pre": tuple(value.id for value in implementations)},
    )
    return target


def compile_cooperative(graph, placement=None, compute_warps=8):
    compiler = Compiler(CompileOptions(placement=placement))
    compiler.target = cooperative_target(compute_warps)
    return compiler.compile(graph).module


def distributed_module(*, tokens, hc, hidden, splits, output_lanes, placement, **attrs):
    """An independent sharded successor makes the ordinary launch use 128 CTAs."""
    class Graph(fm.Module):
        def forward(self):
            count = hc * (hc + 2)
            inputs = tuple(self.input(name, fm.tensor_type(dtype, shape), id=name)
                           for name, dtype, shape in (
                               ("streams", "float32", (tokens, hc, hidden)),
                               ("projection", "bfloat16", (count, hc * hidden)),
                               ("base", "float32", (count,)), ("scale", "float32", (3,))))
            stage = fm.F.nn.mhc_pre(*inputs, epsilon=1e-5, sinkhorn_epsilon=1e-6, sinkhorn_iterations=3,
                                    projection_reduction="stream_fma_128", projection_splits=splits,
                                    output_lanes=output_lanes, **attrs)
            outputs = list(fm.F.tensors.get_items(stage, 0, 1, 2))
            sentinel_type = fm.tensor_type("float32", (128,))
            sentinel = self.input("sentinel", sentinel_type, id="sentinel")
            split = fm.DistributedType(sentinel_type, (fm.SBP.split_contiguous((0, 1), 1),), placement)
            local = fm.F.distributed.force_boxing(sentinel, split)
            outputs.append(fm.F.distributed.force_boxing(fm.F.math.add(local, local), sentinel_type))
            self.function("main", (*inputs, sentinel), outputs)

    return Graph(dialect="distributed", stage="frozen_constants", entry="main",
                 metadata={"auto_distribution": {"placement": placement.to_data()}}).build()


@pytest.mark.parametrize("tokens,hc,hidden,splits", [(1, 4, 4096, 8), (3, 3, 512, 2), (1, 1, 128, 1)])
def test_workspace_and_barrier_contract_survive_normal_lowering(tmp_path, tokens, hc, hidden, splits):
    compiled = compile_cooperative(module(tokens=tokens, hc=hc, hidden=hidden, splits=splits,
                                          output_lanes=(8,), collapse_reduction="sequential_fma"))
    dispatch, = (value.dispatch for value in compiled.kernel_definitions
                 if value.dispatch.semantic_op == "nn.mhc_pre")
    assert dispatch.microkernel.variant == "cooperative_stream_fma"
    assert dispatch.microkernel.facts["internal_grid_barriers"] == 2
    assert {"cooperative_grid", "grid_sync"} <= set(dispatch.microkernel.requires)
    assert dispatch.semantic_attrs["projection_reduction"] == "stream_fma_128"
    assert dispatch.semantic_attrs.get("projection_splits", 1) == splits
    workspace, = dispatch.workspaces
    assert workspace.name == "projection_partials"
    assert workspace.type == fm.tensor_type("float32", (tokens, (hc * (hc + 2) + 1) * splits + hc))
    assert workspace.memory_space == "workspace"
    assert workspace.lifetime == "invocation"
    assert workspace.alignment == 256
    resumed = fm.load_module(fm.emit_module(compiled, tmp_path / "selected.py"))
    assert resumed.semantic_hash == compiled.semantic_hash


@pytest.mark.parametrize("reduction", ["parallel", "contiguous_tf32"])
def test_cooperative_preference_cannot_change_the_projection_arithmetic(reduction):
    compiled = compile_cooperative(module(reduction=reduction, splits=1))
    dispatch, = (value.dispatch for value in compiled.kernel_definitions
                 if value.dispatch.semantic_op == "nn.mhc_pre")
    assert dispatch.microkernel.variant != "cooperative_stream_fma"
    assert not dispatch.workspaces


def test_cooperative_codegen_uses_three_storage_separated_phases(tmp_path):
    compiled = compile_cooperative(module(hc=4, hidden=4096, splits=8, output_lanes=(8,)))
    artifact = write_artifact(compiled, tmp_path / "cooperative", target="nvidia-sm90", emit_executable=True)
    source = (artifact / "generated_kernels.py").read_text()
    assert "# flagmega-kernel: mhc_pre/cooperative_stream_fma" in source
    assert "_flagmega_mhc_project_split_pair(" in source
    assert "__project(" in source and "__mix(" in source and "__collapse(" in source
    assert "tl.zeros((2, 1, 128), tl.float32)" in source


@pytest.mark.parametrize("cooperative", (False, True))
def test_cooperative_results_declare_cross_block_publication(cooperative):
    graph = module(hc=4, hidden=4096, splits=8, output_lanes=(8,))
    compiled = compile_cooperative(graph) if cooperative else Compiler().compile(graph).module
    dispatch, = (d.dispatch for d in compiled.kernel_definitions if d.dispatch.semantic_op == "nn.mhc_pre")
    expected = fm.MemoryAccessScope.CHIP if cooperative else fm.MemoryAccessScope.INFERRED
    assert [dispatch.memory_effect_map[name].scope for name in dispatch.outputs] == [expected] * 3


@pytest.mark.parametrize("tokens,hc,hidden,splits,lanes,warps", [(1, 4, 4096, 8, (8,), 8),
                                                               (3, 3, 512, 2, (2, 4), 8),
                                                               (1, 4, 4096, 8, (8,), 16),
                                                               (2, 1, 128, 1, (8,), 8)])
@pytest.mark.parametrize("cooperative_grid", [False, True])
@pytest.mark.parametrize("collapse_reduction", ["sequential_fma", "sum"])
def test_cooperative_complete_stage_matches_the_original_bitwise(tmp_path, tokens, hc, hidden, splits, lanes, warps,
                                                               cooperative_grid, collapse_reduction):
    import torch
    from triton.flagmega.runtime import load

    from python.test.flagmega.codegen.triton.kernels.mhc.test_mhc_runtime import _run, _skip_without_sm90

    _skip_without_sm90(torch)
    placement = (fm.Placement((8, 16), "yx", "bb") if cooperative_grid
                 else fm.Placement((1, 1), "yx", "bb"))
    builder = distributed_module if cooperative_grid else module
    graph = builder(tokens=tokens, hc=hc, hidden=hidden, splits=splits, output_lanes=lanes,
                    collapsed_dtype="bfloat16", collapse_input_dtype="bfloat16",
                    collapse_reduction=collapse_reduction, **({"placement": placement} if cooperative_grid else {}))
    # Test the selected kernel at an explicit physical topology. The general
    # high-level attention-decomposition stage requires a nontrivial block
    # axis even when this graph contains no attention operation.
    graph = replace(graph, dialect="distributed", stage="frozen_constants",
                    metadata={**graph.metadata, "auto_distribution": {"placement": placement.to_data()}})
    baseline_compiler = Compiler(CompileOptions(placement=placement))
    target = baseline_compiler.target
    target.triton_implementation_model = replace(
        target.triton_implementation_model,
        implementations=tuple(replace(value, parameters={**value.parameters, "compute_num_warps": warps})
                              if value.id == "tir.mhc_pre.stream_fma" else value
                              for value in target.triton_implementation_model.implementations),
    )
    paths = [write_artifact(compiled, tmp_path / name, target="nvidia-sm90", emit_executable=True)
             for name, compiled in (("baseline", baseline_compiler.compile(graph).module),
                                    ("cooperative", compile_cooperative(graph, placement, warps)))]
    runtimes = [load(path, device="cuda:0") for path in paths]
    generator = torch.Generator().manual_seed(739)
    values = {"streams": torch.randn((tokens, hc, hidden), generator=generator, device="cpu"),
              "projection": torch.randn((hc * (hc + 2), hc * hidden), generator=generator) * .02,
              "base": torch.randn((hc * (hc + 2),), generator=generator),
              "scale": torch.tensor([.7, 1.1, .9])}
    shapes = ((tokens, hc), (tokens, hc, hc), (tokens, hidden))
    dtypes = (torch.float32, torch.float32, torch.bfloat16)
    if cooperative_grid:
        values["projection"] = values["projection"].bfloat16()
        values["sentinel"] = torch.arange(128, dtype=torch.float32)
        shapes += ((128,),)
        dtypes += (torch.float32,)
    outputs, guards = [], []
    for runtime in runtimes:
        names = runtime.ir_module.function_map[runtime.ir_module.entry].outputs
        current, current_guards = {}, []
        for name, shape, dtype in zip(names, shapes, dtypes, strict=True):
            count = torch.tensor(shape).prod().item()
            storage = torch.full((count + 32,), 123., dtype=dtype, device="cuda")
            current[name] = storage[:count].reshape(shape)
            current_guards.append(storage[count:])
        outputs.append(current)
        guards.append(current_guards)
    try:
        for _ in range(2):
            device_values = {name: value.cuda() for name, value in values.items()}
            for runtime, current in zip(runtimes, outputs, strict=True):
                _run(runtime, device_values, current, torch)
                assert runtime.resource_report["spill_bytes"] == 0
                assert runtime.resource_report["grid"] == ([128, 1, 1] if cooperative_grid else [1, 1, 1])
                if cooperative_grid:
                    torch.testing.assert_close(tuple(current.values())[-1], device_values["sentinel"] * 2,
                                               rtol=0, atol=0)
            for original, cooperative in zip(outputs[0].values(), outputs[1].values(), strict=True):
                torch.testing.assert_close(cooperative, original, rtol=0, atol=0)
            for guard_set in guards:
                assert all(torch.all(value == 123.) for value in guard_set)
            values["streams"].add_(1 / 128)
    finally:
        for runtime in runtimes:
            runtime.close()


@pytest.fixture(scope="module", params=("stream_fma_128", "contiguous_tf32"))
def cooperative_raw(request):
    from triton.flagmega.codegen.triton.call_abi import describe_kernel_call_abis
    if request.param == "stream_fma_128":
        compiled = compile_cooperative(module(hc=4, hidden=4096, splits=8, output_lanes=(8,)))
    else:
        from python.test.flagmega.codegen.triton.kernels.mhc.test_cooperative_contiguous_tf32 import (
            compile_stage,
        )
        compiled = compile_stage(module(hc=4, hidden=4096, splits=64, output_lanes=(8,),
                                        reduction="contiguous_tf32"), None, cooperative=True)
    raw = next(value for value in describe_kernel_call_abis(compiled) if value["family"] == "mhc_pre")
    for group in ("inputs", "outputs", "workspaces"):
        for value in raw[group]:
            for buffer in value["buffers"]:
                buffer.setdefault("runtime_argument", value["formal"])
    return raw


@pytest.mark.parametrize("group,changes", [
    ("outputs", {"memory_sharing_scope": "block"}),
    ("outputs", {"coordinate_space": "local"}),
    ("workspaces", {"scalar_dtype": "bfloat16"}),
    ("workspaces", {"scalar_storage_strides": (205, 1)}),
    ("workspaces", {"memory_sharing_scope": "block"}),
    ("inputs", {"distributed_backing_type": {"partial": "sum"}}),
    ("inputs", {"distributed_type": {"exclusive": {"axes": [0]}, "axis_policies": []}}),
])
def test_cooperative_rejects_unshared_outputs_and_unproved_read_aliases(cooperative_raw, group, changes):
    from triton.flagmega.codegen.triton.mhc_renderer import mhc_pre_call
    raw = deepcopy(cooperative_raw)
    abi = raw[group][0]["buffers"][0]["abi"]
    if "distributed_type" in changes:
        changes = {**changes, "distributed_type": {**(abi["distributed_type"] or {}), **changes["distributed_type"]}}
    abi.update(changes)
    with pytest.raises(CodegenError, match="Cooperative mHC"):
        mhc_pre_call(raw)


def test_cooperative_renderer_requires_explicit_barrier_facts(cooperative_raw):
    from triton.flagmega.codegen.triton.mhc_renderer import mhc_pre_call
    raw = deepcopy(cooperative_raw)
    raw["facts"]["internal_grid_barriers"] = 0
    with pytest.raises(CodegenError, match="two declared internal grid barriers"):
        mhc_pre_call(raw)


def test_cooperative_renderer_rejects_a_different_projection_arithmetic(cooperative_raw):
    from triton.flagmega.codegen.triton.mhc_renderer import mhc_pre_call
    raw = deepcopy(cooperative_raw)
    raw["variant"] = ("cooperative_contiguous_tf32" if raw["variant"] == "cooperative_stream_fma"
                      else "cooperative_stream_fma")
    with pytest.raises(CodegenError, match="projection_reduction contract"):
        mhc_pre_call(raw)
