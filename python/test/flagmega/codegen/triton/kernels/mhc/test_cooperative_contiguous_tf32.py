# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Independent contiguous TF32 splits cooperate without changing their sums."""

from dataclasses import replace

import pytest
from triton.flagmega import ir as fm
from triton.flagmega.artifacts import write_artifact
from triton.flagmega.compiler import Compiler
from triton.flagmega.options import CompileOptions
from triton.flagmega.targets import NvidiaSm90Target


def graph(*, tokens=1, hc=4, hidden=4096, splits=64, lanes=(8,), grid=True,
          collapse_reduction="sequential_fma", collapsed_dtype="float32", sinkhorn_iterations=3,
          stream_dtype="float32"):
    placement = fm.Placement((8, 16) if grid else (1, 1), "yx", "bb")

    class Graph(fm.Module):
        def forward(self):
            count = hc * (hc + 2)
            inputs = tuple(self.input(name, fm.tensor_type(dtype, shape), id=name)
                           for name, dtype, shape in (
                               ("streams", stream_dtype, (tokens, hc, hidden)),
                               ("projection", "float32", (count, hc * hidden)),
                               ("base", "float32", (count,)), ("scale", "float32", (3,))))
            stage = fm.F.nn.mhc_pre(*inputs, epsilon=1e-5, sinkhorn_epsilon=1e-6,
                sinkhorn_iterations=sinkhorn_iterations, projection_reduction="contiguous_tf32",
                projection_splits=splits, collapse_reduction=collapse_reduction,
                collapsed_dtype=collapsed_dtype, collapse_input_dtype="bfloat16", output_lanes=lanes, name="mhc")
            outputs = list(fm.F.tensors.get_items(stage, 0, 1, 2))
            parameters = list(inputs)
            if grid:
                kind = fm.tensor_type("float32", (128,))
                sentinel = self.input("sentinel", kind, id="sentinel")
                sharded = fm.DistributedType(kind, (fm.SBP.split_contiguous((0, 1), 1),), placement)
                local = fm.F.distributed.force_boxing(sentinel, sharded)
                outputs.append(fm.F.distributed.force_boxing(fm.F.math.add(local, local), kind))
                parameters.append(sentinel)
            self.function("main", parameters, outputs)

    return Graph(dialect="distributed", stage="frozen_constants", entry="main",
                 metadata={"auto_distribution": {"placement": placement.to_data()}}).build(), placement


def compile_stage(source, placement, *, cooperative, warps=8, include_stream=False):
    target = NvidiaSm90Target()
    catalog = target.triton_implementation_model
    if cooperative:
        from triton.flagmega.targets.nvidia.mhc import mhc_cooperative_contiguous_tf32_implementations
        extra = tuple(replace(value, parameters={**value.parameters, "compute_num_warps": warps})
                      for value in mhc_cooperative_contiguous_tf32_implementations())
        if include_stream:
            from triton.flagmega.targets.nvidia.mhc import mhc_cooperative_stream_fma_implementations
            extra += mhc_cooperative_stream_fma_implementations()
        catalog = replace(catalog, implementations=(*catalog.implementations, *extra),
                          preferences={**catalog.preferences, "mhc_pre": tuple(value.id for value in extra)})
    else:
        catalog = replace(catalog, implementations=tuple(
            replace(value, parameters={**value.parameters, "compute_num_warps": warps})
            if value.id == "tir.mhc_pre.contiguous_tf32" else value for value in catalog.implementations))
    target.triton_implementation_model = catalog
    compiler = Compiler(CompileOptions(placement=placement))
    compiler.target = target
    return compiler.compile(source).module


@pytest.mark.parametrize("tokens,hc,hidden,splits", [(1, 4, 4096, 64), (2, 3, 256, 4), (1, 1, 128, 1)])
def test_cooperative_tf32_has_explicit_workspace_and_retains_native_arithmetic(tmp_path, tokens, hc, hidden, splits):
    source, mesh = graph(tokens=tokens, hc=hc, hidden=hidden, splits=splits)
    result = compile_stage(source, mesh, cooperative=True)
    dispatch, = (definition.dispatch for definition in result.kernel_definitions
                 if definition.dispatch.semantic_op == "nn.mhc_pre")
    assert dispatch.microkernel.variant == "cooperative_contiguous_tf32"
    assert dispatch.semantic_attrs["projection_reduction"] == "contiguous_tf32"
    assert dispatch.semantic_attrs.get("projection_splits", 1) == splits
    assert dispatch.microkernel.facts["internal_grid_barriers"] == 2
    workspace, = dispatch.workspaces
    assert workspace.type == fm.tensor_type("float32", (tokens, (hc * (hc + 2) + 1) * splits + hc))
    assert workspace.alignment == 256 and workspace.lifetime == "invocation"
    assert workspace.memory_space == "workspace"
    assert fm.load_module(fm.emit_module(result, tmp_path / "cooperative.py")).semantic_hash == result.semantic_hash


def test_tf32_and_stream_fma_cooperative_variants_share_the_template_without_changing_contracts(tmp_path):
    from triton.flagmega.rules.neutral._utility import make_node

    source, mesh = graph()
    original = next(node for node in source.nodes if node.op == "nn.mhc_pre")
    second = make_node("nn.mhc_pre", "stream", tuple(source.node_map[value] for value in original.inputs),
                       {**original.attrs, "projection_reduction": "stream_fma_128", "projection_splits": 8}, {})
    fields = tuple(make_node("builtin.get_item", f"stream_{index}", (second,), {"index": index}, {}) for index in range(3))
    source = fm.verify_module(replace(source, nodes=(*source.nodes, second, *fields),
        functions=(replace(source.function_map[source.entry],
                           outputs=(*source.function_map[source.entry].outputs, *(node.id for node in fields))),)))
    result = compile_stage(source, mesh, cooperative=True, include_stream=True)
    stages = [definition.dispatch for definition in result.kernel_definitions if definition.dispatch.semantic_op == "nn.mhc_pre"]
    assert {stage.microkernel.variant for stage in stages} == {"cooperative_contiguous_tf32", "cooperative_stream_fma"}
    assert all(stage.microkernel.facts["internal_grid_barriers"] == 2 for stage in stages)
    artifact = write_artifact(result, tmp_path / "mixed", target="nvidia-sm90", emit_executable=True)
    text = (artifact / "generated_kernels.py").read_text()
    assert "# flagmega-kernel: mhc_pre/cooperative_contiguous_tf32" in text
    assert "# flagmega-kernel: mhc_pre/cooperative_stream_fma" in text


@pytest.mark.parametrize("grid", [False, True])
@pytest.mark.parametrize("collapse", ["sequential_fma", "sum"])
@pytest.mark.parametrize("tokens,hc,hidden,splits,lanes,warps,iterations", [
    (1, 4, 4096, 64, (8,), 8, 3), (2, 3, 256, 4, (2, 4), 8, 3),
    (1, 4, 64, 1, (), 4, 3), (1, 4, 4096, 64, (4,), 8, 20),
])
def test_device_stage_matches_serial_native_contract_bitwise(tmp_path, grid, collapse, tokens, hc, hidden, splits, lanes,
                                                           warps, iterations):
    import torch
    from triton.flagmega.runtime import load

    from python.test.flagmega.codegen.triton.kernels.mhc.test_mhc_runtime import _run, _skip_without_sm90

    _skip_without_sm90(torch)
    source, mesh = graph(tokens=tokens, hc=hc, hidden=hidden, splits=splits, lanes=lanes, grid=grid,
                         collapse_reduction=collapse, sinkhorn_iterations=iterations)
    runtimes = [load(write_artifact(compile_stage(source, mesh, cooperative=cooperative, warps=warps),
                                   tmp_path / str(cooperative), target="nvidia-sm90", emit_executable=True), device="cuda:0")
                for cooperative in (False, True)]
    generator = torch.Generator().manual_seed(20261018)
    values = {"streams": torch.randn((tokens, hc, hidden), generator=generator),
              "projection": torch.randn((hc * (hc + 2), hc * hidden), generator=generator) * .02,
              "base": torch.randn((hc * (hc + 2),), generator=generator),
              "scale": torch.tensor([.7, 1.1, .9])}
    shapes = ((tokens, hc), (tokens, hc, hc), (tokens, hidden))
    if grid:
        values["sentinel"] = torch.arange(128, dtype=torch.float32)
        shapes += ((128,),)
    outputs, guards = [], []
    for runtime in runtimes:
        current = {}
        for name, shape in zip(runtime.ir_module.function_map[runtime.ir_module.entry].outputs, shapes, strict=True):
            count = torch.tensor(shape).prod().item()
            whole = torch.full((count + 128,), -123., dtype=torch.float32, device="cuda")
            current[name] = whole[64:-64].reshape(shape)
            guards.extend((whole[:64], whole[-64:]))
        outputs.append(current)
    try:
        for _ in range(3):
            inputs = {name: value.cuda() for name, value in values.items()}
            for runtime, result in zip(runtimes, outputs, strict=True):
                _run(runtime, inputs, result, torch)
                assert runtime.resource_report["spill_bytes"] == 0
                assert runtime.resource_report["grid"] == ([128, 1, 1] if grid else [1, 1, 1])
            for before, after in zip(outputs[0].values(), outputs[1].values(), strict=True):
                assert torch.equal(before.view(torch.uint8), after.view(torch.uint8))
            assert all(bool((guard == -123).all()) for guard in guards)
            if grid:
                assert torch.equal(tuple(outputs[1].values())[-1], inputs["sentinel"] * 2)
            values["streams"].add_(1 / 128)
    finally:
        for runtime in runtimes:
            runtime.close()
