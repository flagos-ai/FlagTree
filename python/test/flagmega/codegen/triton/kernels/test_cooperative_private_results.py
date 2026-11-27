# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Raw blocks are shared; each owner may merge into its own private result."""

import pytest
from triton.flagmega import ir as fm
from triton.flagmega.artifacts import write_artifact
from triton.flagmega.codegen.triton import render_triton_package

from python.test.flagmega.codegen.triton.kernels.block_fp8.test_block_fp8_cooperative_wgmma import (
    TensorCheckpoint,
    compile_projection,
    projection_graph,
    projection_values,
)
from python.test.flagmega.codegen.triton.kernels.matmul_glu.test_matmul_glu_cooperative_wgmma import (
    compile_glu,
    glu_graph,
    glu_values,
)
from python.test.flagmega.codegen.triton.kernels.sparse_experts.test_cooperative_wgmma import (
    compile_stage,
    stage_graph,
    stage_values,
)


def private_stage(family, *, private=True, placement=None):
    from math import prod

    mesh = placement or fm.Placement((8, 16), "yx", "bb")
    rows = 4096 if prod(size for size, level in zip(mesh.hierarchy, mesh.hierarchy_levels) if level != "b") > 1 else 1024
    if family == "block_fp8":
        graph = projection_graph(n=rows, k=256, lanes=(8,), placement=mesh, readonly=True,
                                 private_result=private)
        return graph, compile_projection(graph, mesh)
    if family == "matmul_glu":
        graph = glu_graph(n=rows, k=256, placement=mesh, readonly=True, private_result=private)
        return graph, compile_glu(graph, mesh)
    graph = stage_graph(tokens=1, routes=8, rows=512, reduction=256, experts=11, packed=True,
                        placement=mesh, readonly=True, private_result=private)
    return graph, compile_stage(graph, mesh, True)


@pytest.mark.parametrize("family", ("block_fp8", "matmul_glu", "sparse_experts_gate_up"))
def test_normal_codegen_keeps_owner_local_cooperative_results_private(tmp_path, family):
    _, module = private_stage(family)
    plan = fm.verify_buffer_plan(module)
    calls = [n for n in module.nodes if n.op == "tir.call"
             and (d := fm.kernel_dispatch_for_call(module, n)) is not None
             and d.microkernel.family == family]
    assert len(calls) == 1
    result = plan.buffer_map[calls[0].id]
    assert result.storage == "block_local_data"
    assert result.distributed_storage_kind is fm.DistributedBufferStorageKind.COMPACT_LOCAL
    report = render_triton_package(module, tmp_path / "codegen")
    source = (tmp_path / "codegen/generated_kernels.py").read_text()
    assert "__partials(" in source and "__merge(" in source
    assert any(p["storage"] == "block_local_data" for p in report["runtime_binding"]["pools"])


@pytest.mark.parametrize("family", ("block_fp8", "matmul_glu", "sparse_experts_gate_up"))
def test_private_merge_matches_shared_stage_and_preserves_guards(tmp_path, family):
    from math import prod

    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (9, 0):
        pytest.skip("requires SM90")
    from triton.flagmega.runtime import load
    from triton.flagmega.runtime.module import GeneratedTirCallGraphModule

    graph, private_module = private_stage(family)
    _, shared_module = private_stage(family, private=False)
    data = {"block_fp8": projection_values, "matmul_glu": glu_values,
            "sparse_experts_gate_up": stage_values}[family](torch, graph)
    device_data = {name: value.cuda() for name, value in data.items() if not name.endswith("weight")}
    runtimes, outputs, guards, arguments = [], [], [], []
    try:
        for label, module in (("shared", shared_module), ("private", private_module)):
            artifact = write_artifact(module, tmp_path / label, target="nvidia-sm90", emit_executable=True,
                                      checkpoint=TensorCheckpoint(data))
            runtime = load(artifact, device="cuda:0")
            runtimes.append(runtime)
            current = {}
            for spec in runtime.external_arguments:
                if spec["role"] != "result":
                    continue
                descriptor = runtime.buffer_plan.buffer_map[spec["buffer"]]
                dtype = descriptor.dtype
                lanes = dtype.lanes if isinstance(dtype, fm.VectorType) else ()
                scalar = dtype.elem_type if lanes else dtype
                shape = (*descriptor.component_shape, *lanes)
                storage = torch.full((prod(shape) + 64,), 123., dtype=getattr(torch, scalar.value), device="cuda")
                current[spec["value"]] = storage[:-64].reshape(shape)
                guards.append(storage[-64:])
            outputs.append(tuple(current.values()))
            arguments.append(tuple(({**device_data, **current})[a["value"]] for a in runtime.external_arguments))
        for mutation in range(4):
            if mutation:
                for name, value in device_data.items():
                    if name in {"value", "dispatched"}:
                        value.fill_((mutation - 2) / 8)
                    elif name.endswith("scale"):
                        value.mul_(1.03125)
            for runtime, args in zip(runtimes, arguments, strict=True):
                GeneratedTirCallGraphModule.prepare(runtime, *args)
                GeneratedTirCallGraphModule.run_into(runtime, *args)
                torch.cuda.synchronize()
                assert runtime.resource_report["grid"] == [128, 1, 1]
                assert runtime.resource_report["spill_bytes"] == 0
            for shared, private in zip(outputs[0], outputs[1], strict=True):
                expected = shared.float()
                assert torch.equal(private.view(torch.uint8), expected.view(torch.uint8))
            assert all(torch.all(guard == 123.) for guard in guards)
            torch.testing.assert_close(outputs[1][1], device_data["sentinel"] * 2, rtol=0, atol=0)
    finally:
        for runtime in reversed(runtimes):
            runtime.close()
