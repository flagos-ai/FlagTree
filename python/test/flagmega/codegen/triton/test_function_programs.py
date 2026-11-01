# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Stage boundaries of repeated functions enter ordinary storage/DAG planning."""
from dataclasses import replace

import pytest

from triton.flagmega import ir as fm
from triton.flagmega.artifacts import write_artifact
from triton.flagmega.codegen.triton import render_triton_package
from triton.flagmega.compiler import Compiler
from triton.flagmega.errors import CodegenError, IRVerificationError
from triton.flagmega.passes.functions import outline_function_stages
from triton.flagmega.runtime import load
from triton.flagmega.runtime.module import GeneratedTirCallGraphModule as Runtime
from triton.flagmega.targets import NvidiaSm90Target

from python.test.flagmega.passes.functions.test_outline_stages import STAGES, reusable_graph


def compile_repeated_stages(*, outline=True, packed=False):
    class Target(NvidiaSm90Target):
        def propose_tir(self, module):
            if outline:
                module = outline_function_stages(module, "worker", STAGES)
            module = super().propose_tir(module)
            dependencies = ((), (0,), (1,), (0,), (3,), (2, 4))
            programs = tuple({
                "symbol": f"program_{call}_{stage}",
                "calls": (f"call{call}.stage{stage}" if outline else STAGES[stage].nodes[0],),
                "num_warps": 4,
                "depends_on": tuple(f"program_{call}_{parent}" for parent in parents) or
                              ((f"program_{call - 1}_5",) if call else ()),
            } for call in range(2) for stage, parents in enumerate(dependencies))
            return replace(module, metadata={**module.metadata, "kernel_program_partition": programs})

    compiler = Compiler()
    compiler.target = Target()
    return compiler.compile(reusable_graph(packed=packed)).module


def test_nested_function_operations_are_not_executable_entry_cuts(tmp_path):
    # An explicit valid graph reproduces the missing boundary: v2 cuts address
    # entry calls, so a nested op id cannot name a kernel launch boundary.
    with pytest.raises((CodegenError, IRVerificationError), match="straight-line call|exactly partition|actual entry call"):
        render_triton_package(compile_repeated_stages(outline=False), tmp_path)


@pytest.mark.parametrize("packed", [False, True])
def test_outlined_calls_use_normal_typed_allocation_codegen_and_resume(tmp_path, packed):
    module = compile_repeated_stages(packed=packed)
    reloaded = fm.load_module(fm.emit_module(module, tmp_path / "outlined.py"))
    assert reloaded.semantic_hash == module.semantic_hash
    package = render_triton_package(reloaded, tmp_path / "source")
    assert len(package["kernels"]) == 12
    assert len(package["device_functions"]) == 6
    from triton.flagmega.passes.functions import function_nodes
    assert sum(n.op == "tir.call" and n.attrs["callee"] in module.kernel_callable_map
               for f in module.functions if f.name != module.entry for n in function_nodes(module, f)) == 6
    streams = {p["symbol"]: p["stream_index"] for p in package["kernels"]}
    assert streams["program_0_1"] != streams["program_0_3"]
    assert package["kernels"][6]["depends_on"] == ["program_0_5"]


@pytest.mark.parametrize("packed", [False, True])
def test_device_repeated_stages_preserve_distinct_actuals_rebinding_and_graph(tmp_path, packed):
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (9, 0):
        pytest.skip("requires SM90")
    module = compile_repeated_stages(packed=packed)
    runtime = load(write_artifact(module, tmp_path / "artifact", target="nvidia-sm90", emit_executable=True),
                   device="cuda:0")
    feeds = {}
    generator = torch.Generator(device="cuda").manual_seed(431)
    for value in module.function_map[module.entry].parameters:
        kind = fm.logical_type(module.node_map[value].type)
        shape = (*tuple(d.fixed_value for d in kind.shape), *getattr(kind.dtype, "lanes", ()))
        feeds[value] = torch.empty(shape, dtype=torch.bfloat16, device="cuda")
    output_shape = (1, 16, 8) if packed else (1, 32)
    guards = [torch.full((160 if packed else 64,), -123.0, dtype=torch.bfloat16, device="cuda") for _ in range(6)]
    outputs = [value[16:-16].reshape(output_shape) for value in guards]
    stream = torch.cuda.Stream()

    def arguments():
        results = iter(outputs)
        return tuple(next(results) if spec["role"] == "result" else feeds[spec["value"]]
                     for spec in runtime.external_arguments)

    try:
        Runtime.prepare(runtime, *arguments())
        assert all(report["spill_bytes"] == 0 for report in runtime.resource_report)
        assert len({p.compiled_kernel.hash for p in runtime._prepared}) == 6
        assert all(runtime._prepared[i].compiled_kernel is runtime._prepared[i + 6].compiled_kernel
                   for i in range(6))
        for mutation in range(6):
            for value in feeds.values():
                value.copy_((torch.randint(-4, 5, value.shape, generator=generator, device="cuda").float() / 16).bfloat16())
            if mutation == 5:
                feeds["x0"].zero_()
            for value in outputs:
                value.fill_(float("nan"))
            stream.wait_stream(torch.cuda.current_stream())
            Runtime.run_into(runtime, *arguments(), stream=stream.cuda_stream)
            torch.cuda.synchronize()
            expected = []
            for x in (feeds["x0"], feeds["x1"]):
                prefix = (x.float() + feeds["bias"].float()).bfloat16()
                if packed:
                    left = ((prefix.float() + feeds["a"].float()).bfloat16().float() * feeds["b"].float()).bfloat16()
                    right = ((prefix.float() + feeds["c"].float()).bfloat16().float() * feeds["d"].float()).bfloat16()
                else:
                    left = ((prefix.double() @ feeds["a"].double().T).bfloat16().double() @ feeds["b"].double().T).bfloat16()
                    right = ((prefix.double() @ feeds["c"].double().T).bfloat16().double() @ feeds["d"].double().T).bfloat16()
                expected.extend((left, right, (left.float() + right.float()).bfloat16()))
            assert all(torch.equal(actual, gold) for actual, gold in zip(outputs, expected, strict=True))
            assert all(bool((v[:16] == -123).all() & (v[-16:] == -123).all()) for v in guards)
            if mutation == 2:
                feeds = {name: value.clone() for name, value in feeds.items()}
        with torch.cuda.graph(capture := torch.cuda.CUDAGraph(), stream=stream):
            Runtime.run_into(runtime, *arguments(), stream=stream.cuda_stream)
        for _ in range(3):
            capture.replay()
        torch.cuda.synchronize()
        assert all(torch.equal(actual, gold) for actual, gold in zip(outputs, expected, strict=True))
        # Composite decode captures prefill and decode before replaying the
        # older graph. Exercise shared DAG events and scratch in that order,
        # with fresh data so stale workspace cannot mask a missing edge.
        with torch.cuda.graph(newer := torch.cuda.CUDAGraph(), stream=stream):
            Runtime.run_into(runtime, *arguments(), stream=stream.cuda_stream)
        for mutation, graph in enumerate((capture, newer, capture, newer)):
            for name, value in feeds.items():
                value.fill_(mutation / 16 if name in {"x0", "x1"} else 0. if name == "bias" else 1 / 16)
            for value in outputs:
                value.fill_(float("nan"))
            graph.replay()
            torch.cuda.synchronize()
            branch = (mutation + 1) / 256 if packed else 2 * mutation
            for index, actual in enumerate(outputs):
                assert torch.equal(actual, torch.full_like(actual, branch * (2 if index % 3 == 2 else 1)))
            assert all(bool((v[:16] == -123).all() & (v[-16:] == -123).all()) for v in guards)
    finally:
        runtime.close()
