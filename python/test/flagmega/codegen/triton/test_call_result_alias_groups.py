# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Produced tuple fields and their returned views use one device allocation."""

from dataclasses import replace

import pytest

from triton.flagmega.artifacts import write_artifact
from triton.flagmega.compiler import Compiler
from triton.flagmega.runtime import load
from triton.flagmega.targets import NvidiaSm90Target
from python.test.flagmega.passes.tir.bufferize.test_call_result_alias_groups import result_alias_graph


def compile_alias_results(*, program_dag, view_first):
    class Target(NvidiaSm90Target):
        def propose_tir(self, module):
            module = super().propose_tir(module)
            if not program_dag:
                return module
            programs = tuple(
                dict(symbol=name, calls=(call,), num_warps=4, depends_on=dependencies)
                for name, call, dependencies in (
                    ("first", "call0", ()), ("second", "call1", ("first",)),
                    ("sum", "output", ("second",)),
                )
            )
            return replace(module, metadata={**module.metadata, "kernel_program_partition": programs})

    compiler = Compiler()
    compiler.target = Target()
    return compiler.compile(result_alias_graph(
        dialect="ntt", stage="frozen_constants", view_first=view_first,
    )).module


@pytest.mark.parametrize("program_dag", [False, True])
@pytest.mark.parametrize("view_first", [False, True])
def test_device_returned_view_reads_the_produced_tuple_leaf(tmp_path, program_dag, view_first):
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (9, 0):
        pytest.skip("requires SM90")
    module = compile_alias_results(program_dag=program_dag, view_first=view_first)
    runtime = load(write_artifact(module, tmp_path / "artifact", target="nvidia-sm90", emit_executable=True),
                   device="cuda:0")
    guard = torch.full((48,), -123., dtype=torch.float32, device="cuda")
    output = guard[16:-16].reshape(2, 8)
    values = [torch.zeros_like(output) for _ in range(2)]
    generator = torch.Generator(device="cuda").manual_seed(857)
    stream = torch.cuda.Stream()

    def check():
        gold = (2 * values[0].double() + 2 * values[1].double()).float()
        assert torch.equal(output, gold)
        assert bool((guard[:16] == -123.).all() & (guard[-16:] == -123.).all())

    try:
        runtime.prepare(*values, output=output)
        reports = runtime.resource_report
        if isinstance(reports, dict):
            reports = (reports,)
        assert all(r["spill_bytes"] == 0 for r in reports)
        if program_dag:
            assert runtime._prepared[0].compiled_kernel is runtime._prepared[1].compiled_kernel
        for mutation in range(6):
            if mutation == 2:
                values = [value.clone() for value in values]
            for value in values:
                value.copy_(torch.randint(-16, 17, value.shape, device="cuda", generator=generator).float() / 16)
            if mutation == 5:
                values[0].zero_()
            output.fill_(float("nan"))
            stream.wait_stream(torch.cuda.current_stream())
            runtime.run_into(output, *values, stream=stream.cuda_stream)
            torch.cuda.synchronize()
            check()
        with torch.cuda.graph(graph := torch.cuda.CUDAGraph(), stream=stream):
            runtime.run_into(output, *values, stream=stream.cuda_stream)
        for mutation in range(3):
            values[0].fill_(mutation / 16)
            values[1].fill_((mutation + 1) / 8)
            output.fill_(float("nan"))
            graph.replay()
            torch.cuda.synchronize()
            check()
    finally:
        runtime.close()
