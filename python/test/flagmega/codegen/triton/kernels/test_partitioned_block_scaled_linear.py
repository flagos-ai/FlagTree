# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Run rounded FP8 partition projections through normal Compiler/codegen/runtime."""

from dataclasses import replace
from math import prod

import pytest
import torch
from triton.flagmega import ir as fm
from triton.flagmega.artifacts import write_artifact
from triton.flagmega.compiler import Compiler
from triton.flagmega.options import CompileOptions
from triton.flagmega.runtime import load
from triton.flagmega.runtime.module import GeneratedTirCallGraphModule as Runtime
from triton.flagmega.targets import NvidiaSm90Target

from python.test.flagmega.codegen.triton.kernels.block_fp8.test_block_fp8_cooperative_wgmma import projection_target
from python.test.flagmega.ir.ops.nn.test_partitioned_block_scaled_linear import Graph


@pytest.mark.parametrize("tokens,n,k,input_lanes,output_lanes,quantization,task_tile_n", [
    (1, 259, 1024, (), (), "dynamic_fp8_sm90", 64),
    (1, 80, 1024, (2, 4), (2, 4), "dynamic_fp8_sm90", 64),
    (2, 80, 1024, (), (), "dynamic_fp8", 64),
    (1, 4096, 2048, (), (8,), "dynamic_fp8_sm90", 64),
    pytest.param(1, 1024, 2048, (), (), "dynamic_fp8_sm90", 32, id="native-task32"),
    pytest.param(1, 1024, 2048, (), (), "dynamic_fp8_sm90", 16, id="native-task16"),
])
def test_device_matches_explicit_parts_with_mutations_and_guards(tmp_path, tokens, n, k,
                                                                 input_lanes, output_lanes, quantization, task_tile_n):
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (9, 0):
        pytest.skip("requires SM90")
    placement = fm.Placement((8, 16), "yx", "bb")
    runtimes, arguments, stores, outputs = [], [], [], []
    generator = torch.Generator(device="cuda").manual_seed(3183)
    x = torch.randint(-16, 17, (tokens, k), generator=generator, device="cuda").bfloat16() / 16
    weight = (torch.randint(-16, 17, (n, k), generator=generator, device="cuda").float() / 16).to(torch.float8_e4m3fn)
    scales = torch.linspace(.001, .11, ((n + 127) // 128) * (k // 128), device="cuda").reshape(-1, k // 128)
    x = x.reshape(tokens, k // prod(input_lanes), *input_lanes)
    stream = torch.cuda.Stream()
    try:
        for explicit in (True, False):
            compiler = Compiler(CompileOptions(placement=placement))
            compiler.target = (projection_target(cooperative=False, warps=8)
                               if quantization == "dynamic_fp8_sm90" else NvidiaSm90Target())
            if not explicit:
                base = compiler.target.triton_implementation_model
                compiler.target.triton_implementation_model = replace(base, implementations=tuple(
                    replace(impl, parameters={**impl.parameters, "tile_n": task_tile_n})
                    if impl.family == "partitioned_block_scaled_linear" else impl for impl in base.implementations))
            graph = Graph(tokens=tokens, n=n, k=k, explicit=explicit, input_lanes=input_lanes,
                          output_lanes=output_lanes, quantization=quantization).build()
            module = compiler.compile(graph).module
            calls = [d.dispatch for d in module.kernel_definitions]
            if not explicit:
                producer, = (d for d in calls if d.microkernel.family == "partitioned_block_scaled_linear")
                assert producer.microkernel.implementation == "tir.partitioned_block_scaled_linear.local"
                assert not producer.workspaces
                assert not producer.microkernel.facts.get("internal_grid_barriers", 0)
                assert not any(d.microkernel.family in {"block_fp8", "concat"} for d in calls)
            artifact = write_artifact(module, tmp_path / str(explicit), target="nvidia-sm90", emit_executable=True)
            runtime = load(artifact, device="cuda:0")
            runtimes.append(runtime)
            storage = torch.full((4 * tokens * n + 128,), -123, dtype=torch.bfloat16, device="cuda")
            output = storage[64:-64].reshape(4, tokens, n // prod(output_lanes), *output_lanes)
            values = {"x": x, "weight": weight, "scales": scales}
            args = tuple(output if spec["role"] == "result" else values[spec["value"]]
                         for spec in runtime.external_arguments)
            arguments.append(args); stores.append(storage); outputs.append(output)
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for runtime, args in zip(runtimes, arguments):
                Runtime.prepare(runtime, *args)
                assert runtime.resource_report["spill_bytes"] == 0
            for mutation in range(3):
                x.copy_(torch.randint(-16, 17, x.shape, generator=generator, device="cuda").bfloat16() / 16)
                if mutation == 1:
                    x.zero_()
                for runtime, args in zip(runtimes, arguments):
                    Runtime.run_into(runtime, *args)
                torch.cuda.synchronize()
                assert torch.equal(outputs[0].view(torch.uint8), outputs[1].view(torch.uint8))
                for storage in stores:
                    assert bool((storage[:64] == -123).all() & (storage[-64:] == -123).all())
    finally:
        for runtime in runtimes:
            runtime.close()
