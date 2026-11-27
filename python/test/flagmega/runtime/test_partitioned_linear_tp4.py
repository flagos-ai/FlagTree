# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Run with torchrun --nproc-per-node=4 -m pytest on four idle SM90 GPUs."""

import json
import os

import pytest
from triton.flagmega import ir as fm
from triton.flagmega.artifacts import write_artifact
from triton.flagmega.compiler import Compiler
from triton.flagmega.runtime import load
from triton.flagmega.runtime.module import GeneratedTirCallGraphModule

from python.test.flagmega.ir.ops.nn.test_partitioned_linear import graph


@pytest.mark.skipif(os.environ.get("WORLD_SIZE") != "4", reason="requires torchrun with four SM90 GPUs")
def test_k_sharded_projection_and_bf16_partial_collective(tmp_path):
    import torch
    import torch.distributed as dist

    device = f"cuda:{os.environ['LOCAL_RANK']}"
    torch.cuda.set_device(device)
    if torch.cuda.get_device_capability() != (9, 0):
        pytest.skip("requires SM90")
    dist.init_process_group("gloo")
    runtime = None
    try:
        placement = fm.Placement((4, 8, 16), "cyx", "cbb")
        source = graph(k=2048, n=128, parts=4, lanes=(2, 4), input_lanes=(2, 4),
                       placement=placement, k_policy=fm.SBP.split_contiguous((0,)), materialize=True)
        assert source.node_map["projection"].type.partial == fm.SBPPartial((0,))
        module = Compiler().compile(source).module
        artifact = write_artifact(module, tmp_path / "artifact", target="nvidia-sm90", emit_executable=True)
        runtime = load(artifact, device=device)
        generator = torch.Generator().manual_seed(10031)
        x = (torch.randint(-7, 8, (1, 2048), generator=generator).float() / 128).bfloat16()
        w = (torch.randint(-7, 8, (128, 2048), generator=generator).float() / 128).bfloat16()
        storage = torch.full((160,), 123., dtype=torch.bfloat16, device=device)
        result = storage[16:-16].reshape(1, 16, 2, 4)
        result_spec, = (spec for spec in runtime.external_arguments if spec["role"] == "result")
        values = {"x": x.reshape(1, 256, 2, 4).to(device), "w": w.to(device), result_spec["value"]: result}
        arguments = tuple(values[spec["value"]] for spec in runtime.external_arguments)
        GeneratedTirCallGraphModule.prepare(runtime, *arguments)
        for mutation in range(3):
            if mutation:
                x.add_(1 / 128)
                values["x"].copy_(x.reshape(1, 256, 2, 4))
            result.fill_(float("nan"))
            dist.barrier()
            GeneratedTirCallGraphModule.run_into(runtime, *arguments)
            torch.cuda.synchronize()
            expected = sum((x[:, i:i + 512].float() @ w[:, i:i + 512].float().T).bfloat16().float()
                           for i in range(0, 2048, 512)).bfloat16()
            torch.testing.assert_close(result.cpu().reshape(1, 128), expected, rtol=0, atol=0)
            assert bool(torch.all(storage[:16] == 123.)) and bool(torch.all(storage[-16:] == 123.))
        assert runtime.resource_report["spill_bytes"] == 0
        (tmp_path / "tp4-result.json").write_text(json.dumps({"rank": dist.get_rank(), "passed": True,
            "mutations": 3, "guards": True, "resources": runtime.resource_report,
            "source_sha256": runtime.codegen["source_sha256"]}, indent=2) + "\n")
    finally:
        if runtime is not None:
            runtime.close()
        dist.destroy_process_group()


@pytest.fixture(scope='module')
def fresh_tp4_device_group():
    import torch
    import torch.distributed as dist
    device = f"cuda:{os.environ['LOCAL_RANK']}"
    torch.cuda.set_device(device)
    if torch.cuda.get_device_capability() != (9, 0):
        pytest.skip('requires SM90')
    dist.init_process_group('gloo')
    yield device
    dist.destroy_process_group()


@pytest.mark.skipif(os.environ.get('WORLD_SIZE') != '4', reason='requires torchrun with four SM90 GPUs')
@pytest.mark.parametrize('program_kind', ['ordinary', 'persistent'])
def test_fresh_dynamic_k_partition_handoff_and_graph(tmp_path, program_kind, fresh_tp4_device_group):
    import torch
    import torch.distributed as dist
    from python.test.flagmega.compiler.test_dynamic_partitioned_linear import compile_tp4_dynamic_projection
    device = fresh_tp4_device_group
    runtime = None
    try:
        module = compile_tp4_dynamic_projection(kind=program_kind,
            max_ctas=2 if program_kind == 'persistent' else None)
        partial = fm.verify_buffer_plan(module).buffer_map['projection']
        assert partial.distributed_type.partial == fm.SBP.partial((0,))
        assert partial.distributed_storage_kind == fm.DistributedBufferStorageKind.COMPACT_PER_CHIP
        assert partial.nbytes == 8192
        runtime = load(write_artifact(module, tmp_path / 'artifact', target='nvidia-sm90',
                                     emit_executable=True), device=device)
        k, n, span = 8192, 4096, 2048
        # Integer products and these dyadic inputs make the FP32 reference
        # exact, while retaining each TP partition's required BF16 rounding.
        x_cpu = torch.empty((1, k), dtype=torch.bfloat16)
        w_cpu = ((torch.arange(n*k).reshape(n, k) % 15 - 7).float() / 128).bfloat16()
        weight = w_cpu.to(device)
        x, rebound_x = [torch.empty((1, k // 8, 2, 4), dtype=torch.bfloat16, device=device) for _ in range(2)]
        guards = [torch.full((n + 32,), 123., dtype=torch.bfloat16, device=device) for _ in range(2)]
        outputs = [guard[16:-16].reshape(1, n // 8, 2, 4) for guard in guards]

        def bind(value, output):
            values = {'x': value, 'w': weight}
            return tuple(output if spec['role'] == 'result' else values[spec['value']]
                         for spec in runtime.external_arguments)

        arguments, rebound_arguments = bind(x, outputs[0]), bind(rebound_x, outputs[1])
        caller = torch.cuda.Stream()
        GeneratedTirCallGraphModule.prepare(runtime, *arguments)
        grids = [tuple(program.grid) for program in runtime._prepared]
        assert grids == [(512 if program_kind == 'ordinary' else 2, 1, 1), (128, 1, 1)]

        def populate(mutation, value, output):
            x_cpu.copy_((torch.arange(k) % 15 - 7 + mutation).float().div_(128))
            value.copy_(x_cpu.reshape(value.shape))
            output.fill_(float('nan'))
            return sum((x_cpu[:, start:start+span].float() @ w_cpu[:, start:start+span].float().T)
                       .bfloat16().float() for start in range(0, k, span)).bfloat16()

        def check(index, expected):
            torch.testing.assert_close(outputs[index].cpu().reshape(1, n), expected, rtol=0, atol=0)
            assert bool((guards[index][:16] == 123.).all() & (guards[index][-16:] == 123.).all())

        for mutation in range(3):
            expected = populate(mutation, x, outputs[0])
            dist.barrier()
            caller.wait_stream(torch.cuda.current_stream())
            GeneratedTirCallGraphModule.run_into(runtime, *arguments, stream=caller.cuda_stream)
            caller.synchronize()
            check(0, expected)
        expected = populate(3, rebound_x, outputs[1])
        dist.barrier()
        caller.wait_stream(torch.cuda.current_stream())
        GeneratedTirCallGraphModule.run_into(runtime, *rebound_arguments, stream=caller.cuda_stream)
        caller.synchronize()
        check(1, expected)
        captured = torch.cuda.CUDAGraph()
        dist.barrier()
        with torch.cuda.graph(captured, stream=caller):
            for _ in range(3):
                GeneratedTirCallGraphModule.run_into(runtime, *rebound_arguments, stream=caller.cuda_stream)
        for mutation in range(4, 7):
            expected = populate(mutation, rebound_x, outputs[1])
            dist.barrier()
            caller.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(caller):
                captured.replay()
            caller.synchronize()
            check(1, expected)
        reports = runtime.resource_report
        assert all(report['spill_bytes'] == 0 for report in reports)
        (tmp_path / 'dynamic-tp4-result.json').write_text(json.dumps({
            'passed': True, 'rank': dist.get_rank(), 'program_kind': program_kind,
            'program_grids': grids, 'k': k, 'local_k': span, 'n': n, 'partial_bytes_per_chip': partial.nbytes,
            'eager_mutations': 3, 'pointer_rebind': True, 'captured_requests': 3,
            'graph_replays': 3, 'captured_input_mutations': 3, 'guards': True,
            'resources': reports,
        }, indent=2) + '\n')
    finally:
        if runtime is not None:
            runtime.close()
