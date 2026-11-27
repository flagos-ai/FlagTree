# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Per-chip partial writers must retain all four contributors to a collective."""

import json
import os

import pytest
from triton.flagmega import ir as fm
from triton.flagmega.artifacts import write_artifact
from triton.flagmega.compiler import Compiler
from triton.flagmega.runtime import load
from triton.flagmega.runtime.module import GeneratedTirCallGraphModule as Runtime


@pytest.fixture(scope="module")
def device_group():
    import torch
    import torch.distributed as dist

    device = f"cuda:{os.environ['LOCAL_RANK']}"
    torch.cuda.set_device(device)
    if torch.cuda.get_device_capability() != (9, 0):
        pytest.skip("requires SM90")
    dist.init_process_group("gloo")
    yield device
    dist.destroy_process_group()


@pytest.mark.skipif(os.environ.get("WORLD_SIZE") != "4", reason="requires torchrun on four SM90 GPUs")
@pytest.mark.parametrize("dtype,lanes", [("float32", ()), ("bfloat16", (2, 4))])
def test_partial_sum_publishes_each_device_before_collective(tmp_path, dtype, lanes, device_group):
    import torch
    import torch.distributed as dist

    device = device_group
    runtime = None
    placement = fm.Placement((4, 8, 16), "cyx", "cbb")
    kind = fm.tensor_type(dtype, (4, 1, 4096))
    distributed = fm.DistributedType(kind, (fm.SBP.split_contiguous((0,)), fm.SBP.broadcast(),
        fm.SBP.split_contiguous((1, 2), 32)), placement)

    class Graph(fm.Module):
        def forward(self):
            x = self.input("x", kind, id="x")
            local = fm.F.distributed.sharded_view(x, distributed)
            partial = fm.F.math.reduce_sum(local, axes=(0,), keep_dims=False, output_lanes=lanes, name="partial")
            result = fm.F.distributed.force_boxing(partial, fm.logical_type(partial.type), name="result")
            self.function("main", (x,), (result,))

    try:
        module = Compiler().compile(Graph(dialect="distributed", stage="frozen_constants", entry="main",
            metadata={"auto_distribution": {"placement": placement.to_data()}}).build()).module
        buffer = fm.verify_buffer_plan(module).buffer_map["partial"]
        assert buffer.distributed_storage_kind == fm.DistributedBufferStorageKind.COMPACT_PER_CHIP
        artifact = write_artifact(module, tmp_path / "artifact", target="nvidia-sm90", emit_executable=True)
        runtime = load(artifact, device=device)
        torch_dtype = getattr(torch, dtype)
        x = torch.empty((4, 1, 4096), dtype=torch_dtype, device=device)
        storage = torch.full((4096 + 32,), 123., dtype=torch_dtype, device=device)
        result = storage[16:-16].reshape(1, 4096) if not lanes else storage[16:-16].reshape(1, 512, 2, 4)
        arguments = tuple(x if spec["role"] != "result" else result for spec in runtime.external_arguments)
        Runtime.prepare(runtime, *arguments)
        for mutation in range(3):
            indices = torch.arange(4 * 4096, dtype=torch.int64, device=device).reshape(4, 1, 4096)
            values = ((indices * 3 + mutation * 5) % 31 - 15).float().div_(8).to(torch_dtype)
            x.copy_(values)
            result.fill_(float("nan"))
            dist.barrier()
            Runtime.run_into(runtime, *arguments)
            torch.cuda.synchronize()
            expected = values.float().sum(0).to(torch_dtype)
            torch.testing.assert_close(result.reshape(1, 4096), expected, rtol=0, atol=0)
            assert bool((storage[:16] == 123.).all() & (storage[-16:] == 123.).all())
        assert runtime.resource_report["spill_bytes"] == 0
        (tmp_path / "result.json").write_text(json.dumps({"passed": True, "rank": dist.get_rank(),
            "dtype": dtype, "lanes": lanes, "mutations": 3, "guards": True,
            "source_sha256": runtime.codegen["source_sha256"], "resources": runtime.resource_report}, indent=2) + "\n")
    finally:
        if runtime is not None:
            runtime.close()
        dist.barrier()


@pytest.mark.skipif(os.environ.get('WORLD_SIZE') != '4', reason='requires torchrun on four SM90 GPUs')
@pytest.mark.parametrize('dtype,lanes', [('float32', ()), ('bfloat16', (2, 4))])
def test_cross_mesh_partial_component_is_complete_before_the_collective(tmp_path, dtype, lanes, device_group):
    import torch
    import torch.distributed as dist
    from python.test.flagmega.compiler.test_execution_mesh_partial import partial_mesh_graph
    device = device_group
    runtime = None
    try:
        module = Compiler().compile(partial_mesh_graph(dtype, lanes)).module
        plan = fm.verify_buffer_plan(module)
        source, view = (plan.buffer_map[name] for name in ('partial', 'repartitioned'))
        assert source.mem_span.must_alias(view.mem_span)
        assert source.distributed_type.partial == view.distributed_type.partial == fm.SBP.partial((0,))
        artifact = write_artifact(module, tmp_path / 'artifact', target='nvidia-sm90', emit_executable=True)
        runtime = load(artifact, device=device)
        x = torch.empty((4, 1, 4096), dtype=getattr(torch, dtype), device=device)
        guard = torch.full((4096 + 32,), 123., dtype=x.dtype, device=device)
        result = guard[16:-16].reshape(1, 4096) if not lanes else guard[16:-16].reshape(1, 512, 2, 4)
        rebound_x = torch.empty_like(x)
        rebound_guard = torch.full_like(guard, 123.)
        rebound_result = rebound_guard[16:-16].reshape(result.shape)
        assert rebound_x.data_ptr() != x.data_ptr()
        assert rebound_result.data_ptr() != result.data_ptr()
        arguments = tuple(result if spec['role'] == 'result' else x for spec in runtime.external_arguments)
        rebound_arguments = tuple(rebound_result if spec['role'] == 'result' else rebound_x
                                  for spec in runtime.external_arguments)
        # Compiler global scratch belongs to the first launch stream. Eager
        # and captured requests use that same caller stream throughout.
        caller = torch.cuda.Stream()
        Runtime.prepare(runtime, *arguments)
        assert [tuple(p.grid) for p in runtime._prepared] == [(64, 1, 1), (128, 1, 1)]

        def assert_result(output, storage, values):
            torch.testing.assert_close(output.reshape(1, 4096), values.float().sum(0).to(x.dtype),
                                       rtol=0, atol=0)
            assert bool((storage[:16] == 123.).all() & (storage[-16:] == 123.).all())

        for mutation in range(4):
            indices = torch.arange(4 * 4096, device=device).reshape(4, 1, 4096)
            values = ((indices * 3 + mutation * 5) % 31 - 15).float().div_(8).to(x.dtype)
            x.copy_(values)
            result.fill_(float('nan'))
            dist.barrier()
            caller.wait_stream(torch.cuda.current_stream())
            Runtime.run_into(runtime, *arguments, stream=caller.cuda_stream)
            torch.cuda.synchronize()
            assert_result(result, guard, values)
        rebound_x.copy_(values)
        rebound_result.fill_(float('nan'))
        dist.barrier()
        caller.wait_stream(torch.cuda.current_stream())
        Runtime.run_into(runtime, *rebound_arguments, stream=caller.cuda_stream)
        torch.cuda.synchronize()
        assert_result(rebound_result, rebound_guard, values)
        caller.wait_stream(torch.cuda.current_stream())
        graph = torch.cuda.CUDAGraph()
        dist.barrier()
        with torch.cuda.graph(graph, stream=caller):
            for _ in range(3):
                Runtime.run_into(runtime, *rebound_arguments, stream=caller.cuda_stream)
        for mutation in range(4, 7):
            values = ((indices * 3 + mutation * 5) % 31 - 15).float().div_(8).to(x.dtype)
            rebound_x.copy_(values)
            rebound_result.fill_(float('nan'))
            dist.barrier()
            caller.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(caller):
                graph.replay()
            torch.cuda.synchronize()
            assert_result(rebound_result, rebound_guard, values)
        reports = runtime.resource_report
        assert all(report['spill_bytes'] == 0 for report in reports)
        (tmp_path / 'cross-mesh-result.json').write_text(json.dumps({
            'passed': True, 'rank': dist.get_rank(), 'dtype': dtype, 'lanes': lanes,
            'program_grids': [[64, 1, 1], [128, 1, 1]], 'eager_mutations': 4,
            'pointer_rebind': True, 'captured_requests': 3, 'graph_replays': 3,
            'captured_input_mutations': 3, 'guards': True,
            'resources': reports,
        }, indent=2) + '\n')
    finally:
        if runtime is not None:
            runtime.close()
        dist.barrier()
