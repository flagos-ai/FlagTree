# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Independent FP8 CTAs publish original BF16 parts to a fixed TP4 sum."""

import json
import os
from dataclasses import replace
from pathlib import Path

import pytest

from triton.flagmega import ir as fm
from triton.flagmega.compiler import Compiler
from triton.flagmega.targets import NvidiaSm90Target


def compile_projection_handoff(local_k, *, kind='ordinary'):
    from triton.flagmega.passes.auto_distributed.auto_distributed import AutoDistributedPass

    class Graph(fm.Module):
        def forward(self):
            x = self.input('x', fm.tensor_type(fm.vector_type('bfloat16', (2, 4)),
                                             (1, local_k * 4 // 8)), id='x')
            w = self.input('weight', fm.tensor_type('float8_e4m3fn', (4096, local_k * 4)), id='weight')
            scale = self.input('scales', fm.tensor_type('float32', (32, local_k * 4 // 128)), id='scales')
            parts = fm.F.nn.partitioned_block_scaled_linear(x, w, scale, partitions=4,
                activation_quantization='dynamic_fp8_sm90', output_lanes=(2, 4), name='projection',
                metadata={'execution_mesh': '*', 'kernel_program_kind': kind,
                          **({'kernel_max_ctas': 2} if kind == 'persistent' else {})})
            wide = fm.F.tensors.cast(parts, fm.vector_type('float32', (2, 4)))
            total = fm.F.math.reduce_sum(wide, axes=(0,), keep_dims=False)
            result = fm.F.tensors.cast(total, fm.vector_type('bfloat16', (2, 4)), name='result')
            self.function('main', (x, w, scale), (result,))

    compiler = Compiler()
    target = NvidiaSm90Target().with_placement(fm.Placement((4, 8, 16), 'cyx', 'cbb'))
    model = target.triton_implementation_model
    target.triton_implementation_model = replace(model, implementations=tuple(
        replace(i, parameters={**i.parameters, 'tile_n': 16, 'compute_num_warps': 4,
                               'k_blocks_per_mma': 4, 'k_group_row_order': 'n_major',
                               'activation_layout': 'warp', 'accumulator_layout': 'vector'})
        if i.id == 'tir.partitioned_block_scaled_linear.local' else i for i in model.implementations))
    compiler.target = target
    proposal = compiler.compile(Graph(dialect='high_level', stage='imported', entry='main').build(),
                                stop_after='propose-distribution').module
    search = AutoDistributedPass._build_graph(proposal, target)
    bucket, = (b for b in search.buckets if b.node_id == 'projection')
    split, = (c for c in bucket.candidates
              if c.return_type.axis_policies[0] == fm.SBP.split_contiguous((0,)))
    selected = replace(proposal, selections=tuple(replace(record, candidate_id=split.id,
        origin='agent', rationale='One original material BF16 K partition per TP device')
        if record.point_id == 'distribution.projection' else record for record in proposal.selections))
    return compiler.compile(selected).module


@pytest.mark.parametrize('kind', ('ordinary', 'persistent'))
@pytest.mark.parametrize('local_k', (512, 3072))
def test_complete_fp8_handoff_has_independent_CTA_and_material_part(tmp_path, local_k, kind):
    from triton.flagmega.codegen.triton.tir_package import describe_tir_package
    module = compile_projection_handoff(local_k, kind=kind)
    package = describe_tir_package(module)
    assert [p['grid'] for p in package['kernels']] == [[256 if kind == 'ordinary' else 2, 1, 1], [128, 1, 1]]
    assert package['kernels'][0]['num_warps'] == 4
    assert len(package['kernels'][0]['calls']) == 1
    partial = fm.verify_buffer_plan(module).buffer_map['projection']
    # Material P is a split tensor, unlike a reduced SBPPartial component.
    # Its canonical backing has all four slots; each rank publishes one slot.
    assert partial.nbytes == 4 * 8192
    assert partial.distributed_storage_kind == fm.DistributedBufferStorageKind.CANONICAL_GLOBAL
    assert partial.distributed_type.axis_policies[0] == fm.SBP.split_contiguous((0,))
    restored = fm.load_module(fm.emit_module(module, tmp_path / 'physical.py'))
    assert restored.semantic_hash == module.semantic_hash


@pytest.fixture(scope='module')
def device_group():
    import torch
    import torch.distributed as dist
    device = f"cuda:{os.environ['LOCAL_RANK']}"
    torch.cuda.set_device(device)
    if torch.cuda.get_device_capability() != (9, 0):
        pytest.skip('requires SM90')
    dist.init_process_group('gloo')
    yield device
    dist.destroy_process_group()


@pytest.mark.skipif(os.environ.get('WORLD_SIZE') != '4', reason='requires four SM90 GPUs under torchrun')
@pytest.mark.parametrize('kind', ('ordinary', 'persistent'))
@pytest.mark.parametrize('local_k', (512, 3072))
def test_GPU_dynamic_FP8_part_to_fixed_collective_graph(tmp_path, local_k, kind, device_group):
    import torch
    import torch.distributed as dist
    from triton.flagmega.artifacts import write_artifact
    from triton.flagmega.runtime import load
    from triton.flagmega.runtime.module import GeneratedTirCallGraphModule as Runtime

    device, k, n = device_group, local_k * 4, 4096
    module = compile_projection_handoff(local_k, kind=kind)
    runtime = load(write_artifact(module, tmp_path / 'artifact', target='nvidia-sm90',
                                 emit_executable=True), device=device)
    try:
        index = torch.arange(n * k, dtype=torch.int64, device=device)
        weight = (((((index * 1103515245 + 12345) & 0x7fffffff) >> 16) % 17 - 8).float() / 32)
        weight = weight.to(torch.float8_e4m3fn).reshape(n, k)
        scales = torch.exp2(-4 - ((torch.arange(32, device=device)[:, None] +
                                 torch.arange(k // 128, device=device)[None, :]) % 5).float())
        xs = [torch.empty((1, k // 8, 2, 4), dtype=torch.bfloat16, device=device) for _ in range(2)]
        guards = [torch.full((n + 32,), 123., dtype=torch.bfloat16, device=device) for _ in range(2)]
        outputs = [g[16:-16].reshape(1, n // 8, 2, 4) for g in guards]
        def bind(bank):
            values = dict(x=xs[bank], weight=weight, scales=scales)
            return tuple(outputs[bank] if s['role'] == 'result' else values[s['value']]
                         for s in runtime.external_arguments)
        arguments = [bind(i) for i in range(2)]
        caller = torch.cuda.Stream()
        Runtime.prepare(runtime, *arguments[0])
        grids = [list(p.grid) for p in runtime._prepared]
        assert grids == [[256 if kind == 'ordinary' else 2, 1, 1], [128, 1, 1]]
        def populate(mutation, bank):
            xi = torch.arange(k, dtype=torch.int64, device=device)
            value = (((xi * 48271 + (mutation + 1) * 69621) % 15 - 7).float() * 64).bfloat16()
            value[::128] = 448.
            if mutation == 1:
                value.zero_()
            xs[bank].copy_(value.reshape(xs[bank].shape))
            # All products are exact dyadics. Round each ordered K128 FMA,
            # then the original material BF16 part, then the FP32 TP sum.
            parts = []
            for part in range(4):
                total = torch.zeros((1, n), dtype=torch.float32, device=device)
                for block in range(part * local_k // 128, (part + 1) * local_k // 128):
                    start = block * 128
                    raw = value[start:start + 128].double()[None, :] @ weight[:, start:start + 128].double().T
                    factor = scales[torch.arange(n, device=device) // 128, block].double()[None, :]
                    total = (raw * factor + total.double()).float()
                parts.append(total.bfloat16().float())
            outputs[bank].fill_(float('nan'))
            return sum(parts).bfloat16()
        def check(bank, expected):
            torch.testing.assert_close(outputs[bank].reshape(1, n), expected, rtol=0, atol=0)
            assert bool((guards[bank][:16] == 123.).all() & (guards[bank][-16:] == 123.).all())
        for mutation in range(4):
            bank = int(mutation == 3)
            expected = populate(mutation, bank)
            dist.barrier()
            caller.wait_stream(torch.cuda.current_stream())
            Runtime.run_into(runtime, *arguments[bank], stream=caller.cuda_stream)
            caller.synchronize()
            check(bank, expected)
        graph = torch.cuda.CUDAGraph()
        dist.barrier()
        with torch.cuda.graph(graph, stream=caller):
            for _ in range(3):
                Runtime.run_into(runtime, *arguments[1], stream=caller.cuda_stream)
        for mutation in range(4, 7):
            expected = populate(mutation, 1)
            dist.barrier()
            caller.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(caller):
                graph.replay()
            caller.synchronize()
            check(1, expected)
        reports = runtime.resource_report
        assert all(r['spill_bytes'] == 0 for r in reports)
        receipt_dir = os.environ.get('FLAGMEGA_FP8_TP4_RECEIPT_DIR')
        destination = tmp_path if receipt_dir is None else Path(receipt_dir) / f'rank-{dist.get_rank()}'
        destination.mkdir(parents=True, exist_ok=True)
        (destination / f'{local_k}-{kind}.json').write_text(json.dumps(dict(
            passed=True, rank=dist.get_rank(), local_k=local_k, global_k=k, n=n,
            program_grids=grids, source_sha256=runtime.codegen['source_sha256'],
            material_BF16_parts=True, FP64_ordered_dyadic_oracle=True,
            zero_input=True, pointer_rebind=True, captured_requests=3,
            changed_input_Graph_replays=3, guards=True, resources=reports,
            performance_measurement=False, full_model_acceptance=False), indent=2) + '\n')
    finally:
        runtime.close()
