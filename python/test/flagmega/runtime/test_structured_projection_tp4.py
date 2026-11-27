# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Real peer completion and partial sums in mixed-mesh recurrent layers."""
import json
import os
from pathlib import Path

import pytest

from triton.flagmega import ir as fm
from triton.flagmega.artifacts import write_artifact
from triton.flagmega.runtime import load
from triton.flagmega.runtime.module import GeneratedTirCallGraphModule as Runtime
from python.test.flagmega.compiler.test_structured_projection_programs import compile_structured_projection_loop
from python.test.flagmega.runtime.test_partitioned_linear_tp4 import fresh_tp4_device_group


CASES = [pytest.param(kind, trips, second, id=f'{kind}-{trips if second is None else "shared"}')
         for kind in ('ordinary', 'persistent') for trips, second in ((0, None), (1, None), (3, None), (1, 2))]


@pytest.mark.skipif(os.environ.get('WORLD_SIZE') != '4', reason='requires torchrun with four SM90 GPUs')
@pytest.mark.parametrize('kind,trips,second', CASES)
def test_device_structured_projection_tp4(tmp_path, fresh_tp4_device_group, kind, trips, second):
    import torch
    import torch.distributed as dist
    device = fresh_tp4_device_group
    module = compile_structured_projection_loop(trips=trips, kind=kind, second_trips=second)
    plan = fm.verify_buffer_plan(module)
    partial = plan.buffer_map['projection']
    assert partial.distributed_type.partial == fm.SBPPartial((0,))
    assert partial.distributed_storage_kind is fm.DistributedBufferStorageKind.COMPACT_PER_CHIP
    size, span = 4096, 1024
    rows = torch.arange(size)
    # Four independently rounded TP components, with nonzero work in both
    # accumulation splits and different per-chip columns. These sparse dyadic
    # sums are exactly representable in FP32 and FP64 before each BF16 boundary.
    weight_cpu = torch.zeros((size, size), dtype=torch.bfloat16)
    for owner in range(4):
        columns = (rows * 7 + owner * 13) % span + owner * span
        weight_cpu[rows, columns] = ((rows % 5 - 2).float() * (2.**owner / 32)).bfloat16()
    weight_banks_cpu = (weight_cpu, weight_cpu.flip(1).contiguous())
    weight_banks = tuple(w.to(device) for w in weight_banks_cpu)
    values = [torch.empty((1, size // 8, 2, 4), dtype=torch.bfloat16, device=device) for _ in range(2)]
    guards = [torch.full((size + 32,), 123., dtype=torch.bfloat16, device=device) for _ in range(2)]
    outputs = [g[16:-16].reshape(1, size // 8, 2, 4) for g in guards]
    runtime = load(write_artifact(module, tmp_path / 'artifact', target='nvidia-sm90',
                                 emit_executable=True), device=device)

    def bind(bank):
        arguments = {'value': values[bank], 'weight': weight_banks[bank]}
        return tuple(outputs[bank] if spec['role'] == 'result' else arguments[spec['value']]
                     for spec in runtime.external_arguments)

    def populate(bank, mutation):
        initial = ((rows % 31 - 15 + mutation).float() / 64).bfloat16().reshape(1, size)
        values[bank].copy_(initial.reshape(values[bank].shape))
        outputs[bank].fill_(float('nan'))
        current = initial
        weight = weight_banks_cpu[bank].double()
        for _ in range(trips + (second or 0)):
            total = torch.zeros_like(current, dtype=torch.float32)
            for start in range(0, size, span):
                exact = current[:, start:start + span].double() @ weight[:, start:start + span].T
                total += exact.bfloat16().float()
            current = (total.bfloat16().float() + current.float()).bfloat16()
        return current

    def check(bank, expected):
        torch.testing.assert_close(outputs[bank].cpu().reshape(1, size), expected, rtol=0, atol=0)
        assert bool((guards[bank][:16] == 123.).all() & (guards[bank][-16:] == 123.).all())

    try:
        caller = torch.cuda.Stream()
        arguments, rebound_arguments = bind(0), bind(1)
        Runtime.prepare(runtime, *arguments)
        dynamic = [p for p in runtime.codegen['kernels'] if p.get('execution_mesh', {}).get('kind') == 'dynamic']
        assert len(dynamic) == trips + (second or 0)
        assert all(p['grid'] == [512 if kind == 'ordinary' else 2, 1, 1] for p in dynamic)
        completions = [p for p in runtime.codegen['kernels'] if p['invocation_kind'] == 'device_completion']
        assert completions and all(p['requires_grid_wide_sync'] for p in completions)
        for mutation in range(3):
            expected = populate(0, mutation)
            dist.barrier()
            caller.wait_stream(torch.cuda.current_stream())
            Runtime.run_into(runtime, *arguments, stream=caller.cuda_stream)
            caller.synchronize()
            check(0, expected)
        expected = populate(1, 3)
        dist.barrier()
        caller.wait_stream(torch.cuda.current_stream())
        Runtime.run_into(runtime, *rebound_arguments, stream=caller.cuda_stream)
        caller.synchronize()
        check(1, expected)
        captured = torch.cuda.CUDAGraph()
        dist.barrier()
        with torch.cuda.graph(captured, stream=caller):
            for _ in range(3):
                Runtime.run_into(runtime, *rebound_arguments, stream=caller.cuda_stream)
        for mutation in range(4, 7):
            expected = populate(1, mutation)
            dist.barrier()
            caller.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(caller):
                captured.replay()
            caller.synchronize()
            check(1, expected)
        reports = runtime.resource_report
        assert all(r['spill_bytes'] == 0 for r in reports)
        directory = Path(os.environ.get('FLAGMEGA_STRUCTURED_TP4_RECEIPT_DIR', str(tmp_path)))
        directory.mkdir(parents=True, exist_ok=True)
        case = f'{kind}-{trips if second is None else "shared"}'
        (directory / f'{case}-rank{dist.get_rank()}.json').write_text(json.dumps({
            'passed': True, 'case': case, 'rank': dist.get_rank(), 'artifact': str(tmp_path / 'artifact'),
            'semantic_hash': module.semantic_hash, 'generated_source_sha256': runtime.codegen['source_sha256'],
            'dtype': 'bfloat16', 'k': size, 'local_k': span, 'n': size, 'partitions': 4,
            'trips': trips, 'second_trips': second, 'partial_bytes_per_chip': partial.nbytes,
            'eager_mutations': 3, 'input_weight_output_pointer_rebind': True,
            'captured_requests': 3, 'graph_replays': 3, 'captured_input_mutations': 3, 'guards': True,
            'program_execution': runtime.codegen['program_execution'],
            'actual_invocations': runtime.codegen['kernels'], 'resources': reports,
            'weights_producer_communication_consumer_overlap_proven': False,
            'performance_measurement': False, 'full_model_acceptance': False,
        }, ensure_ascii=False, indent=2) + '\n')
    finally:
        runtime.close()
