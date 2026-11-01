# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Independent row tasks keep numerical partitions and chip-shared data."""
from dataclasses import replace

import pytest

from triton.flagmega import ir as fm
from triton.flagmega.compiler import Compiler
from triton.flagmega.codegen.triton.tir_package import describe_tir_package
from triton.flagmega.ir.launch_recipe import TaskLaunchRecipe
from python.test.flagmega.ir.ops.nn.test_partitioned_linear import graph


def dynamic_projection_graph(*, kind='ordinary', max_ctas=None, accumulation_splits=1, **options):
    module = graph(**options)
    return replace(module, nodes=tuple(replace(node, attrs={**node.attrs,
        **({'accumulation_splits': accumulation_splits} if accumulation_splits != 1 else {})}, metadata={**node.metadata,
        'execution_mesh': '*', 'kernel_program_kind': kind,
        **({'kernel_max_ctas': max_ctas} if max_ctas is not None else {}),
    }) if node.id == 'projection' else replace(node, metadata={**node.metadata,
        'execution_mesh': (8, 16)}) if node.id == 'result' else node for node in module.nodes))


@pytest.mark.parametrize('kind,n,max_ctas', [('ordinary', 83, None), ('persistent', 4096, 2)])
def test_normal_dynamic_projection_has_complete_row_task_coverage(tmp_path, kind, n, max_ctas):
    module = Compiler().compile(dynamic_projection_graph(kind=kind, max_ctas=max_ctas,
        k=2048, n=n, parts=1)).module
    program, = describe_tir_package(module)['kernels']
    assert program['execution_mesh']['kind'] == 'dynamic'
    assert program['program_kind'] == kind
    recipe = TaskLaunchRecipe.from_data(program['launch_recipe'])
    assert recipe.work_size == n and recipe.tile_size == 8
    assert recipe.task_count == (n + 7) // 8
    assert recipe.resolve(sm_count=132) == ((n + 7) // 8 if kind == 'ordinary' else 2, 1, 1)
    dispatch, = module.kernel_definitions
    assert dispatch.dispatch.microkernel.variant == 'simt'
    assert not dispatch.dispatch.microkernel.shared_workspaces
    restored = fm.load_module(fm.emit_module(module, tmp_path / 'dynamic.py'))
    assert restored.semantic_hash == module.semantic_hash


def test_complete_task_recipe_does_not_require_an_elementwise_vector_schedule():
    from triton.flagmega.codegen.triton.execution_programs import selected_task_recipe
    from triton.flagmega.ir.execution_mesh import KernelProgramKind
    recipe = TaskLaunchRecipe(4096, 8, KernelProgramKind.ORDINARY)
    node = fm.Node('projection', 'tir.call', (), fm.tensor_type('bfloat16', (1, 4096)),
                   attrs={'parameters': {'task_mapping': recipe.to_data()}})
    assert selected_task_recipe(node) == recipe


def compile_tp4_dynamic_projection(*, kind='ordinary', max_ctas=None):
    from triton.flagmega.targets import NvidiaSm90Target
    from triton.flagmega.passes.auto_distributed.auto_distributed import AutoDistributedPass
    compiler = Compiler()
    compiler.target = NvidiaSm90Target().with_placement(fm.Placement((4, 8, 16), 'cyx', 'cbb'))
    proposal = compiler.compile(dynamic_projection_graph(kind=kind, max_ctas=max_ctas,
        k=8192, n=4096, parts=4, lanes=(2, 4), input_lanes=(2, 4)),
        stop_after='propose-distribution').module
    search = AutoDistributedPass._build_graph(proposal, compiler.target)
    bucket, = (b for b in search.buckets if b.node_id == 'projection')
    partial, = (c for c in bucket.candidates if c.return_type.partial == fm.SBP.partial((0,)))
    # This graph has exactly one choice. Preserve the complete proposal and
    # let the normal constrained solver regenerate its input/output resharding.
    assert len(proposal.selection_points) == 1
    selected = replace(proposal, selections=tuple(replace(record, candidate_id=partial.id,
        origin='agent', rationale='one complete K partition per TP device')
        if record.point_id == 'distribution.projection' else record for record in proposal.selections))
    return compiler.compile(selected).module


def test_fresh_tp4_distribution_preserves_device_partial_before_fixed_collective():
    module = compile_tp4_dynamic_projection()
    programs = describe_tir_package(module)['kernels']
    assert [p['grid'] for p in programs] == [[512, 1, 1], [128, 1, 1]]
    partial = fm.verify_buffer_plan(module).buffer_map['projection']
    assert partial.distributed_type.partial == fm.SBP.partial((0,))
    assert partial.distributed_type.placement.hierarchy == (4, 1, 1)
    assert partial.distributed_storage_kind == fm.DistributedBufferStorageKind.COMPACT_PER_CHIP
    assert partial.nbytes == 4096 * 2


@pytest.mark.parametrize('kind,k,n,parts,lanes,splits', [
    ('ordinary', 320, 83, 4, (), 1),
    ('persistent', 2048, 4096, 1, (2, 4), 1),
    ('persistent', 320, 80, 4, (2, 4), 4),
    ('ordinary', 4, 1, 2, (), 1),
])
def test_gpu_dynamic_projection_preserves_parts_lanes_tails_and_graph(tmp_path, kind, k, n, parts, lanes, splits):
    import torch
    from triton.flagmega.artifacts import write_artifact
    from triton.flagmega.runtime import load
    from triton.flagmega.runtime.module import GeneratedTirCallGraphModule as Runtime
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (9, 0):
        pytest.skip('requires SM90')
    module = Compiler().compile(dynamic_projection_graph(kind=kind,
        max_ctas=2 if kind == 'persistent' else None, k=k, n=n, parts=parts,
        lanes=lanes, input_lanes=lanes, accumulation_splits=splits)).module
    runtime = load(write_artifact(module, tmp_path / 'artifact', target='nvidia-sm90',
                                 emit_executable=True), device='cuda:0')
    x = torch.empty((1, k), dtype=torch.bfloat16, device='cuda')
    weight = torch.empty((n, k), dtype=torch.bfloat16, device='cuda')
    guard = torch.full((n + 32,), 123., dtype=torch.bfloat16, device='cuda')
    result = guard[16:-16]
    shapes = {spec['value']: tuple(d.fixed_value for d in fm.logical_type(module.node_map[spec['value']].type).shape)
              + tuple(getattr(fm.logical_type(module.node_map[spec['value']].type).dtype, 'lanes', ()))
              for spec in runtime.external_arguments}
    values = {'x': x, 'w': weight}
    arguments = tuple((result if spec['role'] == 'result' else values[spec['value']]).reshape(shapes[spec['value']])
                      for spec in runtime.external_arguments)
    caller = torch.cuda.Stream()
    try:
        Runtime.prepare(runtime, *arguments)
        for mutation in range(3):
            if k == 4:
                x.fill_(1.)
                weight.copy_(torch.tensor([[1., 1/256, 1/256, 0.]], device='cuda', dtype=torch.bfloat16))
            else:
                x.copy_((torch.arange(k, device='cuda') % 15 - 7 + mutation).float().div_(128))
                weight.copy_((torch.arange(n*k, device='cuda').reshape(n, k) % 15 - 7).float().div_(128))
            result.fill_(float('nan'))
            caller.wait_stream(torch.cuda.current_stream())
            Runtime.run_into(runtime, *arguments, stream=caller.cuda_stream)
            caller.synchronize()
            part_span, inner = k // parts, k // parts // splits
            expected = torch.zeros((n,), dtype=torch.float64, device='cuda')
            for part in range(parts):
                accumulated = torch.zeros_like(expected)
                for split in range(splits):
                    start = part * part_span + split * inner
                    accumulated += (weight[:, start:start+inner].double() * x[0, start:start+inner].double()).sum(1)
                expected += accumulated.bfloat16().double()
            torch.testing.assert_close(result, expected.bfloat16(), rtol=0, atol=0)
            assert bool((guard[:16] == 123.).all() & (guard[-16:] == 123.).all())
        captured = torch.cuda.CUDAGraph()
        with torch.cuda.graph(captured, stream=caller):
            for _ in range(3):
                Runtime.run_into(runtime, *arguments, stream=caller.cuda_stream)
        with torch.cuda.stream(caller):
            captured.replay()
        caller.synchronize()
        torch.testing.assert_close(result, expected.bfloat16(), rtol=0, atol=0)
        assert runtime.resource_report['spill_bytes'] == 0
        assert not runtime.resource_report['cooperative_grid']
    finally:
        runtime.close()


@pytest.mark.parametrize('k,parts,splits', [(2048, 1, 1), (8192, 4, 2)])
def test_gpu_nonperiodic_bf16_matches_stable_rounding_and_fp32_error_bound(tmp_path, k, parts, splits):
    """Use FP64 product sums and the actual FP32 reduction depth as an oracle.

    The existing op allows different FP32 trees, but still rounds every part
    and the final result to BF16. Stable rounding intervals must match exactly.
    """
    import json
    import math
    import torch
    from triton.flagmega.artifacts import write_artifact
    from triton.flagmega.runtime import load
    from triton.flagmega.runtime.module import GeneratedTirCallGraphModule as Runtime
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (9, 0):
        pytest.skip('requires SM90')
    n = 4096
    module = Compiler().compile(dynamic_projection_graph(k=k, n=n, parts=parts,
        lanes=(2, 4), input_lanes=(2, 4), accumulation_splits=splits)).module
    runtime = load(write_artifact(module, tmp_path / 'artifact', target='nvidia-sm90',
                                 emit_executable=True), device='cuda:0')
    generator = torch.Generator(device='cuda').manual_seed(8937)
    x = (torch.randn((1, k), generator=generator, device='cuda') *
         torch.exp2(torch.randint(-6, 7, (1, k), generator=generator, device='cuda').float())).bfloat16()
    w = (torch.randn((n, k), generator=generator, device='cuda') *
         torch.exp2(torch.randint(-6, 7, (n, k), generator=generator, device='cuda').float())).bfloat16()
    guard = torch.full((n + 32,), 123., dtype=torch.bfloat16, device='cuda')
    out = guard[16:-16]
    values = {'x': x.reshape(1, k // 8, 2, 4), 'w': w}
    arguments = tuple(out.reshape(1, n // 8, 2, 4) if spec['role'] == 'result' else values[spec['value']]
                      for spec in runtime.external_arguments)
    try:
        Runtime.prepare(runtime, *arguments)
        Runtime.run_into(runtime, *arguments)
        torch.cuda.synchronize()
        span = k // parts
        depth = math.ceil(span / splits / 512) + 9 + splits - 1
        eps = 2.**-24
        gamma = depth * eps / (1 - depth * eps)
        lower, upper = [torch.zeros((n,), dtype=torch.float32, device='cuda') for _ in range(2)]
        for part in range(parts):
            start = part * span
            # These bounded-exponent BF16 products are exactly representable
            # in FP32. Only the summation tree contributes rounding error.
            products = w[:, start:start+span].double() * x[0, start:start+span].double()
            exact, error = products.sum(1), gamma * products.abs().sum(1)
            lo, hi = (exact - error).bfloat16().float(), (exact + error).bfloat16().float()
            # Rounded FP32 addition is monotone. Execute the required ordered
            # part sums on interval endpoints; a gamma bound here needlessly
            # widens exact BF16 midpoint sums that FP32 represents exactly.
            lower += lo
            upper += hi
        lo, hi = lower.bfloat16(), upper.bfloat16()
        stable = lo == hi
        assert bool(((out >= lo) & (out <= hi)).all())
        torch.testing.assert_close(out[stable], lo[stable], rtol=0, atol=0)
        assert bool((guard[:16] == 123.).all() & (guard[-16:] == 123.).all())
        (tmp_path / 'arbitrary-bf16.json').write_text(json.dumps({'passed': True, 'k': k, 'n': n,
            'parts': parts, 'accumulation_splits': splits, 'stable_exact_outputs': int(stable.sum()),
            'ambiguous_outputs': int((~stable).sum()), 'fp32_depth_bound': depth,
            'resources': runtime.resource_report}, indent=2) + '\n')
    finally:
        runtime.close()
