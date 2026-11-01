# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Independent FP8 tiles retain complete material P/T/N and K128 domains."""
from dataclasses import replace
from math import prod

import pytest

from triton.flagmega import ir as fm
from triton.flagmega.codegen.triton.tir_package import describe_tir_package
from triton.flagmega.compiler import Compiler
from triton.flagmega.ir.launch_recipe import TaskLaunchRecipe
from python.test.flagmega.ir.ops.nn.test_partitioned_block_scaled_linear import Graph


def dynamic_fp8_projection_graph(*, kind='ordinary', max_ctas=None, **options):
    module = Graph(**options).build()
    return replace(module, nodes=tuple(replace(n, metadata={**n.metadata,
        'execution_mesh': '*', 'kernel_program_kind': kind,
        **({'kernel_max_ctas': max_ctas} if max_ctas is not None else {})})
        if n.id == 'projection' else n for n in module.nodes))


@pytest.mark.parametrize('kind,tokens,n,k,parts,lanes', [
    ('ordinary', 1, 83, 4096, 4, ()),
    ('persistent', 2, 80, 512, 4, (2, 4)),
    ('ordinary', 1, 4096, 1024, 1, (2, 4)),
])
def test_dynamic_fp8_recipe_covers_whole_part_token_row_tiles(tmp_path, kind, tokens, n, k, parts, lanes):
    module = Compiler().compile(dynamic_fp8_projection_graph(kind=kind,
        max_ctas=2 if kind == 'persistent' else None, tokens=tokens, n=n, k=k, parts=parts,
        input_lanes=lanes, output_lanes=lanes)).module
    descriptor = describe_tir_package(module)
    program, = descriptor['kernels']
    call, = (c for c in descriptor['render_calls'] if c['family'] == 'partitioned_block_scaled_linear')
    assert program['execution_mesh']['kind'] == 'dynamic'
    assert program['program_kind'] == kind and len(program['calls']) == 1
    recipe = TaskLaunchRecipe.from_data(program['launch_recipe'])
    assert recipe.work_size == parts * tokens * ((n + call['task_tile_n'] - 1) // call['task_tile_n'])
    assert recipe.tile_size == 1 and recipe.task_count == call['tasks']
    assert recipe.resolve(sm_count=132) == (2 if kind == 'persistent' else call['tasks'], 1, 1)
    assert not program['requires_grid_wide_sync']
    assert call['part_k'] == k // parts
    assert call['activation_quantization'] == 'dynamic_fp8_sm90'
    assert call['tile_n'] >= 64
    restored = fm.load_module(fm.emit_module(module, tmp_path / 'physical.py'))
    assert restored.semantic_hash == module.semantic_hash
    result = fm.verify_buffer_plan(restored).buffer_map['projection']
    assert result.nbytes == parts * tokens * n * 2


@pytest.mark.parametrize('wrong_mapping', [{'work_size': 6}, {'tile_size': 64}])
def test_dynamic_fp8_rejects_scalar_flattening_across_partition_tails(wrong_mapping):
    from copy import deepcopy
    from triton.flagmega.codegen.triton.partitioned_block_scaled_linear_renderer import partitioned_block_scaled_linear_call
    from triton.flagmega.errors import CodegenError
    module = Compiler().compile(dynamic_fp8_projection_graph(tokens=1, n=83, k=4096, parts=4)).module
    raw, = deepcopy(describe_tir_package(module)['render_calls'])
    raw['parameters']['task_mapping'].update(wrong_mapping)
    with pytest.raises(CodegenError, match='complete P/T/N tiles'):
        partitioned_block_scaled_linear_call(raw)


@pytest.mark.parametrize('dynamic', (False, True))
def test_rejects_mma_columns_that_promote_native_fp8_to_fp16(dynamic):
    from triton.flagmega.errors import CodegenError
    from triton.flagmega.targets import NvidiaSm90Target
    compiler = Compiler()
    target = NvidiaSm90Target()
    model = target.triton_implementation_model
    target.triton_implementation_model = replace(model, implementations=tuple(
        replace(i, parameters={**i.parameters, 'accumulator_layout': 'mma',
                               'compute_num_warps': 4, 'mma_columns': 8})
        if i.id == 'tir.partitioned_block_scaled_linear.local' else i for i in model.implementations))
    compiler.target = target
    source = dynamic_fp8_projection_graph(tokens=1,n=64,k=1024,parts=1) if dynamic else Graph(tokens=1,n=64,k=1024,parts=1).build()
    with pytest.raises(CodegenError, match='native FP8 MMA.*16'):
        describe_tir_package(compiler.compile(source).module)


@pytest.mark.parametrize('dynamic', (False, True))
@pytest.mark.parametrize('execution_warps', (4, 8))
def test_activation_layout_selection_survives_normal_fp8_codegen(dynamic, execution_warps):
    from triton.flagmega.codegen.triton.tir_package import render_tir_package
    from triton.flagmega.targets import NvidiaSm90Target
    compiler = Compiler()
    target = NvidiaSm90Target()
    model = target.triton_implementation_model
    target.triton_implementation_model = replace(model, implementations=tuple(
        replace(i, parameters={**i.parameters, 'activation_layout': 'warp',
                               'compute_num_warps': 4})
        if i.id == 'tir.partitioned_block_scaled_linear.local' else i for i in model.implementations))
    compiler.target = target
    source = dynamic_fp8_projection_graph(tokens=1, n=64, k=1024, parts=1) if dynamic else Graph(tokens=1, n=64, k=1024, parts=1).build()
    source = replace(source, nodes=tuple(replace(n, metadata={**n.metadata,
        'execution_mesh': '*' if dynamic else {'kind': 'fixed', 'shape': [1]},
        'kernel_num_warps': execution_warps}) if n.id == 'projection' else n for n in source.nodes))
    descriptor = describe_tir_package(compiler.compile(source).module)
    call, = descriptor['render_calls']
    assert call['activation_layout'] == 'warp'
    assert descriptor['kernels'][0]['num_warps'] == execution_warps
    assert f'BlockEncoding([4, 1], [32, 1], [1, {execution_warps}], [1, 0])' in render_tir_package(descriptor, 'test')


@pytest.mark.parametrize('dynamic', (False, True))
@pytest.mark.parametrize('parameters,message', [
    ({'activation_layout': 'unknown'}, 'activation layout'),
    ({'compute_num_warps': 2}, 'native FP8 MMA.*warps'),
])
def test_rejects_invalid_activation_or_native_fp8_warp_layout(dynamic, parameters, message):
    from triton.flagmega.errors import CodegenError
    from triton.flagmega.targets import NvidiaSm90Target
    compiler = Compiler()
    target = NvidiaSm90Target()
    model = target.triton_implementation_model
    target.triton_implementation_model = replace(model, implementations=tuple(
        replace(i, parameters={**i.parameters, **parameters})
        if i.id == 'tir.partitioned_block_scaled_linear.local' else i for i in model.implementations))
    compiler.target = target
    source = dynamic_fp8_projection_graph(tokens=1, n=64, k=1024, parts=1) if dynamic else Graph(tokens=1, n=64, k=1024, parts=1).build()
    with pytest.raises(CodegenError, match=message):
        describe_tir_package(compiler.compile(source).module)


@pytest.mark.parametrize('kind,quantization', [
    ('ordinary', 'dynamic_fp8_sm90'), ('persistent', 'dynamic_fp8_sm90'),
    ('ordinary', 'dynamic_fp8'), ('persistent', 'dynamic_fp8'),
])
@pytest.mark.parametrize('accumulator_layout,num_warps,mma_columns,k,activation_layout,execution_warps,task_tile_n', [
    ('vector',8,16,1024,'auto',8,64), ('mma',8,16,1024,'auto',8,64),
    ('mma',4,16,1024,'auto',4,64), ('mma',4,16,4096,'auto',4,64),
    ('vector',8,16,1024,'warp',8,64), ('mma',8,16,1024,'warp',8,64),
    ('mma',4,16,4096,'warp',4,64), ('mma',4,16,4096,'warp',8,64),
    ('mma',4,16,4096,'warp',4,32),
])
def test_gpu_dynamic_fp8_preserves_original_mma_parts_tails_and_graph(tmp_path, kind, quantization, accumulator_layout, num_warps, mma_columns, k, activation_layout, execution_warps, task_tile_n, k_blocks_per_mma=1, k_group_row_order='group_major'):
    import json
    import os
    from pathlib import Path
    import torch
    from triton.flagmega.artifacts import write_artifact
    from triton.flagmega.runtime import load
    from triton.flagmega.runtime.module import GeneratedTirCallGraphModule as Runtime
    from triton.flagmega.targets import NvidiaSm90Target
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (9, 0):
        pytest.skip('requires SM90')
    tokens, n, parts, lanes = 2, 80, 4, (2, 4)
    options = dict(tokens=tokens, n=n, k=k, parts=parts, input_lanes=lanes,
                   output_lanes=lanes, quantization=quantization)
    compiler = Compiler()
    target = NvidiaSm90Target()
    model = target.triton_implementation_model
    target.triton_implementation_model = replace(model, implementations=tuple(
        replace(i, parameters={**i.parameters, 'accumulator_layout': accumulator_layout,
                               'compute_num_warps': num_warps, 'mma_columns': mma_columns,
                               'activation_layout': activation_layout, 'tile_n': task_tile_n,
                               'k_blocks_per_mma': k_blocks_per_mma,
                               'k_group_row_order': k_group_row_order})
        if i.id == 'tir.partitioned_block_scaled_linear.local' else i for i in model.implementations))
    compiler.target = target
    dynamic = dynamic_fp8_projection_graph(kind=kind,
        max_ctas=2 if kind == 'persistent' else None, **options)
    dynamic = replace(dynamic, nodes=tuple(replace(n, metadata={**n.metadata,
        'kernel_num_warps': execution_warps}) if n.id == 'projection' else n for n in dynamic.nodes))
    modules = [Compiler().compile(Graph(**options).build()).module, compiler.compile(dynamic).module]
    runtimes = [load(write_artifact(m, tmp_path / str(i), target='nvidia-sm90',
        emit_executable=True), device='cuda:0') for i, m in enumerate(modules)]
    generator = torch.Generator(device='cuda').manual_seed(3187)
    values = [dict(x=torch.empty((tokens, k // prod(lanes), *lanes), dtype=torch.bfloat16, device='cuda'),
                   weight=(torch.randn((n, k), generator=generator, device='cuda') / 4).to(torch.float8_e4m3fn),
                   scales=(torch.rand(((n + 127) // 128, k // 128), generator=generator, device='cuda') + .125) / 128)
              for _ in range(2)]
    guards = [torch.full((parts * tokens * n + 32,), 123., dtype=torch.bfloat16, device='cuda') for _ in range(2)]
    outputs = [g[16:-16].reshape(parts, tokens, n // prod(lanes), *lanes) for g in guards]
    caller = torch.cuda.Stream()
    def arguments(runtime, bank, output):
        return tuple(output if s['role'] == 'result' else values[bank][s['value']]
                     for s in runtime.external_arguments)
    try:
        for i, runtime in enumerate(runtimes):
            Runtime.prepare(runtime, *arguments(runtime, 0, outputs[i]))
        if k_blocks_per_mma > 1 and k_group_row_order == 'n_major':
            import re
            # The scale consumer must not distribute one K128 amax across
            # warps. Inspect the actual lowered reduction, not the template.
            ttgir = runtimes[1]._prepared[0].compiled_kernel.asm['ttgir']
            reductions = re.findall(
                r'"tt.reduce"\([^)]*\) <\{axis = (\d+) : i32\}>.*? : \(tensor<(\d+)x(\d+)xf32, (#[\w]+)>\)',
                ttgir, re.S)
            assert reductions
            for axis, rows, columns, name in reductions:
                assert (int(rows), int(columns))[int(axis)] == 128
                # Quantize each distinct group once. The MMA's additional
                # columns replicate those codes after the reduction.
                assert (int(rows), int(columns))[1 - int(axis)] == k_blocks_per_mma
                layout = next(line for line in ttgir.splitlines()
                              if line.startswith(name + ' = '))
                for field, expected in (
                    ('sizePerThread', 4), ('threadsPerWarp', 32), ('warpsPerCTA', 1),
                ):
                    dimensions = re.search(field + r' = \[(\d+), (\d+)\]', layout)
                    assert dimensions and int(dimensions[1 + int(axis)]) == expected, layout
        for bank in range(2):
            for mutation in range(3):
                values[bank]['x'].copy_((torch.randn((tokens, k // prod(lanes), *lanes), generator=generator,
                    device='cuda') * (0 if mutation == 1 else 1 + bank)).bfloat16())
                values[bank]['scales'].mul_(1.03125)
                for i, runtime in enumerate(runtimes):
                    outputs[i].fill_(float('nan'))
                    caller.wait_stream(torch.cuda.current_stream())
                    Runtime.run_into(runtime, *arguments(runtime, bank, outputs[i]), stream=caller.cuda_stream)
                caller.synchronize()
                torch.testing.assert_close(outputs[1], outputs[0], rtol=0, atol=0)
        captures = []
        for i, runtime in enumerate(runtimes):
            with torch.cuda.graph(graph := torch.cuda.CUDAGraph(), stream=caller):
                for _ in range(3):
                    Runtime.run_into(runtime, *arguments(runtime, 1, outputs[i]), stream=caller.cuda_stream)
            captures.append(graph)
        for _ in range(3):
            values[1]['x'].copy_(torch.randn(values[1]['x'].shape, generator=generator, device='cuda').bfloat16())
            caller.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(caller):
                for graph in captures:
                    graph.replay()
            caller.synchronize()
            torch.testing.assert_close(outputs[1], outputs[0], rtol=0, atol=0)
        assert all(bool((g[:16] == 123.).all() & (g[-16:] == 123.).all()) for g in guards)
        assert runtimes[1].resource_report['spill_bytes'] == 0
        assert not runtimes[1].resource_report['cooperative_grid']
        directory = Path(os.environ.get('FLAGMEGA_DYNAMIC_FP8_RECEIPT_DIR', str(tmp_path)))
        directory.mkdir(parents=True, exist_ok=True)
        (directory / f'{kind}-{quantization}-{accumulator_layout}-warps{num_warps}-cols{mma_columns}-k{k}-{activation_layout}-exec{execution_warps}-tile{task_tile_n}-groups{k_blocks_per_mma}-{k_group_row_order}.json').write_text(json.dumps(dict(passed=True,
            normal_artifact=str(tmp_path / '1'), original_mma_bitwise=True, kind=kind,
            quantization=quantization, accumulator_layout=accumulator_layout, num_warps=num_warps, mma_columns=mma_columns, k=k, parts=parts,
            activation_layout=activation_layout,
            execution_warps=execution_warps,
            task_tile_n=task_tile_n,
            k_blocks_per_mma=k_blocks_per_mma,
            k_group_row_order=k_group_row_order,
            input_weight_scale_pointer_rebind=True,
            graph_changed_input_replays=3, guards=True, resources=runtimes[1].resource_report,
            semantic_hash=modules[1].semantic_hash, source_sha256=runtimes[1].codegen['source_sha256'],
            performance_measurement=False, full_model_acceptance=False), indent=2) + '\n')
    finally:
        for runtime in runtimes:
            runtime.close()
