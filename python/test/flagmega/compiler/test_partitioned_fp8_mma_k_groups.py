# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: MIT
"""One native K128 dot may carry independent K groups in its row/column pairs."""
from dataclasses import replace

import pytest
from triton.flagmega import ir as fm
from triton.flagmega.compiler import Compiler
from triton.flagmega.codegen.triton.tir_package import describe_tir_package, render_tir_package
from triton.flagmega.ir.launch_recipe import TaskLaunchRecipe
from triton.flagmega.targets import NvidiaSm90Target
from python.test.flagmega.compiler.test_dynamic_partitioned_fp8_linear import (
    dynamic_fp8_projection_graph,
    test_gpu_dynamic_fp8_preserves_original_mma_parts_tails_and_graph as _gpu_contract,
)
from python.test.flagmega.ir.ops.nn.test_partitioned_block_scaled_linear import Graph


def grouped_compiler(groups, tile=16, accumulator='vector', row_order='group_major'):
    compiler = Compiler()
    target = NvidiaSm90Target()
    model = target.triton_implementation_model
    target.triton_implementation_model = replace(model, implementations=tuple(
        replace(i, parameters={**i.parameters, 'k_blocks_per_mma': groups,
                               'k_group_row_order': row_order,
                               'tile_n': tile, 'compute_num_warps': 4,
                               'activation_layout': 'warp', 'accumulator_layout': accumulator})
        if i.id == 'tir.partitioned_block_scaled_linear.local' else i for i in model.implementations))
    compiler.target = target
    return compiler


def test_existing_selection_without_K_groups_keeps_default_contract(tmp_path):
    from triton.flagmega.targets.nvidia.partitioned_block_scaled_linear import (
        partitioned_block_scaled_linear_implementations,
    )
    implementation, = partitioned_block_scaled_linear_implementations()
    assert 'k_blocks_per_mma' not in implementation.parameters
    assert 'k_group_row_order' not in implementation.parameters
    compiler = Compiler()
    compiler.target = NvidiaSm90Target()
    module = compiler.compile(dynamic_fp8_projection_graph(tokens=1, n=80, k=3072, parts=1)).module
    restored = fm.load_module(fm.emit_module(module, tmp_path / 'default.py'))
    assert restored.semantic_hash == module.semantic_hash
    call, = describe_tir_package(restored)['render_calls']
    assert call['k_blocks_per_mma'] == 1 and call['dot_k'] == 128
    assert call['k_group_row_order'] == 'group_major'


@pytest.mark.parametrize('dynamic', (False, True))
@pytest.mark.parametrize('groups,tile', ((2, 32), (4, 32), (8, 16), (16, 16)))
@pytest.mark.parametrize('row_order', ('group_major', 'n_major'))
def test_grouped_dot_keeps_K128_and_complete_P_T_N_task_domain(tmp_path, dynamic, groups, tile, row_order):
    options = dict(tokens=2, n=80, k=12288, parts=4, input_lanes=(2, 4), output_lanes=(2, 4))
    source = dynamic_fp8_projection_graph(**options) if dynamic else Graph(**options).build()
    module = grouped_compiler(groups, tile, row_order=row_order).compile(source).module
    package = describe_tir_package(module)
    compile(render_tir_package(package, 'grouped_fp8'), '<grouped_fp8>', 'exec')
    call, = package['render_calls']
    assert call['k_blocks_per_mma'] == groups
    assert call['k_group_row_order'] == row_order
    assert call['tile_n'] == max(64, groups * tile)
    assert call['part_k'] == 3072 and call['dot_k'] == 128
    assert call['tasks'] == 4 * 2 * ((80 + tile - 1) // tile)
    if dynamic:
        program, = package['kernels']
        assert len(program['calls']) == 1
        assert TaskLaunchRecipe.from_data(program['launch_recipe']).task_count == call['tasks']
        assert not program['requires_grid_wide_sync']
    restored = fm.load_module(fm.emit_module(module, tmp_path / 'physical.py'))
    assert restored.semantic_hash == module.semantic_hash
    assert fm.verify_buffer_plan(restored).buffer_map['projection'].nbytes == 4 * 2 * 80 * 2


@pytest.mark.parametrize('groups', (0, 3, 32))
def test_rejects_invalid_K_group_geometry(groups):
    from triton.flagmega.errors import CodegenError
    with pytest.raises(CodegenError, match='K groups'):
        describe_tir_package(grouped_compiler(groups).compile(
            dynamic_fp8_projection_graph(tokens=1, n=80, k=3072, parts=1)).module)


def test_grouped_dot_rejects_replicated_MMA_accumulator_contract():
    from triton.flagmega.errors import CodegenError
    with pytest.raises(CodegenError, match='ordered vector accumulator'):
        describe_tir_package(grouped_compiler(4, accumulator='mma').compile(
            dynamic_fp8_projection_graph(tokens=1, n=80, k=3072, parts=1)).module)


def test_rejects_unknown_K_group_row_mapping():
    from triton.flagmega.errors import CodegenError
    with pytest.raises(CodegenError, match='K group row order'):
        describe_tir_package(grouped_compiler(4, row_order='unknown').compile(
            dynamic_fp8_projection_graph(tokens=1, n=80, k=3072, parts=1)).module)


@pytest.mark.parametrize('kind', ('ordinary', 'persistent'))
@pytest.mark.parametrize('quantization', ('dynamic_fp8_sm90', 'dynamic_fp8'))
@pytest.mark.parametrize('groups,tile', ((2, 32), (4, 32), (8, 16), (16, 16)))
@pytest.mark.parametrize('row_order', ('group_major', 'n_major'))
def test_GPU_grouped_dot_preserves_K128_rounding_parts_tails_and_Graph(tmp_path, kind, quantization, groups, tile, row_order):
    _gpu_contract(tmp_path, kind, quantization, 'vector', 4, 16, 12288, 'warp', 4, tile,
                  k_blocks_per_mma=groups, k_group_row_order=row_order)


@pytest.mark.parametrize('kind', ('ordinary', 'persistent'))
@pytest.mark.parametrize('quantization', ('dynamic_fp8_sm90', 'dynamic_fp8'))
def test_GPU_N_major_uses_enclosing_eight_warps_with_selected_four(tmp_path, kind, quantization):
    _gpu_contract(tmp_path, kind, quantization, 'vector', 4, 16, 12288, 'warp', 8, 32,
                  k_blocks_per_mma=4, k_group_row_order='n_major')
