# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: MIT
from dataclasses import replace

import pytest
from triton.flagmega import ir as fm
from triton.flagmega.errors import IRVerificationError
from triton.flagmega.passes.tir.fuse_partial_rms_packed_dense_matmul_glu import fuse_partial_rms_packed_dense_matmul_glu


def graph(*, extra=None, norm_attrs=None, vector=True):
    p, b = fm.Placement((8, 16), 'yx', 'bb'), fm.SBP.broadcast()
    lane = 4 if vector else 1
    split = fm.SBP.split_contiguous((0, 1), granularity=16 // lane)
    dtype = fm.vector_type('float32', (4, )) if vector else fm.DType.FLOAT32
    affine_dtype = fm.vector_type('bfloat16', (4, )) if vector else fm.DType.BFLOAT16
    raw_type = fm.DistributedType(fm.tensor_type(dtype, (1, 2048 // lane)), (b, split), p)
    param_type = fm.DistributedType(fm.tensor_type(affine_dtype, (2048 // lane, )), (split, ), p)
    wt = fm.DistributedType(fm.tensor_type('bfloat16', (128, 768, 2, 64)),
                            (b, fm.SBP.split_contiguous((0, 1), granularity=6), b, b), p)

    class Graph(fm.Module):

        def __init__(self):
            super().__init__(dialect='high_level', stage='vector_contracts_lowered', entry='main')

        def forward(self):
            x = self.input('x', raw_type, id='x')
            y = self.input('y', raw_type, id='y')
            scale = self.input('scale', param_type, id='scale')
            gate = self.input('gate', wt, id='gate')
            up = self.input('up', wt, id='up')
            bias = fm.F.builtin.splat_const(param_type, 0.0, name='bias')
            raw = (fm.F.math.vectorized_binary(x, y, binary_op='add', name='raw') if vector else fm.F.math.add(
                x, y, name='raw'))
            stats = fm.F.nn.norm_stats(raw, axis=-1, use_mean=False, name='stats')
            materialized = fm.F.distributed.boxing(stats, replace(stats.type, partial=None), name='materialized')
            attrs = {'axis': -1, 'epsilon': 1e-6, 'use_mean': False, 'output_dtype': 'bfloat16', **(norm_attrs or {})}
            norm = fm.F.nn.norm_apply(raw, materialized, scale, bias, name='norm', **attrs)
            broad = fm.F.distributed.sharded_view(norm, replace(norm.type, axis_policies=(b, b)), name='broad')
            scalar = fm.F.tensors.bitcast(broad, 'bfloat16', name='scalar') if vector else broad
            glu = fm.F.nn.packed_dense_matmul_glu(scalar, gate, up, round_activation=False, name='glu')
            outputs = [glu]
            if extra is not None:
                outputs.append(
                    {'stats': stats, 'materialized': materialized, 'norm': norm, 'scalar': scalar, 'raw': raw}[extra])
            self.function('main', (x, y, scale, gate, up), outputs)

    module = Graph().build()
    return replace(module, metadata={'auto_distribution': {'placement': p.to_data()}})


@pytest.mark.parametrize('vector', (False, True))
def test_local_stats_retained_with_explicit_partial_and_raw_late_reader(vector, tmp_path):
    before = graph(extra='raw', vector=vector)
    after = fuse_partial_rms_packed_dense_matmul_glu(before, partial_group_size=32)
    assert after.node_map['stats'] == before.node_map['stats']
    assert after.node_map['raw'] == before.node_map['raw']
    assert after.function_map['main'].outputs[-1] == 'raw'
    fused = after.node_map['glu']
    assert fused.op == 'ntt.partial_rms_packed_dense_matmul_glu'
    assert len(fused.inputs) == 6 and fused.inputs[-1] == 'stats'
    assert {'materialized', 'norm', 'broad', 'scalar'}.isdisjoint(after.node_map)
    assert after.node_map['stats'].type.partial.axes == (0, 1)
    assert fm.load_module(fm.emit_module(after, tmp_path / 'partial.py')).semantic_hash == after.semantic_hash


@pytest.mark.parametrize('extra', ('stats', 'materialized', 'norm', 'scalar'))
def test_shared_intermediate_is_not_silently_absorbed(extra):
    before = graph(extra=extra)
    assert fuse_partial_rms_packed_dense_matmul_glu(before, partial_group_size=32) is before


def test_late_placement_rejected():
    before = graph()
    late = replace(
        before, nodes=tuple(
            replace(n, metadata={'bufferization.memory_space': 'block_local_data'}) if n.id == 'raw' else n
            for n in before.nodes))
    with pytest.raises(IRVerificationError, match='late memory placement'):
        fuse_partial_rms_packed_dense_matmul_glu(late, partial_group_size=32)


def explicit_target():
    from triton.flagmega.targets import NvidiaSm90Target
    model = NvidiaSm90Target().triton_implementation_model
    base = model.implementation('tir.dense_matmul_glu.packed_tensor_descriptor_table_paired_smem_pipeline_inline_gemv')
    donor = model.implementation('tir.dense_matmul_glu.packed_tensor_descriptor_paired_smem_pipeline_full_lhs_gemv')
    variant = replace(
        base, id='tir.dense_matmul_glu.test_explicit_partial_rms',
        parameters={**base.parameters, 'lhs_stage_extent':
                    2048}, contract={**base.contract, 'fusion': 'explicit_partial_rms', 'norm_math_modes': ('fast', )},
        shared_workspaces=(*base.shared_workspaces, donor.shared_workspaces[2]), transfer_pipeline=replace(
            base.transfer_pipeline, consumer_shared_workspace_indices=(2, ),
            channels=(replace(base.transfer_pipeline.channels[0], source_argument_indices=(3, 4)), )))
    return NvidiaSm90Target(triton_implementation_model=replace(
        model, implementations=(*model.implementations, variant), preferences={
            **model.preferences, 'dense_matmul_glu': (variant.id, *model.preferences.get('dense_matmul_glu', ()))
        }, name='explicit-partial-rms-test'))


@pytest.fixture(scope='module')
def compiled_partial():
    from triton.flagmega.compiler import Compiler
    compiler = Compiler()
    compiler.target = explicit_target()
    return compiler.compile(fuse_partial_rms_packed_dense_matmul_glu(graph(extra='raw'), partial_group_size=32)).module


def test_normal_planner_has_real_raw_and_partial_chip_backing(compiled_partial):
    from triton.flagmega.codegen.triton import describe_tir_package
    plan = fm.verify_buffer_plan(compiled_partial)
    raw, stats = (plan.buffer_map[name] for name in ('raw', 'stats'))
    for buffer in (raw, stats):
        assert plan.memory_space_map[buffer.mem_span.buffer.memory_space].sharing_scope.value == 'chip'
    assert raw.mem_span.nbytes == 8192
    assert raw.distributed_storage_kind.value == 'canonical_global'
    assert stats.mem_span.nbytes == 4  # One logical owner, not the whole allocation.
    assert plan.physical_buffer_map[stats.physical_id].size.fixed_value == 512
    assert stats.distributed_storage_kind.value == 'compact_per_owner'
    assert not stats.mem_span.may_alias(raw.mem_span)
    package = describe_tir_package(compiled_partial)
    call, = [c for c in package['render_calls'] if c['family'] == 'dense_matmul_glu']
    norm = call['norm_stage']
    assert norm['schema'] == 'flagmega.rms-glu-stage/v2'
    assert norm['partial_owner_count'] == 128
    assert 'dense_glu_norm_partial_lane' in norm['partial_stats_offset']
    assert tuple(call['pipeline_contract']['channels'][0]['source_argument_indices']) == (3, 4)


@pytest.mark.parametrize('field,value', [('memory_sharing_scope', 'block'), ('owner_count', 1),
                                         ('component_stride_scalar_elements', 0), ('storage_kind', 'canonical_global'),
                                         ('scalar_dtype', 'bfloat16'), ('pool_scope_local', True)])
def test_encoder_rejects_invalid_partial_backing(compiled_partial, field, value):
    from copy import deepcopy
    from triton.flagmega.codegen.triton.runtime_binding import describe_function_runtime_binding
    from triton.flagmega.codegen.triton.kernel_call_renderers import _dense_matmul_glu_call
    from triton.flagmega.errors import CodegenError
    binding = describe_function_runtime_binding(compiled_partial)
    raw = deepcopy(next(c for c in binding['call_abi']['kernel_calls'] if c['family'] == 'dense_matmul_glu'))
    raw['inputs'][5]['buffers'][0]['abi'][field] = value
    with pytest.raises(CodegenError, match='partial statistics'):
        _dense_matmul_glu_call(raw)


def test_partial_variant_not_offered_to_three_input_glu():
    from triton.flagmega.compiler import Compiler
    compiler = Compiler()
    compiler.target = explicit_target()
    proposed = compiler.compile(graph(), stop_after='propose-tir').module
    point = next(p for p in proposed.selection_points if p.owner == 'glu')
    assert 'tir.dense_matmul_glu.test_explicit_partial_rms' not in {c.id for c in point.candidates}
