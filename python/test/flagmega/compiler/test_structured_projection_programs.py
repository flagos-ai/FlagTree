# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: MIT
"""A recurrent projection preserves TP partitions across independent entries."""
from dataclasses import replace

import pytest

from triton.flagmega import ir as fm
from triton.flagmega.codegen.triton.tir_package import describe_tir_package
from triton.flagmega.compiler import Compiler
from triton.flagmega.ir.ops.core import construction_scope


def structured_projection_loop(*, trips=3, kind='ordinary', second_trips=None, size=4096):
    b = fm.IRBuilder(dialect='high_level', stage='imported')
    placement = fm.Placement((4, 8, 16), 'cyx', 'cbb')
    state_type = fm.tensor_type(fm.vector_type('bfloat16', (2, 4)), (1, size // 8))
    weight_type = fm.tensor_type('bfloat16', (size, size))
    index = fm.tensor_type('int64', ())
    iv = b.var('iv', index, id='iv')
    state = b.var('state', state_type, id='state')
    weights = b.var('weights', weight_type, id='weights')
    with construction_scope(b):
        partial = fm.F.nn.partitioned_linear(state, weights, partitions=4, output_lanes=(2, 4),
            accumulation_splits=2, name='projection', metadata={'execution_mesh': '*',
                'kernel_program_kind': kind, **({'kernel_max_ctas': 2} if kind == 'persistent' else {})})
        published = fm.F.math.add(partial, state, name='published',
                                  metadata={'execution_mesh': (4, 16)})
    body = b.region('layer', (iv, state, weights), (published,), iter_count=1)
    value = b.var('value', state_type, id='value')
    weight = b.var('weight', weight_type, id='weight')
    with construction_scope(b):
        lower = fm.F.builtin.scalar_const(index, 0, name='lower')
        upper = fm.F.builtin.scalar_const(index, trips, name='upper')
        step = fm.F.builtin.scalar_const(index, 1, name='step')
        loop = fm.F.control.for_loop(lower, upper, step, value, body=body, captures=(weight,), name='loop')
        output = fm.F.tensors.get_item(loop, 0, name='output')
        if second_trips is not None:
            second_upper = fm.F.builtin.scalar_const(index, second_trips, name='second_upper')
            other = fm.F.control.for_loop(lower, second_upper, step, output, body=body,
                                         captures=(weight,), name='other_loop')
            output = fm.F.tensors.get_item(other, 0, name='other_output')
    b.function('main', (value, weight), (output,))
    return replace(b.build(entry='main'), metadata={'auto_distribution': {'placement': placement.to_data()},
                                                   'execution_mesh': (8, 16)})


def compile_structured_projection_loop(**options):
    from triton.flagmega.targets import NvidiaSm90Target
    from triton.flagmega.passes.auto_distributed import solve_search_graph
    from triton.flagmega.passes.auto_distributed.auto_distributed import AutoDistributedPass
    from triton.flagmega.passes.auto_distributed.materializer import distribution_selection_state
    from triton.flagmega.selection import override_plan
    compiler = Compiler()
    compiler.target = NvidiaSm90Target().with_placement(fm.Placement((4, 8, 16), 'cyx', 'cbb'))
    proposal = compiler.compile(structured_projection_loop(**options), stop_after='propose-distribution').module
    search = AutoDistributedPass._build_graph(proposal, compiler.target)
    bucket, = (b for b in search.buckets if b.node_id == 'projection')
    partial, = (c for c in bucket.candidates if c.return_type.partial == fm.SBPPartial((0,)))
    result = solve_search_graph(search, fixed_selections={bucket.node_id: partial.id})
    _, records = distribution_selection_state(result, policy=compiler.target.distribution_policy.identity)
    selected = compiler.run_stage(proposal, 'auto-distributed', plan=override_plan(proposal,
        tuple((r.point_id, r.candidate_id) for r in records))).module
    return compiler.compile(selected).module


@pytest.mark.parametrize('kind', ['ordinary', 'persistent'])
@pytest.mark.parametrize('trips', [0, 1, 3])
def test_mixed_projection_loop_has_real_partial_collective_and_carry_entries(tmp_path, kind, trips):
    module = compile_structured_projection_loop(trips=trips, kind=kind)
    restored = fm.load_module(fm.emit_module(module, tmp_path / 'projection_loop.py'))
    assert restored.semantic_hash == module.semantic_hash
    plan = fm.verify_buffer_plan(restored)
    partial = plan.buffer_map['projection']
    assert partial.distributed_type.partial == fm.SBPPartial((0,))
    assert partial.distributed_storage_kind is fm.DistributedBufferStorageKind.COMPACT_PER_CHIP
    assert partial.nbytes == 8192
    descriptor = describe_tir_package(restored)
    assert descriptor['kernel_program_schema'] == 'flagmega.kernel-programs/v6-structured'
    launches = descriptor['kernels']
    dynamic = [p for p in launches if p.get('execution_mesh', {}).get('kind') == 'dynamic']
    assert len(dynamic) == trips
    assert all(p['program_kind'] == kind and len(p['calls']) == 1 for p in dynamic)
    collective, = (c for c in descriptor['render_calls']
                   if c['implementation'] == 'tir.distributed_boxing.gather_reduce_scatter'
                   and c['mode'] == 'partial_sum')
    assert collective['partial_owner_count'] == 1
    assert collective['remote_peers'] == 4
    for index, launch in enumerate(launches):
        if launch in dynamic:
            assert launches[index - 1]['invocation_kind'] == 'device_completion'
            assert launches[index - 1]['grid'] == [128, 1, 1]
            assert not launch['requires_grid_wide_sync']
    copies = [p for p in launches if p['invocation_kind'] == 'loop_copy']
    assert len(copies) == 2 + 2 * trips
    assert copies[0]['transfer'] == ['initial', 'current']
    assert copies[-1]['transfer'] == ['current', 'result']


def test_two_projection_loops_share_device_definitions_and_have_distinct_frames():
    module = compile_structured_projection_loop(trips=1, second_trips=2)
    plan = fm.verify_buffer_plan(module)
    first, second = (plan.call_map[name] for name in ('loop', 'other_loop'))
    a, = first.loop_carries
    b, = second.loop_carries
    assert a.result == b.initial
    assert a.formal == b.formal
    assert not plan.buffer_map[a.current].physical_access_span.may_alias(
        plan.buffer_map[b.current].physical_access_span)
    kernels = describe_tir_package(module)['kernels']
    dynamic = [p for p in kernels if p.get('execution_mesh', {}).get('kind') == 'dynamic']
    assert len(dynamic) == 3
    assert len({p['entry_symbol'] for p in dynamic}) == 1
    assert dynamic[0]['argument_bindings'] != dynamic[1]['argument_bindings']


def test_dynamic_algorithm_internal_grid_wait_is_still_rejected():
    from copy import deepcopy
    from triton.flagmega.codegen.triton.structured_programs import describe_structured_programs
    from triton.flagmega.errors import CodegenError
    module = compile_structured_projection_loop(trips=1)
    descriptor = deepcopy(describe_tir_package(module))
    call, = (c for c in descriptor['render_calls'] if c['call'] == 'projection')
    call['facts']['internal_grid_barriers'] = 1
    with pytest.raises(CodegenError, match='Independent dynamic tiles cannot contain grid/device waits'):
        describe_structured_programs(module, {'programs': module.metadata['kernel_program_partition']}, descriptor)


def test_chip_local_carry_entry_compiles_with_its_own_physical_signature(tmp_path, monkeypatch):
    import triton
    from triton.backends.compiler import GPUTarget
    from triton.compiler import ASTSource
    from triton.flagmega.codegen.triton import render_triton_package
    from python.test.flagmega.codegen.triton.test_kernel_programs import import_source
    import triton._flagtree_backend as flagtree_backend
    # The explicit offline CUDA target has no active GPU to autodetect.
    monkeypatch.setattr(flagtree_backend, 'FLAGTREE_BACKEND', 'nvidia')
    module = compile_structured_projection_loop(trips=1)
    package = render_triton_package(module, tmp_path)
    source = import_source(tmp_path / 'generated_kernels.py')
    transfer, = (p for p in package['kernels'] if p.get('transfer') == ['initial', 'current'])
    kernel = getattr(source, transfer['symbol'])
    assert kernel.arg_names == ['_fm_loop_src', '_fm_loop_dst']
    binary = triton.compile(ASTSource(kernel, {'_fm_loop_src': '*bf16', '_fm_loop_dst': '*bf16'}),
                            target=GPUTarget('cuda', 90, 32), options={'num_warps': 4})
    assert binary.metadata.num_warps == 4
