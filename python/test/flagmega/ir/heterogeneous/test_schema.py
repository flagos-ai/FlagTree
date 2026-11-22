# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: MIT

from dataclasses import replace

import pytest

from triton.flagmega import ir as fm
from triton.flagmega.errors import IRSchemaError, IRVerificationError
from triton.flagmega.ir.tir.heterogeneous import (
    ChannelBarrier, ExecutionPlacement, HeterogeneousDrain, HeterogeneousLaunch,
    HeterogeneousPlan, HeterogeneousProgram, KernelPhaseInvoke, PipelineExchange,
    WorkerCall, WorkerFunction, verify_heterogeneous_programs,
)


def test_empty_optional_schema_preserves_default_hash():
    module = fm.IRModule('test', 'imported', (), (fm.Function('main', (), ()),), 'main')
    old = module.semantic_data()
    assert 'heterogeneous_plans' not in old
    assert 'heterogeneous_programs' not in old
    assert fm.IRModule.from_data(old).semantic_hash == module.semantic_hash


def make_module(*, repeated=False, projections=1):
    source = 'layer' if repeated else 'main'
    value_type = fm.tensor_type('float32', (1,))
    call = fm.KernelInvoke('projection', 'native')
    definition = fm.T.kernel_definition('native', 'cpp', (
        fm.T.prim_parameter('result', value_type, fm.T.PrimParameterRole.OUTPUT),
    ), fm.T.kernel_dispatch(
        semantic_op='math.add', arguments=(), outputs=('result',), writes=('result',),
        microkernel=fm.T.microkernel_selection('native.test.v1', 'test', 'native'),
    ), fm.T.return_((fm.T.return_binding(fm.T.value_ref('result', value_type), 'result'),)))
    source_function = fm.ExecutionFunction(source, (), (), fm.Sequential((call,)))
    functions = [source_function]
    if repeated:
        functions.append(fm.ExecutionFunction('main', (), (), fm.Sequential((
            fm.PrimFunctionCall('layer0', 'layer'), fm.PrimFunctionCall('layer1', 'layer'),
        ))))
    sentinel = fm.IRBuilder(dialect='bufferized_tir', stage='bufferized_tir').var('unused', value_type)
    module = fm.IRModule('bufferized_tir', 'bufferized_tir', (sentinel,),
        tuple(fm.Function(value.name, (), ()) for value in functions), 'main',
        kernel_definitions=(definition,), execution_functions=tuple(functions))
    resource = fm.T.workspace_requirement('pipeline.mmz', fm.tensor_type('uint8', (1024,)),
        'pipeline_mmz', 64, fm.T.WorkspaceLifetime.FUNCTION)
    plan = HeterogeneousPlan('split', 'main', 'test.exchange/v1', (
        ExecutionPlacement(source, call.call_id, call.kernel, 'kpu', 'native.test.v1', projections),
    ), (resource,), source_hash=fm.heterogeneous_source_hash(module))
    allocation = fm.PhysicalBuffer('pipeline.mmz', 'pipeline_mmz', 1024, 64)
    exchanges = []
    cpu, device = [], []
    from triton.flagmega.ir.tir.heterogeneous_verify import exchange_ranges
    for index in range(projections):
        name = f'exchange{index}'
        def buffer(field, offset):
            return fm.T.buffer(f'{name}.{field}', 'uint8', fm.MemSpan(allocation, offset, 64), (64,), (1,))
        exchange = PipelineExchange(name, source, call.call_id, index,
            (buffer('input', index * 128),), (buffer('output', index * 128 + 64),))
        exchanges.append(exchange)
        def phase(kind):
            return KernelPhaseInvoke(f'{name}.{kind}', call.call_id, kind, index, name, call)
        def event(kind):
            return ChannelBarrier(f'{name}.{kind}', name, kind, index + 1, (), exchange_ranges(exchange, kind))
        cpu.extend((phase('prepare'), event('publish'), event('observe'), phase('finish'), event('reclaim')))
        device.extend((event('acquire'), phase('execute'), event('complete')))
    workers = [
        WorkerFunction(source, 'cpu0', fm.ExecutionFunction(f'{source}.cpu0', (), (), fm.Sequential(tuple(cpu))), projections),
        WorkerFunction(source, 'kpu', fm.ExecutionFunction(f'{source}.kpu', (), (), fm.Sequential(tuple(device))), projections),
    ]
    if repeated:
        for domain in ('cpu0', 'kpu'):
            calls = tuple(WorkerCall(replace(value, callee=f'layer.{domain}'), index * projections)
                for index, value in enumerate(functions[-1].body.fields))
            workers.append(WorkerFunction('main', domain,
                fm.ExecutionFunction(f'main.{domain}', (), (), fm.Sequential(calls)), 2 * projections))
    program = HeterogeneousProgram('program', 'split', tuple(workers), tuple(exchanges),
        HeterogeneousLaunch('main.cpu0', 'main.kpu'), HeterogeneousDrain(('main.cpu0', 'main.kpu')))
    result = replace(module, heterogeneous_plans=(plan,), heterogeneous_programs=(program,))
    verify_heterogeneous_programs(result)
    return result


def edit_worker(module, index, edit):
    program = module.heterogeneous_programs[0]
    workers = list(program.functions)
    worker = workers[index]
    workers[index] = replace(worker, function=replace(worker.function,
        body=fm.Sequential(tuple(edit(worker.function.body.fields)))))
    return replace(module, heterogeneous_programs=(replace(program, functions=tuple(workers)),))


@pytest.mark.parametrize('repeated', (False, True))
def test_typed_program_roundtrip_and_existing_execution_contract(repeated, tmp_path):
    module = make_module(repeated=repeated)
    fm.verify_module(module)
    loaded = fm.IRModule.from_data(module.to_data())
    fm.verify_module(loaded)
    assert loaded.semantic_hash == module.semantic_hash
    assert loaded.heterogeneous_programs == module.heterogeneous_programs
    from triton.flagmega.ir.python_ir import emit_module, load_module
    path = tmp_path / 'module.py'
    emit_module(module, path)
    result = load_module(path)
    fm.verify_module(result)
    assert result.semantic_hash == module.semantic_hash
    assert 'T.kernel_phase_invoke(' in path.read_text()
    assert 'T.channel_barrier(' in path.read_text()


@pytest.mark.parametrize('edit,match', [
    (lambda fields: fields[:-1], 'each exchange'),
    (lambda fields: (*fields[:2], fields[3], fields[2], *fields[4:]), 'cycle'),
    (lambda fields: (replace(fields[0], projection=9), *fields[1:]), 'projection'),
    (lambda fields: (replace(fields[0], dependencies=('missing',)), *fields[1:]), 'unresolved'),
    (lambda fields: (*fields[:1], replace(fields[1], sequence=2), *fields[2:]), 'sequence'),
    (lambda fields: (*fields[:1], replace(fields[1], epoch_resource='stale'), *fields[2:]), 'epoch'),
    (lambda fields: (fields[0], replace(fields[1], ranges=(replace(fields[1].ranges[0], nbytes=32),)), *fields[2:]), 'ranges'),
])
def test_invalid_event_and_phase_edits_fail_closed(edit, match):
    with pytest.raises(IRVerificationError, match=match):
        verify_heterogeneous_programs(edit_worker(make_module(), 0, edit))


def test_repeated_callee_requires_distinct_sequence_ranges():
    module = make_module(repeated=True)
    for index in (2, 3):
        module = edit_worker(module, index, lambda fields: (fields[0], replace(fields[1], phase_offset=0)))
    with pytest.raises(IRVerificationError, match='reuse one invocation phase'):
        verify_heterogeneous_programs(module)


def test_final_drain_cannot_omit_or_replace_worker():
    module = make_module()
    program = module.heterogeneous_programs[0]
    edited = replace(program, drain=HeterogeneousDrain(('main.cpu0', 'absent')))
    with pytest.raises(IRVerificationError, match='final drain'):
        verify_heterogeneous_programs(replace(module, heterogeneous_programs=(edited,)))


def test_program_does_not_weaken_ordinary_execution_function_coverage():
    module = make_module()
    duplicate = module.heterogeneous_programs[0].functions[0].function
    with pytest.raises(IRVerificationError, match='exactly cover graph functions'):
        fm.verify_module(replace(module, execution_functions=(*module.execution_functions, duplicate)))


def test_unordered_exchange_alias_is_rejected():
    module = make_module(projections=2)
    program = module.heterogeneous_programs[0]
    first, second = program.exchanges
    second = replace(second, inputs=(replace(second.inputs[0], mem_span=first.inputs[0].mem_span),))
    from triton.flagmega.ir.tir.heterogeneous_verify import exchange_ranges
    def reorder(fields):
        fields = tuple(replace(value, ranges=exchange_ranges(second, value.event))
            if isinstance(value, ChannelBarrier) and value.exchange == second.name else value for value in fields)
        return (*fields[:2], *fields[5:7], *fields[2:5], *fields[7:])
    module = replace(module, heterogeneous_programs=(replace(program, exchanges=(first, second)),))
    module = edit_worker(module, 0, reorder)
    # Update device acquire/complete access ranges for the edited physical alias.
    module = edit_worker(module, 1, lambda fields: tuple(
        replace(value, ranges=exchange_ranges(second, value.event))
        if isinstance(value, ChannelBarrier) and value.exchange == second.name else value for value in fields))
    with pytest.raises(IRVerificationError, match='unordered alias hazard'):
        verify_heterogeneous_programs(module)


@pytest.mark.parametrize('change', [dict(slots=1), dict(slots=True), dict(cpu_domain='kpu'), dict(source_hash='x')])
def test_plan_rejects_invalid_protocol_shape(change):
    with pytest.raises(IRSchemaError):
        replace(make_module().heterogeneous_plans[0], **change)


def test_placement_and_resource_edits_change_hash_and_are_verified():
    module = make_module()
    plan = module.heterogeneous_plans[0]
    bad = replace(plan, placements=(replace(plan.placements[0], domain='unknown'),))
    changed = replace(module, heterogeneous_plans=(bad,))
    assert changed.semantic_hash != module.semantic_hash
    with pytest.raises(IRVerificationError, match='unknown placement'):
        verify_heterogeneous_programs(changed)
    bad = replace(plan, resources=())
    with pytest.raises(IRVerificationError, match='declared static'):
        verify_heterogeneous_programs(replace(module, heterogeneous_plans=(bad,)))


def test_visitors_and_rewriters_traverse_program():
    module = make_module()
    visited = []
    class Visitor(fm.TIRVisitor):
        def visit_channel_barrier(self, value):
            visited.append(value.event)
    Visitor().visit(module.heterogeneous_programs[0])
    assert set(visited) == {'publish', 'acquire', 'complete', 'observe', 'reclaim'}
    class Rewrite(fm.TIRRewriter):
        def visit_channel_barrier(self, value):
            return replace(value, epoch_resource='new')
    edited = Rewrite().rewrite(module.heterogeneous_programs[0])
    assert edited != module.heterogeneous_programs[0]


@pytest.mark.parametrize('program_present', (False, True))
def test_existing_codegen_cannot_silently_ignore_new_program(program_present):
    from triton.flagmega.codegen.cpp.k230 import describe_k230_package
    from triton.flagmega.codegen.cpp.renderer import describe_cpp_package
    from triton.flagmega.codegen.triton.tir_package import describe_tir_package
    from triton.flagmega.errors import CodegenError
    module = make_module()
    if not program_present:
        module = replace(module, heterogeneous_programs=())
    for describe in (describe_k230_package, describe_cpp_package, describe_tir_package):
        with pytest.raises(CodegenError, match='heterogeneous'):
            describe(module)


def test_two_slots_reject_three_unreclaimed_publications():
    module = make_module(projections=3)
    def reorder(fields):
        return tuple(value for index in (0, 1, 5, 6, 10, 11, 2, 3, 4, 7, 8, 9, 12, 13, 14) for value in (fields[index],))
    with pytest.raises(IRVerificationError, match='cycle'):
        verify_heterogeneous_programs(edit_worker(module, 0, reorder))


def test_checkpoint_rejects_unknown_heterogeneous_fields():
    data = make_module().heterogeneous_plans[0].to_data()
    data['pretend_runtime_ready'] = True
    with pytest.raises(ValueError, match='Unknown heterogeneous TIR fields'):
        fm.tir_from_data(data)


def test_old_default_hash_is_the_original_canonical_payload():
    import hashlib
    import json
    module = replace(make_module(), heterogeneous_plans=(), heterogeneous_programs=())
    old = module.semantic_data()
    assert 'heterogeneous_plans' not in old and 'heterogeneous_programs' not in old
    canonical = json.dumps(old, sort_keys=True, separators=(',', ':'), ensure_ascii=False)
    assert module.semantic_hash == hashlib.sha256(canonical.encode()).hexdigest()


def test_companion_dump_shows_both_worker_graphs(tmp_path):
    from triton.flagmega.ir.python_ir import emit_module
    emit_module(make_module(repeated=True), tmp_path / 'module.py')
    source = (tmp_path / 'module.script').read_text()
    for expected in ('worker layer.cpu0', 'worker layer.kpu', 'worker main.cpu0', 'worker main.kpu',
                     'T.WorkerCall(phase_offset=1)', 'T.ChannelBarrier', 'drain after='):
        assert expected in source


def test_odd_callee_phase_count_changes_absolute_slot_and_checks_capacity():
    from triton.flagmega.ir.tir.heterogeneous import exchange_ranges
    module = make_module(repeated=True)
    program = module.heterogeneous_programs[0]
    exchange = replace(program.exchanges[0], slot_stride_bytes=128)
    program = replace(program, exchanges=(exchange,))
    module = replace(module, heterogeneous_programs=(program,))
    verify_heterogeneous_programs(module)
    loaded = fm.IRModule.from_data(module.to_data())
    verify_heterogeneous_programs(loaded)
    assert loaded.semantic_hash == module.semantic_hash
    assert [call.phase_offset for call in program.functions[2].function.body.fields] == [0, 1]
    assert exchange_ranges(exchange, 'observe', sequence=1)[0].offset == 64
    assert exchange_ranges(exchange, 'observe', sequence=2)[0].offset == 192
    assert exchange_ranges(exchange, 'observe', sequence=3)[0].offset == 64
    plan = module.heterogeneous_plans[0]
    small = replace(plan.resources[0], type=fm.tensor_type('uint8', (192,)))
    with pytest.raises(IRVerificationError, match='declared resource'):
        verify_heterogeneous_programs(replace(module, heterogeneous_plans=(replace(plan, resources=(small,)),)))


def source_edit(module, functions):
    module = replace(module, execution_functions=functions)
    return replace(module, heterogeneous_plans=(replace(module.heterogeneous_plans[0],
        source_hash=fm.heterogeneous_source_hash(module)),))


def with_cpu_predecessor(*, barrier=False, dependency=False, opaque_effect=False, repeated=False):
    module = make_module(repeated=repeated)
    source = module.execution_functions[-1]
    projection = source.body.fields[0]
    prior = fm.KernelInvoke('prior', 'native', effect_kind='write' if opaque_effect else 'pure')
    if dependency:
        projection = replace(projection, dependencies=('prior',))
    if opaque_effect:
        projection = replace(projection, effect_kind='write')
    local_barrier = fm.Barrier(fm.BarrierScope.DEVICE, ('prior',), projection.call_id, hazards=('RAW',))
    source = replace(source, body=fm.Sequential((prior, *((local_barrier,) if barrier else ()),
                                                projection, *source.body.fields[1:])))
    module = source_edit(module, (*module.execution_functions[:-1], source))
    plan = module.heterogeneous_plans[0]
    place = ExecutionPlacement(source.name, 'prior', 'native', 'cpu0', 'native.test.v1')
    module = replace(module, heterogeneous_plans=(replace(plan, placements=(place, *plan.placements)),))
    cpu_index = 2 if repeated else 0
    for index in ((2, 3) if repeated else (0, 1)):
        def edit(fields):
            values = tuple(replace(value, invocation=replace(value.invocation, dependencies=projection.dependencies,
                                  effect_kind=projection.effect_kind))
                           if isinstance(value, (KernelPhaseInvoke, WorkerCall)) and value.invocation.call_id == projection.call_id
                           else value for value in fields)
            return (prior, *((local_barrier,) if barrier else ()), *values) if index == cpu_index else values
        module = edit_worker(module, index, edit)
    return module, cpu_index


def test_slot_stride_must_fit_actual_backing_not_only_declared_resource():
    module = make_module(repeated=True)
    program = module.heterogeneous_programs[0]
    exchange = program.exchanges[0]
    allocation = fm.PhysicalBuffer('pipeline.mmz', 'pipeline_mmz', 128, 64)
    def shrink(value):
        return replace(value, mem_span=fm.MemSpan(allocation, value.mem_span.start, value.mem_span.size))
    exchange = replace(exchange, inputs=tuple(map(shrink, exchange.inputs)),
                       outputs=tuple(map(shrink, exchange.outputs)), slot_stride_bytes=128)
    with pytest.raises(IRVerificationError, match='declared resource'):
        fm.verify_module(replace(module, heterogeneous_programs=(replace(program, exchanges=(exchange,)),)))


def test_source_barrier_preserved_and_its_order_verified():
    module, index = with_cpu_predecessor(barrier=True)
    fm.verify_module(module)
    with pytest.raises(IRVerificationError, match='source barriers'):
        fm.verify_module(edit_worker(module, index, lambda fields: (fields[0], *fields[2:])))
    with pytest.raises(IRVerificationError, match='source barrier'):
        fm.verify_module(edit_worker(module, index, lambda fields: (*fields[2:], *fields[:2])))
    with pytest.raises(IRVerificationError, match='source barriers'):
        fm.verify_module(edit_worker(module, index, lambda fields: (fields[0],
            replace(fields[1], scope=fm.BarrierScope.BLOCK), *fields[2:])))


@pytest.mark.parametrize('repeated', (False, True))
def test_source_cpu_dependency_is_discharged_before_device_publication(repeated):
    module, index = with_cpu_predecessor(dependency=True, repeated=repeated)
    fm.verify_module(module)
    fm.verify_module(fm.IRModule.from_data(module.to_data()))
    with pytest.raises(IRVerificationError, match='source dependency'):
        fm.verify_module(edit_worker(module, index, lambda fields: (*fields[1:], fields[0])))


def test_source_opaque_effect_order_cannot_be_reversed_without_a_dependency():
    module, index = with_cpu_predecessor(opaque_effect=True)
    fm.verify_module(module)
    with pytest.raises(IRVerificationError, match='memory/effect order'):
        fm.verify_module(edit_worker(module, index, lambda fields: (*fields[1:], fields[0])))


def test_source_distinct_allocations_in_reused_arena_still_conflict():
    from triton.flagmega.ir.bufferization import BufferDescriptor, BufferPlan, MemorySpace, AllocationStrategy
    from triton.flagmega.ir.tir.heterogeneous_verify import source_call_conflicts
    module = make_module()
    buffers = tuple(BufferDescriptor(name, fm.tensor_type('float32', (1,)).dtype, (1,), (1,),
        'workspace', 64, fm.MemSpan(fm.PhysicalBuffer(name, 'workspace', 64, 64), 0, 4), function='main')
        for name in ('a', 'b'))
    space = MemorySpace('workspace', 'host', 64, 1024, AllocationStrategy.REUSE)
    plan = BufferPlan(buffers, tuple(value.mem_span.buffer for value in buffers), (space,), (), 64, 0, 64, (), ())
    left = fm.KernelInvoke('left', 'native', results=(fm.PrimCallBinding('result', 'a'),), writes=('a',))
    right = fm.KernelInvoke('right', 'native', arguments=(fm.PrimCallBinding('input', 'b'),), reads=('b',))
    assert source_call_conflicts(module, 'main', left, right, buffer_plan=plan)
    moved = replace(buffers[1], mem_span=fm.MemSpan(replace(buffers[1].mem_span.buffer, start=fm.dim(64)), 0, 4))
    assert not source_call_conflicts(module, 'main', left, right, buffer_plan=replace(plan, buffers=(buffers[0], moved)))
