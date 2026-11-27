# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Mixed-mesh loops retain one typed body and a structured invocation plan."""
from dataclasses import replace

import pytest

from triton.flagmega import ir as fm
from triton.flagmega.compiler import Compiler
from python.test.flagmega.passes.auto_distributed.test_loop_yield_reshard import mixed_loop


def _walk(statement):
    from triton.flagmega.ir.tir.visitor import iter_tir_children
    yield statement
    for child in iter_tir_children(statement):
        yield from _walk(child)


@pytest.mark.parametrize('kind', ['ordinary', 'persistent'])
@pytest.mark.parametrize('trips', [0, 1, 3])
def test_mixed_loop_has_a_scoped_program_plan_before_allocation(trips, kind, tmp_path):
    source = mixed_loop(trips, kind)
    module = Compiler().compile(source, stop_after='finalize-tir-package').module
    execution = module.metadata['kernel_program_execution']
    assert execution['schema'] == 'flagmega.structured-program-execution/v1'
    functions = {f['function']: f['steps'] for f in execution['functions']}
    assert functions['main'] == ({'kind': 'loop', 'call': 'loop', 'callee': 'body',
                                   'start': 0, 'stop': trips, 'step': 1},)
    programs = module.metadata['kernel_program_partition']
    assert all(p['function'] == 'body' for p in programs)
    assert len([p for p in programs if p['execution_mesh']['kind'] == 'dynamic']) == 1
    assert [s['symbol'] for s in functions['body']] == [p['symbol'] for p in programs]
    assert module.node_map['loop'].attrs == source.node_map['loop'].attrs
    loaded = fm.load_module(fm.emit_module(module, tmp_path/'scoped.py'))
    assert loaded.semantic_hash == module.semantic_hash
    fm.verify_module(loaded)


def test_two_loops_share_program_definitions_and_keep_their_own_invocations():
    module = Compiler().compile(mixed_loop(1, second_stop=2), stop_after='finalize-tir-package').module
    functions = {f['function']: f['steps'] for f in module.metadata['kernel_program_execution']['functions']}
    assert [s['call'] for s in functions['main']] == ['loop', 'other_loop']
    assert [s['stop'] for s in functions['main']] == [1, 2]
    assert all(s['callee'] == 'body' for s in functions['main'])
    assert len(functions) == 2
    assert len(module.metadata['kernel_program_partition']) == len(functions['body'])


@pytest.mark.parametrize('case', ['loop', 'calls'])
def test_module_mesh_is_an_inherited_starting_point_for_mixed_loop_edits(case):
    source = replace(mixed_loop(1) if case == 'loop' else mixed_calls_module(),
                     metadata={'execution_mesh': (8, 16)})
    module = Compiler().compile(source, stop_after='finalize-tir-package').module
    assert module.metadata['kernel_program_execution']['functions'][0]['steps'][0]['kind'] == (
        'loop' if case == 'loop' else 'call')


@pytest.mark.parametrize('trips', [0, 1, 3])
@pytest.mark.parametrize('kind', ['ordinary', 'persistent'])
def test_full_mixed_loop_retains_physical_carry_and_capture_bindings(trips, kind, tmp_path):
    module = Compiler().compile(mixed_loop(trips, kind)).module
    plan = fm.verify_buffer_plan(module)
    record = plan.call_map['loop']
    carry, = record.loop_carries
    assert dict(record.arguments)[carry.formal] == carry.current
    assert dict(record.arguments)['capture'] == 'increment'
    buffers = [plan.buffer_map[value] for value in carry.to_data().values()]
    assert all((b.dtype, b.shape, b.strides, b.distributed_type) ==
               (buffers[0].dtype, buffers[0].shape, buffers[0].strides, buffers[0].distributed_type)
               for b in buffers)
    current, staging = plan.buffer_map[carry.current], plan.buffer_map[carry.staging]
    assert not current.physical_access_span.may_alias(staging.physical_access_span)
    assert current.live_start == staging.live_start and current.live_end == staging.live_end
    loop, = (item for item in _walk(module.execution_function_map['main'].body)
              if isinstance(item, fm.For))
    assert loop.call_id == record.call
    from dataclasses import asdict
    assert [asdict(c) for c in loop.carry_bindings] == [c.to_data() for c in record.loop_carries]
    restored = fm.load_module(fm.emit_module(module, tmp_path/'physical.py'))
    assert restored.semantic_hash == module.semantic_hash
    fm.verify_buffer_plan(restored)


def test_shared_body_uses_distinct_caller_frames_and_forwards_the_first_result():
    module = Compiler().compile(mixed_loop(1, second_stop=2)).module
    plan = fm.verify_buffer_plan(module)
    first, second = (plan.call_map[value] for value in ('loop', 'other_loop'))
    a, = first.loop_carries
    b, = second.loop_carries
    assert a.result == b.initial
    assert a.formal == b.formal
    assert dict(first.arguments)['capture'] == dict(second.arguments)['capture'] == 'increment'
    frames = [{p.memory_space: p for p in record.memory_pools} for record in (first, second)]
    assert frames[0]['workspace'].offset != frames[1]['workspace'].offset
    assert not plan.buffer_map[a.current].physical_access_span.may_alias(
        plan.buffer_map[b.current].physical_access_span)


def mixed_calls_module():
    b = fm.IRBuilder(dialect='high_level', stage='imported')
    tensor = fm.tensor_type('float32', (4096,))
    x, y = b.var('x', tensor, id='x'), b.var('y', tensor, id='y')
    from triton.flagmega.ir.ops.core import construction_scope
    with construction_scope(b):
        a = fm.F.math.add(x, y, name='a', metadata={'execution_mesh': (8, 16)})
        c = fm.F.math.add(a, y, name='c', metadata={'execution_mesh': '*'})
        d = fm.F.math.add(c, y, name='d', metadata={'execution_mesh': (4, 16)})
    b.function('layer', (x, y), (d,), attrs={'reusable': True})
    value, inc = b.var('value', tensor, id='value'), b.var('inc', tensor, id='inc')
    first = b.call('builtin.call', (value, inc), tensor, id='first', attrs={'callee': 'layer'})
    second = b.call('builtin.call', (first, inc), tensor, id='second', attrs={'callee': 'layer'})
    b.function('main', (value, inc), (second,))
    return b.build(entry='main')


def test_plain_mixed_calls_share_definitions_and_preserve_call_bindings():
    module = Compiler().compile(mixed_calls_module()).module
    functions = {f['function']: f['steps'] for f in module.metadata['kernel_program_execution']['functions']}
    calls = [n.id for n in module.nodes if n.op == 'tir.call' and n.attrs['callee'] == 'layer']
    assert len(calls) == 2
    assert functions['main'] == tuple({'kind': 'call', 'call': value, 'callee': 'layer'} for value in calls)
    assert all(p['function'] == 'layer' for p in module.metadata['kernel_program_partition'])
    plan = fm.verify_buffer_plan(module)
    assert all(plan.call_map[value].callee == 'layer' for value in calls)


def test_structured_program_types_are_public_and_reject_unknown_fields():
    from triton.flagmega.errors import IRSchemaError
    plan = fm.ProgramExecution('main', (fm.FunctionProgramExecution('main', (
        fm.ProgramInvocation('one'), fm.LoopInvocation('loop', 'body', 0, 3, 1))),
        fm.FunctionProgramExecution('body', (fm.ProgramInvocation('body_one'),))))
    assert fm.ProgramExecution.from_data(plan.to_data()) == plan
    data = plan.to_data()
    data['functions'][0]['steps'][1]['invented'] = True
    with pytest.raises(IRSchemaError):
        fm.ProgramExecution.from_data(data)


def test_dynamic_module_default_remains_dynamic_and_never_fuses():
    b = fm.IRBuilder(dialect='high_level', stage='imported')
    tensor = fm.tensor_type('float32', (4096,))
    x, y = b.var('x', tensor, id='x'), b.var('y', tensor, id='y')
    from triton.flagmega.ir.ops.core import construction_scope
    with construction_scope(b):
        first = fm.F.math.add(x, y, name='first')
        second = fm.F.math.add(first, y, name='second')
    b.function('main', (x, y), (second,))
    source = replace(b.build(entry='main'), metadata={'execution_mesh': '*'})
    module = Compiler().compile(source, stop_after='finalize-tir-package').module
    programs = module.metadata['kernel_program_partition']
    assert len(programs) == 2
    assert all(p['execution_mesh']['kind'] == 'dynamic' and len(p['calls']) == 1
               and 'launch_recipe' in p for p in programs)


def test_scoped_dynamic_program_keeps_its_own_warp_and_register_request():
    source = mixed_loop(1)
    source = replace(source, nodes=tuple(replace(n, metadata={**n.metadata,
        'kernel_num_warps': 2, 'kernel_maxnreg': 64}) if n.id == 'dynamic' else n for n in source.nodes))
    module = Compiler().compile(source, stop_after='finalize-tir-package').module
    dynamic, = (p for p in module.metadata['kernel_program_partition'] if p['execution_mesh']['kind'] == 'dynamic')
    assert dynamic['num_warps'] == 2 and dynamic['maxnreg'] == 64
    assert dynamic['function'] == 'body' and len(dynamic['calls']) == 1


@pytest.mark.parametrize('corrupt', ['bounds', 'order', 'scope', 'omission'])
def test_scoped_execution_plan_rejects_an_edit_that_disagrees_with_typed_ir(corrupt):
    from triton.flagmega.errors import IRVerificationError
    module = Compiler().compile(mixed_loop(1, second_stop=2), stop_after='finalize-tir-package').module
    from triton.flagmega.codegen.triton.package_plan import plain_package_value
    execution = plain_package_value(module.metadata['kernel_program_execution'])
    programs = plain_package_value(module.metadata['kernel_program_partition'])
    if corrupt == 'bounds':
        execution['functions'][0]['steps'][0]['stop'] = 99
    elif corrupt == 'order':
        execution['functions'][0]['steps'].reverse()
    elif corrupt == 'scope':
        programs[0]['function'] = 'main'
    else:
        execution['functions'][1]['steps'].pop()
    changed = replace(module, metadata={**module.metadata, 'kernel_program_execution': execution,
                                       'kernel_program_partition': programs})
    with pytest.raises(IRVerificationError, match='[Ss]tructured|[Ss]coped'):
        fm.verify_module(changed)
