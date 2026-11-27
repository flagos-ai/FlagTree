# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Interprocedural program metadata stays complete across function dump views."""
from dataclasses import replace

import pytest

from triton.flagmega import ir as fm
from triton.flagmega.diagnostics import DumpFlags, DumpManager
from triton.flagmega.errors import CheckpointError, IRVerificationError
from triton.flagmega.ir.program_execution import verify_program_execution
from python.test.flagmega.ir.tir.test_prim_function import make_prim_module


def structured_module(kind='call'):
    primitive = make_prim_module().prim_functions[0]
    tensor = fm.tensor_type('float32', (4,))
    b = fm.IRBuilder(dialect='semantic_tir', stage='tir_selected')
    b.prim_function(primitive)
    worker_arg = b.var('worker_arg', tensor, id='worker_arg')
    first = b.call('tir.call', (worker_arg,), tensor, id='first', attrs={'callee': 'copy_4'})
    second = b.call('tir.call', (first,), tensor, id='second', attrs={'callee': 'copy_4'})
    value = b.var('value', tensor, id='value')
    if kind == 'loop':
        from triton.flagmega.ir.ops.core import construction_scope
        index_type = fm.tensor_type('int64', ())
        iv = b.var('iv', index_type, id='iv')
        body = b.region('worker', (iv, worker_arg), (second,), iter_count=1)
        with construction_scope(b):
            lower = fm.F.builtin.scalar_const(index_type, 0, name='lower')
            upper = fm.F.builtin.scalar_const(index_type, 3, name='upper')
            stride = fm.F.builtin.scalar_const(index_type, 1, name='stride')
            loop = fm.F.control.for_loop(lower, upper, stride, value, body=body, name='nested')
            out = fm.F.tensors.get_item(loop, 0, name='output')
        main_steps = (fm.LoopInvocation('nested', 'worker', 0, 3, 1),)
    else:
        b.function('worker', (worker_arg,), (second,))
        out = b.call('tir.call', (value,), tensor, id='nested', attrs={'callee': 'worker'})
        main_steps = (fm.CallInvocation('nested', 'worker'),)
    b.function('main', (value,), (out,))
    plan = fm.ProgramExecution('main', (
        fm.FunctionProgramExecution('main', main_steps),
        fm.FunctionProgramExecution('worker', (fm.ProgramInvocation('p0'), fm.ProgramInvocation('p1'))),
    ))
    source = b.build(entry='main')
    return fm.verify_module(replace(source, metadata={
        'kernel_program_execution': plan.to_data(),
        'kernel_program_partition': (
            {'symbol': 'p0', 'function': 'worker', 'calls': ('first',)},
            {'symbol': 'p1', 'function': 'worker', 'calls': ('second',)},
        ),
    }))


@pytest.mark.parametrize('kind', ['call', 'loop'])
def test_structured_program_function_dumps_load_and_merge(tmp_path, kind):
    module = structured_module(kind)
    dumped = DumpManager(tmp_path, DumpFlags.PASS_IR).root.dump_module(
        module, 'After', category=DumpFlags.PASS_IR)
    for name in ('worker', 'main'):
        fragment = fm.load_module(dumped.directory / f'{name}.py')
        assert fragment.entry == name
        assert fragment.metadata['kernel_program_execution'] == module.metadata['kernel_program_execution']
        assert fragment.metadata['kernel_program_partition'] == module.metadata['kernel_program_partition']
        assert verify_program_execution(fragment).entry == 'main'
    merged = fm.load_module(dumped.directory)
    assert merged == module
    assert merged.semantic_hash == module.semantic_hash
    assert '_dump_function_fragment' not in merged.metadata
    assert 'function_signatures' not in merged.metadata


@pytest.mark.parametrize('scope', ['whole', 'main', 'worker'])
@pytest.mark.parametrize('corrupt', ['program_order', 'call_target', 'scope', 'omission'])
def test_fragments_keep_plan_structure_and_local_body_checks(scope, corrupt):
    module = structured_module()
    data = module.to_data()['metadata']
    execution, programs = data['kernel_program_execution'], data['kernel_program_partition']
    # Change the relevant owned body when checking a fragment. Errors in the
    # external body are checked by that body's checkpoint and the merged IR.
    if corrupt == 'program_order':
        execution['functions'][1]['steps'].reverse()
        scope = 'worker' if scope != 'whole' else scope
    elif corrupt == 'call_target':
        execution['functions'][0]['steps'][0]['callee'] = 'missing'
    elif corrupt == 'scope':
        programs[0]['function'] = 'main'
    else:
        execution['functions'][1]['steps'].pop()
    changed = replace(module, metadata=data)
    if scope != 'whole':
        changed = fm.function_module(changed, scope)
    with pytest.raises(IRVerificationError, match='Structured'):
        fm.verify_module(changed)


@pytest.mark.parametrize('scope', ['whole', 'main'])
def test_loop_fragment_keeps_typed_bounds_check(scope):
    module = structured_module('loop')
    data = module.to_data()['metadata']
    data['kernel_program_execution']['functions'][0]['steps'][0]['stop'] = 9
    changed = replace(module, metadata=data)
    if scope == 'main':
        changed = fm.function_module(changed, 'main')
    with pytest.raises(IRVerificationError, match='loop bounds'):
        fm.verify_module(changed)


@pytest.mark.parametrize('corrupt', ['marker_absent', 'partial_signatures', 'multiple_functions'])
def test_fragment_exception_requires_explicit_complete_context(corrupt):
    module = structured_module()
    view = fm.function_module(module, 'worker')
    data = dict(view.metadata)
    if corrupt == 'marker_absent':
        data.pop('_dump_function_fragment')
    elif corrupt == 'partial_signatures':
        data['function_signatures'] = {'worker': data['function_signatures']['worker']}
    else:
        view = module
        data = {**module.metadata, **data}
    with pytest.raises(IRVerificationError, match='Structured'):
        fm.verify_module(replace(view, metadata=data))


def test_module_entry_mismatch_remains_invalid():
    module = structured_module()
    with pytest.raises(IRVerificationError, match='entry differs'):
        fm.verify_module(replace(module, entry='worker'))


def test_merge_rejects_inconsistent_plan_edits(tmp_path):
    module = structured_module()
    dumped = DumpManager(tmp_path, DumpFlags.PASS_IR).root.dump_module(
        module, 'After', category=DumpFlags.PASS_IR)
    # The worker view cannot inspect main's absent typed call; the main view
    # still checks it, and the directory rejects conflicting complete plans.
    worker_path = dumped.directory / 'worker.py'
    source = worker_path.read_text()
    assert "'call': 'nested'" in source
    worker_path.write_text(source.replace("'call': 'nested'", "'call': 'renamed'"))
    fm.load_module(worker_path)
    with pytest.raises(CheckpointError, match='conflicting module metadata'):
        fm.load_module(dumped.directory)
