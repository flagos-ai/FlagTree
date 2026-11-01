# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Structured typed calls lower to shared device code and physical invocations."""
import pytest

from triton.flagmega.compiler import Compiler
from triton.flagmega.codegen.triton.tir_package import describe_tir_package
from python.test.flagmega.passes.auto_distributed.test_loop_yield_reshard import mixed_loop
from python.test.flagmega.compiler.test_structured_execution_programs import mixed_calls_module


def _record_device_receipt(tmp_path, runtime, module, case):
    import json
    import os
    from pathlib import Path
    from triton.flagmega.passes.functions.graph import function_nodes
    def counts(nodes):
        result = dict(Pack=0, Slice=0, Cast=0, Concat=0, Unpack=0, Pad=0)
        for node in nodes:
            key = {'pack': 'Pack', 'slice': 'Slice', 'slice_to_shape': 'Slice', 'cast': 'Cast',
                   'vectorized_cast': 'Cast', 'concat': 'Concat', 'unpack': 'Unpack', 'pad': 'Pad'}.get(node.op.rsplit('.', 1)[-1])
            if key:
                result[key] += 1
        return result
    record = dict(case=case, passed=True, normal_artifact=str(tmp_path/'artifact'),
                  semantic_hash=module.semantic_hash, generated_source_sha256=runtime.codegen['source_sha256'],
                  program_execution=runtime.codegen['program_execution'], invocations=runtime.codegen['kernels'],
                  resources=runtime.resource_report, eager_and_graph=True, output_guards=True,
                  ir_counts=dict(whole_module=counts(module.nodes),
                      functions={f.name: counts(function_nodes(module, f)) for f in module.functions}),
                  full_model_acceptance=False, performance_measurement=False)
    directory = Path(os.environ.get('FLAGMEGA_STRUCTURED_RECEIPT_DIR', str(tmp_path)))
    directory.mkdir(parents=True, exist_ok=True)
    (directory/(case+'.json')).write_text(json.dumps(record, ensure_ascii=False, indent=2)+'\n')


@pytest.mark.parametrize('kind', ['ordinary', 'persistent'])
@pytest.mark.parametrize('trips', [0, 1, 3])
def test_mixed_loop_has_executable_shared_entries_and_declared_carry_transfers(kind, trips):
    module = Compiler().compile(mixed_loop(trips, kind)).module
    descriptor = describe_tir_package(module)
    assert descriptor['kernel_program_schema'] == 'flagmega.kernel-programs/v6-structured'
    invocations = descriptor['kernels']
    compute = [v for v in invocations if v['invocation_kind'] == 'program']
    copies = [v for v in invocations if v['invocation_kind'] == 'loop_copy']
    assert len(compute) == 3*trips
    assert len(copies) == 2+2*trips
    assert len({v['entry_symbol'] for v in compute}) == (3 if trips else 0)
    assert all('argument_bindings' in v for v in invocations)
    assert copies[0]['transfer'] == ['initial', 'current']
    assert copies[-1]['transfer'] == ['current', 'result']
    assert [v['transfer'] for v in copies[1:-1]] == [['yielded', 'staging'], ['staging', 'current']]*trips


def test_mixed_shared_calls_reuse_code_with_distinct_actual_bindings():
    module = Compiler().compile(mixed_calls_module()).module
    descriptor = describe_tir_package(module)
    invocations = descriptor['kernels']
    assert len(invocations) == 6
    assert [v['entry_symbol'] for v in invocations[:3]] == [v['entry_symbol'] for v in invocations[3:]]
    assert invocations[0]['argument_bindings'] != invocations[3]['argument_bindings']


def test_two_mixed_loops_keep_distinct_frames_and_one_body_code():
    module = Compiler().compile(mixed_loop(1, second_stop=2)).module
    descriptor = describe_tir_package(module)
    compute = [v for v in descriptor['kernels'] if v['invocation_kind'] == 'program']
    assert len(compute) == 9
    assert len({v['entry_symbol'] for v in compute}) == 3
    assert compute[0]['argument_bindings'] != compute[3]['argument_bindings']


def embedded_calls_module():
    from dataclasses import replace
    from triton.flagmega import ir as fm
    module = mixed_calls_module()
    first = module.node_map['first']
    outside = fm.Node('outside', 'math.add', ('first', 'inc'), first.type,
                      metadata={'execution_mesh': '*'})
    nodes = []
    for node in module.nodes:
        if node.id == 'second':
            nodes.append(outside)
            node = replace(node, inputs=('outside', 'inc'))
        nodes.append(node)
    return replace(module, nodes=tuple(nodes))


def test_scoped_root_program_between_shared_calls_preserves_exact_execution_order():
    module = Compiler().compile(embedded_calls_module()).module
    descriptor = describe_tir_package(module)
    invocations = descriptor['kernels']
    assert len(invocations) == 7
    assert invocations[3]['call_path'] == []
    assert invocations[3]['calls'] == ['outside.vectorized.compute']
    assert [v['entry_symbol'] for v in invocations[:3]] == [v['entry_symbol'] for v in invocations[4:]]


@pytest.mark.parametrize('corrupt', ['pointer', 'transfer', 'path', 'bounds'])
def test_structured_runtime_rejects_joint_manifest_and_source_forgery(corrupt):
    from copy import deepcopy
    from types import SimpleNamespace
    from triton.flagmega.errors import ArtifactError
    from triton.flagmega.runtime.kernel_programs import verified_kernel_programs
    module = Compiler().compile(mixed_loop(1)).module
    descriptor = describe_tir_package(module)
    forged = deepcopy({k: descriptor[k] for k in ('kernel_program_schema', 'kernels', 'program_execution')})
    if corrupt == 'pointer':
        forged['kernels'][0]['argument_bindings'][1]['byte_offset'] += 256
    elif corrupt == 'transfer':
        forged['kernels'][0]['transfer'] = ['staging', 'current']
    elif corrupt == 'path':
        forged['kernels'][0]['call_path'] = ['invented']
    else:
        forged['program_execution']['functions'][0]['steps'][0]['stop'] = 3
    namespace = {'FLAGMEGA_KERNEL_PROGRAMS': forged['kernels'], 'FLAGMEGA_PROGRAM_EXECUTION': forged['program_execution']}
    entries = {v['entry_symbol']: SimpleNamespace(fn=SimpleNamespace(__name__=v['entry_symbol'], __globals__=namespace))
               for v in forged['kernels']}
    kernels = tuple(entries[v['entry_symbol']] for v in forged['kernels'])
    with pytest.raises(ArtifactError, match='typed TIR'):
        verified_kernel_programs(forged, kernels, module)


@pytest.mark.parametrize('kind,stop,shared', [('ordinary', 0, False), ('ordinary', 1, False),
                                           ('ordinary', 3, False), ('persistent', 3, False),
                                           ('persistent', 0, False), ('persistent', 1, False),
                                           ('ordinary', 1, True), ('persistent', 1, True),
                                           ('calls', 2, False), ('embedded', 2, False)])
def test_device_structured_calls_carry_mutation_rebinding_and_graph(tmp_path, kind, stop, shared):
    torch = pytest.importorskip('torch')
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (9, 0):
        pytest.skip('requires SM90')
    from triton.flagmega.artifacts import write_artifact
    from triton.flagmega.runtime import load
    from triton.flagmega.runtime.module import GeneratedTirCallGraphModule as Runtime
    source = embedded_calls_module() if kind == 'embedded' else mixed_calls_module() if kind == 'calls' else mixed_loop(stop, kind, second_stop=2 if shared else None)
    module = Compiler().compile(source).module
    artifact = write_artifact(module, tmp_path/'artifact', target='nvidia-sm90', emit_executable=True)
    runtime = load(artifact, device='cuda:0')
    generator = torch.Generator(device='cuda').manual_seed(20261027)
    value, inc = [torch.empty((4096,), device='cuda') for _ in range(2)]
    guard = torch.full((4128,), -123.0, device='cuda')
    output = guard[16:-16]
    trips = stop+2 if shared else stop

    def arguments():
        return tuple(output if s['role'] == 'result' else value if s['value'] == 'value' else inc
                     for s in runtime.external_arguments)

    def mutate():
        for tensor in (value, inc):
            tensor.copy_(torch.randint(-32, 33, tensor.shape, generator=generator, device='cuda').float()/16)

    def check():
        torch.cuda.synchronize()
        assert torch.equal(output, value+(7 if kind == 'embedded' else 3*trips)*inc)
        assert bool((guard[:16] == -123).all() & (guard[-16:] == -123).all())

    try:
        mutate()
        Runtime.prepare(runtime, *arguments())
        for mutation in range(4):
            mutate()
            output.fill_(float('nan'))
            Runtime.run_into(runtime, *arguments())
            check()
            if mutation == 1:
                value, inc = value.clone(), inc.clone()
                guard = torch.full_like(guard, -123.0)
                output = guard[16:-16]
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            Runtime.prepare(runtime, *arguments())
            Runtime.run_into(runtime, *arguments())
        with torch.cuda.graph(graph := torch.cuda.CUDAGraph(), stream=stream):
            Runtime.run_into(runtime, *arguments())
            Runtime.run_into(runtime, *arguments())
        for _ in range(3):
            mutate()
            graph.replay()
            check()
        compute = [v for v in runtime.codegen['kernels'] if v['invocation_kind'] == 'program']
        assert len({v['entry_symbol'] for v in compute}) == (4 if kind == 'embedded' else 3 if trips else 0)
        assert all(r['spill_bytes'] == 0 for r in runtime.resource_report)
        compiled = {}
        for spec, prepared in zip(runtime.codegen['kernels'], runtime._prepared, strict=True):
            compiled.setdefault(spec['entry_symbol'], set()).add(prepared.compiled_kernel.hash)
        assert all(len(identities) == 1 for identities in compiled.values())
        _record_device_receipt(tmp_path, runtime, module, f'{kind}-{stop}-shared{int(shared)}')
    finally:
        runtime.close()


def coupled_carry_module():
    from triton.flagmega import ir as fm
    from triton.flagmega.ir.ops.core import construction_scope
    b = fm.IRBuilder(dialect='high_level', stage='imported')
    tensor, index = fm.tensor_type('float32', (4096,)), fm.tensor_type('int64', ())
    iv, a, c, inc = (b.var('iv', index, id='iv'), b.var('a', tensor, id='a'),
                     b.var('b', tensor, id='b'), b.var('capture', tensor, id='capture'))
    with construction_scope(b):
        first = fm.F.math.add(c, inc, name='dynamic', metadata={'execution_mesh': '*'})
        second = fm.F.math.add(first, inc, name='fixed', metadata={'execution_mesh': (4, 16)})
    body = b.region('body', (iv, a, c, inc), (second, a), iter_count=2)
    x, y, increment = (b.var(v, tensor, id=v) for v in ('value', 'other', 'increment'))
    with construction_scope(b):
        lower = fm.F.builtin.scalar_const(index, 0, name='lower')
        upper = fm.F.builtin.scalar_const(index, 3, name='upper')
        stride = fm.F.builtin.scalar_const(index, 1, name='stride')
        result = fm.F.control.for_loop(lower, upper, stride, x, y, body=body,
                                      captures=(increment,), name='loop')
        left = fm.F.tensors.get_item(result, 0, name='left')
        right = fm.F.tensors.get_item(result, 1, name='right')
    b.function('main', (x, y, increment), (left, right))
    return b.build(entry='main')


def induction_module():
    from triton.flagmega import ir as fm
    from triton.flagmega.ir.ops.core import construction_scope
    b = fm.IRBuilder(dialect='high_level', stage='imported')
    tensor, index = fm.tensor_type('int64', (4096,)), fm.tensor_type('int64', ())
    iv, carried, capture = (b.var('iv', index, id='iv'), b.var('carried', tensor, id='carried'),
                            b.var('capture', tensor, id='capture'))
    with construction_scope(b):
        indices = fm.F.tensors.broadcast_to(iv, shape=(4096,), name='indices')
        first = fm.F.math.add(carried, indices, name='index_add', metadata={'execution_mesh': (8, 16)})
        second = fm.F.math.add(first, capture, name='dynamic', metadata={'execution_mesh': '*',
                              'kernel_num_warps': 2, 'kernel_maxnreg': 64})
        third = fm.F.math.add(second, capture, name='fixed', metadata={'execution_mesh': (4, 16)})
    body = b.region('body', (iv, carried, capture), (third,), iter_count=1)
    value, increment = (b.var(v, tensor, id=v) for v in ('value', 'increment'))
    with construction_scope(b):
        lower = fm.F.builtin.scalar_const(index, 1, name='lower')
        upper = fm.F.builtin.scalar_const(index, 6, name='upper')
        stride = fm.F.builtin.scalar_const(index, 2, name='stride')
        result = fm.F.control.for_loop(lower, upper, stride, value, body=body,
                                      captures=(increment,), name='loop')
        output = fm.F.tensors.get_item(result, 0, name='output')
    b.function('main', (value, increment), (output,))
    return b.build(entry='main')


@pytest.mark.parametrize('case', ['coupled', 'induction'])
def test_structured_physical_plan_preserves_simultaneous_carries_and_real_induction(case):
    module = Compiler().compile(coupled_carry_module() if case == 'coupled' else induction_module()).module
    descriptor = describe_tir_package(module)
    if case == 'coupled':
        copies = [v['transfer'] for v in descriptor['kernels'] if v['invocation_kind'] == 'loop_copy']
        assert copies == [['initial', 'current']]*2 + (
            [['yielded', 'staging']]*2 + [['staging', 'current']]*2)*3 + [['current', 'result']]*2
    else:
        compute = [v for v in descriptor['kernels'] if v['invocation_kind'] == 'program']
        assert [v['argument_bindings'][0]['value'] for v in compute[::3]] == [1, 3, 5]


@pytest.mark.parametrize('case', ['coupled', 'induction'])
def test_device_structured_simultaneous_carries_and_induction(tmp_path, case):
    torch = pytest.importorskip('torch')
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (9, 0):
        pytest.skip('requires SM90')
    from triton.flagmega.artifacts import write_artifact
    from triton.flagmega.runtime import load
    from triton.flagmega.runtime.module import GeneratedTirCallGraphModule as Runtime
    module = Compiler().compile(coupled_carry_module() if case == 'coupled' else induction_module()).module
    dtype = torch.float32 if case == 'coupled' else torch.int64
    feeds = {v: torch.arange(4096, device='cuda', dtype=dtype)+(2**35 if case == 'induction' else 0)
             for v in module.function_map['main'].parameters}
    feeds['increment'].fill_(1)
    if case == 'coupled':
        feeds['other'].neg_()
    guards = [torch.full((4128,), -123, device='cuda', dtype=dtype)
              for _ in module.function_map['main'].outputs]
    outputs = [v[16:-16] for v in guards]
    runtime = load(write_artifact(module, tmp_path/'artifact', target='nvidia-sm90', emit_executable=True), device='cuda:0')
    results = iter(outputs)
    arguments = tuple(next(results) if s['role'] == 'result' else feeds[s['value']] for s in runtime.external_arguments)

    def check():
        torch.cuda.synchronize()
        gold = [feeds['other']+4, feeds['value']+2] if case == 'coupled' else [feeds['value']+15]
        assert all(torch.equal(a, b) for a, b in zip(outputs, gold, strict=True))
        assert all(bool((v[:16] == -123).all() & (v[-16:] == -123).all()) for v in guards)

    try:
        Runtime.prepare(runtime, *arguments)
        Runtime.run_into(runtime, *arguments)
        check()
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            Runtime.prepare(runtime, *arguments)
            Runtime.run_into(runtime, *arguments)
        with torch.cuda.graph(graph := torch.cuda.CUDAGraph(), stream=stream):
            Runtime.run_into(runtime, *arguments)
        feeds['value'].add_(3)
        graph.replay()
        check()
        assert all(v['spill_bytes'] == 0 for v in runtime.resource_report)
        if case == 'induction':
            for spec, prepared in zip(runtime.codegen['kernels'], runtime._prepared, strict=True):
                if spec.get('maxnreg') == 64:
                    assert prepared.resource_report['num_warps'] == 2
                    assert prepared.resource_report['registers_per_thread'] <= 64
        _record_device_receipt(tmp_path, runtime, module, case)
    finally:
        runtime.close()
