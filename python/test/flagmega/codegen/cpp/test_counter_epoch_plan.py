from copy import deepcopy
from dataclasses import replace
import json

import pytest

from triton.flagmega.errors import IRSchemaError, IRVerificationError, CodegenError
from triton.flagmega.ir.counter_epoch_plan import (
    CounterEpochPlan, CounterEpochRow, CounterReset, CounterBarrier,
    EntryCounterEffect, CounterEffectContract, identity,
    register_counter_effect_contract, make_counter_epoch_plan, verify_counter_epoch_plan,
)

DEVICE_SHA = 'd' * 64


def fixture(name='test-counter-epoch/v1'):
    entries = [dict(entry_id=i, definition_sha256=str(i) * 64,
                    scratch_nbytes=n, signature=[{'tag': 1}], grid=[g, 1, 1],
                    selected_implementation='fixture', launch_kind='cooperative' if n else 'ordinary')
               for i, n, g in ((0, 100, 96), (1, 52, 48), (2, 100, 96), (3, 0, 1))]
    invocations = [dict(ordinal=i, entry_id=i, call_path=['call'+str(i)],
                        argument_bindings=[{'kind': 'argument', 'source': i}],
                        invocation_kind='program', frame_identity=identity([i])) for i in range(4)]
    programs = dict(schema='flagmega.cpp-programs/v1', semantic_hash='a'*64,
                    program_execution={'entry': 'main'}, entries=entries, invocations=invocations)
    programs['plan_sha256'] = identity(programs)
    effects = tuple(EntryCounterEffect(i, identity(e),
                ((CounterBarrier(0, 96 if i != 1 else 48),),) if i < 2 else
                ((), (CounterBarrier(0, 96),)) if i == 2 else ((),))
                for i, e in enumerate(entries))
    contract = CounterEffectContract(name, programs['plan_sha256'], DEVICE_SHA, 'f'*64, 100,
                  effects, (('baseline', (CounterReset(0, 100), CounterReset(1, 52), CounterReset(2, 100))),
                            ('candidate', (CounterReset(0, 100), CounterReset(2, 100)))))
    register_counter_effect_contract(contract)
    return programs, contract


def rehash(data):
    data['plan_sha256'] = identity({k: v for k, v in data.items() if k != 'plan_sha256'})
    return data


@pytest.mark.parametrize('schedule,resets', [('baseline', 3), ('candidate', 2)])
def test_typed_roundtrip_and_all_branch_states(schedule, resets):
    programs, contract = fixture()
    plan = make_counter_epoch_plan(contract.name, schedule, programs, DEVICE_SHA)
    assert CounterEpochPlan.from_data(json.loads(json.dumps(plan.to_data()))) == plan
    assert CounterEffectContract.from_data(json.loads(json.dumps(contract.to_data()))) == contract
    proof = verify_counter_epoch_plan(plan, programs, DEVICE_SHA)
    assert proof['calls'] == 4 and proof['resets'] == resets
    assert proof['terminal_counter_values'][0] == [0, 96]
    assert proof['max_counter_values'][0] == (96 if schedule == 'baseline' else 144)


@pytest.mark.parametrize('key,value', [
    ('schema', 'v99'), ('mode', 'diagnostic'), ('mode', 'multistream'), ('counter_bits', True),
    ('counter_bits', 64), ('counter_count', 1), ('counter_count', True), ('frame_nbytes', 101),
    ('frame_nbytes', 0), ('safe', True), ('effects', []), ('unknown', 'field'),
])
def test_closed_schema_rejects_even_with_rehashed_input(key, value):
    programs, contract = fixture()
    data = make_counter_epoch_plan(contract.name, 'candidate', programs, DEVICE_SHA).to_data()
    data[key] = value
    with pytest.raises(IRSchemaError):
        CounterEpochPlan.from_data(rehash(data))


@pytest.mark.parametrize('mutation', ['missing', 'extra', 'reordered', 'duplicate', 'changed_binding', 'entry_id'])
def test_full_rows_reject_sliced_or_changed_invocations(mutation):
    programs, contract = fixture()
    data = make_counter_epoch_plan(contract.name, 'candidate', programs, DEVICE_SHA).to_data()
    if mutation == 'missing': data['rows'].pop()
    elif mutation == 'extra': data['rows'].append({**data['rows'][-1], 'ordinal': 4})
    elif mutation == 'reordered': data['rows'][1:3] = data['rows'][1:3][::-1]
    elif mutation == 'duplicate': data['rows'][1] = data['rows'][0]
    elif mutation == 'changed_binding': data['rows'][1]['invocation_sha256'] = 'e'*64
    else: data['rows'][1]['entry_id'] = 2
    with pytest.raises((IRSchemaError, IRVerificationError)):
        verify_counter_epoch_plan(rehash(data), programs, DEVICE_SHA)


@pytest.mark.parametrize('mutation', ['initial', 'transition', 'duplicate', 'extra', 'span', 'reordered'])
def test_reset_whitelist_cannot_be_replaced_by_a_self_hash(mutation):
    programs, contract = fixture()
    data = make_counter_epoch_plan(contract.name, 'candidate', programs, DEVICE_SHA).to_data()
    if mutation == 'initial': data['resets'].pop(0)
    elif mutation == 'transition': data['resets'].pop()
    elif mutation == 'duplicate': data['resets'].append(data['resets'][-1])
    elif mutation == 'extra': data['resets'].insert(1, {'ordinal': 1, 'nbytes': 52})
    elif mutation == 'span': data['resets'][1]['nbytes'] = 52
    else: data['resets'].reverse()
    with pytest.raises((IRSchemaError, IRVerificationError)):
        verify_counter_epoch_plan(rehash(data), programs, DEVICE_SHA)


@pytest.mark.parametrize('component', ['execution', 'definition', 'geometry', 'scratch', 'argument', 'call_path', 'conditional'])
def test_changed_physical_program_invalidates_reviewed_effects(component):
    programs, contract = fixture()
    plan = make_counter_epoch_plan(contract.name, 'candidate', programs, DEVICE_SHA)
    changed = deepcopy(programs)
    if component == 'execution': changed['program_execution']['entry'] = 'other'
    elif component == 'definition': changed['entries'][1]['definition_sha256'] = '9'*64
    elif component == 'geometry': changed['entries'][1]['grid'] = [96, 1, 1]
    elif component == 'scratch': changed['entries'][1]['scratch_nbytes'] = 100
    elif component == 'argument': changed['invocations'][1]['argument_bindings'][0]['source'] = 2
    elif component == 'call_path': changed['invocations'][1]['call_path'] = ['other']
    else: changed['entries'][2]['conditional'] = 'constant_true'
    changed['plan_sha256'] = identity({k: v for k, v in changed.items() if k != 'plan_sha256'})
    with pytest.raises(IRVerificationError):
        verify_counter_epoch_plan(plan, changed, DEVICE_SHA)


def test_unknown_contract_and_rebinding_cannot_bless_changed_effects():
    programs, contract = fixture()
    plan = make_counter_epoch_plan(contract.name, 'candidate', programs, DEVICE_SHA)
    with pytest.raises(IRVerificationError, match='not registered'):
        verify_counter_epoch_plan(replace(plan, contract='unknown/v1'), programs, DEVICE_SHA)
    changed = replace(contract, effects=tuple(replace(e, alternatives=((),)) for e in contract.effects))
    with pytest.raises(IRVerificationError, match='rebound'):
        register_counter_effect_contract(changed)
    with pytest.raises(IRVerificationError, match='implementation'):
        verify_counter_epoch_plan(plan, programs, 'a'*64)
    with pytest.raises(IRVerificationError, match='implementation'):
        verify_counter_epoch_plan(replace(plan, contract_sha256='a'*64), programs, DEVICE_SHA)


def test_finite_proof_rejects_unsafe_reviewed_schedule_without_gpu():
    programs, contract = fixture('test-unsafe-counter-epoch/v1')
    unsafe = replace(contract, name='test-unsafe-counter-epoch-registered/v1',
                     schedules=(('unsafe', (CounterReset(0, 100),)),))
    register_counter_effect_contract(unsafe)
    # 96 -> +48 =144, then a96-participant barrier would release a partial
    # generation. Even a registered schedule must pass the arithmetic proof.
    with pytest.raises(IRVerificationError, match='partial generation'):
        make_counter_epoch_plan(unsafe.name, 'unsafe', programs, DEVICE_SHA)


def test_counter_wrap_and_out_of_span_effects_are_rejected():
    programs, contract = fixture('test-counter-wrap/v1')
    # A single generation fits uint32, but two cannot be represented without
    # wrapping. Use a compact fixture rather than billions of fake rows.
    effect = replace(contract.effects[0], alternatives=((CounterBarrier(0, 2**31), CounterBarrier(0, 2**31)),))
    overflow = replace(contract, name='test-counter-wrap-registered/v1', effects=(effect, *contract.effects[1:]))
    register_counter_effect_contract(overflow)
    with pytest.raises(IRVerificationError, match='wrap'):
        make_counter_epoch_plan(overflow.name, 'baseline', programs, DEVICE_SHA)
    bad = replace(contract, name='test-counter-span-registered/v1',
                  effects=(contract.effects[0], replace(contract.effects[1], alternatives=((CounterBarrier(13, 4),),)), *contract.effects[2:]))
    register_counter_effect_contract(bad)
    with pytest.raises(IRVerificationError, match='exact per-entry'):
        make_counter_epoch_plan(bad.name, 'baseline', programs, DEVICE_SHA)


def test_explicit_invalid_or_diagnostic_descriptor_does_not_fall_back():
    from triton.flagmega.codegen.cpp.programs import configure_counter_epoch_descriptor
    with pytest.raises(CodegenError):
        configure_counter_epoch_descriptor({'platform': 'nvidia'}, None)
    with pytest.raises(CodegenError, match='normal NVIDIA'):
        configure_counter_epoch_descriptor({'platform': 'nvidia', 'program_profile': {'mode': 'trace'}}, {})


def test_runtime_projection_preserves_companion():
    from triton.flagmega.codegen.cpp.package import cpp_runtime_metadata
    programs, contract = fixture()
    plan = make_counter_epoch_plan(contract.name, 'candidate', programs, DEVICE_SHA).to_data()
    projected = cpp_runtime_metadata({'cpp_programs': programs, 'cpp_counter_epoch_plan': plan,
                                      'programs': ['render-only'], 'native_program_contexts': ['render-only']})
    assert json.loads(json.dumps(projected)) == {'cpp_programs': programs, 'cpp_counter_epoch_plan': plan}


def test_real_codegen_companion_roundtrip_and_body_binding(tmp_path, monkeypatch):
    from triton.flagmega import ir as fm
    from triton.flagmega.compiler import Compiler
    from triton.flagmega.codegen.cpp import renderer
    from triton.flagmega.codegen.cpp.programs import device_source_identity
    from python.test.flagmega.compiler.test_structured_execution_programs import mixed_calls_module
    module = Compiler().compile(mixed_calls_module()).module
    normal = renderer.describe_cpp_package(module)
    programs = normal['cpp_programs']
    frame = max(entry['scratch_nbytes'] for entry in programs['entries'])
    resets = tuple(CounterReset(row['ordinal'], programs['entries'][row['entry_id']]['scratch_nbytes'])
                   for row in programs['invocations'] if programs['entries'][row['entry_id']]['scratch_nbytes'])
    # A baseline clear before each cooperative entry is used only to test the
    # serializer/codegen boundary. It is not a proof for this module's effects.
    contract = CounterEffectContract('test-codegen-companion/v1', programs['plan_sha256'],
        device_source_identity(normal), 'b'*64, frame,
        tuple(EntryCounterEffect(entry['entry_id'], identity(entry), ((),)) for entry in programs['entries']),
        (('baseline', resets),))
    register_counter_effect_contract(contract)
    plan = make_counter_epoch_plan(contract.name, 'baseline', programs, contract.device_source_sha256)
    planned = replace(module, metadata={**module.metadata, 'counter_epoch_plan': plan.to_data()})
    fm.emit_module(planned, tmp_path / 'planned.py')
    loaded = fm.load_module(tmp_path / 'planned.py')
    assert loaded.semantic_hash == planned.semantic_hash != module.semantic_hash
    package = renderer.describe_cpp_package(loaded)
    assert package['cpp_programs'] == programs
    assert package['cpp_counter_epoch_plan'] == plan.to_data()
    captured = {}
    original = renderer.CppTemplateRegistry

    class ObserveHost(original):
        def render(self, name, context):
            if name == 'entrypoints/program_launch.cpp.jinja':
                captured['plan'] = json.loads(json.loads(context['native_counter_epoch_manifest_literal']))
            return super().render(name, context)

    monkeypatch.setattr(renderer, 'CppTemplateRegistry', ObserveHost)
    renderer.render_cpp_package(package, 'test')
    assert captured['plan'] == plan.to_data()

    class ChangedBody(original):
        def render(self, name, context):
            text = super().render(name, context)
            return text + '\n// changed device implementation' if name == 'entrypoints/call_graph.cpp.jinja' else text

    monkeypatch.setattr(renderer, 'CppTemplateRegistry', ChangedBody)
    with pytest.raises(CodegenError, match='implementation'):
        renderer.render_cpp_package(package, 'test')
    monkeypatch.setattr(renderer, 'CppTemplateRegistry', original)
    from triton.flagmega.codegen.cpp.program_profile import configure_descriptor, MODE
    diagnostic = configure_descriptor(package, MODE)
    with pytest.raises(CodegenError, match='normal NVIDIA'):
        renderer.render_cpp_package(diagnostic, 'test')
