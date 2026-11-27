# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Closed companion for reviewed counter epochs of a complete serial program.

A plan names a registered, implementation-bound effect contract. Metadata may
select a reviewed schedule but cannot supply its own effects or a safety flag.
The finite proof covers every declared alternative and rejects counter wrap.
"""
from collections.abc import Mapping
from dataclasses import dataclass
import hashlib
import json
import re

from triton.flagmega.errors import IRSchemaError, IRVerificationError

COUNTER_EPOCH_SCHEMA = 'flagmega.counter-epoch-plan/v1'
COUNTER_EFFECT_SCHEMA = 'flagmega.counter-effect-contract/v1'
COUNTER_EPOCH_MODE = 'normal_serial_complete'


def canonical_json(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False)


def identity(value):
    return hashlib.sha256(canonical_json(value).encode()).hexdigest()


def _closed(data, keys):
    if not isinstance(data, Mapping) or set(data) != set(keys):
        raise IRSchemaError('Counter epoch schema has missing or unknown fields.')


def _uint(value, *, positive=False):
    if type(value) is not int or value < int(positive) or value >= 2**32:
        raise IRSchemaError('Counter epoch requires an exact bounded unsigned integer.')
    return value


def _digest(value):
    if not isinstance(value, str) or re.fullmatch('[0-9a-f]{64}', value) is None:
        raise IRSchemaError('Counter epoch requires a canonical SHA256 identity.')
    return value


def _name(value):
    if not isinstance(value, str) or not value or value.strip() != value:
        raise IRSchemaError('Counter epoch requires a nonempty unpadded name.')
    return value


def _sequence(value):
    if not isinstance(value, (tuple, list)):
        raise IRSchemaError('Counter epoch requires a sequence.')
    return value


@dataclass(frozen=True)
class CounterReset:
    ordinal: int
    nbytes: int

    def __post_init__(self):
        _uint(self.ordinal)
        _uint(self.nbytes, positive=True)
        if self.nbytes % 4:
            raise IRSchemaError('Counter reset span must contain complete uint32 counters.')

    def to_data(self):
        return {'ordinal': self.ordinal, 'nbytes': self.nbytes}

    @classmethod
    def from_data(cls, data):
        _closed(data, ('ordinal', 'nbytes'))
        return cls(**data)


@dataclass(frozen=True)
class CounterEpochRow:
    ordinal: int
    entry_id: int
    invocation_sha256: str

    def __post_init__(self):
        _uint(self.ordinal)
        _uint(self.entry_id)
        _digest(self.invocation_sha256)

    def to_data(self):
        return {'ordinal': self.ordinal, 'entry_id': self.entry_id,
                'invocation_sha256': self.invocation_sha256}

    @classmethod
    def from_data(cls, data):
        _closed(data, ('ordinal', 'entry_id', 'invocation_sha256'))
        return cls(**data)


@dataclass(frozen=True)
class CounterEpochPlan:
    contract: str
    contract_sha256: str
    schedule: str
    execution_sha256: str
    device_source_sha256: str
    frame_nbytes: int
    rows: tuple[CounterEpochRow, ...]
    resets: tuple[CounterReset, ...]

    def __post_init__(self):
        _name(self.contract)
        _name(self.schedule)
        for value in (self.contract_sha256, self.execution_sha256, self.device_source_sha256):
            _digest(value)
        _uint(self.frame_nbytes, positive=True)
        if self.frame_nbytes % 4:
            raise IRSchemaError('Counter frame must contain uint32 counters.')
        object.__setattr__(self, 'rows', tuple(_sequence(self.rows)))
        object.__setattr__(self, 'resets', tuple(_sequence(self.resets)))
        if not self.rows or any(not isinstance(row, CounterEpochRow) for row in self.rows):
            raise IRSchemaError('Counter plan requires typed complete rows.')
        if any(not isinstance(reset, CounterReset) for reset in self.resets):
            raise IRSchemaError('Counter plan requires typed reset spans.')
        if [row.ordinal for row in self.rows] != list(range(len(self.rows))):
            raise IRSchemaError('Counter plan rows must be complete, ordered and unique.')
        ordinals = [reset.ordinal for reset in self.resets]
        if (not ordinals or ordinals != sorted(set(ordinals)) or ordinals[-1] >= len(self.rows)
                or self.resets[0] != CounterReset(0, self.frame_nbytes)
                or any(reset.nbytes > self.frame_nbytes for reset in self.resets)):
            raise IRSchemaError('Counter plan requires ordered bounded resets and a full initial reset.')

    def _body(self):
        return {'schema': COUNTER_EPOCH_SCHEMA, 'mode': COUNTER_EPOCH_MODE,
                'contract': self.contract, 'contract_sha256': self.contract_sha256,
                'schedule': self.schedule, 'execution_sha256': self.execution_sha256,
                'device_source_sha256': self.device_source_sha256,
                'counter_bits': 32, 'counter_count': self.frame_nbytes // 4,
                'frame_nbytes': self.frame_nbytes,
                'rows': [row.to_data() for row in self.rows],
                'resets': [reset.to_data() for reset in self.resets]}

    @property
    def plan_sha256(self):
        return identity(self._body())

    def to_data(self):
        return {**self._body(), 'plan_sha256': self.plan_sha256}

    @classmethod
    def from_data(cls, data):
        _closed(data, ('schema', 'mode', 'contract', 'contract_sha256', 'schedule',
                      'execution_sha256', 'device_source_sha256', 'counter_bits',
                      'counter_count', 'frame_nbytes', 'rows', 'resets', 'plan_sha256'))
        if data['schema'] != COUNTER_EPOCH_SCHEMA or data['mode'] != COUNTER_EPOCH_MODE:
            raise IRSchemaError('Unsupported counter epoch schema or execution mode.')
        if type(data['counter_bits']) is not int or data['counter_bits'] != 32:
            raise IRSchemaError('Counter epochs currently require uint32 counters.')
        plan = cls(data['contract'], data['contract_sha256'], data['schedule'],
                   data['execution_sha256'], data['device_source_sha256'], data['frame_nbytes'],
                   tuple(CounterEpochRow.from_data(row) for row in _sequence(data['rows'])),
                   tuple(CounterReset.from_data(reset) for reset in _sequence(data['resets'])))
        if _uint(data['counter_count'], positive=True) != plan.frame_nbytes // 4:
            raise IRSchemaError('Counter count differs from the frame span.')
        if _digest(data['plan_sha256']) != plan.plan_sha256:
            raise IRSchemaError('Counter plan identity differs from its canonical contents.')
        return plan


@dataclass(frozen=True)
class CounterBarrier:
    slot: int
    population: int

    def __post_init__(self):
        _uint(self.slot)
        _uint(self.population, positive=True)

    def to_data(self):
        return {'slot': self.slot, 'population': self.population}

    @classmethod
    def from_data(cls, data):
        _closed(data, ('slot', 'population'))
        return cls(**data)


@dataclass(frozen=True)
class EntryCounterEffect:
    entry_id: int
    entry_sha256: str
    alternatives: tuple[tuple[CounterBarrier, ...], ...]

    def __post_init__(self):
        _uint(self.entry_id)
        _digest(self.entry_sha256)
        object.__setattr__(self, 'alternatives', tuple(tuple(_sequence(branch))
                           for branch in _sequence(self.alternatives)))
        if (not self.alternatives or any(not isinstance(effect, CounterBarrier)
                for branch in self.alternatives for effect in branch)):
            raise IRSchemaError('Counter effects require all typed branch alternatives.')

    def to_data(self):
        return {'entry_id': self.entry_id, 'entry_sha256': self.entry_sha256,
                'alternatives': [[effect.to_data() for effect in branch] for branch in self.alternatives]}

    @classmethod
    def from_data(cls, data):
        _closed(data, ('entry_id', 'entry_sha256', 'alternatives'))
        return cls(data['entry_id'], data['entry_sha256'], tuple(
            tuple(CounterBarrier.from_data(effect) for effect in _sequence(branch))
            for branch in _sequence(data['alternatives'])))


@dataclass(frozen=True)
class CounterEffectContract:
    name: str
    execution_sha256: str
    device_source_sha256: str
    proof_sha256: str
    frame_nbytes: int
    effects: tuple[EntryCounterEffect, ...]
    schedules: tuple[tuple[str, tuple[CounterReset, ...]], ...]

    def __post_init__(self):
        _name(self.name)
        for value in (self.execution_sha256, self.device_source_sha256, self.proof_sha256):
            _digest(value)
        _uint(self.frame_nbytes, positive=True)
        if self.frame_nbytes % 4:
            raise IRSchemaError('Effect frame must contain uint32 counters.')
        object.__setattr__(self, 'effects', tuple(_sequence(self.effects)))
        object.__setattr__(self, 'schedules', tuple((_name(name), tuple(_sequence(resets)))
                           for name, resets in _sequence(self.schedules)))
        if (not self.effects or any(not isinstance(effect, EntryCounterEffect) for effect in self.effects)
                or [effect.entry_id for effect in self.effects] != list(range(len(self.effects)))
                or any(e.slot >= self.frame_nbytes // 4 for f in self.effects for b in f.alternatives for e in b)):
            raise IRSchemaError('Effects must cover every entry and stay inside the counter frame.')
        if (not self.schedules or len({name for name, _ in self.schedules}) != len(self.schedules)
                or any(not isinstance(r, CounterReset) for _, resets in self.schedules for r in resets)):
            raise IRSchemaError('Reviewed counter schedules must be named, typed and unique.')

    def to_data(self):
        return {'schema': COUNTER_EFFECT_SCHEMA, 'name': self.name,
                'execution_sha256': self.execution_sha256,
                'device_source_sha256': self.device_source_sha256, 'proof_sha256': self.proof_sha256,
                'frame_nbytes': self.frame_nbytes, 'effects': [effect.to_data() for effect in self.effects],
                'schedules': [{'name': name, 'resets': [reset.to_data() for reset in resets]}
                              for name, resets in self.schedules]}

    @property
    def contract_sha256(self):
        return identity(self.to_data())

    @classmethod
    def from_data(cls, data):
        _closed(data, ('schema', 'name', 'execution_sha256', 'device_source_sha256',
                      'proof_sha256', 'frame_nbytes', 'effects', 'schedules'))
        if data['schema'] != COUNTER_EFFECT_SCHEMA:
            raise IRSchemaError('Unsupported counter effect contract schema.')
        schedules = []
        for row in _sequence(data['schedules']):
            _closed(row, ('name', 'resets'))
            schedules.append((row['name'], tuple(CounterReset.from_data(r) for r in _sequence(row['resets']))))
        return cls(data['name'], data['execution_sha256'], data['device_source_sha256'],
                   data['proof_sha256'], data['frame_nbytes'],
                   tuple(EntryCounterEffect.from_data(e) for e in _sequence(data['effects'])), tuple(schedules))


# Trusted implementation extensions register immutable contracts, analogous to
# selected family encoders. Artifact/module data cannot install a contract.
_CONTRACTS = {}


def register_counter_effect_contract(contract):
    if not isinstance(contract, CounterEffectContract):
        raise IRSchemaError('Register a typed reviewed effect contract.')
    previous = _CONTRACTS.get(contract.name)
    if previous is not None and previous != contract:
        raise IRVerificationError('Counter effect contract names cannot be rebound.')
    _CONTRACTS[contract.name] = contract
    return contract


def counter_effect_contract(name):
    try:
        return _CONTRACTS[name]
    except (KeyError, TypeError) as error:
        raise IRVerificationError('Counter effect contract is not registered by an implementation extension.') from error


def _program_identity(programs):
    body = {key: value for key, value in programs.items() if key != 'plan_sha256'}
    if programs.get('plan_sha256') != identity(body):
        raise IRVerificationError('Ordinary program manifest identity is stale.')
    return programs['plan_sha256']


def make_counter_epoch_plan(contract_name, schedule, programs, device_source_sha256):
    contract = counter_effect_contract(contract_name)
    try:
        resets = dict(contract.schedules)[schedule]
    except (KeyError, TypeError) as error:
        raise IRVerificationError('Counter schedule is not reviewed by the registered contract.') from error
    plan = CounterEpochPlan(contract.name, contract.contract_sha256, schedule, _program_identity(programs),
                            device_source_sha256, contract.frame_nbytes,
                            tuple(CounterEpochRow(row['ordinal'], row['entry_id'], identity(row))
                                  for row in programs['invocations']), resets)
    verify_counter_epoch_plan(plan, programs, device_source_sha256)
    return plan


def verify_counter_epoch_plan(plan, programs, device_source_sha256):
    if not isinstance(plan, CounterEpochPlan):
        plan = CounterEpochPlan.from_data(plan)
    contract = counter_effect_contract(plan.contract)
    if (plan.contract_sha256 != contract.contract_sha256
            or plan.execution_sha256 != _program_identity(programs)
            or plan.execution_sha256 != contract.execution_sha256
            or plan.device_source_sha256 != device_source_sha256
            or plan.device_source_sha256 != contract.device_source_sha256
            or plan.frame_nbytes != contract.frame_nbytes
            or dict(contract.schedules).get(plan.schedule) != plan.resets):
        raise IRVerificationError('Counter plan differs from the reviewed implementation/schedule identity.')
    expected_rows = tuple(CounterEpochRow(row['ordinal'], row['entry_id'], identity(row))
                          for row in programs['invocations'])
    if plan.rows != expected_rows or len(contract.effects) != len(programs['entries']):
        raise IRVerificationError('Counter plan does not bind the complete invocation order and entries.')
    for effect, entry in zip(contract.effects, programs['entries'], strict=True):
        if (entry['entry_id'] != effect.entry_id or effect.entry_sha256 != identity(entry)
                or entry['scratch_nbytes'] > plan.frame_nbytes
                or entry['scratch_nbytes'] % 4):
            raise IRVerificationError('Counter effect differs from selected entry/geometry/scratch identity.')
        if any(e.slot >= entry['scratch_nbytes'] // 4 for branch in effect.alternatives for e in branch):
            raise IRVerificationError('Counter effect is outside the exact per-entry scratch span.')
    resets = {reset.ordinal: reset.nbytes // 4 for reset in plan.resets}
    states = [{0} for _ in range(plan.frame_nbytes // 4)]
    maxima = [0] * len(states)
    for row in plan.rows:
        for slot in range(resets.get(row.ordinal, 0)):
            states[slot] = {0}
        effect = contract.effects[row.entry_id]
        branch_states = []
        for branch in effect.alternatives:
            current = [set(values) for values in states]
            for barrier in branch:
                values = current[barrier.slot]
                if any(value % barrier.population or value + barrier.population >= 2**32 for value in values):
                    raise IRVerificationError('Counter epoch has a partial generation or uint32 wrap.')
                current[barrier.slot] = {value + barrier.population for value in values}
            branch_states.append(current)
        states = [set().union(*(branch[slot] for branch in branch_states)) for slot in range(len(states))]
        maxima = [max(old, max(values)) for old, values in zip(maxima, states, strict=True)]
    return {'plan_sha256': plan.plan_sha256, 'contract_sha256': contract.contract_sha256,
            'proof_sha256': contract.proof_sha256, 'calls': len(plan.rows), 'resets': len(plan.resets),
            'nodes': len(plan.rows) + len(plan.resets), 'max_counter_values': maxima,
            'terminal_counter_values': [sorted(values) for values in states]}
