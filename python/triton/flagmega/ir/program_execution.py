# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Structured invocations over shared, function-scoped program definitions.

This contract describes control order. Physical call frames and loop transfers
remain owned by FunctionBufferPlan and executable TIR, rather than being
invented by an invocation list or a runtime pointer-expression parser.
"""
from collections.abc import Mapping
from dataclasses import dataclass

from triton.flagmega.errors import IRSchemaError, IRVerificationError

PROGRAM_EXECUTION_SCHEMA = 'flagmega.structured-program-execution/v1'


def _name(value):
    if not isinstance(value, str) or not value or value.strip() != value:
        raise IRSchemaError('Structured execution requires nonempty, unpadded names.')
    return value


@dataclass(frozen=True)
class ProgramInvocation:
    symbol: str

    def __post_init__(self):
        _name(self.symbol)

    def to_data(self):
        return {'kind': 'program', 'symbol': self.symbol}


@dataclass(frozen=True)
class CallInvocation:
    call: str
    callee: str

    def __post_init__(self):
        _name(self.call)
        _name(self.callee)

    def to_data(self):
        return {'kind': 'call', 'call': self.call, 'callee': self.callee}


@dataclass(frozen=True)
class LoopInvocation:
    call: str
    callee: str
    start: int
    stop: int
    step: int

    def __post_init__(self):
        _name(self.call)
        _name(self.callee)
        if any(type(v) is not int for v in (self.start, self.stop, self.step)) or self.step == 0:
            raise IRSchemaError('Structured loop bounds must be integers with a nonzero step.')

    def to_data(self):
        return {'kind': 'loop', 'call': self.call, 'callee': self.callee,
                'start': self.start, 'stop': self.stop, 'step': self.step}


def invocation_from_data(data):
    if not isinstance(data, Mapping):
        raise IRSchemaError('Structured invocation must be a mapping.')
    kind = data.get('kind')
    if not isinstance(kind, str):
        raise IRSchemaError('Structured invocation kind must be named.')
    cls = {'program': ProgramInvocation, 'call': CallInvocation, 'loop': LoopInvocation}.get(kind)
    if cls is None or set(data) != {'kind', *cls.__dataclass_fields__}:
        raise IRSchemaError('Structured invocation has an unknown kind or fields.')
    return cls(**{key: data[key] for key in cls.__dataclass_fields__})


@dataclass(frozen=True)
class FunctionProgramExecution:
    function: str
    steps: tuple[ProgramInvocation | CallInvocation | LoopInvocation, ...]

    def __post_init__(self):
        _name(self.function)
        object.__setattr__(self, 'steps', tuple(self.steps))
        if any(not isinstance(v, (ProgramInvocation, CallInvocation, LoopInvocation)) for v in self.steps):
            raise IRSchemaError('Structured function steps require typed invocations.')

    def to_data(self):
        return {'function': self.function, 'steps': [step.to_data() for step in self.steps]}


@dataclass(frozen=True)
class ProgramExecution:
    entry: str
    functions: tuple[FunctionProgramExecution, ...]

    def __post_init__(self):
        _name(self.entry)
        object.__setattr__(self, 'functions', tuple(self.functions))
        if any(not isinstance(v, FunctionProgramExecution) for v in self.functions):
            raise IRSchemaError('Structured execution functions require typed definitions.')
        names = tuple(value.function for value in self.functions)
        if len(names) != len(set(names)) or self.entry not in names:
            raise IRSchemaError('Structured execution requires unique functions and a defined entry.')

    def to_data(self):
        return {'schema': PROGRAM_EXECUTION_SCHEMA, 'entry': self.entry,
                'functions': [value.to_data() for value in self.functions]}

    @classmethod
    def from_data(cls, data):
        if (not isinstance(data, Mapping) or set(data) != {'schema', 'entry', 'functions'}
                or data['schema'] != PROGRAM_EXECUTION_SCHEMA):
            raise IRSchemaError('Unknown structured execution schema or fields.')
        functions = []
        if not isinstance(data['functions'], (tuple, list)):
            raise IRSchemaError('Structured execution functions must be a sequence.')
        for value in data['functions']:
            if (not isinstance(value, Mapping) or set(value) != {'function', 'steps'}
                    or not isinstance(value['steps'], (tuple, list))):
                raise IRSchemaError('Unknown structured function execution fields.')
            functions.append(FunctionProgramExecution(value['function'], tuple(
                invocation_from_data(step) for step in value['steps'])))
        return cls(data['entry'], tuple(functions))


def loop_invocation(module, node):
    bounds = tuple(module.node_map[value] for value in node.inputs[:3])
    if any(v.op not in {'builtin.scalar_const', 'tir.scalar_const'} for v in bounds):
        raise IRVerificationError('Structured execution requires constant typed For bounds.', node_id=node.id)
    return LoopInvocation(node.id, str(node.attrs['callee']), *(v.attrs['value'] for v in bounds))


def execution_item_identity(function, node):
    return f'{function}.{node.id}_iv' if node.op == 'control.for' else node.id


def verify_program_execution(module):
    data = module.metadata.get('kernel_program_execution')
    requests = module.metadata.get('kernel_program_partition', ())
    if data is None:
        if isinstance(requests, (tuple, list)) and any(isinstance(p, Mapping) and 'function' in p for p in requests):
            raise IRVerificationError('Scoped programs require a structured execution plan.', stage=module.stage)
        return None

    def fail(message):
        raise IRVerificationError('Structured execution ' + message, stage=module.stage)

    try:
        plan = ProgramExecution.from_data(data)
    except IRSchemaError as error:
        fail(str(error))
    # A per-function checkpoint retains the complete interprocedural plan,
    # just like ExecutionFunction schedules. Verify its local typed body and
    # the complete plan structure; directory loading restores the entry and
    # verifies every body together after removing the explicit fragment mark.
    fragment = module.metadata.get('_dump_function_fragment') is True
    graph_names = set(module.function_map)
    declared_names = graph_names
    if fragment:
        signatures = module.metadata.get('function_signatures')
        if (len(graph_names) != 1 or not isinstance(signatures, Mapping)
                or not graph_names.issubset(signatures)
                or plan.entry not in signatures):
            fail('dump fragment requires complete function signatures.')
        declared_names = set(signatures)
    elif plan.entry != module.entry:
        fail('entry differs from the typed module.')
    if not isinstance(requests, (tuple, list)) or not requests:
        fail('requires scoped program definitions.')
    programs = {}
    for request in requests:
        if (not isinstance(request, Mapping) or not isinstance(request.get('symbol'), str)
                or request['symbol'] in programs or not isinstance(request.get('function'), str)
                or request['function'] not in declared_names
                or not isinstance(request.get('calls'), (tuple, list)) or not request['calls']):
            fail('has invalid or duplicate scoped program definitions.')
        if 'depends_on' in request or request.get('tasks_per_owner', 1) != 1:
            fail('requires serial invocation order and each selected program task contract.')
        programs[request['symbol']] = request
    functions = {f.function: f for f in plan.functions}
    # Use the same graph reachability as function buffer planning; bound
    # parameters stop the walk, even for a body shared by multiple loops.
    from triton.flagmega.passes.functions.graph import function_nodes
    used = set()
    for function in plan.functions:
        if function.function not in declared_names:
            fail('references an unknown typed function.')
        local = function.function in module.function_map
        nodes = (tuple(n for n in function_nodes(module, module.function_map[function.function])
                       if n.op in {'tir.call', 'control.for'}) if local else ())
        by_id = {n.id: n for n in nodes}
        actual = []
        for step in function.steps:
            if isinstance(step, ProgramInvocation):
                request = programs.get(step.symbol)
                if request is None or request['function'] != function.function or step.symbol in used:
                    fail('program reference disagrees with its function scope or repeats a definition.')
                used.add(step.symbol)
                actual.extend(request['calls'])
                continue
            if step.callee not in functions:
                fail('call reference differs from its typed invocation or callee scope.')
            if not local:
                continue
            node = by_id.get(step.call)
            if node is None or str(node.attrs.get('callee', '')) != step.callee:
                fail('call reference differs from its typed invocation or callee scope.')
            if isinstance(step, LoopInvocation):
                if node.op != 'control.for' or step != loop_invocation(module, node):
                    fail('loop bounds or kind differ from the typed For.')
            elif node.op != 'tir.call' or step.callee not in declared_names:
                fail('function invocation differs from its typed call.')
            actual.append(execution_item_identity(function.function, node))
        expected = [execution_item_identity(function.function, n) for n in nodes]
        if local and actual != expected:
            fail('must preserve every scoped executable item in its original order.')
    if used != set(programs):
        fail('contains unreferenced program definitions.')
    visited, active = set(), set()

    def visit(name):
        if name in active:
            fail('does not permit a recursive invocation graph.')
        if name in visited:
            return
        active.add(name)
        for step in functions[name].steps:
            if not isinstance(step, ProgramInvocation):
                visit(step.callee)
        active.remove(name)
        visited.add(name)
    visit(plan.entry)
    if visited != set(functions):
        fail('contains unreachable function definitions.')
    return plan
