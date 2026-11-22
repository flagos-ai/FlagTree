# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Coverage, ownership and wait-graph proofs for projected worker programs."""

from collections import defaultdict, deque
from dataclasses import replace
import hashlib
import json
from math import prod

from triton.flagmega.errors import IRVerificationError
from triton.flagmega.ir.tir.barrier import Barrier
from triton.flagmega.ir.tir.heterogeneous import (
    ChannelBarrier, HeterogeneousPlan, HeterogeneousProgram, KernelPhaseInvoke,
    WorkerCall, projected_function_attrs,
)
from triton.flagmega.ir.tir.kernel_invoke import KernelInvoke
from triton.flagmega.ir.tir.prim_function_call import PrimFunctionCall
from triton.flagmega.ir.tir.sequential import Sequential
from triton.flagmega.ir.tir.synchronization_range import SynchronizationRange


def heterogeneous_source_hash(module):
    from triton.flagmega.ir.model import _to_data
    payload = {name: [value.to_data() for value in getattr(module, name)] for name in (
        "prim_functions", "kernel_definitions", "execution_functions",
    )}
    payload["buffer_plan"] = _to_data(module.metadata.get("buffer_plan"))
    if "k230_value_synchronization" in module.metadata:
        payload["k230_value_synchronization"] = _to_data(module.metadata["k230_value_synchronization"])
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _reject(module, message):
    raise IRVerificationError("Heterogeneous " + message, stage=module.stage)


def _unique(module, values, key, label):
    result = {}
    for value in values:
        identity = key(value)
        if identity in result:
            _reject(module, f"duplicate {label} {identity!r}.")
        result[identity] = value
    return result


def _fields(module, body):
    if not isinstance(body, Sequential):
        _reject(module, "workers require static sequential bodies.")
    result = []
    for value in body.fields:
        if isinstance(value, Sequential):
            result.extend(_fields(module, value))
        else:
            result.append(value)
    return tuple(result)


def exchange_ranges(exchange, event, *, sequence=None):
    if sequence is not None and (type(sequence) is not int or not 0 < sequence < 2 ** 32):
        raise ValueError("Exchange range resolution requires a positive uint32 sequence.")
    offset = 0 if sequence is None else (sequence - 1) % 2 * exchange.slot_stride_bytes
    def ranges(buffers, mode):
        return tuple(SynchronizationRange(
            value.mem_span.buffer.memory_space, value.mem_span.buffer.id,
            value.mem_span.absolute_start.fixed_value + offset, value.mem_span.size.fixed_value, mode,
        ) for value in buffers)
    if event == "publish":
        return ranges(exchange.inputs, "write")
    if event == "acquire":
        return ranges(exchange.inputs, "read")
    if event == "complete":
        return (*ranges(exchange.inputs, "read"), *ranges(exchange.outputs, "write"))
    if event == "observe":
        return ranges(exchange.outputs, "read")
    if event == "reclaim":
        return ranges((*exchange.inputs, *exchange.outputs), "read_write")
    raise ValueError(f"Unknown exchange event {event!r}.")


def verify_heterogeneous_programs(module):
    if any(not isinstance(value, HeterogeneousPlan) for value in module.heterogeneous_plans):
        _reject(module, "plans must be typed HeterogeneousPlan values.")
    if any(not isinstance(value, HeterogeneousProgram) for value in module.heterogeneous_programs):
        _reject(module, "programs must be typed HeterogeneousProgram values.")
    plans = _unique(module, module.heterogeneous_plans, lambda value: value.name, "plan")
    programs = _unique(module, module.heterogeneous_programs, lambda value: value.name, "program")
    if not plans and not programs:
        return
    sources = module.execution_function_map
    calls = {}
    for source in sources.values():
        for call in _fields(module, source.body):
            if not isinstance(call, (KernelInvoke, PrimFunctionCall, Barrier)):
                _reject(module, "initial program source must contain only static calls and local barriers.")
            if not isinstance(call, Barrier):
                calls[(source.name, call.call_id)] = call
    for plan in plans.values():
        fragment_signatures = module.metadata.get("function_signatures", {}) if module.metadata.get("_dump_function_fragment") else {}
        if plan.source_entry not in module.function_map and plan.source_entry not in sources and plan.source_entry not in fragment_signatures:
            _reject(module, "plan source entry does not exist.")
        placements = _unique(module, plan.placements, lambda value: (value.source_function, value.call_id), "placement")
        _unique(module, plan.resources, lambda value: value.name, "resource")
        for key, place in placements.items():
            if place.domain not in {plan.cpu_domain, plan.device_domain}:
                _reject(module, f"unknown placement domain {place.domain!r}.")
            if plan.protocol not in {"k230.glb-values/v1", "k230.functions/v1"} and (place.domain == plan.cpu_domain) != (place.projections == 0):
                _reject(module, "CPU placements require zero device projections; device placements require projections.")
            if sources:
                call = calls.get(key)
                if not isinstance(call, KernelInvoke) or call.kernel != place.kernel:
                    _reject(module, f"placement {key!r} differs from the source kernel call.")
                definition = module.kernel_callable_map.get(place.kernel)
                dispatch = getattr(definition, "dispatch", None)
                selection = getattr(dispatch, "microkernel", None)
                if selection is None or selection.implementation != place.implementation:
                    _reject(module, f"placement {key!r} differs from its selected implementation.")
        if sources and set(placements) != {key for key, value in calls.items() if isinstance(value, KernelInvoke)}:
            _reject(module, "placement must exactly cover source kernel calls.")
        if plan.source_hash is not None and plan.source_hash != heterogeneous_source_hash(module):
            _reject(module, "source execution fingerprint is stale.")
    used_plans = set()
    if any(plan.protocol == "k230.functions/v1" for plan in plans.values()):
        from triton.flagmega.passes.tir.k230_functions import verify_function_program
        verify_function_program(module)
    for program in programs.values():
        plan = plans.get(program.plan)
        if plan is None or plan.name in used_plans:
            _reject(module, "each program requires its own existing plan.")
        used_plans.add(plan.name)
        if not sources or plan.source_hash is None:
            _reject(module, "physical programs require a frozen source execution fingerprint.")
        if plan.protocol == "k230.glb-values/v1":
            from triton.flagmega.passes.tir.heterogeneous_values import verify_value_program
            verify_value_program(module, program, plan)
        else:
            _verify_program(module, program, plan, calls)


def _verify_program(module, program, plan, sources):
    workers = _unique(module, program.functions, lambda value: value.function.name, "worker function")
    projections = _unique(module, program.functions, lambda value: (value.source_function, value.domain), "function projection")
    domains = {plan.cpu_domain, plan.device_domain}
    if set(projections) != {(name, domain) for name in module.execution_function_map for domain in domains}:
        _reject(module, "workers must exactly project every source function into both domains.")
    for worker in workers.values():
        source = module.execution_function_map[worker.source_function]
        if (worker.function.parameters != source.parameters or worker.function.results != source.results
                or dict(worker.function.attrs) != projected_function_attrs(source, cpu=worker.domain == plan.cpu_domain)):
            _reject(module, "worker physical signature must preserve its source function ABI.")
    entries = (program.launch.cpu_entry, program.launch.device_entry)
    for name, domain in zip(entries, (plan.cpu_domain, plan.device_domain)):
        if name not in workers or (workers[name].source_function, workers[name].domain) != (plan.source_entry, domain):
            _reject(module, "launch entry does not select the correct worker projection.")
    if set(program.drain.after) != set(entries) or program.drain.epoch_resource != program.launch.epoch_resource:
        _reject(module, "final drain must cover both workers in the launch epoch.")
    if workers[entries[0]].phase_count != workers[entries[1]].phase_count:
        _reject(module, "worker entry phase counts differ.")
    placements = {(value.source_function, value.call_id): value for value in plan.placements}
    exchanges = _unique(module, program.exchanges, lambda value: value.name, "exchange")
    exchange_keys = _unique(module, program.exchanges,
        lambda value: (value.source_function, value.origin_call_id, value.projection), "projection exchange")
    expected = {(place.source_function, place.call_id, index) for place in plan.placements for index in range(place.projections)}
    if set(exchange_keys) != expected:
        _reject(module, "exchanges must exactly cover device projections.")
    resources = {value.name: value for value in plan.resources}
    for exchange in exchanges.values():
        for field in (*exchange.inputs, *exchange.outputs):
            span = field.mem_span
            resource = resources.get(span.buffer.id)
            if (resource is None or resource.memory_space != span.buffer.memory_space
                    or resource.lifetime.value != "function"
                    or any(not value.is_fixed for value in resource.type.shape)):
                _reject(module, "exchange storage requires a declared static function-lifetime resource.")
            size = prod(value.fixed_value for value in resource.type.shape) * resource.type.dtype.itemsize
            if (not span.buffer.size.is_fixed or span.absolute_end.fixed_value + exchange.slot_stride_bytes > size
                    or span.end.fixed_value + exchange.slot_stride_bytes > span.buffer.size.fixed_value
                    or span.buffer.alignment > resource.alignment):
                _reject(module, "exchange storage exceeds its declared resource.")
        fields = (*exchange.inputs, *exchange.outputs)
        for index, left in enumerate(fields):
            for right in fields[index + 1:]:
                if (left.mem_span.buffer.id == right.mem_span.buffer.id
                        and max(left.mem_span.offset, right.mem_span.offset)
                        < min(left.mem_span.absolute_end.fixed_value, right.mem_span.absolute_end.fixed_value)):
                    _reject(module, "exchange input/output fields overlap.")
    for worker in workers.values():
        _verify_worker(module, worker, workers, projections, sources, placements, exchanges, plan)
    _verify_wait_graph(module, program, plan, workers, exchanges)


def _verify_worker(module, worker, workers, projections, sources, placements, exchanges, plan):
    source_calls = {key[1]: value for key, value in sources.items() if key[0] == worker.source_function}
    seen = defaultdict(list)
    events = defaultdict(list)
    statements = _fields(module, worker.function.body)
    original_barriers = tuple(value for value in _fields(module, module.execution_function_map[worker.source_function].body)
                              if isinstance(value, Barrier))
    projected_barriers = tuple(value for value in statements if isinstance(value, Barrier))
    if projected_barriers != (original_barriers if worker.domain == plan.cpu_domain else ()):
        _reject(module, "source barriers must be preserved exactly in the CPU projection.")
    _unique(module, statements, lambda value: getattr(value, "call_id", id(value)), "worker event")
    for value in statements:
        if isinstance(value, Barrier):
            continue
        if isinstance(value, WorkerCall):
            original = source_calls.get(value.call_id)
            target = workers.get(value.invocation.callee)
            if not isinstance(original, PrimFunctionCall) or target is None:
                _reject(module, "worker call has no corresponding source call/callee.")
            if target.source_function != original.callee or target.domain != worker.domain:
                _reject(module, "worker call changes source callee or execution domain.")
            if replace(value.invocation, callee=original.callee) != original:
                _reject(module, "worker call changes physical bindings or effects.")
            if value.phase_offset + target.phase_count > worker.phase_count:
                _reject(module, "callee sequence range exceeds its worker function.")
            seen[value.call_id].append("call")
        elif isinstance(value, KernelInvoke):
            original = source_calls.get(value.call_id)
            place = placements.get((worker.source_function, value.call_id))
            if original != value or place is None or place.domain != worker.domain or place.projections:
                _reject(module, "unsplit kernel is not an exact CPU source call.")
            seen[value.call_id].append("cpu")
        elif isinstance(value, KernelPhaseInvoke):
            original = source_calls.get(value.origin_call_id)
            place = placements.get((worker.source_function, value.origin_call_id))
            exchange = exchanges.get(value.exchange)
            if original != value.invocation or place is None or not 0 <= value.projection < place.projections:
                _reject(module, "kernel phase changes or invents a source projection.")
            expected_domain = plan.device_domain if value.phase == "execute" else plan.cpu_domain
            if worker.domain != expected_domain or exchange is None or (
                exchange.source_function, exchange.origin_call_id, exchange.projection
            ) != (worker.source_function, value.origin_call_id, value.projection):
                _reject(module, "kernel phase domain/exchange does not match its source.")
            seen[value.origin_call_id].append((value.projection, value.phase))
        elif isinstance(value, ChannelBarrier):
            exchange = exchanges.get(value.exchange)
            expected_domain = plan.device_domain if value.event in {"acquire", "complete"} else plan.cpu_domain
            if (exchange is None or exchange.source_function != worker.source_function or worker.domain != expected_domain
                    or value.sequence > worker.phase_count):
                _reject(module, "channel event has invalid domain, exchange or sequence.")
            if value.ranges != exchange_ranges(exchange, value.event):
                _reject(module, "channel event does not cover its exact physical access ranges.")
            events[value.exchange].append(value.event)
        else:
            _reject(module, f"unsupported worker statement {type(value).__name__}.")
    for call_id, original in source_calls.items():
        if isinstance(original, PrimFunctionCall):
            expected = ["call"]
        else:
            place = placements[(worker.source_function, call_id)]
            expected = (["cpu"] if worker.domain == place.domain else []) if not place.projections else [
                (index, phase) for index in range(place.projections)
                for phase in (("execute",) if worker.domain == plan.device_domain else ("prepare", "finish"))
            ]
        if sorted(seen[call_id], key=str) != sorted(expected, key=str):
            _reject(module, f"worker call coverage differs for {call_id!r}.")
    for exchange in exchanges.values():
        if exchange.source_function != worker.source_function:
            continue
        expected = {"acquire", "complete"} if worker.domain == plan.device_domain else {"publish", "observe", "reclaim"}
        if len(events[exchange.name]) != len(expected) or set(events[exchange.name]) != expected:
            _reject(module, "each exchange requires exactly one event of each worker role.")


def _verify_wait_graph(module, program, plan, workers, exchanges):
    graph = defaultdict(set)
    events = defaultdict(dict)
    phases = defaultdict(dict)
    accesses = []
    sequence_keys = {}
    channel_sequences = {}
    expanded_count = 0
    source_contexts = []

    def expand(name, path, offset, stack):
        nonlocal expanded_count
        if name in stack:
            _reject(module, "recursive worker calls are unsupported.")
        worker = workers[name]
        local, starts, bounds, barriers = {}, {}, {}, []
        order = []
        statements = _fields(module, worker.function.body)
        for index, value in enumerate(statements):
            if isinstance(value, WorkerCall):
                child = expand(value.invocation.callee, (*path, value.call_id), offset + value.phase_offset, (*stack, name))
                if child:
                    local[value.call_id] = child[-1]
                    starts[value.call_id] = child[0]
                    bounds[value.call_id] = (child[0], child[-1])
                    order.extend(child)
                continue
            expanded_count += 1
            if expanded_count > 100000:
                _reject(module, "bounded worker expansion exceeds 100000 events.")
            if isinstance(value, Barrier):
                key = (worker.domain, *path, ("barrier", index))
                barriers.append((value, key))
            else:
                key = (worker.domain, *path, value.call_id)
                local[value.call_id] = key
                starts[value.call_id] = key
            graph[key]
            order.append(key)
            if isinstance(value, ChannelBarrier):
                if value.epoch_resource != program.launch.epoch_resource:
                    _reject(module, "channel epoch differs from the program launch.")
                channel = (*path, value.exchange)
                events[channel][value.event] = key
                sequence = offset + value.sequence
                if sequence >= 2 ** 32:
                    _reject(module, "phase sequence overflows uint32.")
                previous = sequence_keys.setdefault(sequence, channel)
                if previous != channel:
                    _reject(module, "different exchanges reuse one invocation phase sequence.")
                if channel_sequences.setdefault(channel, sequence) != sequence:
                    _reject(module, "events for one exchange disagree on their phase sequence.")
            elif isinstance(value, KernelPhaseInvoke):
                channel = (*path, value.exchange)
                phases[channel][value.phase] = key
                exchange = exchanges[value.exchange]
                ranges = exchange_ranges(exchange, {"prepare": "publish", "execute": "complete", "finish": "observe"}[value.phase])
                accesses.extend((key, r, channel) for r in ranges)
                if worker.domain == plan.cpu_domain:
                    first = bounds.get(value.origin_call_id, (key, key))[0]
                    bounds[value.origin_call_id] = (first, key)
                    if value.phase == "finish":
                        local[value.origin_call_id] = key
            elif isinstance(value, KernelInvoke):
                bounds[value.call_id] = (key, key)
        for before, after in zip(order, order[1:]):
            graph[before].add(after)
        for value in statements:
            dependencies = value.after if isinstance(value, ChannelBarrier) else (
                value.dependencies if isinstance(value, KernelPhaseInvoke) else ())
            for dependency in dependencies:
                if dependency not in local:
                    _reject(module, f"worker dependency {dependency!r} is unresolved.")
                graph[local[dependency]].add(starts[value.call_id])
        if worker.domain == plan.cpu_domain:
            source_contexts.append((worker.source_function, bounds, barriers))
        return order

    expand(program.launch.cpu_entry, (), 0, ())
    expand(program.launch.device_entry, (), 0, ())
    count = workers[program.launch.cpu_entry].phase_count
    if count != len(sequence_keys) or set(sequence_keys) != set(range(1, count + 1)):
        _reject(module, "invocation phase sequences must exactly cover the bounded range.")
    if set(events) != set(phases):
        _reject(module, "channel events and computation phases differ.")
    for channel in events:
        e, p = events[channel], phases[channel]
        chain = (p["prepare"], e["publish"], e["acquire"], p["execute"], e["complete"], e["observe"], p["finish"], e["reclaim"])
        for before, after in zip(chain, chain[1:]):
            graph[before].add(after)
    for sequence, channel in sequence_keys.items():
        if sequence > plan.slots:
            prior = sequence_keys[sequence - plan.slots]
            graph[events[prior]["reclaim"]].add(phases[channel]["prepare"])
    indegree = dict.fromkeys(graph, 0)
    for targets in graph.values():
        for target in targets:
            indegree[target] += 1
    ready = deque(key for key, value in indegree.items() if not value)
    ordered = []
    while ready:
        key = ready.popleft()
        ordered.append(key)
        for target in graph[key]:
            indegree[target] -= 1
            if not indegree[target]:
                ready.append(target)
    if len(ordered) != len(graph):
        _reject(module, "worker dependencies and publication barriers form a cycle.")
    reachable = {}
    for key in reversed(ordered):
        reachable[key] = set(graph[key])
        for target in graph[key]:
            reachable[key].update(reachable[target])
    _verify_source_order(module, source_contexts, reachable)
    resolved_accesses = [(key, replace(value, offset=value.offset +
        (channel_sequences[channel] - 1) % plan.slots * exchanges[channel[-1]].slot_stride_bytes))
        for key, value, channel in accesses]
    for index, (left, a) in enumerate(resolved_accesses):
        for right, b in resolved_accesses[index + 1:]:
            if left == right or a.physical_id != b.physical_id or a.storage != b.storage:
                continue
            if a.mode == b.mode == "read" or max(a.offset, b.offset) >= min(a.offset + a.nbytes, b.offset + b.nbytes):
                continue
            if right not in reachable[left] and left not in reachable[right]:
                _reject(module, "unordered alias hazard between exchange ownership phases.")


def source_call_conflicts(module, function, left, right, *, buffer_plan=None):
    """Conservatively retain source order for physical reuse and opaque effects."""
    if (left.effect_kind != "pure" and right.effect_kind != "pure" and
            (left.effect_resource is None or right.effect_resource is None
             or left.effect_resource == right.effect_resource)):
        return True
    from triton.flagmega.ir.bufferization import BufferPlan
    plan = module.metadata.get("buffer_plan") if buffer_plan is None else buffer_plan
    if plan is None:
        return bool(set(left.writes) & set((*right.reads, *right.writes))
                    or set(right.writes) & set(left.reads))
    if not isinstance(plan, BufferPlan):
        plan = BufferPlan.from_data(plan)
    def ranges(call):
        result = []
        scratch = {b.actual for b in call.workspaces}
        for name in set((*call.reads, *call.writes)) | scratch:
            descriptor = plan.buffer_map.get(name)
            if descriptor is None:
                _reject(module, "source access has no physical buffer descriptor.")
            span = descriptor.physical_access_span
            space = plan.memory_space_map[span.buffer.memory_space]
            identity = span.buffer.id
            if space.supports_lifetime_reuse and space.allocation_scope.value == "function" and space.kind != "shared":
                identity = (space.name, function)
            lo, hi = span.absolute_start.minimum, span.absolute_end.maximum
            if lo is None or hi is None:
                _reject(module, "source access requires a bounded physical span.")
            result.append((space.name, identity, lo, hi, name in call.writes or name in scratch))
        for pool in call.memory_pools:
            result.append((pool.memory_space, (pool.memory_space, function), pool.offset, pool.offset + pool.nbytes, True))
        return result
    return any(a[:2] == b[:2] and (a[4] or b[4]) and max(a[2], b[2]) < min(a[3], b[3])
               for a in ranges(left) for b in ranges(right))


def _verify_source_order(module, contexts, reachable):
    from triton.flagmega.ir.bufferization import BufferPlan
    buffer_plan = module.metadata.get("buffer_plan")
    if buffer_plan is not None and not isinstance(buffer_plan, BufferPlan):
        buffer_plan = BufferPlan.from_data(buffer_plan)
    conflicts = {}
    for function in {value[0] for value in contexts}:
        calls = tuple(value for value in _fields(module, module.execution_function_map[function].body)
                      if isinstance(value, (KernelInvoke, PrimFunctionCall)))
        conflicts[function] = tuple((left.call_id, right.call_id) for index, left in enumerate(calls)
            for right in calls[index + 1:] if source_call_conflicts(module, function, left, right, buffer_plan=buffer_plan))
    def require(before, after, reason):
        if after not in reachable[before]:
            _reject(module, f"source {reason} is not preserved by projected worker order.")
    for function, bounds, barriers in contexts:
        source = _fields(module, module.execution_function_map[function].body)
        calls = tuple(value for value in source if isinstance(value, (KernelInvoke, PrimFunctionCall)))
        if set(bounds) != {value.call_id for value in calls}:
            _reject(module, "CPU projection must bound every source call completion.")
        for call in calls:
            for dependency in call.dependencies:
                if dependency not in bounds:
                    _reject(module, f"source dependency {dependency!r} is unresolved.")
                require(bounds[dependency][1], bounds[call.call_id][0], "dependency")
        for barrier, key in barriers:
            if barrier.before not in bounds or any(value not in bounds for value in barrier.after):
                _reject(module, "source barrier refers to an unresolved call.")
            for predecessor in barrier.after:
                require(bounds[predecessor][1], key, "barrier predecessor")
            require(key, bounds[barrier.before][0], "barrier successor")
        for left, right in conflicts[function]:
            require(bounds[left][1], bounds[right][0], "memory/effect order")
