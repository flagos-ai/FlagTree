# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Static value dependencies for the K230 shared-word program protocol."""

from dataclasses import dataclass, replace

from triton.flagmega.errors import IRVerificationError
from triton.flagmega.ir import Buffer, ExecutionFunction, KernelInvoke, MemSpan, PhysicalBuffer, PrimCallBinding, PrimFunctionCall, Sequential
from triton.flagmega.ir.bufferization import BufferPlan
from triton.flagmega.ir.tir.barrier import Barrier, BarrierScope
from triton.flagmega.ir.tir.heterogeneous import (
    ChannelBarrier, ExecutionPlacement, HeterogeneousDrain, HeterogeneousLaunch,
    HeterogeneousPlan, HeterogeneousProgram, ValueExchange, WorkerCall, WorkerFunction, heterogeneous_source_hash,
)
from triton.flagmega.ir.tir.heterogeneous_verify import _fields, source_call_conflicts
from triton.flagmega.ir.tir.synchronization_range import SynchronizationRange


PROTOCOL = "k230.glb-values/v1"
ENTRY = "$entry"
EXIT = "$exit"


def _reject(module, message):
    raise IRVerificationError("K230 value program " + message, stage=module.stage)


def validate_placements(module, target, contracts, plan):
    expected = set(contracts)
    actual = {(p.source_function, p.call_id): p for p in plan.placements}
    if len(actual) != len(plan.placements) or set(actual) != expected:
        _reject(module, "placement must cover every selected source call exactly once.")
    for key, place in actual.items():
        definition, contract = contracts[key]
        domains = contract.legal_domains or (contract.domain,)
        if (place.kernel != definition.name or place.implementation != contract.implementation
                or place.domain not in domains or place.projections != len(contract.projections)):
            _reject(module, "edited placement differs from its implementation capability.")
        if place.projections:
            _reject(module, "requires a direct device implementation; legacy projection phases are not value kernels.")
    if (plan.protocol != target.heterogeneous_protocol or plan.protocol not in {PROTOCOL, "k230.functions/v1"} or plan.source_entry != module.entry
            or plan.cpu_domain != "cpu0" or plan.device_domain != "kpu"
            or plan.resources != target.heterogeneous_resource_requirements(tuple(c for _, c in contracts.values()))):
        _reject(module, "protocol, entry, domains or resource requirements changed.")
    return actual


def plan_value_execution(module, target, contracts):
    placements = tuple(ExecutionPlacement(f, c, d.name, k.domain, k.implementation, len(k.projections))
                       for (f, c), (d, k) in contracts.items())
    plan = HeterogeneousPlan("heterogeneous", module.entry, target.heterogeneous_protocol, placements,
                            target.heterogeneous_resource_requirements(tuple(c for _, c in contracts.values())))
    if module.heterogeneous_plans:
        if len(module.heterogeneous_plans) != 1:
            _reject(module, "requires exactly one placement plan.")
        plan = module.heterogeneous_plans[0]
    selected = validate_placements(module, target, contracts, plan)
    domains = {call: place.domain for (_, call), place in selected.items()}
    function_calls = [n for n in module.nodes if n.op in {"builtin.call", "tir.call"} and n.attrs["callee"] in module.function_map]
    if target.heterogeneous_protocol == "k230.functions/v1":
        from .bufferize.graph import function_nodes
        owners = {name: {p.domain for p in selected.values() if p.source_function == name} for name in module.function_map}
        children = {function.name: {n.attrs["callee"] for n in function_nodes(module, function)
                                   if n.op in {"builtin.call", "tir.call"} and n.attrs["callee"] in owners}
                    for function in module.functions}
        for _ in range(len(owners)):
            for function, callees in children.items():
                for callee in callees:
                    owners[function].update(owners[callee])
        domains.update({call.id: next(iter(owners[call.attrs["callee"]])) if len(owners[call.attrs["callee"]]) == 1 else "cpu0"
                        for call in function_calls})
    users = {}
    for node in module.nodes:
        for operand in node.inputs:
            users.setdefault(operand, []).append(node.id)
        if (target.heterogeneous_protocol == "k230.functions/v1" and node.op in {"builtin.call", "tir.call"}
                and node.attrs["callee"] in module.function_map):
            callee = module.function_map[node.attrs["callee"]]
            for actual, formal in zip(node.inputs, callee.parameters):
                users.setdefault(actual, []).append(formal)
    exported = {value for function in module.functions for value in function.outputs}
    consumer_domains = {}

    def consumers(value, visited=frozenset()):
        if value in visited:
            _reject(module, "source value graph contains a cycle.")
        if value in consumer_domains:
            return consumer_domains[value]
        result = {"cpu0"} if value in exported else set()
        for user in users.get(value, ()):
            if user in domains:
                result.add(domains[user])
            else:
                result.update(consumers(user, visited | {value}))
        consumer_domains[value] = frozenset(result)
        return consumer_domains[value]

    nodes = []
    formals = {value for function in module.functions if function.name != module.entry for value in function.outputs}
    boundary_calls = {call.id for call in function_calls}
    for node in module.nodes:
        if node.id in domains:
            owner = domains[node.id]
            readers = consumers(node.id)
            space = ("k230_shared" if target.heterogeneous_protocol == "k230.functions/v1" and node.id in boundary_calls and owner == "kpu"
                     else "k230_values" if owner == "kpu" and readers <= {"kpu"}
                     else "k230_shared" if "kpu" in {owner, *readers} else None)
            if target.heterogeneous_protocol == "k230.functions/v1" and node.id in formals:
                space = None
            metadata = dict(node.metadata)
            previous = metadata.pop("bufferization.memory_space", None)
            if previous is not None and previous not in {"k230_values", "k230_shared", space}:
                _reject(module, "placement conflicts with an explicit buffer memory space.")
            if space is not None:
                metadata["bufferization.memory_space"] = space
            node = replace(node, metadata=metadata)
        nodes.append(node)
    return replace(module, nodes=tuple(nodes), heterogeneous_plans=(plan,))


def _buffer_plan(module):
    raw = module.metadata.get("buffer_plan")
    if raw is None:
        _reject(module, "requires the ordinary BufferPlan.")
    return raw if isinstance(raw, BufferPlan) else BufferPlan.from_data(raw)


def _accesses(module, buffers, function, call):
    result = []
    scratch = {b.actual for b in call.workspaces}
    reads, writes = set(call.reads) | scratch, set(call.writes) | scratch
    for name in sorted(reads | writes):
        descriptor = buffers.buffer_map.get(name)
        if descriptor is None:
            _reject(module, f"source access {name!r} has no BufferPlan descriptor.")
        span = descriptor.physical_access_span
        if not span.absolute_start.is_fixed or not span.size.is_fixed:
            _reject(module, "first value program requires static physical spans.")
        space = buffers.memory_space_map[span.buffer.memory_space]
        identity = (function + ":" + space.name if space.supports_lifetime_reuse
                    and space.allocation_scope.value == "function" else span.buffer.id)
        mode = "read_write" if name in reads and name in writes else "write" if name in writes else "read"
        value = Buffer(name, descriptor.dtype, span, descriptor.shape, descriptor.strides)
        result.append((value, SynchronizationRange(space.name, identity, span.absolute_start.fixed_value,
                                                   span.size.fixed_value, mode)))
    return tuple(result)


def _immutable_resolver(module, buffers):
    resources = {r["name"] for r in module.metadata.get("k230_static_program", {}).get("resources", ()) if r["read_only"]}
    source_entry = module.heterogeneous_plans[0].source_entry if module.heterogeneous_plans else module.entry
    callers = {}
    for source in module.execution_functions:
        for call in _fields(module, source.body):
            if isinstance(call, PrimFunctionCall):
                for binding in call.arguments:
                    callers.setdefault((call.callee, binding.formal), []).append((source.name, binding.actual))

    def immutable(function, name, seen=frozenset()):
        key = (function, name)
        if key in seen:
            _reject(module, "recursive immutable binding provenance.")
        descriptor = buffers.buffer_map[name]
        space = buffers.memory_space_map[descriptor.mem_span.buffer.memory_space]
        if space.kind == "host_readonly" or descriptor.role == "immediate":
            return True
        if function == source_entry and descriptor.source_node in resources:
            return True
        actuals = callers.get(key, ())
        return bool(actuals) and all(immutable(f, n, seen | {key}) for f, n in actuals)
    return immutable


def _argument_accesses(module, buffers, function, call):
    result = _accesses(module, buffers, function, call)
    present = {value.name for value, _ in result}
    missing = tuple(dict.fromkeys(b.actual for b in call.arguments if b.actual not in present))
    if missing:
        immutable = _immutable_resolver(module, buffers)
        if any(buffers.buffer_map[name].storage != "scalar" or not immutable(function, name) for name in missing):
            _reject(module, "an argument without memory effects must be a proven immutable scalar.")
        scalar_call = KernelInvoke(call.call_id, call.kernel, arguments=tuple(PrimCallBinding(n, n) for n in missing), reads=missing)
        result += _accesses(module, buffers, function, scalar_call)
    return result


def _overlap(left, right):
    return (left.storage == right.storage and left.physical_id == right.physical_id
            and max(left.offset, right.offset) < min(left.offset + left.nbytes, right.offset + right.nbytes))


def value_barrier_ranges(exchange, event):
    word = exchange.state_word.mem_span
    payload = exchange.producer_ranges if event == "publish" else exchange.consumer_ranges
    return (*payload, SynchronizationRange(word.buffer.memory_space, word.buffer.id,
        word.absolute_start.fixed_value, 4, "write" if event == "publish" else "read"))


@dataclass(frozen=True)
class _ValueDependency:
    producer: str
    consumer: str
    values: tuple[Buffer, ...]
    producer_ranges: tuple[SynchronizationRange, ...]
    consumer_ranges: tuple[SynchronizationRange, ...]


def _merge_dependency(left, right):
    return replace(left, values=tuple(dict.fromkeys((*left.values, *right.values))),
                   producer_ranges=tuple(dict.fromkeys((*left.producer_ranges, *right.producer_ranges))),
                   consumer_ranges=tuple(dict.fromkeys((*left.consumer_ranges, *right.consumer_ranges))))


def _reduce_value_dependencies(module, calls, domains, edges):
    names = [call.call_id for call in calls]
    positions = {name: i for i, name in enumerate(names)}
    ordered = {name: set() for name in names}
    for domain in dict.fromkeys(domains.values()):
        own = [name for name in names if domains[name] == domain]
        for left, right in zip(own, own[1:]):
            ordered[left].add(right)
    unique = {}
    for edge in edges:
        key = edge.producer, edge.consumer
        unique[key] = _merge_dependency(unique[key], edge) if key in unique else edge
    edges = tuple(unique.values())
    for edge in edges:
        ordered[edge.producer].add(edge.consumer)
    reachable = {name: 0 for name in names}
    for name in reversed(names):
        for consumer in ordered[name]:
            if positions[consumer] <= positions[name]:
                _reject(module, "value dependencies must follow the source call order.")
            reachable[name] |= (1 << positions[consumer]) | reachable[consumer]

    def reaches(left, right):
        return left == right or bool(reachable[left] & (1 << positions[right]))

    kept = [edge for edge in edges if not any(
        other != edge.consumer and reaches(other, edge.consumer) for other in ordered[edge.producer])]
    result = list(kept)
    for edge in edges:
        if edge in kept:
            continue
        # Route cache obligations too; an ordering path alone cannot replace their publication.
        index = next((i for i, target in enumerate(kept)
                      if domains[target.producer] == domains[edge.producer]
                      and reaches(edge.producer, target.producer) and reaches(target.consumer, edge.consumer)), None)
        if index is None:
            _reject(module, "reduced dependency has no same-direction payload route.")
        result[index] = _merge_dependency(result[index], edge)
    return tuple(result)


def _function_schedule(module, plan, source, buffers, counts, word_offset, immutable, reduce_synchronization):
    fields = _fields(module, source.body)
    if any(not isinstance(x, (KernelInvoke, PrimFunctionCall, Barrier)) for x in fields):
        _reject(module, "supports static kernel and reusable function calls only.")
    calls = [x for x in fields if isinstance(x, (KernelInvoke, PrimFunctionCall))]
    if any(x.call_id in {ENTRY, EXIT} for x in calls):
        _reject(module, "source call uses a reserved boundary identity.")
    placements = {p.call_id: p for p in plan.placements if p.source_function == source.name}
    domains = {key: p.domain for key, p in placements.items()}
    domains.update({call.call_id: plan.cpu_domain for call in calls if isinstance(call, PrimFunctionCall)})
    domains.update({ENTRY: plan.cpu_domain, EXIT: plan.cpu_domain})
    entry_values = tuple(name for name in source.parameters if name in buffers.buffer_map
                         and not immutable(source.name, name))
    calls = [KernelInvoke(ENTRY, ENTRY, results=tuple(PrimCallBinding(n, n) for n in entry_values), writes=entry_values), *calls,
             KernelInvoke(EXIT, EXIT, arguments=tuple(PrimCallBinding(n, n) for n in source.results), reads=source.results)]
    accesses = {call.call_id: _accesses(module, buffers, source.name, call) for call in calls}
    if source.name != plan.source_entry:
        for name in (*entry_values, *source.results):
            boundary = _accesses(module, buffers, source.name, KernelInvoke(
                ENTRY, ENTRY, arguments=(PrimCallBinding(name, name),), reads=(name,)))[0][1]
            touching = [call for call in calls[1:-1]
                        if any(_overlap(boundary, r) for _, r in accesses[call.call_id])]
            if touching and ((name in entry_values and domains[touching[0].call_id] != plan.cpu_domain)
                             or (name in source.results and domains[touching[-1].call_id] != plan.cpu_domain)):
                _reject(module, "reusable mutable boundaries require CPU ownership; device-owned call boundaries are not implemented.")
    for call in calls:
        if call.call_id in {ENTRY, EXIT}:
            continue
        for value, _ in accesses[call.call_id]:
            space = value.mem_span.buffer.memory_space
            if isinstance(call, KernelInvoke) and domains[call.call_id] == plan.cpu_domain and space == "k230_values":
                _reject(module, "CPU call cannot access device-private storage; regenerate placement-dependent buffers.")
            if domains[call.call_id] == plan.device_domain and space == "workspace":
                _reject(module, "device call cannot access private CPU workspace.")
    positions = {call.call_id: index for index, call in enumerate(calls)}
    explicit = set()
    for call in calls:
        for predecessor in call.dependencies:
            explicit.add((predecessor, call.call_id))
    for barrier in (x for x in fields if isinstance(x, Barrier)):
        if barrier.scope not in {BarrierScope.BLOCK, BarrierScope.CHIP} or barrier.axis_group_axes:
            _reject(module, "requires single-owner BLOCK or CHIP source synchronization.")
        explicit.update((before, barrier.before) for before in barrier.after)
    for before, after in explicit:
        if before not in positions or after not in positions or positions[before] >= positions[after]:
            _reject(module, "source dependency is unresolved or not acyclic.")
    resource = next((r for r in plan.resources if r.name == "values.glb_words"), None)
    if resource is None:
        _reject(module, "has no declared shared GLB word resource.")
    size = resource.type.shape[0].fixed_value
    words = PhysicalBuffer(resource.name, resource.memory_space, size, resource.alignment)
    crossing_edges, local_edges = [], []
    for index, left in enumerate(calls):
        for right in calls[index + 1:]:
            pairs = [(a, ar, b, br) for a, ar in accesses[left.call_id] for b, br in accesses[right.call_id]
                     if _overlap(ar, br) and (ar.mode != "read" or br.mode != "read")]
            needed = (bool(pairs) or (left.call_id, right.call_id) in explicit
                      or source_call_conflicts(module, source.name, left, right, buffer_plan=buffers))
            if not needed:
                continue
            if domains[left.call_id] == domains[right.call_id]:
                if left.call_id != ENTRY and right.call_id != EXIT:
                    local_edges.append((left.call_id, right.call_id, tuple(dict.fromkeys(p[3] for p in pairs))))
                continue
            for composite, peer in ((left, right), (right, left)):
                if isinstance(composite, PrimFunctionCall):
                    # Postorder callee validation proves its CPU return joins all device work.
                    if composite is left and composite.callee in counts:
                        continue
                    for pool in composite.memory_pools:
                        if any(r.storage == pool.memory_space and max(pool.offset, r.offset) < min(pool.offset + pool.nbytes, r.offset + r.nbytes)
                               for _, r in accesses[peer.call_id]):
                            _reject(module, f"cross-domain caller pool reuse {composite.call_id}/{peer.call_id}/{pool.memory_space} needs a device-owned frame summary; regenerate disjoint pools.")
            values = tuple(dict.fromkeys(value for pair in pairs for value in (pair[0], pair[2])))
            for value in values:
                storage = value.mem_span.buffer.memory_space
                if storage not in {"k230_shared", "external", "rdata"}:
                    _reject(module, f"cross-device value {value.name!r} is not in shared storage ({storage}).")
            crossing_edges.append(_ValueDependency(left.call_id, right.call_id, values,
                tuple(dict.fromkeys(p[1] for p in pairs)), tuple(dict.fromkeys(p[3] for p in pairs))))
    if reduce_synchronization:
        crossing_edges = _reduce_value_dependencies(module, calls, domains, crossing_edges)
    exchanges = []
    for sequence, edge in enumerate(crossing_edges, 1):
        if (word_offset + sequence) * 4 > size:
            _reject(module, "crossing edges exceed the allocated shared-word capacity.")
        word = Buffer(f"edge{sequence}.word", "int32", MemSpan(words, (word_offset + sequence - 1) * 4, 4), (1,), (1,))
        exchanges.append(ValueExchange(f"{source.name}.edge{sequence}", source.name,
            edge.producer, edge.consumer, domains[edge.producer], domains[edge.consumer], edge.values, word,
            edge.producer_ranges, edge.consumer_ranges))
    bodies = {plan.cpu_domain: [], plan.device_domain: []}
    child_offset = len(exchanges)
    for call in calls:
        body = bodies[domains[call.call_id]]
        for sequence, edge in enumerate(exchanges, 1):
            if edge.consumer == call.call_id:
                body.append(ChannelBarrier(edge.name + ".acquire", edge.name, "acquire", sequence, (),
                                           value_barrier_ranges(edge, "acquire")))
        predecessors = [(before, ranges) for before, after, ranges in local_edges if after == call.call_id]
        if reduce_synchronization and predecessors:
            body.append(Barrier(BarrierScope.BLOCK, tuple(dict.fromkeys(before for before, _ in predecessors)),
                                call.call_id, ("source_order",), tuple(dict.fromkeys(r for _, ranges in predecessors for r in ranges))))
        elif predecessors:
            for before, ranges in predecessors:
                body.append(Barrier(BarrierScope.BLOCK, (before,), call.call_id, ("source_order",), ranges))
        if call.call_id not in {ENTRY, EXIT}:
            if isinstance(call, PrimFunctionCall):
                for domain in bodies:
                    bodies[domain].append(WorkerCall(replace(call, callee=call.callee + "." + domain), child_offset))
                child_offset += counts[call.callee]
                if child_offset >= 2 ** 32:
                    _reject(module, "expanded invocation ticket count exceeds uint32.")
            else:
                body.append(call)
        for sequence, edge in enumerate(exchanges, 1):
            if edge.producer == call.call_id:
                body.append(ChannelBarrier(edge.name + ".publish", edge.name, "publish", sequence, (),
                                           value_barrier_ranges(edge, "publish")))
    workers = []
    for domain, body in bodies.items():
        attrs = dict(source.attrs)
        owned = {name for call in calls if domains[call.call_id] == domain and call.call_id not in {ENTRY, EXIT}
                 for name in call.writes}
        for key in ("written_parameters", "transfer_source_parameters"):
            if key in attrs:
                attrs[key] = tuple(name for name in attrs[key] if name in owned)
        function = ExecutionFunction(source.name + "." + domain, source.parameters, source.results,
                                     Sequential(tuple(body)), attrs)
        workers.append(WorkerFunction(source.name, domain, function, child_offset))
    if source.name != plan.source_entry:
        # A caller can reuse its pools only after its CPU role observes all device work.
        observed = {edge.producer for edge in exchanges if edge.consumer_domain == plan.cpu_domain}
        for left, right, _ in reversed(local_edges):
            if right in observed:
                observed.add(left)
        joined = False
        for call in reversed(calls):
            if domains[call.call_id] == plan.device_domain:
                joined = joined or call.call_id in observed
                if joined:
                    observed.add(call.call_id)
        if any(domains[call.call_id] == plan.device_domain and call.call_id not in observed for call in calls):
            _reject(module, "reusable function has device work not observed before CPU return.")
    return workers, exchanges, child_offset


def value_schedule(module, plan):
    """Project physical order into two reusable worker call graphs."""
    policy = module.metadata.get("k230_value_synchronization", "transitive_payload_v1")
    if policy not in ("all_dependencies_v1", "transitive_payload_v1"):
        _reject(module, "k230_value_synchronization must be all_dependencies_v1 or transitive_payload_v1.")
    reduce_synchronization = policy == "transitive_payload_v1"
    buffers = _buffer_plan(module)
    immutable = _immutable_resolver(module, buffers)
    order, visited, active = [], set(), set()

    def visit(name):
        if name in active or name not in module.execution_function_map:
            _reject(module, "requires a closed acyclic execution call graph.")
        if name in visited:
            return
        active.add(name)
        source = module.execution_function_map[name]
        for call in _fields(module, source.body):
            if isinstance(call, PrimFunctionCall):
                visit(call.callee)
        active.remove(name)
        visited.add(name)
        order.append(source)

    # Function dumps retain the complete physical closure beside one graph fragment.
    for source in module.execution_functions:
        visit(source.name)
    workers, exchanges, counts = [], [], {}
    for source in order:
        roles, edges, count = _function_schedule(module, plan, source, buffers, counts, len(exchanges), immutable,
                                               reduce_synchronization)
        workers.extend(roles)
        exchanges.extend(edges)
        counts[source.name] = count
    entries = (plan.source_entry + "." + plan.cpu_domain, plan.source_entry + "." + plan.device_domain)
    return HeterogeneousProgram("heterogeneous", plan.name, tuple(workers), tuple(exchanges),
                                HeterogeneousLaunch(*entries), HeterogeneousDrain(entries))


def verify_value_program(module, program, plan):
    if any(p.projections for p in plan.placements):
        _reject(module, "cannot mix descriptor phases with direct value kernels.")
    expected = value_schedule(module, plan)
    if program != expected:
        _reject(module, "worker calls, typed edges, synchronization or ownership differ from the source value schedule.")


def form_value_execution(module, target, contracts):
    plan, = module.heterogeneous_plans
    validate_placements(module, target, contracts, plan)
    program = value_schedule(module, plan)
    result = replace(module, heterogeneous_plans=(replace(plan, source_hash=heterogeneous_source_hash(module)),),
                     heterogeneous_programs=(program,))
    from triton.flagmega.ir.tir.heterogeneous import verify_heterogeneous_programs
    verify_heterogeneous_programs(result)
    target.verify_heterogeneous_program(result)
    return result
