"""Scoped prototype: prove device completion at physical hazards and return.

The normal physical DAG verifier and all materialized device barriers remain
authoritative. This policy rejects a graph whose actual execution schedule
cannot prove every cross-chip byte hazard before it changes completion flags.
"""
from functools import wraps

from triton.flagmega.errors import CodegenError
from triton.flagmega.ir import verify_buffer_plan
from triton.flagmega.ir.program_order import program_ancestors
from triton.flagmega.ir.tir.barrier import Barrier, BarrierScope
from triton.flagmega.ir.tir.kernel_invoke import KernelInvoke
from triton.flagmega.ir.tir.prim_function_call import PrimFunctionCall
from triton.flagmega.passes.tir.bufferize.call_accesses import CallAccessResolver
from triton.flagmega.passes.tir.bufferize.synchronization import (
    _node_accesses, _requires_device, _conflicts,
)


def completion_proof(module, programs, eager):
    """Validate actual instantiated execution before choosing terminal epochs."""
    ancestors = program_ancestors(programs)
    if ancestors is None:
        raise CodegenError("Deferred completion requires an explicit program DAG.")
    peers = [p["symbol"] for p in programs if eager[p["symbol"]]]
    completion = {p["symbol"]: False for p in programs}
    if not peers:
        return completion, dict(peer_programs=0, terminal_programs=(), device_hazard_pairs=0)
    last = peers[-1]
    if any(p not in ancestors[last] for p in peers[:-1]):
        raise CodegenError("Deferred completion requires an ordered device epoch chain.")
    completion[last] = True
    plan = verify_buffer_plan(module)
    resolver = CallAccessResolver(module, plan, _node_accesses)
    functions = {f.name: f for f in module.execution_functions}
    if module.entry not in functions:
        raise CodegenError("Deferred completion requires materialized execution functions.")
    cache, active = {}, set()

    def expand(name):
        if name in cache:
            return cache[name]
        if name in active:
            raise CodegenError("Deferred completion cannot summarize recursive execution.")
        active.add(name)
        result = []
        bindings = dict(plan.function_map[name].values)
        for statement in functions[name].body.fields:
            if isinstance(statement, Barrier):
                if statement.scope is BarrierScope.DEVICE:
                    result.append(("device", name + ":before:" + statement.before, ()))
                continue
            if not isinstance(statement, (PrimFunctionCall, KernelInvoke)):
                raise CodegenError("Deferred completion requires straight-line execution regions.")
            node = module.node_map[statement.call_id]
            call = plan.call_map.get(statement.call_id)
            if call is None:
                result.append(("access", statement.call_id,
                               resolver.accesses(name, node, bindings)))
            else:
                if call.loop_carries:
                    raise CodegenError("Deferred completion requires explicit loop-copy epochs.")
                for kind, identity, accesses in expand(call.callee):
                    result.append((kind, identity, resolver.instantiate(name, node, accesses)
                                   if accesses else ()))
        active.remove(name)
        cache[name] = tuple(result)
        return cache[name]

    owner = {c: p["symbol"] for p in programs for c in p["calls"]}
    events = {p["symbol"]: [] for p in programs}
    prior = []
    entry = module.entry
    bindings = dict(plan.function_map[entry].values)
    actual_calls = []
    for statement in functions[entry].body.fields:
        if isinstance(statement, Barrier):
            if statement.scope is BarrierScope.DEVICE:
                prior.append(("device", entry + ":before:" + statement.before, ()))
            continue
        if not isinstance(statement, (PrimFunctionCall, KernelInvoke)):
            raise CodegenError("Deferred completion requires straight-line entry regions.")
        identity = statement.call_id
        if identity not in owner:
            raise CodegenError("Deferred completion found an execution call outside the program DAG.")
        actual_calls.append(identity)
        symbol = owner[identity]
        events[symbol].extend(prior)
        prior.clear()
        node = module.node_map[identity]
        call = plan.call_map.get(identity)
        if call is None:
            events[symbol].append(("access", identity, resolver.accesses(entry, node, bindings)))
        else:
            if call.loop_carries:
                raise CodegenError("Deferred completion requires explicit loop-copy epochs.")
            for kind, identity, accesses in expand(call.callee):
                events[symbol].append((kind, identity, resolver.instantiate(entry, node, accesses)
                                       if accesses else ()))
    if prior or actual_calls != [c for p in programs for c in p["calls"]]:
        raise CodegenError("Deferred completion requires a complete ordered execution partition.")

    # Independent dynamic grids cannot execute a device-wide rendezvous.
    # Close their preceding peer epoch in its latest ordered peer ancestor;
    # the normal range proof must still discharge the original leading fence.
    for program in programs:
        if 'launch_recipe' not in program:
            continue
        leading = []
        for event in events[program['symbol']]:
            if event[0] != 'device':
                break
            leading.append(event)
        predecessors = [peer for peer in peers if peer in ancestors[program['symbol']]]
        if leading and predecessors:
            completion[predecessors[-1]] = True

    pending, original_barriers, leaves, checked = [], 0, 0, 0
    for program in programs:
        symbol = program["symbol"]
        for kind, identity, accesses in events[symbol]:
            if kind == "device":
                original_barriers += 1
                pending = [p for p in pending if p[0] != symbol and p[0] not in ancestors[symbol]]
                continue
            leaves += 1
            for old_symbol, old_identity, previous in pending:
                for current in accesses:
                    if not (_requires_device(previous) or _requires_device(current)):
                        continue
                    checked += 1
                    if _conflicts(previous, current):
                        raise CodegenError(
                            "Deferred completion has an uncovered device memory hazard: "
                            f"{old_symbol}:{old_identity}:{previous.buffer} -> "
                            f"{symbol}:{identity}:{current.buffer}."
                        )
            pending.extend((symbol, identity, a) for a in accesses)
        if completion[symbol]:
            pending = [p for p in pending if p[0] != symbol and p[0] not in ancestors[symbol]]
    if any(_requires_device(a) for _, _, a in pending):
        raise CodegenError("Deferred completion leaves an unfinished peer effect at return.")
    return completion, dict(peer_programs=len(peers),
        terminal_programs=tuple(p['symbol'] for p in programs if completion[p['symbol']]),
        original_device_barriers=original_barriers, leaf_calls=leaves,
        device_hazard_pairs=checked, uncovered_hazards=0, pending_peer_effects_at_return=0)


def activate():
    """Install an explicit source-identified policy in this trial process only."""
    from triton.flagmega.codegen.triton import program_dependencies
    original = program_dependencies.verify_program_dependencies
    if getattr(original, "_deferred_completion_trial", False):
        return
    cached = [None, None, None]

    @wraps(original)
    def verified(module, programs, runtime_binding, *, with_device_completion=False,
                 with_accesses=False):
        writes, eager, accesses = original(module, programs, runtime_binding, with_accesses=True)
        if cached[0] is not module or cached[1] != programs:
            completion, proof = completion_proof(module, programs, eager)
            cached[:] = [module, programs, (completion, proof)]
        completion, proof = cached[2]
        verified.last_proof = proof
        if with_accesses:
            return writes, completion, accesses
        return (writes, completion) if with_device_completion else writes

    verified._deferred_completion_trial = True
    program_dependencies.verify_program_dependencies = verified
