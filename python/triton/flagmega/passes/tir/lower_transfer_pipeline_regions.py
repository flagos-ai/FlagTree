# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Lower typed transfer contracts to explicit producer/consumer TIR regions."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, replace

from triton.flagmega.errors import IRVerificationError
from triton.flagmega.ir.model import IRModule
from triton.flagmega.ir import verify_buffer_plan
from triton.flagmega.ir.tir import (
    Block,
    Barrier,
    synchronization_scope_name,
    BufferLoad,
    BufferStore,
    For,
    IfThenElse,
    KernelDispatch,
    Let,
    PipelineDrain,
    PipelineHandoff,
    PipelineStage,
    PrimFunction,
    PrimFunctionCall,
    KernelInvoke,
    ProducerConsumerRegion,
    Sequential,
    ExecutionFunction,
    execution_calls_of,
    iter_tir_children,
)


@dataclass(frozen=True)
class _ByteRange:
    arena: tuple[str, str | None]
    start: int
    end: int


@dataclass(frozen=True)
class _Owner:
    statement: object
    stage: PipelineStage | None
    ranges: tuple[_ByteRange, ...]


def lower_transfer_pipeline_regions(module: IRModule) -> IRModule:
    """Materialize every selected transfer pipeline after Bufferize.

    KernelInvoke operations share their containing execution function's
    region. KernelDefinition contracts never receive regions or lifetimes.
    Explicitly authored/legacy PrimFunctions with straight-line dispatches
    retain support, with the same Shared drain/handoff rules. A pipeline below
    structured control flow requires explicit task control-flow lowering.
    """

    host_functions = program_host_functions(module)
    if host_functions:
        verify_program_publication(module, host_functions)
    pipeline_functions = _pipeline_execution_functions(module)
    repeated = _repeated_execution_functions(module)
    execution_functions = tuple(
        _lower_execution_function(
            function, module, pipeline_functions,
            repeated=function.name in repeated,
        )
        if function.name in pipeline_functions and function.name not in host_functions
        else function
        for function in module.execution_functions
    )
    functions = tuple(_lower_function(function) for function in module.prim_functions)
    if (
        all(before is after for before, after in zip(module.prim_functions, functions))
        and all(
            before is after
            for before, after in zip(
                module.execution_functions, execution_functions
            )
        )
    ):
        return module
    return replace(
        module,
        prim_functions=functions,
        execution_functions=execution_functions,
    )


def _pipeline_execution_functions(module: IRModule) -> frozenset[str]:
    host_functions = program_host_functions(module)
    result = {
        function.name
        for function in module.execution_functions
        if any(
            _call_has_direct_transfer_pipeline(call, module)
            for call in execution_calls_of(function)
        )
    }
    changed = True
    while changed:
        changed = False
        for function in module.execution_functions:
            if function.name in result or function.name in host_functions:
                continue
            if any(
                call.callee in result
                for call in execution_calls_of(function)
            ):
                result.add(function.name)
                changed = True
    return frozenset(result - host_functions)


def program_host_functions(module):
    """Host control scopes cannot inherit a callee's CTA pipeline lifetime."""
    from triton.flagmega.ir.program_execution import verify_program_execution, ProgramInvocation
    plan = verify_program_execution(module)
    if plan is None:
        return frozenset()
    return frozenset(function.function for function in plan.functions
        if len(function.steps) != 1 or not isinstance(function.steps[0], ProgramInvocation))


def verify_program_publication(module, host_functions):
    """Check actual frame effects at ordered, chip-global launch boundaries."""
    from triton.flagmega.ir.program_execution import verify_program_execution, ProgramInvocation
    from triton.flagmega.ir.memory_effect import MemoryAccessMode
    from triton.flagmega.ir.bufferization import MemorySharingScope
    from triton.flagmega.passes.tir.bufferize.call_accesses import CallAccessResolver
    from triton.flagmega.passes.tir.bufferize.synchronization import _node_accesses
    plan = verify_buffer_plan(module)
    execution = verify_program_execution(module)
    requests = {p['symbol']: p for p in module.metadata['kernel_program_partition']}
    if any('depends_on' in p for p in requests.values()):
        raise IRVerificationError('Host pipeline publication requires the verified sequential program stream.')
    launch = module.metadata.get('launch_contract', {}).get('grid_mesh', {})
    if any(size != 1 and level in ('n', 'c') for size, level in
           zip(launch.get('hierarchy', ()), launch.get('hierarchy_levels', ''), strict=True)):
        raise IRVerificationError('Chip-local launch completion cannot publish a peer-device pipeline source.')
    resolver = CallAccessResolver(module, plan, _node_accesses)
    for function in execution.functions:
        if function.function not in host_functions:
            continue
        bindings = dict(plan.function_map[function.function].values)
        history = []
        for ordinal, step in enumerate(function.steps):
            identities = requests[step.symbol]['calls'] if isinstance(step, ProgramInvocation) else (step.call,)
            accesses = []
            for identity in identities:
                # A uniform native loop is an indivisible typed program item.
                node = module.node_map.get(identity)
                if node is None:
                    node = next((n for n in module.nodes if n.op == 'control.for'
                                 and identity == f'{function.function}.{n.id}_iv'), None)
                if node is None:
                    raise IRVerificationError('Program publication has no physical executable item.')
                accesses.extend(resolver.accesses(function.function, node, bindings))
            _verify_publication_accesses(accesses, history, function.function, ordinal, plan)
            history.extend((ordinal, access) for access in accesses)


def _verify_publication_accesses(accesses, history, function, ordinal, plan):
    """A retired physical frame is dead only after complete ordered writes."""
    from triton.flagmega.ir.memory_effect import MemoryAccessMode
    from triton.flagmega.ir.bufferization import MemorySharingScope
    initialized = []
    for access in accesses:
        for before, writer in history if access.effect.physical_mode & MemoryAccessMode.READ else ():
            if (before == ordinal or writer.physical_id != access.physical_id
                    or not writer.effect.physical_mode & MemoryAccessMode.WRITE
                    or max(writer.offset, access.offset) >= min(writer.offset + writer.nbytes, access.offset + access.nbytes)):
                continue
            begin, end = max(writer.offset, access.offset), min(writer.offset + writer.nbytes, access.offset + access.nbytes)
            cursor = begin
            for lo, hi in sorted((value.offset, value.offset + value.nbytes) for value in initialized
                    if value.physical_id == access.physical_id):
                if lo > cursor:
                    break
                cursor = max(cursor, hi)
            if cursor >= end:
                # Earlier writes in this native invocation completely
                # initialize the reused byte span. A retired frame's
                # old value is no longer live at this read.
                continue
            if access.sharing_scope is not MemorySharingScope.CHIP or writer.sharing_scope is not MemorySharingScope.CHIP:
                raise IRVerificationError(f'A Shared/BLOCK/private value crosses a native program publication boundary: '
                    f'@{function} step{before} {writer.node}/{writer.buffer} '
                    f'{writer.sharing_scope.value} [{writer.offset},{writer.offset+writer.nbytes}) -> '
                    f'step{ordinal} {access.node}/{access.buffer} {access.sharing_scope.value} '
                    f'[{access.offset},{access.offset+access.nbytes}), role={plan.buffer_map[access.buffer].role}.')
        if access.effect.physical_mode & MemoryAccessMode.WRITE:
            initialized.append(access)


def native_program_function(module, function, request):
    """Lower one verified contiguous native program with a fresh pipe arena."""
    owned = set(request['calls'])
    selected, pending = [], []
    for statement in function.body.fields:
        if isinstance(statement, Barrier):
            pending.append(statement)
            continue
        identity = (f'{function.name}.{statement.call_id}_iv' if isinstance(statement, For)
                    else getattr(statement, 'call_id', None))
        if identity in owned:
            selected.extend((*pending, statement))
        pending = []
    if not selected:
        raise IRVerificationError('Native program has no complete physical call span.')
    local = replace(function, body=Sequential(tuple(selected)))
    return _lower_execution_function(local, module, _pipeline_execution_functions(module), repeated=False)


def _repeated_execution_functions(module: IRModule) -> frozenset[str]:
    """Name every function that executes more than once per launch.

    A loop body is repeated, and so is anything it calls: ``decode_layer``
    owns the stage arenas but runs once per trip because the body calls it.
    Arena reuse therefore wraps around for these functions even though their
    own bodies are straight-line.
    """

    result = {
        function.name
        for function in module.execution_functions
        if function.attrs.get("region_kind") == "for"
    }
    # A structured loop names its body as a callee, so a body reached only
    # through a For in another function is repeated too.
    for function in module.execution_functions:
        for loop in _structured_loop_bodies(function):
            result.add(loop)
    changed = True
    while changed:
        changed = False
        for function in module.execution_functions:
            if function.name not in result:
                continue
            for call in execution_calls_of(function):
                if call.callee not in result and any(
                    value.name == call.callee
                    for value in module.execution_functions
                ):
                    result.add(call.callee)
                    changed = True
    return frozenset(result)


def _structured_loop_bodies(function) -> tuple[str, ...]:
    result = []

    def visit(node) -> None:
        if isinstance(node, For) and node.body_function:
            result.append(node.body_function)
        for child in iter_tir_children(node):
            visit(child)

    visit(function.body)
    return tuple(result)


def _call_has_direct_transfer_pipeline(call, module) -> bool:
    primitive = module.kernel_callable_map.get(call.callee)
    if primitive is None:
        return False
    dispatch = _single_kernel_dispatch(primitive)
    return dispatch is not None and _transfer_pipeline(dispatch) is not None


def _single_kernel_dispatch(function):
    dispatches = tuple(
        child
        for child in iter_tir_children(function.body)
        if isinstance(child, KernelDispatch)
    )
    if len(dispatches) == 1:
        return dispatches[0]
    # ``iter_tir_children`` does not yield the root field itself.
    direct = tuple(
        value for value in function.body.fields
        if isinstance(value, KernelDispatch)
    )
    return direct[0] if len(direct) == 1 else None


def _lower_execution_function(
    function: ExecutionFunction,
    module: IRModule,
    pipeline_functions: frozenset[str],
    *,
    repeated: bool = False,
) -> ExecutionFunction:
    from triton.flagmega.passes.tir.bufferize.synchronization import transfer_source_dependencies

    if _contains_region(function.body):
        raise IRVerificationError(
            f"ExecutionFunction @{function.name} already contains a "
            "producer/consumer region."
        )
    calls = _straight_line_calls(
        function.body, function.name, module, pipeline_functions
    )
    execution_order = _execution_order(function.body)
    pipeline_calls = tuple(
        call
        for call in calls
        if _is_pipeline_call(call, module, pipeline_functions)
    )
    if not pipeline_calls:
        return function
    stages = {
        id(call): PipelineStage(
            f"{function.name}_transfer_stage_{index}", call
        )
        for index, call in enumerate(pipeline_calls)
    }
    owners = _execution_shared_owners(
        execution_order, stages, module, pipeline_functions, function.name
    )
    drains, consumer_after, producer_before = _shared_synchronization(
        owners, function.name,
        # A repeated function's arena users wrap around: the next execution's
        # first producer refills bytes this one's last consumer still reads.
        repeated=repeated,
    )
    plan = verify_buffer_plan(module) if "buffer_plan" in module.metadata else None
    source_spans = (
        {} if plan is None
        else {value.id: value.physical_access_span for value in plan.buffers}
    )
    source_barrier_indices = _add_execution_source_handoffs(
        execution_order,
        stages,
        consumer_after,
        producer_before,
        function.name,
        source_spans,
        None if plan is None else transfer_source_dependencies(module, plan, function.name, calls),
    )
    consumer = _rewrite_consumer(
        function.body, stages, drains, consumer_after
    )
    producer = _build_producer(
        function.body, stages, drains, producer_before,
        source_barrier_paths=_source_publication_paths(
            function.body, execution_order, stages, source_spans,
            source_barrier_indices, repeated=repeated,
        ),
    )
    return replace(
        function,
        body=Sequential((ProducerConsumerRegion(
            producer, consumer
        ),)),
    )


def _straight_line_calls(
    body: Sequential,
    function_name: str,
    module: IRModule,
    pipeline_functions: frozenset[str],
):
    _validate_structured_pipeline_placement(
        body,
        lambda call: (
            isinstance(call, (PrimFunctionCall, KernelInvoke))
            and _is_pipeline_call(call, module, pipeline_functions)
        ),
        function_name,
        "ExecutionFunction",
    )
    return tuple(
        statement
        for statement in _execution_order(body)
        if isinstance(statement, (PrimFunctionCall, KernelInvoke))
    )


def _is_pipeline_call(call, module, pipeline_functions) -> bool:
    return (
        call.callee in pipeline_functions
        or _call_has_direct_transfer_pipeline(call, module)
    )


def _execution_shared_owners(
    execution_order,
    stages,
    module,
    pipeline_functions,
    function_name,
):
    owners = []
    for statement in execution_order:
        stage = stages.get(id(statement))
        buffers = _statement_shared_buffers(statement)
        if stage is not None:
            call = statement
            primitive = module.kernel_callable_map.get(call.callee)
            dispatch = (
                None if primitive is None else _single_kernel_dispatch(primitive)
            )
            pipeline = None if dispatch is None else _transfer_pipeline(dispatch)
            if pipeline is not None:
                buffers = tuple(
                    buffers[index]
                    for index in pipeline.shared_workspace_indices
                )
            elif call.callee not in pipeline_functions:
                raise IRVerificationError(
                    f"Pipeline call {call.call_id!r} in @{function_name} has "
                    "no direct or interprocedural transfer contract."
                )
            if not buffers:
                raise IRVerificationError(
                    f"Transfer-pipeline stage {stage.stage_id!r} in "
                    f"@{function_name} has no caller-owned Shared workspace "
                    "after Bufferize."
                )
        ranges = tuple(
            _fixed_shared_range(buffer, function_name) for buffer in buffers
        )
        if ranges:
            owners.append(_Owner(statement, stage, ranges))
    return tuple(owners)


def _add_execution_source_handoffs(
    execution_order,
    stages,
    consumer_after,
    producer_before,
    function_name,
    source_spans,
    source_requirements=None,
):
    publication_indices = set()
    handoff_index = sum(len(values) for values in consumer_after.values())
    call_indices = {value.call_id: index for index, value in enumerate(execution_order)
                    if isinstance(value, (PrimFunctionCall, KernelInvoke))}
    for index, statement in enumerate(execution_order):
        stage = stages.get(id(statement))
        call = statement if isinstance(statement, (PrimFunctionCall, KernelInvoke)) else None
        if stage is None or not call.transfer_sources:
            continue
        sources = set(call.transfer_sources)
        requirements = (
            tuple((predecessor.call_id, ("block", (), None))
                  for predecessor in execution_order[:index]
                  if isinstance(predecessor, (PrimFunctionCall, KernelInvoke))
                  and _writes_transfer_source(sources, _statement_writes(predecessor), source_spans))
            if source_requirements is None else source_requirements[call.call_id]
        )
        release_index = None
        for predecessor_id, requirement in requirements:
            predecessor_index = call_indices[predecessor_id]
            predecessor = execution_order[predecessor_index]
            if id(predecessor) in stages and requirement[0] == "block":
                ready = predecessor_index
            else:
                publication = _publication_barrier_indices(
                    execution_order, predecessor_index + 1, index, requirement
                )
                ready = None if publication is None else publication[-1]
                if ready is None:
                    scope = "Block" if requirement[0] == "block" else "Chip"
                    raise IRVerificationError(
                        f"Transfer-pipeline stage {stage.stage_id!r} in @{function_name} "
                        f"reads a source after {predecessor_id!r} without an effective "
                        f"{scope} barrier (source publication barrier required)."
                    )
                publication_indices.update(publication)
            release_index = ready if release_index is None else max(release_index, ready)
        if release_index is not None:
            handoff = PipelineHandoff(f"{function_name}_source_handoff_{handoff_index}")
            handoff_index += 1
            _append_unique(consumer_after[id(execution_order[release_index])], handoff)
            _append_unique(producer_before[id(call)], handoff)
    return frozenset(publication_indices)


def _first_publication(order, start, end, requirement):
    indices = _publication_barrier_indices(order, start, end, requirement)
    return None if indices is None else indices[-1]


def _publication_barrier_indices(order, start, end, requirement):
    """Select exact occurrences proving the existing first publication point.

    Barrier ranges describe the consumer hazard that created each barrier;
    their scope can publish other sources too. Keep the scope proof, including
    multiple axis-group barriers when one alone does not cover the source.
    """
    from triton.flagmega.passes.tir.bufferize.barrier_coverage import BarrierCoverage

    def covers(indices):
        coverage = BarrierCoverage()
        for index in indices:
            barrier = order[index]
            scope = synchronization_scope_name(barrier.scope)
            coverage.apply((scope, barrier.axis_group_axes, requirement[2]), None)
        return coverage.covers(requirement)

    candidates = []
    for index in range(start, end):
        if not isinstance(order[index], Barrier):
            continue
        candidates.append(index)
        if covers(candidates):
            # Drop redundant earlier scopes without moving the handoff. The
            # last barrier is necessary: no preceding prefix covered the edge.
            for candidate in tuple(candidates[:-1]):
                without = [value for value in candidates if value != candidate]
                if covers(without):
                    candidates = without
            return tuple(candidates)
    return None


def _writes_transfer_source(sources, writes, spans):
    if sources.intersection(writes):
        return True
    # Views/subspans retain their own logical IDs. The physical span, not
    # name equality, determines whether the producer observes a prior write.
    return any(spans[source].may_alias(spans[written])
               for source in sources if source in spans
               for written in writes if written in spans)


def _lower_function(function: PrimFunction) -> PrimFunction:
    if _contains_region(function.body):
        raise IRVerificationError(
            f"PrimFunction @{function.name} already contains a producer/consumer region."
        )
    dispatches = _straight_line_dispatches(function.body, function.name)
    execution_order = _execution_order(function.body)
    pipeline_dispatches = tuple(
        dispatch for dispatch in dispatches if _transfer_pipeline(dispatch) is not None
    )
    if not pipeline_dispatches:
        return function

    stages = {
        id(dispatch): PipelineStage(
            f"{function.name}_transfer_stage_{index}", dispatch
        )
        for index, dispatch in enumerate(pipeline_dispatches)
    }
    owners = _shared_owners(execution_order, stages, function.name)
    drains, consumer_after, producer_before = _shared_synchronization(
        owners, function.name
    )
    _add_source_handoffs(
        execution_order,
        stages,
        consumer_after,
        producer_before,
        function.name,
    )
    consumer = _rewrite_consumer(
        function.body, stages, drains, consumer_after
    )
    producer = _build_producer(
        function.body, stages, drains, producer_before
    )
    region = ProducerConsumerRegion(
        producer,
        consumer,
    )
    return replace(function, body=Sequential((region,)))


def _straight_line_dispatches(
    body: Sequential, function_name: str
) -> tuple[KernelDispatch, ...]:
    _validate_structured_pipeline_placement(
        body,
        lambda dispatch: (
            isinstance(dispatch, KernelDispatch)
            and _transfer_pipeline(dispatch) is not None
        ),
        function_name,
        "PrimFunction",
    )
    return tuple(
        statement
        for statement in _execution_order(body)
        if isinstance(statement, KernelDispatch)
    )


def _contains_region(statement) -> bool:
    if isinstance(statement, ProducerConsumerRegion):
        return True
    return any(_contains_region(child) for child in iter_tir_children(statement))


def _transfer_pipeline(dispatch: KernelDispatch):
    selection = dispatch.microkernel
    return None if selection is None else selection.transfer_pipeline


def _shared_owners(
    execution_order: tuple[object, ...],
    stages: dict[int, PipelineStage],
    function_name: str,
) -> tuple[_Owner, ...]:
    owners = []
    for statement in execution_order:
        stage = stages.get(id(statement))
        pipeline = (
            _transfer_pipeline(statement)
            if isinstance(statement, KernelDispatch)
            else None
        )
        if stage is None:
            buffers = _statement_shared_buffers(statement)
        else:
            buffers = tuple(
                statement.shared_workspace_buffers[index]
                for index in pipeline.shared_workspace_indices
            )
            if not buffers:
                raise IRVerificationError(
                    f"Transfer-pipeline stage {stage.stage_id!r} in "
                    f"@{function_name} has no physical Shared workspace after Bufferize."
                )
        ranges = tuple(_fixed_shared_range(buffer, function_name) for buffer in buffers)
        if ranges:
            owners.append(_Owner(statement, stage, ranges))
    return tuple(owners)


def _fixed_shared_range(buffer, function_name: str) -> _ByteRange:
    physical = buffer.mem_span.buffer
    if physical.memory_space != "shared" or physical.function != function_name:
        raise IRVerificationError(
            f"Shared resource {buffer.name!r} in @{function_name} has no "
            "function-owned post-Bufferize MemSpan."
        )
    try:
        start = buffer.mem_span.absolute_start.fixed_value
        end = buffer.mem_span.absolute_end.fixed_value
    except ValueError as error:
        raise IRVerificationError(
            f"Shared resource {buffer.name!r} in @{function_name} has no fixed "
            "post-Bufferize byte range."
        ) from error
    if end <= start:
        raise IRVerificationError(
            f"Shared resource {buffer.name!r} in @{function_name} has empty "
            f"range [{start}, {end})."
        )
    return _ByteRange((physical.memory_space, physical.function), start, end)


def _shared_synchronization(
    owners: tuple[_Owner, ...], function_name: str, *, repeated: bool = False
):
    boundaries: dict[tuple[str, str | None], set[int]] = defaultdict(set)
    for owner in owners:
        for span in owner.ranges:
            boundaries[span.arena].update((span.start, span.end))
    segments = {
        arena: tuple(zip(sorted(values), sorted(values)[1:]))
        for arena, values in boundaries.items()
    }
    frontier: dict[tuple[tuple[str, str | None], int], int] = {}
    drains: set[str] = set()
    consumer_after: dict[int, list[PipelineHandoff]] = defaultdict(list)
    producer_before: dict[int, list[PipelineHandoff]] = defaultdict(list)
    handoffs: dict[tuple[int, int], PipelineHandoff] = {}

    # A loop body's owners repeat, so the last writer of an arena segment is
    # the predecessor of the body's first user on the next trip.  Walking the
    # sequence once leaves that pair unprotected: the next trip's producer
    # starts filling bytes the previous trip's consumer is still reading.
    # Running the sequence twice, keeping only the second pass's results,
    # seeds the frontier with the wrap-around predecessors.
    passes = (owners, owners) if repeated else (owners,)
    for index, sequence in enumerate(passes):
        final = index == len(passes) - 1
        if not final:
            # Dry run: establish the frontier the body inherits from the
            # previous trip, without recording its own synchronization.
            for owner_index, owner in enumerate(sequence):
                for span in owner.ranges:
                    for segment_index, (start, end) in enumerate(segments[span.arena]):
                        if span.start >= end or start >= span.end:
                            continue
                        frontier[(span.arena, segment_index)] = owner_index
            continue
        for owner_index, owner in enumerate(sequence):
            predecessors: dict[int, int] = {}
            for span in owner.ranges:
                for segment_index, (start, end) in enumerate(segments[span.arena]):
                    if span.start >= end or start >= span.end:
                        continue
                    key = (span.arena, segment_index)
                    predecessor = frontier.get(key)
                    # Within one execution an owner cannot race itself, but a
                    # repeated one does: the next execution refills the bytes
                    # this one is still reading, so it is its own predecessor
                    # across the boundary.
                    if predecessor is not None and (
                        predecessor != owner_index or repeated
                    ):
                        predecessors[predecessor] = min(
                            predecessors.get(predecessor, start), start
                        )
                    frontier[key] = owner_index
            for predecessor_index, offset in predecessors.items():
                predecessor = owners[predecessor_index]
                # A repeated function cannot use a drain for any arena
                # hazard.  One pipe instance serves every execution, so its
                # close sits after the loop, while `tle.pipe.drain` needs
                # that close in the same async task and in both tasks at
                # once -- so the drain gets hoisted out and stops protecting
                # the reuse it was created for.  A handoff is a per-execution
                # commit/wait/release triple, which is the right shape.  A
                # wrap-around pair additionally depends on the previous
                # execution, so it is primed before the loop.
                wrapped = repeated and predecessor_index >= owner_index
                if predecessor.stage is not None and not repeated:
                    drains.add(predecessor.stage.stage_id)
                elif owner.stage is not None:
                    pair = (predecessor_index, owner_index)
                    handoff = handoffs.setdefault(
                        pair,
                        PipelineHandoff(
                            f"{function_name}_shared_handoff_{len(handoffs)}_at_{offset}",
                            wrap_around=wrapped,
                        ),
                    )
                    _append_unique(consumer_after[id(predecessor.statement)], handoff)
                    _append_unique(producer_before[id(owner.statement)], handoff)
    return drains, consumer_after, producer_before


def _add_source_handoffs(
    execution_order,
    stages,
    consumer_after,
    producer_before,
    function_name,
) -> None:
    handoff_index = sum(len(values) for values in consumer_after.values())
    for index, statement in enumerate(execution_order):
        stage = stages.get(id(statement))
        dispatch = statement if isinstance(statement, KernelDispatch) else None
        if stage is None:
            continue
        pipeline = _transfer_pipeline(dispatch)
        sources = {
            dispatch.arguments[source_index]
            for source_index in pipeline.read_argument_indices
        }
        # Resolve each source separately. A set-level intersection says only
        # that *some* source is written here, so stopping at the first hit
        # leaves every source whose writer sits further back unresolved -- and
        # an ordinary (unsynchronized) write to one of those is exactly the
        # violation below, silently unreported. Scanning backwards per source
        # keeps the nearest writer of each, which is the one that determines
        # that source's readiness.
        release_index = None
        for source in sorted(sources):
            for offset in reversed(range(index)):
                predecessor = execution_order[offset]
                if source not in _statement_writes(predecessor):
                    continue
                if stages.get(id(predecessor)) is None:
                    raise IRVerificationError(
                        f"Transfer-pipeline stage {stage.stage_id!r} in "
                        f"@{function_name} reads a source after an ordinary "
                        "write; a first-class block barrier is required "
                        "before task handoff."
                    )
                # One edge orders them all: the consumer body is straight-line,
                # so a commit after the latest stage predecessor already
                # implies every earlier one has run. A handoff is a real
                # mbarrier rendezvous, so emitting one per intersecting source
                # would add redundant waits carrying no extra ordering.
                release_index = (
                    offset if release_index is None
                    else max(release_index, offset)
                )
                break
        if release_index is not None:
            handoff = PipelineHandoff(
                f"{function_name}_source_handoff_{handoff_index}"
            )
            handoff_index += 1
            _append_unique(
                consumer_after[id(execution_order[release_index])], handoff
            )
            _append_unique(producer_before[id(dispatch)], handoff)


def _append_unique(values: list[PipelineHandoff], value: PipelineHandoff) -> None:
    if value.handoff_id not in {item.handoff_id for item in values}:
        values.append(value)


def _rewrite_consumer(body, stages, drains, consumer_after) -> Sequential:
    fields = []
    for field in body.fields:
        if isinstance(field, Sequential):
            rewritten = _rewrite_consumer(field, stages, drains, consumer_after)
        elif isinstance(field, For):
            # A structured loop is mirrored on both sides of the region: the
            # pipe is built once at function scope and each iteration performs
            # one push/pop pair, so the loop keeps its consumer body here.
            rewritten = replace(field, body=_rewrite_consumer(
                field.body, stages, drains, consumer_after))
        else:
            rewritten = stages.get(id(field), field)
        fields.append(rewritten)
        stage = stages.get(id(field))
        if stage is not None and stage.stage_id in drains:
            fields.append(PipelineDrain(stage.stage_id, arena_reuse=True))
        fields.extend(consumer_after.get(id(field), ()))
    return replace(body, fields=tuple(fields))


def _producer_needs_source_barriers(execution_order, stages, source_spans) -> bool:
    """Retain partition acquire points only when a producer source can change.

    Include all writers, not only predecessors: a loop's trailing writer can
    precede the first transfer on its next trip. Physical subspans also cover
    aliases and arena reuse that logical binding names do not express.
    """
    sources = {
        source
        for statement in execution_order
        if id(statement) in stages
        for source in statement.transfer_sources
    }
    return any(
        _writes_transfer_source(sources, _statement_writes(statement), source_spans)
        for statement in execution_order
    )


def _source_publication_paths(
    body, execution_order, stages, source_spans, publication_indices, *, repeated
):
    """Use predecessor proof only where it covers the source lifetime.

    Current source requirements do not describe loop backedges. Keep the
    existing participation for repeated mutable functions and counted loops;
    pruning those needs a separate backedge proof. A later write in an acyclic
    schedule is not a predecessor of the earlier source read.
    """
    mutable = _producer_needs_source_barriers(execution_order, stages, source_spans)
    cyclic = repeated or any(isinstance(value, For) for value in _descendants(body))
    occurrences = _execution_occurrences(body)
    if mutable and cyclic:
        return frozenset(path for path, statement in occurrences
                         if isinstance(statement, Barrier))
    return frozenset(occurrences[index][0] for index in publication_indices)


def _build_producer(
    body, stages, drains, producer_before, *, keep_source_barriers=True,
    source_barrier_paths=None, path=(),
) -> Sequential:
    fields = []
    for index, field in enumerate(body.fields):
        field_path = (*path, index)
        fields.extend(producer_before.get(id(field), ()))
        stage = stages.get(id(field))
        if stage is not None:
            fields.append(stage)
            if stage.stage_id in drains:
                fields.append(PipelineDrain(stage.stage_id, arena_reuse=True))
            continue
        if isinstance(field, Barrier):
            # Producer global reads acquire the selected source-publication
            # scopes. Slot reuse remains guarded by unchanged drains/handoffs.
            # Legacy PrimFunctions retain their existing barrier membership.
            keep = (keep_source_barriers if source_barrier_paths is None
                    else field_path in source_barrier_paths)
            if keep:
                fields.append(field)
            continue
        if isinstance(field, Sequential):
            nested = _build_producer(
                field, stages, drains, producer_before,
                keep_source_barriers=keep_source_barriers,
                source_barrier_paths=source_barrier_paths, path=field_path,
            )
            if nested.fields:
                fields.append(nested)
        elif isinstance(field, For):
            # Mirror the loop so the producer issues its stages once per
            # iteration, matching the consumer's pops exactly.
            nested = _build_producer(
                field.body, stages, drains, producer_before,
                keep_source_barriers=keep_source_barriers,
                source_barrier_paths=source_barrier_paths, path=(*field_path, -1),
            )
            if nested.fields:
                fields.append(replace(field, body=nested))
    return replace(body, fields=tuple(fields))


def _execution_occurrences(body: Sequential, path=()):
    """Stable pre-rewrite paths: Sequential field indices, -1 for For.body.

    No names, value equality, or object identity define a barrier occurrence.
    Both selection and rewriting use the unchanged input schedule tree.
    """
    result = []
    for index, statement in enumerate(body.fields):
        field_path = (*path, index)
        if isinstance(statement, Sequential):
            result.extend(_execution_occurrences(statement, field_path))
        elif isinstance(statement, For) and statement.domain.step is not None:
            result.extend(_execution_occurrences(statement.body, (*field_path, -1)))
        else:
            result.append((field_path, statement))
    return tuple(result)


def _execution_order(body: Sequential) -> tuple[object, ...]:
    """Flatten Sequential and counted loops; other structures stay atomic."""
    return tuple(statement for _, statement in _execution_occurrences(body))


def _validate_structured_pipeline_placement(
    body: Sequential,
    is_pipeline_operation,
    function_name: str,
    function_kind: str,
) -> None:
    for statement in _structural_expressions(body):
        if isinstance(statement, For) and statement.domain.step is not None:
            # A counted loop executes its body a proven number of times, so a
            # stage inside it keeps push/pop paired once the region mirrors the
            # loop. Conditional and scoped forms below have no such guarantee.
            continue
        if not isinstance(statement, (For, IfThenElse, Let, Block)):
            continue
        if any(
            is_pipeline_operation(child)
            for child in _descendants(statement)
        ):
            raise IRVerificationError(
                f"{function_kind} @{function_name} contains a transfer-pipeline "
                f"stage under {type(statement).__name__}; producer/consumer "
                "lowering requires a straight-line stage order."
            )


def _structural_expressions(statement):
    yield statement
    for child in iter_tir_children(statement):
        yield from _structural_expressions(child)


def _descendants(statement):
    for child in iter_tir_children(statement):
        yield child
        yield from _descendants(child)


def _statement_calls(statement) -> tuple[PrimFunctionCall, ...]:
    if isinstance(statement, (PrimFunctionCall, KernelInvoke)):
        return (statement,)
    return tuple(
        child
        for child in _descendants(statement)
        if isinstance(child, (PrimFunctionCall, KernelInvoke))
    )


def _statement_dispatches(statement) -> tuple[KernelDispatch, ...]:
    if isinstance(statement, KernelDispatch):
        return (statement,)
    return tuple(
        child
        for child in _descendants(statement)
        if isinstance(child, KernelDispatch)
    )


def _statement_shared_buffers(statement) -> tuple:
    operations = (*_statement_calls(statement), *_statement_dispatches(statement))
    result = []
    seen = set()
    for operation in operations:
        for buffer in operation.shared_workspace_buffers:
            identity = (
                buffer.mem_span.buffer.id,
                str(buffer.mem_span.absolute_start),
                str(buffer.mem_span.absolute_end),
            )
            if identity not in seen:
                seen.add(identity)
                result.append(buffer)
    return tuple(result)


def _statement_writes(statement) -> tuple[str, ...]:
    operations = (*_statement_calls(statement), *_statement_dispatches(statement))
    return tuple(dict.fromkeys(
        value for operation in operations for value in operation.writes
    ))


def _statement_call_ids(statement) -> tuple[str, ...]:
    return tuple(call.call_id for call in _statement_calls(statement))


__all__ = ["lower_transfer_pipeline_regions"]
