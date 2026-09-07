# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Lower typed transfer contracts to explicit producer/consumer TIR regions."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, replace

from triton.flagmega.errors import IRVerificationError
from triton.flagmega.ir.model import IRModule
from triton.flagmega.ir.tir import (
    Block,
    Barrier,
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

    pipeline_functions = _pipeline_execution_functions(module)
    execution_functions = tuple(
        _lower_execution_function(function, module, pipeline_functions)
        if function.name in pipeline_functions
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
            if function.name in result:
                continue
            if any(
                call.callee in result
                for call in execution_calls_of(function)
            ):
                result.add(function.name)
                changed = True
    return frozenset(result)


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
) -> ExecutionFunction:
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
        owners, function.name
    )
    _add_execution_source_handoffs(
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
):
    handoff_index = sum(len(values) for values in consumer_after.values())
    for index, statement in enumerate(execution_order):
        stage = stages.get(id(statement))
        call = statement if isinstance(statement, (PrimFunctionCall, KernelInvoke)) else None
        if stage is None or not call.transfer_sources:
            continue
        sources = set(call.transfer_sources)
        for predecessor_index in range(index - 1, -1, -1):
            predecessor = execution_order[predecessor_index]
            if not sources.intersection(_statement_writes(predecessor)):
                continue
            predecessor_stage = stages.get(id(predecessor))
            if predecessor_stage is None:
                between = execution_order[predecessor_index + 1:index]
                barrier = next((
                    value for value in reversed(between)
                    if isinstance(value, Barrier)
                    and value.before == call.call_id
                    and value.scope.value in {"block", "chip"}
                ), None)
                if barrier is None:
                    raise IRVerificationError(
                        f"Transfer-pipeline stage {stage.stage_id!r} in "
                        f"@{function_name} reads a source after an ordinary "
                        "write without an effective Block barrier."
                    )
                handoff = PipelineHandoff(
                    f"{function_name}_source_handoff_{handoff_index}"
                )
                handoff_index += 1
                _append_unique(consumer_after[id(barrier)], handoff)
                _append_unique(producer_before[id(call)], handoff)
                break
            handoff = PipelineHandoff(
                f"{function_name}_source_handoff_{handoff_index}"
            )
            handoff_index += 1
            _append_unique(consumer_after[id(predecessor)], handoff)
            _append_unique(producer_before[id(call)], handoff)
            break


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
    owners: tuple[_Owner, ...], function_name: str
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

    for owner_index, owner in enumerate(owners):
        predecessors: dict[int, int] = {}
        for span in owner.ranges:
            for segment_index, (start, end) in enumerate(segments[span.arena]):
                if span.start >= end or start >= span.end:
                    continue
                key = (span.arena, segment_index)
                predecessor = frontier.get(key)
                if predecessor is not None and predecessor != owner_index:
                    predecessors[predecessor] = min(
                        predecessors.get(predecessor, start), start
                    )
                frontier[key] = owner_index
        for predecessor_index, offset in predecessors.items():
            predecessor = owners[predecessor_index]
            if predecessor.stage is not None:
                drains.add(predecessor.stage.stage_id)
            elif owner.stage is not None:
                pair = (predecessor_index, owner_index)
                handoff = handoffs.setdefault(
                    pair,
                    PipelineHandoff(
                        f"{function_name}_shared_handoff_{len(handoffs)}_at_{offset}"
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
            for source_index in pipeline.source_argument_indices
        }
        for predecessor in reversed(execution_order[:index]):
            if not sources.intersection(_statement_writes(predecessor)):
                continue
            predecessor_stage = stages.get(id(predecessor))
            if predecessor_stage is None:
                raise IRVerificationError(
                    f"Transfer-pipeline stage {stage.stage_id!r} in "
                    f"@{function_name} reads a source after an ordinary write; "
                    "a first-class block barrier is required before task handoff."
                )
            handoff = PipelineHandoff(
                f"{function_name}_source_handoff_{handoff_index}"
            )
            handoff_index += 1
            _append_unique(consumer_after[id(predecessor)], handoff)
            _append_unique(producer_before[id(dispatch)], handoff)
            break


def _append_unique(values: list[PipelineHandoff], value: PipelineHandoff) -> None:
    if value.handoff_id not in {item.handoff_id for item in values}:
        values.append(value)


def _rewrite_consumer(body, stages, drains, consumer_after) -> Sequential:
    fields = []
    for field in body.fields:
        if isinstance(field, Sequential):
            rewritten = _rewrite_consumer(field, stages, drains, consumer_after)
        else:
            rewritten = stages.get(id(field), field)
        fields.append(rewritten)
        stage = stages.get(id(field))
        if stage is not None and stage.stage_id in drains:
            fields.append(PipelineDrain(stage.stage_id))
        fields.extend(consumer_after.get(id(field), ()))
    return replace(body, fields=tuple(fields))


def _build_producer(body, stages, drains, producer_before) -> Sequential:
    fields = []
    for field in body.fields:
        fields.extend(producer_before.get(id(field), ()))
        stage = stages.get(id(field))
        if stage is not None:
            fields.append(stage)
            if stage.stage_id in drains:
                fields.append(PipelineDrain(stage.stage_id))
            continue
        if isinstance(field, Sequential):
            nested = _build_producer(field, stages, drains, producer_before)
            if nested.fields:
                fields.append(nested)
    return replace(body, fields=tuple(fields))


def _execution_order(body: Sequential) -> tuple[object, ...]:
    """Flatten only Sequential; structured statements remain atomic owners."""

    result = []
    for statement in body.fields:
        if isinstance(statement, Sequential):
            result.extend(_execution_order(statement))
        else:
            result.append(statement)
    return tuple(result)


def _validate_structured_pipeline_placement(
    body: Sequential,
    is_pipeline_operation,
    function_name: str,
    function_kind: str,
) -> None:
    for statement in _structural_expressions(body):
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
