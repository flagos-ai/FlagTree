# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Route proved source publication through a readonly producer prefix.

This pass changes only the location of one existing producer-side wait. It
does not relax transfer-source declarations, publication barriers or drains.
"""

from dataclasses import replace

from triton.flagmega.errors import IRVerificationError
from triton.flagmega.ir.tir import (
    Barrier, KernelInvoke, PipelineDrain, PipelineHandoff, PipelineReadiness,
    PipelineStage, PrimFunctionCall, ProducerConsumerRegion, Sequential,
    execution_calls_of, iter_tir_children,
)
from triton.flagmega.ir.bufferization import BufferPlan


def _region(function):
    fields = function.body.fields
    if len(fields) != 1 or not isinstance(fields[0], ProducerConsumerRegion):
        raise IRVerificationError("Imported readiness requires one explicit pipeline region.")
    result = fields[0]
    allowed = (Barrier, PipelineDrain, PipelineHandoff, PipelineStage)
    if any(not isinstance(x, allowed) for x in result.produce_body.fields):
        raise IRVerificationError("Imported readiness requires straight-line producer control flow.")
    return result


def _readonly_buffers(module, plan):
    """Close readonly storage through aliases and every actual/formal edge."""
    result = {b.id for b in plan.buffers
              if plan.memory_space_map[b.mem_span.buffer.memory_space].kind == "readonly_device"}
    actuals = {}
    for function in module.execution_functions:
        for call in execution_calls_of(function):
            if call.callee in module.execution_function_map:
                for edge in call.arguments:
                    actuals.setdefault(edge.formal, []).append(edge.actual)
    changed = True
    while changed:
        old = len(result)
        for formal, bound in actuals.items():
            if bound and all(value in result for value in bound):
                result.add(formal)
        for buffer in plan.buffers:
            if buffer.alias_of in result:
                result.add(buffer.id)
        changed = len(result) != old
    return frozenset(result)


def _target(module, call, readonly):
    plan = BufferPlan.from_data(module.metadata["buffer_plan"])
    for edge in call.arguments:
        actual = plan.buffer_map[edge.actual]
        if actual.offset_bindings:
            raise IRVerificationError("Imported readiness requires static caller argument addresses.")
        if not actual.shape and actual.id not in readonly:
            source = module.node_map.get(actual.source_node or actual.id)
            if source is None or source.op not in {"builtin.scalar_const", "tir.scalar_const"}:
                raise IRVerificationError(
                    "Imported readiness cannot move a mutable scalar argument evaluation before publication.")
    callee = module.execution_function_map.get(call.callee)
    if callee is None:
        raise IRVerificationError("Imported readiness cannot target a leaf kernel.")
    region = _region(callee)
    prefix = []
    first_mutable = None
    for statement in region.produce_body.fields:
        if isinstance(statement, PipelineHandoff) and statement.wrap_around:
            raise IRVerificationError("Imported readiness does not support repeated handoffs.")
        if not isinstance(statement, PipelineStage):
            continue
        leaf = statement.operation
        if not isinstance(leaf, KernelInvoke) or statement.source_readiness is not None:
            raise IRVerificationError("Imported readiness requires direct, non-nested leaf transfer stages.")
        if not leaf.transfer_sources:
            raise IRVerificationError("A producer prefix must declare its complete transfer-source set.")
        if all(value in readonly for value in leaf.transfer_sources):
            prefix.append(leaf)
            continue
        first_mutable = leaf
        break
    if not prefix or first_mutable is None:
        raise IRVerificationError("Imported readiness requires a readonly prefix followed by a mutable transfer.")
    # Declared immutable sources must remain unwritten throughout the callee.
    sources = {source for leaf in prefix for source in leaf.transfer_sources}
    if any(readonly.intersection(leaf.writes) for leaf in execution_calls_of(callee)):
        raise IRVerificationError("A readonly producer prefix aliases a callee write.")
    return first_mutable.call_id, tuple(value for value in callee.parameters if value in readonly)


def _publication(module, plan, function, region, stage, handoff):
    from .bufferize.synchronization import transfer_source_dependencies
    from .lower_transfer_pipeline_regions import _publication_barrier_indices

    order = tuple(x.operation if isinstance(x, PipelineStage) else x
                  for x in region.consume_body.fields)
    commits = [i for i, value in enumerate(order)
               if isinstance(value, PipelineHandoff) and value.handoff_id == handoff]
    if len(commits) != 1 or order[commits[0]].wrap_around:
        raise IRVerificationError("Imported readiness requires exactly one local one-shot commit.")
    commit = commits[0]
    calls = execution_calls_of(function)
    indices = {x.call_id: i for i, x in enumerate(order)
               if isinstance(x, (PrimFunctionCall, KernelInvoke))}
    if stage.operation.call_id not in indices or commit >= indices[stage.operation.call_id]:
        raise IRVerificationError("Imported readiness commit must precede its consumer invocation.")
    requirements = transfer_source_dependencies(module, plan, function.name, calls)[stage.operation.call_id]
    if not requirements:
        raise IRVerificationError("Imported readiness has no mutable source publication dependency.")
    stages = {x.operation.call_id for x in region.consume_body.fields if isinstance(x, PipelineStage)}
    for predecessor, requirement in requirements:
        before = indices[predecessor]
        if before >= commit:
            raise IRVerificationError("Imported readiness commits before a required source write.")
        if predecessor in stages and requirement[0] == "block":
            continue
        if _publication_barrier_indices(order, before + 1, commit, requirement) is None:
            raise IRVerificationError("Imported readiness lacks its required source publication barrier.")


def verify_pipeline_readiness(module):
    contracts = []
    for function in module.execution_functions:
        def visit(node):
            if isinstance(node, PipelineStage) and node.source_readiness is not None:
                contracts.append((function, node))
                return
            for child in iter_tir_children(node):
                visit(child)
        visit(function.body)
    if not contracts:
        return
    plan = BufferPlan.from_data(module.metadata["buffer_plan"])
    readonly = _readonly_buffers(module, plan)
    from .lower_transfer_pipeline_regions import _repeated_execution_functions
    repeated = _repeated_execution_functions(module)
    checked = set()
    for function, stage in contracts:
        if function.name in repeated or stage.operation.callee in repeated:
            raise IRVerificationError("Imported readiness requires non-repeated invocation lifetimes.")
        region = _region(function)
        binding = stage.source_readiness
        target, parameters = _target(module, stage.operation, readonly)
        if (binding.before_call_id, binding.readonly_parameters) != (target, parameters):
            raise IRVerificationError("Imported readiness target or readonly formal closure differs from its proof.")
        occurrences = [x for x in (*region.produce_body.fields, *region.consume_body.fields)
                       if isinstance(x, PipelineStage) and x.stage_id == stage.stage_id]
        if len(occurrences) != 2 or any(x != stage for x in occurrences):
            raise IRVerificationError("Both call roles must carry the same imported readiness contract.")
        key = (function.name, stage.stage_id)
        if key not in checked:
            _publication(module, plan, function, region, stage, binding.handoff_id)
            checked.add(key)


def route_pipeline_source_readiness(module, *, callees=()):
    """Opt in selected callees after ordinary Bufferize/transfer lowering.

    An empty selection is a strict identity. Unsupported selected edges fail
    explicitly; calls without an incoming source handoff remain unchanged.
    """
    if not callees:
        return module
    selected = frozenset(callees)
    missing = selected - module.execution_function_map.keys()
    if missing:
        raise IRVerificationError(f"Unknown readiness callees: {sorted(missing)}")
    plan = BufferPlan.from_data(module.metadata["buffer_plan"])
    readonly = _readonly_buffers(module, plan)
    functions = []
    count = 0
    for function in module.execution_functions:
        if not (len(function.body.fields) == 1
                and isinstance(function.body.fields[0], ProducerConsumerRegion)):
            functions.append(function)
            continue
        region = function.body.fields[0]
        producer = list(region.produce_body.fields)
        replacements, removed = {}, set()
        for index, stage in enumerate(producer):
            if not isinstance(stage, PipelineStage) or stage.operation.callee not in selected:
                continue
            if stage.source_readiness is not None:
                continue
            if index == 0 or not isinstance(producer[index - 1], PipelineHandoff):
                continue
            handoff = producer[index - 1]
            if handoff.wrap_around:
                raise IRVerificationError("Cannot route a wrap-around publication as one-shot readiness.")
            target, parameters = _target(module, stage.operation, readonly)
            _publication(module, plan, function, region, stage, handoff.handoff_id)
            binding = PipelineReadiness(handoff.handoff_id, target, parameters)
            replacements[stage.stage_id] = replace(stage, source_readiness=binding)
            removed.add(index - 1)
            count += 1
        if replacements:
            def rewrite(values):
                return Sequential(tuple(replacements.get(x.stage_id, x)
                                        if isinstance(x, PipelineStage) else x for x in values))
            region = ProducerConsumerRegion(
                rewrite([x for i, x in enumerate(producer) if i not in removed]),
                rewrite(region.consume_body.fields))
            function = replace(function, body=Sequential((region,)))
        functions.append(function)
    if not count:
        verify_pipeline_readiness(module)
        return module
    result = replace(module, execution_functions=tuple(functions))
    verify_pipeline_readiness(result)
    return result


__all__ = ["route_pipeline_source_readiness", "verify_pipeline_readiness"]
