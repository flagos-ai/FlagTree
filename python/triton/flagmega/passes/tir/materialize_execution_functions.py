# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Lower graph call order to first-class bufferized execution functions."""

from __future__ import annotations

from dataclasses import replace

from triton.flagmega.errors import IRVerificationError
from triton.flagmega.ir import (
    ExecutionFunction,
    KernelDefinition,
    KernelInvoke,
    IRModule,
    MemoryPoolFrame,
    MemSpan,
    PrimCallBinding,
    PrimFunctionCall,
    Sequential,
    kernel_dispatch_of,
    verify_buffer_plan,
)
from triton.flagmega.passes.functions.graph import (
    callee_first_functions,
    function_nodes,
)


def materialize_execution_functions(module: IRModule) -> IRModule:
    """Create function-level op/call schedules fixed by Bufferize.

    This is deliberately a TIR pass rather than a codegen descriptor builder.
    Every call stores its concrete caller/callee buffer edges and caller-owned
    Shared views, so synchronization and transfer-pipeline passes operate on
    editable IR and codegen only renders that IR.
    """

    if module.execution_functions:
        raise IRVerificationError(
            "MaterializeExecutionFunctions requires an unmaterialized module; "
            "resume from its output stage to edit an existing schedule.",
            stage=module.stage,
        )
    plan = verify_buffer_plan(module)
    result: dict[str, ExecutionFunction] = {}
    for graph_function in callee_first_functions(module):
        function_plan = plan.function_map[graph_function.name]
        values = dict(function_plan.values)
        nested = {value.call: value for value in function_plan.calls}
        kernel_workspaces = {
            value.call: value for value in function_plan.kernel_calls
        }
        nodes = function_nodes(module, graph_function)
        executable_ids = frozenset(
            node.id
            for node in nodes
            if node.id in nested or _kernel_primitive(module, node) is not None
        )
        dependencies = _execution_dependencies(nodes, executable_ids)
        calls: list[PrimFunctionCall | KernelInvoke] = []
        for node in nodes:
            primitive = _kernel_primitive(module, node)
            dispatch = None if primitive is None else kernel_dispatch_of(primitive)
            if primitive is not None and dispatch is not None:
                calls.append(_kernel_call(
                    module,
                    graph_function.name,
                    node,
                    primitive,
                    dispatch,
                    values,
                    kernel_workspaces.get(node.id),
                    dependencies[node.id],
                ))
                continue
            nested_call = nested.get(node.id)
            if nested_call is not None:
                try:
                    callee = result[nested_call.callee]
                except KeyError as error:
                    raise IRVerificationError(
                        f"ExecutionFunction @{graph_function.name} calls "
                        f"unmaterialized @{nested_call.callee}.",
                        stage=module.stage,
                        node_id=node.id,
                    ) from error
                calls.append(_nested_call(
                    graph_function.name,
                    node,
                    nested_call,
                    callee,
                    dependencies[node.id],
                ))

        parameters = tuple(
            buffer_id
            for _, buffer_ids in function_plan.parameters
            for buffer_id in buffer_ids
        )
        outputs = tuple(
            buffer_id
            for _, buffer_ids in function_plan.outputs
            for buffer_id in buffer_ids
        )
        written = {value for call in calls for value in call.writes}
        sources = {value for call in calls for value in call.transfer_sources}
        result[graph_function.name] = ExecutionFunction(
            graph_function.name,
            parameters,
            outputs,
            Sequential(tuple(calls)),
            attrs={
                **dict(graph_function.attrs),
                "written_parameters": tuple(
                    value for value in parameters if value in written
                ),
                "transfer_source_parameters": tuple(
                    value for value in parameters if value in sources
                ),
            },
        )
    return replace(
        module,
        # Kernel contracts survive, executable per-op functions do not. This
        # happens before synchronization/transfer lowering so only the real
        # function body can establish a P/C region or drain boundary.
        prim_functions=tuple(value for value in module.prim_functions if kernel_dispatch_of(value) is None),
        kernel_definitions=(*module.kernel_definitions, *(KernelDefinition(
            value.name, value.module_kind, value.parameters, kernel_dispatch_of(value),
            value.results, value.attrs,
        ) for value in module.prim_functions if kernel_dispatch_of(value) is not None)),
        execution_functions=tuple(
            result[function.name] for function in module.functions
        ),
    )


def _kernel_call(
    module,
    function_name,
    node,
    primitive,
    dispatch,
    values,
    workspace_record,
    dependencies,
) -> KernelInvoke:
    arguments: list[PrimCallBinding] = []
    actuals_by_parameter: dict[str, tuple[str, ...]] = {}
    for parameter, actual_value in zip(
        primitive.runtime_parameters, node.inputs, strict=True
    ):
        actual_buffers = (
            ()
            if parameter.role.value == "metadata"
            else tuple(values.get(actual_value, ()))
        )
        if len(actual_buffers) != len(parameter.buffers):
            raise IRVerificationError(
                f"Kernel call {node.id!r} parameter {parameter.name!r} has "
                f"{len(actual_buffers)} actual buffers, expected "
                f"{len(parameter.buffers)}.",
                stage=module.stage,
                node_id=node.id,
            )
        actuals_by_parameter[parameter.name] = actual_buffers
        arguments.extend(
            PrimCallBinding(formal.name, actual)
            for formal, actual in zip(
                parameter.buffers, actual_buffers, strict=True
            )
        )

    result_actuals = tuple(values.get(node.id, ()))
    result_formals = tuple(
        buffer
        for parameter in primitive.output_parameters
        for buffer in parameter.buffers
    )
    if len(result_actuals) != len(result_formals):
        raise IRVerificationError(
            f"Kernel call {node.id!r} result buffer arity differs from "
            f"@{primitive.name}.",
            stage=module.stage,
            node_id=node.id,
        )
    results = tuple(
        PrimCallBinding(formal.name, actual)
        for formal, actual in zip(result_formals, result_actuals, strict=True)
    )
    cursor = 0
    for parameter in primitive.output_parameters:
        count = len(parameter.buffers)
        actuals_by_parameter[parameter.name] = result_actuals[cursor:cursor + count]
        cursor += count

    workspace_map = (
        {} if workspace_record is None else dict(workspace_record.workspaces)
    )
    workspaces: list[PrimCallBinding] = []
    for parameter in primitive.workspaces:
        parameter_actuals = []
        for formal in parameter.buffers:
            try:
                actual = workspace_map[parameter.name]
            except KeyError as error:
                raise IRVerificationError(
                    f"Kernel call {node.id!r} has no workspace binding for "
                    f"{parameter.name!r}.",
                    stage=module.stage,
                    node_id=node.id,
                ) from error
            workspaces.append(PrimCallBinding(formal.name, actual))
            parameter_actuals.append(actual)
        actuals_by_parameter[parameter.name] = tuple(parameter_actuals)

    def resolve_effect(names) -> tuple[str, ...]:
        return tuple(dict.fromkeys(
            actual
            for name in names
            for actual in actuals_by_parameter.get(name, ())
        ))

    transfer_sources: list[str] = []
    pipeline = (
        None if dispatch.microkernel is None
        else dispatch.microkernel.transfer_pipeline
    )
    if pipeline is not None:
        for channel in pipeline.channels:
            for argument_index in channel.source_argument_indices:
                parameter_name = dispatch.arguments[argument_index]
                transfer_sources.extend(actuals_by_parameter[parameter_name])

    shared = tuple(
        _caller_shared_buffer(function_name, node.id, value)
        for value in dispatch.shared_workspace_buffers
    )
    return KernelInvoke(
        call_id=node.id,
        kernel=primitive.name,
        arguments=tuple(arguments),
        results=results,
        workspaces=tuple(workspaces),
        shared_workspace_buffers=shared,
        transfer_sources=tuple(dict.fromkeys(transfer_sources)),
        reads=resolve_effect(dispatch.reads),
        writes=resolve_effect(dispatch.writes),
        effect_kind=node.effect.kind.value,
        effect_resource=node.effect.resource,
        dependencies=dependencies,
    )


def _nested_call(
    function_name, node, record, callee, dependencies
) -> PrimFunctionCall:
    argument_map = dict(record.arguments)
    result_map = dict(record.results)
    arguments = tuple(
        PrimCallBinding(formal, argument_map[formal])
        for formal in callee.parameters
    )
    results = tuple(
        PrimCallBinding(formal, result_map[formal])
        for formal in callee.results
    )
    written_formals = set(callee.attrs.get("written_parameters", ()))
    source_formals = set(callee.attrs.get("transfer_source_parameters", ()))
    writes = tuple(dict.fromkeys((
        *(argument_map[value] for value in callee.parameters if value in written_formals),
        *(result_map[value] for value in callee.results),
    )))
    shared = tuple(
        _caller_shared_buffer(function_name, node.id, buffer)
        for call in callee.body.fields
        for buffer in _call_shared_buffers(call)
    )
    return PrimFunctionCall(
        call_id=node.id,
        callee=record.callee,
        arguments=arguments,
        results=results,
        shared_workspace_buffers=shared,
        transfer_sources=tuple(
            argument_map[value]
            for value in callee.parameters
            if value in source_formals
        ),
        reads=tuple(value.actual for value in arguments),
        writes=writes,
        effect_kind=node.effect.kind.value,
        effect_resource=node.effect.resource,
        memory_pools=tuple(
            MemoryPoolFrame(
                pool.memory_space,
                pool.allocation,
                pool.offset,
                pool.scope_bytes,
            )
            for pool in record.memory_pools
        ),
        dependencies=dependencies,
    )


def _kernel_primitive(module: IRModule, node):
    if node.op != "tir.call":
        return None
    primitive = module.kernel_callable_map.get(str(node.attrs.get("callee", "")))
    if primitive is None or kernel_dispatch_of(primitive) is None:
        return None
    return primitive


def _execution_dependencies(nodes, executable_ids) -> dict[str, tuple[str, ...]]:
    """Lower SSA and resource-effect order to explicit execution edges.

    Physical buffers cannot define this relation: bufferization intentionally
    aliases views and in-place call results.  The graph is consulted exactly
    once here, while it still owns SSA semantics.  Later TIR passes, editable
    dumps, verification, and codegen consume only these call-id edges.
    """

    node_map = {node.id: node for node in nodes}
    memo: dict[str, tuple[str, ...]] = {}

    def nearest_executable(node_id: str) -> tuple[str, ...]:
        if node_id in executable_ids:
            return (node_id,)
        if node_id in memo:
            return memo[node_id]
        node = node_map.get(node_id)
        if node is None:
            return ()
        result = tuple(dict.fromkeys(
            dependency
            for input_id in node.inputs
            for dependency in nearest_executable(input_id)
        ))
        memo[node_id] = result
        return result

    result: dict[str, tuple[str, ...]] = {}
    last_writer: dict[str, str] = {}
    readers: dict[str, list[str]] = {}
    for node in nodes:
        if node.id not in executable_ids:
            continue
        required = [
            dependency
            for input_id in node.inputs
            for dependency in nearest_executable(input_id)
            if dependency != node.id
        ]
        resource = node.effect.resource
        kind = node.effect.kind.value
        if resource is not None:
            writer = last_writer.get(resource)
            if writer is not None:
                required.append(writer)
            if kind in {"write", "read_write"}:
                required.extend(readers.get(resource, ()))
                readers[resource] = []
                last_writer[resource] = node.id
            elif kind == "read":
                readers.setdefault(resource, []).append(node.id)
        result[node.id] = tuple(dict.fromkeys(required))
    return result


def _call_shared_buffers(statement) -> tuple:
    if isinstance(statement, (PrimFunctionCall, KernelInvoke)):
        return statement.shared_workspace_buffers
    # Existing edited schedule checkpoints may already contain a pipeline
    # region. Its consumer view owns the semantic call once.
    from triton.flagmega.ir import PipelineStage, ProducerConsumerRegion, execution_calls_of
    if isinstance(statement, PipelineStage):
        return statement.operation.shared_workspace_buffers
    if isinstance(statement, ProducerConsumerRegion):
        temporary = ExecutionFunction("_region", (), (), statement.consume_body)
        return tuple(
            buffer
            for call in execution_calls_of(temporary)
            for buffer in call.shared_workspace_buffers
        )
    return ()


def _caller_shared_buffer(function_name: str, call_id: str, buffer):
    physical = replace(
        buffer.mem_span.buffer,
        id=f"execution:{function_name}:{call_id}:{buffer.mem_span.buffer.id}",
        function=function_name,
    )
    return replace(
        buffer,
        name=f"{call_id}.{buffer.name}",
        mem_span=MemSpan(physical, buffer.mem_span.start, buffer.mem_span.size),
    )


__all__ = ["materialize_execution_functions"]
