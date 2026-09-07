# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Bind selected PrimFunction ABI parameters to formal Buffer/MemSpan views."""

from __future__ import annotations

from dataclasses import dataclass, replace
from triton.flagmega.ir.tir.kernel_definition import replace_kernel_dispatch, replace_kernel_callables

from triton.flagmega.errors import IRVerificationError
from triton.flagmega.ir.bufferization import BufferPlan, MemSpan, PhysicalBuffer
from triton.flagmega.ir.dim_expr import dim
from triton.flagmega.ir.distributed_type import (
    is_fully_sharded_across_placement,
    local_shape,
    placement_owner_count,
)
from triton.flagmega.ir.model import (
    DistributedType,
    IRModule,
    NoneType,
    RefType,
    TensorType,
    TupleType,
    logical_type,
)
from triton.flagmega.ir.tir import (
    Buffer,
    DistributedBufferStorageKind,
    PrimParameter,
    PrimParameterRole,
    Return,
    ReturnBinding,
    kernel_dispatch_of,
)
from triton.flagmega.passes.tir.shared_workspace import (
    shared_workspace_buffer_id,
)


def bind_prim_function_buffers(
    module: IRModule,
    *,
    plan: BufferPlan | None = None,
) -> IRModule:
    """Attach formal typed buffers to every selected-kernel PrimFunction ABI."""

    functions = []
    for function in module.kernel_callable_map.values():
        dispatch = kernel_dispatch_of(function)
        if dispatch is None:
            functions.append(function)
            continue
        if any(parameter.buffers for parameter in function.parameters):
            if any(
                parameter.role is not PrimParameterRole.METADATA
                and _leaf_types(parameter.type)
                and not parameter.buffers
                for parameter in function.parameters
            ):
                raise IRVerificationError(
                    f"PrimFunction @{function.name} has a partially buffer-bound ABI."
                )
            functions.append(_bind_shared_workspace_buffers(function, dispatch, plan))
            continue
        distributed_inputs, distributed_outputs = _distributed_signature(dispatch, function)
        workspace_map = {value.name: value for value in dispatch.workspaces}
        transfer_alignments = _transfer_source_alignments(dispatch)
        actual_layouts = _actual_parameter_layouts(module, function, plan)
        parameters = []
        output_index = 0
        for index, parameter in enumerate(function.parameters):
            if parameter.role is PrimParameterRole.METADATA:
                parameters.append(parameter)
                continue
            if parameter.role is PrimParameterRole.OUTPUT:
                physical_type = distributed_outputs[output_index]
                output_index += 1
            elif index < len(distributed_inputs):
                physical_type = distributed_inputs[index]
            else:
                physical_type = parameter.type
            leaves = _leaf_types(physical_type)
            layouts = actual_layouts.get(parameter.name)
            if layouts is not None and len(layouts) != len(leaves):
                raise IRVerificationError(
                    f"PrimFunction @{function.name} parameter {parameter.name!r} "
                    "layout leaf arity differs from its type."
                )
            buffers = tuple(
                _formal_buffer(
                    function.name,
                    parameter,
                    path,
                    leaf,
                    layout=None if layouts is None else layouts[ordinal],
                    alignment=(
                        workspace_map[parameter.name].alignment
                        if parameter.role is PrimParameterRole.WORKSPACE
                        else transfer_alignments.get(parameter.name)
                    ),
                )
                for ordinal, (path, leaf) in enumerate(leaves)
            )
            parameters.append(replace(parameter, buffers=buffers))
        parameter_map = {value.name: value for value in parameters}
        results = Return(tuple(
            replace(binding, value=parameter_map[binding.storage].buffer_value)
            for binding in function.results.values
        ))
        rewritten = replace(function, parameters=tuple(parameters), results=results)
        functions.append(_bind_shared_workspace_buffers(
            rewritten,
            kernel_dispatch_of(rewritten),
            plan,
        ))
    return replace_kernel_callables(module, functions)


def _bind_shared_workspace_buffers(function, dispatch, plan):
    if dispatch is None or not dispatch.shared_workspace_buffers or plan is None:
        return function
    descriptors = plan.buffer_map
    rebound = []
    for buffer in dispatch.shared_workspace_buffers:
        descriptor_id = shared_workspace_buffer_id(function.name, buffer.name)
        try:
            descriptor = descriptors[descriptor_id]
        except KeyError as error:
            raise IRVerificationError(
                f"PrimFunction @{function.name} shared workspace {buffer.name!r} "
                "is absent from the buffer plan."
            ) from error
        rebound.append(replace(buffer, mem_span=descriptor.mem_span))
    rewritten_dispatch = replace(
        dispatch, shared_workspace_buffers=tuple(rebound)
    )
    return replace_kernel_dispatch(function, rewritten_dispatch)


def _transfer_source_alignments(dispatch) -> dict[str, int]:
    selection = dispatch.microkernel
    pipeline = None if selection is None else selection.transfer_pipeline
    if pipeline is None:
        return {}
    result: dict[str, int] = {}
    for channel in pipeline.channels:
        for index in channel.source_argument_indices:
            name = dispatch.arguments[index]
            result[name] = max(
                result.get(name, 1), channel.source_alignment_bytes
            )
    return result


def _distributed_signature(dispatch, function):
    distribution = dispatch.parameters.get("distribution")
    if not hasattr(distribution, "get"):
        return function.runtime_parameter_types, tuple(
            value.type for value in function.output_parameters
        )
    inputs = tuple(distribution.get("input_types", function.runtime_parameter_types))
    output = distribution.get("output_type", function.runtime_return_type)
    if len(inputs) != len(function.runtime_parameters):
        raise IRVerificationError(
            f"KernelDispatch {dispatch.candidate!r} distributed input ABI has the wrong arity."
        )
    if len(function.output_parameters) == 1:
        outputs = (output,)
    elif isinstance(output, TupleType) and len(output.fields) == len(function.output_parameters):
        outputs = output.fields
    else:
        raise IRVerificationError(
            f"KernelDispatch {dispatch.candidate!r} distributed output ABI has the wrong arity."
        )
    return inputs, outputs


def _formal_buffer(
    function_name,
    parameter: PrimParameter,
    path: str,
    value_type,
    *,
    layout,
    alignment: int | None,
) -> Buffer:
    logical = logical_type(value_type)
    if not isinstance(logical, TensorType):
        raise IRVerificationError(
            f"PrimFunction @{function_name} parameter {parameter.name!r} has a non-tensor ABI leaf."
        )
    name = parameter.name if not path else f"{parameter.name}.{path}"
    distributed = value_type if isinstance(value_type, DistributedType) else None
    storage_kind = (
        layout.distributed_storage_kind
        if layout is not None
        else _default_storage_kind(distributed, parameter.role)
    )
    backing_type = (
        layout.distributed_backing_type
        if layout is not None
        else None
    )
    component_dimensions = (
        logical.shape
        if distributed is None
        or storage_kind.exposes_logical_coordinates
        else local_shape(backing_type or distributed)
    )
    size = dim(logical.dtype.itemsize)
    for dimension in component_dimensions:
        size = (size * dimension).simplify()
    physical_size = (
        (size * placement_owner_count(distributed)).simplify()
        if distributed is not None
        and storage_kind is DistributedBufferStorageKind.COMPACT_PER_OWNER
        else size
    )
    alignment = alignment or _power_of_two_alignment(logical.dtype.itemsize)
    physical = PhysicalBuffer(
        f"abi:{function_name}:{name}",
        parameter.memory_space or parameter.role.value,
        physical_size,
        alignment,
        function=function_name,
        role=f"formal_{parameter.role.value}",
    )
    return Buffer(
        name,
        logical.dtype,
        MemSpan(physical, dim(0), size),
        logical.shape,
        (
            tuple(dim(value) for value in layout.strides)
            if layout is not None
            else _dense_strides(component_dimensions)
        ),
        distributed,
        storage_kind,
        backing_type,
    )


def _actual_parameter_layouts(module, function, plan):
    if plan is None:
        return {}
    observed: dict[str, set[tuple[str, ...]]] = {}
    descriptor_map = plan.buffer_map
    for graph_function in module.functions:
        values = dict(plan.function_map[graph_function.name].values)
        kernel_calls = {
            value.call: value
            for value in plan.function_map[graph_function.name].kernel_calls
        }
        for node in module.nodes:
            if node.op != "tir.call" or str(node.attrs.get("callee", "")) != function.name:
                continue
            if node.id not in values:
                continue
            bindings: dict[str, tuple[str, ...]] = {}
            for parameter, input_id in zip(function.runtime_parameters, node.inputs):
                if (
                    parameter.role is not PrimParameterRole.METADATA
                    and _leaf_types(parameter.type)
                ):
                    bindings[parameter.name] = values[input_id]
            outputs = values[node.id]
            cursor = 0
            for parameter in function.output_parameters:
                count = len(_leaf_types(parameter.type))
                bindings[parameter.name] = outputs[cursor:cursor + count]
                cursor += count
            workspace_record = kernel_calls.get(node.id)
            if workspace_record is not None:
                bindings.update(
                    (formal, (actual,))
                    for formal, actual in workspace_record.workspaces
                )
            for parameter_name, actual_ids in bindings.items():
                observed.setdefault(parameter_name, set()).add(tuple(actual_ids))
    result = {}
    for parameter_name, signatures in observed.items():
        layouts = {
            tuple(
                (
                    descriptor_map[buffer_id].distributed_type,
                    descriptor_map[buffer_id].distributed_storage_kind,
                    descriptor_map[buffer_id].distributed_backing_type,
                    descriptor_map[buffer_id].strides,
                    descriptor_map[buffer_id].nbytes,
                )
                for buffer_id in signature
            )
            for signature in signatures
        }
        if len(layouts) != 1:
            raise IRVerificationError(
                f"PrimFunction @{function.name} parameter {parameter_name!r} has conflicting "
                "caller buffer layouts; specialize the PrimFunction before binding."
            )
        signature = next(iter(signatures))
        result[parameter_name] = tuple(descriptor_map[value] for value in signature)
    return result


def _default_storage_kind(distributed, role):
    if distributed is None:
        return DistributedBufferStorageKind.COMPACT_LOCAL
    if role is PrimParameterRole.OUTPUT and (
        distributed.partial is not None
        or is_fully_sharded_across_placement(distributed)
    ):
        return DistributedBufferStorageKind.COMPACT_PER_OWNER
    return DistributedBufferStorageKind.CANONICAL_GLOBAL


def _leaf_types(value_type, prefix: str = ""):
    if isinstance(value_type, NoneType):
        return ()
    if isinstance(value_type, DistributedType):
        return ((prefix, value_type),)
    if isinstance(value_type, TensorType):
        return ((prefix, value_type),)
    if isinstance(value_type, RefType):
        result = []
        for name, field in value_type.fields:
            path = name if not prefix else f"{prefix}.{name}"
            result.extend(_leaf_types(field, path))
        return tuple(result)
    if isinstance(value_type, TupleType):
        result = []
        for index, field in enumerate(value_type.fields):
            path = str(index) if not prefix else f"{prefix}.{index}"
            result.extend(_leaf_types(field, path))
        return tuple(result)
    raise IRVerificationError(f"Cannot create a buffer ABI for {type(value_type).__name__}.")


def _dense_strides(shape):
    strides = []
    current = dim(1)
    for dimension in reversed(shape):
        strides.append(current)
        current = (current * dimension).simplify()
    return tuple(reversed(strides))


def _power_of_two_alignment(itemsize: int) -> int:
    limit = min(max(int(itemsize), 1), 16)
    return 1 << (limit.bit_length() - 1)


@dataclass(frozen=True)
class BindPrimFunctionBuffersPass:
    name: str = "BindPrimFunctionBuffers"
    preserves: frozenset[str] = frozenset()

    def run(self, module: IRModule) -> IRModule:
        return bind_prim_function_buffers(module)


__all__ = ["BindPrimFunctionBuffersPass", "bind_prim_function_buffers"]
