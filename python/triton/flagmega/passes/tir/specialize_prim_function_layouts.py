# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Specialize shared selected PrimFunctions at physical ABI boundaries."""

from __future__ import annotations

import json
from dataclasses import replace
from triton.flagmega.ir.tir.kernel_definition import replace_kernel_dispatch, replace_kernel_callables

from triton.flagmega.errors import IRVerificationError
from triton.flagmega.ir.bufferization import BufferPlan, MemSpan
from triton.flagmega.ir.model import (
    DistributedType,
    IRModule,
    NoneType,
    RefType,
    TensorType,
    TupleType,
)
from triton.flagmega.ir.tir import kernel_dispatch_of


def specialize_prim_functions_for_buffer_layouts(
    module: IRModule,
    plan: BufferPlan,
) -> IRModule:
    """Clone a selected PrimFunction only when callers require distinct ABIs.

    Semantic kernel materialization deliberately happens before bufferization,
    so equivalent kernels can be shared across graph functions and decode-layer
    invocations.  Physical storage is known only after the first buffer plan.
    One formal :class:`Buffer` cannot describe both canonical-global and compact
    per-owner storage (or distinct strides), therefore this pass partitions call
    sites by their complete physical layout and creates one reusable function per
    partition.  Calls with the same layout continue to share code.
    """

    calls_by_callee = _collect_call_layouts(module, plan)
    occupied = {
        *(function.name for function in module.functions),
        *(function.name for function in module.kernel_callable_map.values()),
    }
    rewrites: dict[str, str] = {}
    result_functions = []
    specialization_records = []

    for function in module.kernel_callable_map.values():
        calls = calls_by_callee.get(function.name, ())
        groups = _group_calls(calls)
        if len(groups) <= 1:
            result_functions.append(function)
            continue
        if any(parameter.buffers for parameter in function.parameters):
            raise IRVerificationError(
                f"PrimFunction @{function.name} must be layout-specialized before "
                "formal buffers are bound."
            )

        # Keep the most frequently used ABI on the original symbol.  Ties use
        # first call order, making dumps stable while preserving the dominant
        # decode-layer implementation and its instruction-cache reuse.
        groups = tuple(sorted(groups, key=lambda value: (-len(value[1]), value[2])))
        variants = []
        for index, (signature, call_ids, _) in enumerate(groups):
            name = function.name
            if index:
                name = _unique_variant_name(function.name, index, occupied)
                occupied.add(name)
            variant = _clone_for_layout(
                function,
                name=name,
                source=function.name,
                index=index,
                signature=signature,
            )
            result_functions.append(variant)
            rewrites.update((call_id, name) for call_id in call_ids)
            variants.append({
                "function": name,
                "calls": tuple(call_ids),
                "layout": _signature_data(signature),
            })
        specialization_records.append({
            "source": function.name,
            "variants": tuple(variants),
        })

    if not rewrites:
        return module
    nodes = tuple(
        replace(node, attrs={**dict(node.attrs), "callee": rewrites[node.id]})
        if node.id in rewrites
        else node
        for node in module.nodes
    )
    return replace_kernel_callables(
        module,
        result_functions,
        nodes=nodes,
        metadata={
            **dict(module.metadata),
            "buffer_layout_specializations": tuple(specialization_records),
        },
    )


def _collect_call_layouts(module, plan):
    descriptor_map = plan.buffer_map
    result: dict[str, list[tuple[str, tuple[object, ...], int]]] = {}
    order = 0
    for graph_function in module.functions:
        function_plan = plan.function_map[graph_function.name]
        values = dict(function_plan.values)
        kernel_calls = {
            value.call: value for value in function_plan.kernel_calls
        }
        for node_id in values:
            node = module.node_map[node_id]
            if node.op != "tir.call":
                continue
            callee = str(node.attrs.get("callee", ""))
            function = module.kernel_callable_map.get(callee)
            if function is None or kernel_dispatch_of(function) is None:
                continue
            bindings = _call_bindings(function, node, values, kernel_calls)
            signature = tuple(
                (
                    parameter.name,
                    tuple(_descriptor_layout_key(descriptor_map[value]) for value in bindings.get(parameter.name, ())),
                )
                for parameter in function.parameters
            )
            result.setdefault(callee, []).append((node.id, signature, order))
            order += 1
    return {name: tuple(values) for name, values in result.items()}


def _call_bindings(function, node, values, kernel_calls):
    if len(node.inputs) != len(function.runtime_parameters):
        raise IRVerificationError(
            f"Kernel call {node.id!r} argument arity differs from @{function.name}."
        )
    bindings = {}
    for parameter, input_id in zip(function.runtime_parameters, node.inputs):
        count = (
            0
            if parameter.role.value == "metadata"
            else _leaf_count(parameter.type)
        )
        if count:
            try:
                actuals = tuple(values[input_id])
            except KeyError as error:
                raise IRVerificationError(
                    f"Kernel call {node.id!r} input {input_id!r} has no buffer binding."
                ) from error
            if len(actuals) != count:
                raise IRVerificationError(
                    f"Kernel call {node.id!r} input {parameter.name!r} has the wrong "
                    "buffer leaf arity."
                )
            bindings[parameter.name] = actuals

    try:
        outputs = tuple(values[node.id])
    except KeyError as error:
        raise IRVerificationError(
            f"Kernel call {node.id!r} has no result buffer binding."
        ) from error
    cursor = 0
    for parameter in function.output_parameters:
        count = _leaf_count(parameter.type)
        bindings[parameter.name] = outputs[cursor:cursor + count]
        cursor += count
    if cursor != len(outputs):
        raise IRVerificationError(
            f"Kernel call {node.id!r} result buffer leaf arity differs from @{function.name}."
        )

    workspace_record = kernel_calls.get(node.id)
    if function.workspaces:
        if workspace_record is None:
            raise IRVerificationError(
                f"Kernel call {node.id!r} has no workspace buffer binding."
            )
        bindings.update(
            (formal, (actual,))
            for formal, actual in workspace_record.workspaces
        )
    return bindings


def _group_calls(calls):
    groups: dict[tuple[object, ...], list[str]] = {}
    first_order = {}
    for call_id, signature, order in calls:
        groups.setdefault(signature, []).append(call_id)
        first_order.setdefault(signature, order)
    return tuple(
        (signature, tuple(call_ids), first_order[signature])
        for signature, call_ids in groups.items()
    )


def _descriptor_layout_key(descriptor):
    distributed = (
        None
        if descriptor.distributed_type is None
        else json.dumps(
            descriptor.distributed_type.to_data(),
            sort_keys=True,
            separators=(",", ":"),
        )
    )
    return (
        distributed,
        descriptor.distributed_storage_kind.value,
        tuple(descriptor.strides),
        descriptor.nbytes,
    )


def _signature_data(signature):
    return tuple({
        "parameter": parameter,
        "leaves": tuple({
            "distributed_type": distributed,
            "storage_kind": storage_kind,
            "strides": strides,
            "nbytes": nbytes,
        } for distributed, storage_kind, strides, nbytes in leaves),
    } for parameter, leaves in signature)


def _clone_for_layout(function, *, name, source, index, signature):
    dispatch = kernel_dispatch_of(function)
    assert dispatch is not None
    if name != function.name and dispatch.shared_workspace_buffers:
        buffers = []
        for ordinal, buffer in enumerate(dispatch.shared_workspace_buffers):
            physical = replace(
                buffer.mem_span.buffer,
                id=f"shared:{name}:{ordinal}:{buffer.name}",
                function=name,
            )
            buffers.append(replace(
                buffer,
                mem_span=MemSpan(
                    physical,
                    buffer.mem_span.start,
                    buffer.mem_span.size,
                ),
            ))
        dispatch = replace(dispatch, shared_workspace_buffers=tuple(buffers))
    return replace_kernel_dispatch(
        function,
        dispatch,
        name=name,
        attrs={
            **dict(function.attrs),
            "buffer_layout_specialized_from": source,
            "buffer_layout_specialization_index": index,
            "buffer_layout_signature": _signature_data(signature),
        },
    )


def _unique_variant_name(source, index, occupied):
    ordinal = index
    while True:
        candidate = f"{source}__layout_{ordinal}"
        if candidate not in occupied:
            return candidate
        ordinal += 1


def _leaf_count(value_type):
    if isinstance(value_type, NoneType):
        return 0
    if isinstance(value_type, (TensorType, DistributedType)):
        return 1
    if isinstance(value_type, RefType):
        return sum(_leaf_count(field) for _, field in value_type.fields)
    if isinstance(value_type, TupleType):
        return sum(_leaf_count(field) for field in value_type.fields)
    raise IRVerificationError(
        f"Cannot specialize buffer layout for {type(value_type).__name__}."
    )


__all__ = ["specialize_prim_functions_for_buffer_layouts"]
