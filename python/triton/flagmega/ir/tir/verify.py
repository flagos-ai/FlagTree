# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Structural verification for PrimFunction bodies and explicit buffer effects."""

from __future__ import annotations

from triton.flagmega.errors import IRVerificationError
from triton.flagmega.ir.dim_expr import dim
from triton.flagmega.ir.tir.block import Block
from triton.flagmega.ir.tir.buffer import Buffer
from triton.flagmega.ir.tir.buffer_tuple import BufferTuple
from triton.flagmega.ir.tir.buffer_load import BufferLoad
from triton.flagmega.ir.tir.buffer_store import BufferStore
from triton.flagmega.ir.tir.for_loop import For
from triton.flagmega.ir.tir.let import Let
from triton.flagmega.ir.tir.kernel_dispatch import KernelDispatch
from triton.flagmega.ir.tir.prim_function import PrimFunction, PrimParameterRole
from triton.flagmega.ir.tir.value_ref import ValueRef
from triton.flagmega.ir.tir.visitor import iter_tir_children
from triton.flagmega.ir.model import DistributedType, RefType, TensorType, TupleType, logical_type


def verify_prim_function(function: PrimFunction) -> PrimFunction:
    parameters = function.parameter_map
    buffer_names: set[str] = set()
    bound = [
        bool(value.buffers)
        for value in function.parameters
        if value.role is not PrimParameterRole.METADATA
        and _logical_leaves(value.type)
    ]
    if any(bound) and not all(bound):
        raise IRVerificationError(f"PrimFunction @{function.name} has a partially bound buffer ABI.")
    for parameter in function.parameters:
        if (
            parameter.role is PrimParameterRole.METADATA
            and parameter.buffers
        ):
            raise IRVerificationError(
                f"PrimFunction @{function.name} metadata parameter "
                f"{parameter.name!r} cannot own physical buffers."
            )
        leaves = (
            ()
            if parameter.role is PrimParameterRole.METADATA
            else _logical_leaves(parameter.type)
        )
        if parameter.buffers and (
            len(parameter.buffers) != len(leaves)
            or any(
                logical_type(buffer.type) != logical_type(leaf)
                for buffer, leaf in zip(parameter.buffers, leaves)
            )
        ):
            raise IRVerificationError(
                f"PrimFunction @{function.name} parameter {parameter.name!r} buffers do not match its type."
            )
        for buffer in parameter.buffers:
            if buffer.name in buffer_names:
                raise IRVerificationError(
                    f"PrimFunction @{function.name} has duplicate formal buffer {buffer.name!r}."
                )
            buffer_names.add(buffer.name)
            if buffer.mem_span.buffer.function != function.name:
                raise IRVerificationError(
                    f"PrimFunction @{function.name} formal buffer {buffer.name!r} has wrong ownership."
                )
            expected_space = parameter.memory_space or parameter.role.value
            if buffer.mem_span.buffer.memory_space != expected_space:
                raise IRVerificationError(
                    f"PrimFunction @{function.name} formal buffer {buffer.name!r} has wrong ABI role."
                )
    output_names = {
        value.name for value in function.parameters
        if value.role is PrimParameterRole.OUTPUT
    }
    result_storages = {value.storage for value in function.results.values}
    if output_names != result_storages.intersection(output_names):
        missing = sorted(output_names - result_storages)
        raise IRVerificationError(
            f"PrimFunction @{function.name} output parameters without logical results: {missing}."
        )

    _verify_node(
        function.body,
        function=function,
        parameters=parameters,
        allocated={},
        loop_vars=set(),
        scalar_vars=set(),
    )
    for binding in function.results.values:
        _verify_buffer_value(binding.value, function, parameters, {})
    return function


def _verify_node(
    node,
    *,
    function: PrimFunction,
    parameters,
    allocated: dict[str, Buffer],
    loop_vars: set[str],
    scalar_vars: set[str],
) -> None:
    if isinstance(node, For):
        name = node.loop_var.name
        if name in loop_vars:
            raise IRVerificationError(
                f"PrimFunction @{function.name} shadows active loop variable {name!r}."
            )
        _verify_node(
            node.body,
            function=function,
            parameters=parameters,
            allocated=allocated,
            loop_vars=loop_vars | {name},
            scalar_vars=scalar_vars,
        )
        return
    if isinstance(node, Let):
        if node.var.name in scalar_vars:
            raise IRVerificationError(
                f"PrimFunction @{function.name} shadows active scalar variable {node.var.name!r}."
            )
        _verify_node(
            node.body,
            function=function,
            parameters=parameters,
            allocated=allocated,
            loop_vars=loop_vars,
            scalar_vars=scalar_vars | {node.var.name},
        )
        return
    if isinstance(node, Block):
        local = dict(allocated)
        for buffer in node.alloc_buffers:
            if buffer.name in parameters or buffer.name in local:
                raise IRVerificationError(
                    f"PrimFunction @{function.name} block {node.name!r} duplicates buffer {buffer.name!r}."
                )
            local[buffer.name] = buffer
        reads, writes = _collect_effect_buffers(node.body)
        declared_reads = {value.buffer.name for value in node.reads}
        declared_writes = {value.buffer.name for value in node.writes}
        if not reads.issubset(declared_reads) or not writes.issubset(declared_writes):
            raise IRVerificationError(
                f"PrimFunction @{function.name} block {node.name!r} effects are not covered by "
                f"declared regions: missing reads={sorted(reads - declared_reads)}, "
                f"writes={sorted(writes - declared_writes)}."
            )
        for region in (*node.reads, *node.writes):
            _verify_buffer_value(region.buffer, function, parameters, local)
        _verify_node(
            node.init_body,
            function=function,
            parameters=parameters,
            allocated=local,
            loop_vars=loop_vars,
            scalar_vars=scalar_vars,
        )
        _verify_node(
            node.body,
            function=function,
            parameters=parameters,
            allocated=local,
            loop_vars=loop_vars,
            scalar_vars=scalar_vars,
        )
        return
    if isinstance(node, KernelDispatch):
        runtime_names = tuple(
            value.name for value in function.parameters
            if value.role in {
                PrimParameterRole.INPUT,
                PrimParameterRole.INOUT,
                PrimParameterRole.METADATA,
            }
        )
        output_names = tuple(
            value.name for value in function.parameters
            if value.role is PrimParameterRole.OUTPUT
        )
        workspace_names = tuple(value.name for value in function.workspaces)
        if node.arguments != runtime_names or node.outputs != output_names:
            raise IRVerificationError(
                f"PrimFunction @{function.name} KernelDispatch ABI does not match its parameters."
            )
        if tuple(value.name for value in node.workspaces) != workspace_names:
            raise IRVerificationError(
                f"PrimFunction @{function.name} KernelDispatch workspace ABI does not match its parameters."
            )
        if any(
            requirement.type != parameter.type
            or requirement.memory_space != parameter.memory_space
            for requirement, parameter in zip(node.workspaces, function.workspaces)
        ):
            raise IRVerificationError(
                f"PrimFunction @{function.name} KernelDispatch workspace types/locations differ."
            )
        writable = {
            value.name for value in function.parameters
            if value.role in {PrimParameterRole.INOUT, PrimParameterRole.OUTPUT}
        }
        if not set(node.writes).issubset(writable):
            raise IRVerificationError(
                f"PrimFunction @{function.name} KernelDispatch writes read-only ABI values "
                f"{sorted(set(node.writes) - writable)}."
            )
        output_parameters = {
            value.name: value
            for value in function.parameters
            if value.role is PrimParameterRole.OUTPUT
        }
        missing_output_writes = {
            name
            for name in set(node.outputs) - set(node.writes)
            if not isinstance(logical_type(output_parameters[name].type), RefType)
        }
        if missing_output_writes:
            raise IRVerificationError(
                f"PrimFunction @{function.name} KernelDispatch does not write "
                f"tensor outputs {sorted(missing_output_writes)}."
            )
        _verify_microkernel_resources(node, function)
        return
    if isinstance(node, BufferLoad):
        _verify_buffer_value(node.buffer, function, parameters, allocated)
    if isinstance(node, BufferStore):
        _verify_buffer_value(node.buffer, function, parameters, allocated)
        parameter = parameters.get(node.buffer.name)
        if parameter is not None and parameter.role is PrimParameterRole.INPUT:
            raise IRVerificationError(
                f"PrimFunction @{function.name} writes read-only input buffer {node.buffer.name!r}."
            )
    for child in iter_tir_children(node):
        _verify_node(
            child,
            function=function,
            parameters=parameters,
            allocated=allocated,
            loop_vars=loop_vars,
            scalar_vars=scalar_vars,
        )


def _verify_buffer_value(value, function, parameters, allocated) -> None:
    if isinstance(value, BufferTuple):
        for buffer in value.buffers:
            _verify_buffer_value(buffer, function, parameters, allocated)
        return
    if isinstance(value, ValueRef):
        parameter = parameters.get(value.name)
        if parameter is None or parameter.type != value.type:
            raise IRVerificationError(
                f"PrimFunction @{function.name} uses unbound or mistyped value %{value.name}."
            )
        return
    if not isinstance(value, Buffer):
        return
    parameter = parameters.get(value.name)
    if parameter is None:
        parameter = next(
            (item for item in parameters.values() if value in item.buffers),
            None,
        )
    allocation = allocated.get(value.name)
    if parameter is None and allocation is None:
        raise IRVerificationError(
            f"PrimFunction @{function.name} uses unbound buffer {value.name!r}."
        )
    if parameter is not None:
        if parameter.buffers:
            if value not in parameter.buffers:
                raise IRVerificationError(
                    f"PrimFunction @{function.name} buffer {value.name!r} is not its ABI parameter view."
                )
        elif parameter.type != value.type:
            raise IRVerificationError(
                f"PrimFunction @{function.name} buffer {value.name!r} type does not match its ABI parameter."
            )
    if allocation is not None and allocation != value:
        raise IRVerificationError(
            f"PrimFunction @{function.name} uses conflicting views for allocated buffer {value.name!r}."
        )


def _collect_effect_buffers(node) -> tuple[set[str], set[str]]:
    reads: set[str] = set()
    writes: set[str] = set()

    def visit(value) -> None:
        if isinstance(value, BufferLoad):
            reads.add(value.buffer.name)
        elif isinstance(value, BufferStore):
            writes.add(value.buffer.name)
        for child in iter_tir_children(value):
            visit(child)

    visit(node)
    return reads, writes


def _verify_microkernel_resources(
    dispatch: KernelDispatch,
    function: PrimFunction,
) -> None:
    selection = dispatch.microkernel
    buffers = dispatch.shared_workspace_buffers
    if selection is None:
        if buffers:
            raise IRVerificationError(
                f"PrimFunction @{function.name} has shared buffers without a microkernel."
            )
        return
    descriptors = selection.shared_workspaces
    if len(buffers) != len(descriptors):
        raise IRVerificationError(
            f"PrimFunction @{function.name} must materialize every selected shared workspace."
        )
    for descriptor, buffer in zip(descriptors, buffers):
        physical = buffer.mem_span.buffer
        if (
            descriptor.name != buffer.name
            or descriptor.type != buffer.type
            or physical.memory_space != "shared"
            or physical.function != function.name
            or physical.alignment < descriptor.alignment_bytes
            or buffer.mem_span.size.maximum != descriptor.maximum_nbytes
        ):
            raise IRVerificationError(
                f"PrimFunction @{function.name} shared workspace {descriptor.name!r} "
                "does not match its selected descriptor."
            )
        dense_strides = []
        stride = dim(1)
        for dimension in reversed(descriptor.type.shape):
            dense_strides.append(stride)
            stride = (stride * dimension).simplify()
        if buffer.strides != tuple(reversed(dense_strides)):
            raise IRVerificationError(
                f"PrimFunction @{function.name} shared workspace {descriptor.name!r} "
                "strides do not match its selected dense payload."
            )
        for start, alignment in (
            (physical.start, physical.alignment),
            (buffer.mem_span.absolute_start, descriptor.alignment_bytes),
        ):
            remainder = (start % alignment).simplify()
            if not remainder.is_fixed or remainder.fixed_value != 0:
                raise IRVerificationError(
                    f"PrimFunction @{function.name} shared workspace {descriptor.name!r} "
                    "violates its allocation/view alignment."
                )
    pipeline = selection.transfer_pipeline
    if pipeline is None:
        return
    for channel in pipeline.channels:
        for argument_index in channel.source_argument_indices:
            if argument_index >= len(dispatch.arguments):
                raise IRVerificationError(
                    f"PrimFunction @{function.name} transfer channel {channel.name!r} "
                    f"has invalid source operand {argument_index}."
                )
            argument = dispatch.arguments[argument_index]
            if argument not in dispatch.reads or argument in dispatch.writes:
                raise IRVerificationError(
                    f"PrimFunction @{function.name} transfer channel {channel.name!r} "
                    f"has non-read-only source operand {argument_index} ({argument!r})."
                )


def _logical_leaves(value_type):
    if isinstance(value_type, (TensorType, DistributedType)):
        return (value_type,)
    if isinstance(value_type, RefType):
        return tuple(leaf for _, field in value_type.fields for leaf in _logical_leaves(field))
    if isinstance(value_type, TupleType):
        return tuple(leaf for field in value_type.fields for leaf in _logical_leaves(field))
    return ()


__all__ = ["verify_prim_function"]
