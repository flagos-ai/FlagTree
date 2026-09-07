# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Static logical tensor reshape."""

from __future__ import annotations

from math import prod
from typing import Mapping, Sequence

from triton.flagmega.errors import IRSchemaError
from triton.flagmega.ir.distributed_type import (
    BlockCyclicSplit,
    SBP,
    SBPBroadCast,
    SBPSplit,
    SplitStage,
    is_distributable,
    scale_split_units,
)
from triton.flagmega.ir.model import DistributedType, IRType, Node, TensorType, tensor_type
from triton.flagmega.ir.distributed_inference import tensor_of
from triton.flagmega.ir.ops.core import OpCost, OpDefinition, attribute_parameter, input_parameter, op_definition, tensor_nbytes
from triton.flagmega.ir.type_pattern import is_tensor


@op_definition("tensors.reshape", namespace="tensors", functional_name="reshape", display_name="Tensors.Reshape")
class Reshape(OpDefinition):
    const_evaluable = True
    numpy_materializable = True
    value = input_parameter(is_tensor())
    byte_preserving_input_parameters = (value,)
    shape = attribute_parameter()

    @classmethod
    def normalize_attrs(cls, attributes: Mapping[str, object]) -> dict[str, object]:
        attrs = super().normalize_attrs(attributes)
        shape = attrs["shape"]
        if not isinstance(shape, (tuple, list)) or not shape:
            raise IRSchemaError("F.tensors.reshape shape must be a non-empty integer sequence.")
        values = tuple(shape)
        if any(isinstance(value, bool) or not isinstance(value, int) or value == 0 or value < -1 for value in values):
            raise IRSchemaError("F.tensors.reshape dimensions must be positive integers or one -1.")
        if values.count(-1) > 1:
            raise IRSchemaError("F.tensors.reshape accepts at most one inferred dimension.")
        return {"shape": values}

    @classmethod
    def infer_type(cls, inputs: Sequence[Node], attrs: Mapping[str, object]) -> IRType:
        source_type = cls.value.type_of(inputs)
        value = tensor_of(source_type)
        if any(not dimension.is_fixed for dimension in value.shape):
            raise IRSchemaError("F.tensors.reshape currently requires a static input shape.")
        input_elements = prod(dimension.fixed_value for dimension in value.shape)
        shape = list(int(dimension) for dimension in attrs["shape"])
        known = prod(dimension for dimension in shape if dimension != -1)
        if -1 in shape:
            if input_elements % known:
                raise IRSchemaError("F.tensors.reshape inferred dimension is not integral.")
            shape[shape.index(-1)] = input_elements // known
        elif known != input_elements:
            raise IRSchemaError(
                f"F.tensors.reshape changes logical element count from {input_elements} to {known}.")
        reshaped = tensor_type(value.dtype, shape, layout=value.layout)
        if not isinstance(source_type, DistributedType):
            return reshaped
        policies = _reshape_axis_policies(source_type, reshaped)
        return DistributedType(
            reshaped,
            policies,
            source_type.placement,
            source_type.partial,
        )

    @classmethod
    def evaluate(cls, node, arguments, context):
        value = cls.value.read(arguments)
        input_type = tensor_of(context.types[cls.value.read(node.inputs)])
        output_type = tensor_of(node.type)
        shape = tuple(dimension.fixed_value for dimension in output_type.shape)
        lane_shape = (
            tuple(int(lane) for lane in input_type.dtype.lanes)
            if hasattr(input_type.dtype, "lanes")
            else ()
        )
        return value.reshape((*shape, *lane_shape)).contiguous()

    @classmethod
    def materialize_numpy(cls, node, arguments, context):
        value = cls.value.read(arguments)
        input_type = tensor_of(context.types[cls.value.read(node.inputs)])
        output_type = tensor_of(node.type)
        shape = tuple(dimension.fixed_value for dimension in output_type.shape)
        lane_shape = (
            tuple(int(lane) for lane in input_type.dtype.lanes)
            if hasattr(input_type.dtype, "lanes")
            else ()
        )
        return context.as_contiguous(value.reshape((*shape, *lane_shape)))

    @classmethod
    def cost(cls, node: Node) -> OpCost:
        size = tensor_nbytes(tensor_of(node.type))
        return OpCost(bytes_read=size, bytes_written=size, notes=("reshape",))


def _reshape_axis_policies(
    source: DistributedType,
    output: TensorType,
) -> tuple[SBP, ...]:
    """Port nncase ``ReshapeEvaluator.VisitDistributedType``.

    A reshape is a zero-copy layout transform only when every split boundary
    can be expressed on the new logical axes.  Explicit split units are
    rescaled rather than silently discarded, including ordered block-cyclic
    stages produced by PyNTT's hierarchy-aware split provider.
    """

    input_shape = tuple(dimension.fixed_value for dimension in source.tensor.shape)
    output_shape = tuple(dimension.fixed_value for dimension in output.shape)
    matrix = _reshape_shape_map_matrix(input_shape, output_shape)
    if matrix is None:
        if all(isinstance(policy, SBPBroadCast) for policy in source.axis_policies):
            return tuple(SBP.broadcast() for _ in output_shape)
        raise _distributed_reshape_error(source, output)

    forward, backward = _complete_shape_map(matrix)
    policies: list[SBP | None] = [None] * len(output_shape)

    # One input axis expanded into one or more output axes.
    for input_axis, output_axes in forward.items():
        input_policy = source.axis_policies[input_axis]
        if not isinstance(input_policy, SBPSplit):
            for output_axis in output_axes:
                policies[output_axis] = input_policy
            continue
        split_position = next(
            (index for index, axis in enumerate(output_axes) if output_shape[axis] != 1),
            0,
        )
        split_axis = output_axes[split_position]
        if (
            output_shape[split_axis] != input_shape[input_axis]
            and output_shape[split_axis] % _split_divisor(input_policy, source) != 0
        ):
            raise _distributed_reshape_error(source, output)
        reshaped_policy = input_policy
        if output_shape[split_axis] != input_shape[input_axis]:
            trailing_extent = prod(
                output_shape[axis]
                for axis in output_axes
                if axis != split_axis
            )
            reshaped_policy = scale_split_units(input_policy, 1, trailing_extent)
            if reshaped_policy is None:
                raise _distributed_reshape_error(source, output)
        for output_axis in output_axes:
            policies[output_axis] = (
                reshaped_policy if output_axis == split_axis else SBP.broadcast()
            )

    # One or more input axes flattened into one output axis.
    for output_axis, input_axes in backward.items():
        if policies[output_axis] is not None:
            continue
        split_axes = [
            input_axis
            for input_axis in input_axes
            if isinstance(source.axis_policies[input_axis], SBPSplit)
        ]
        if len(split_axes) > 1:
            flattened = _flatten_block_cyclic_splits(
                source,
                input_axes,
                input_shape,
            )
            if flattened is None:
                raise _distributed_reshape_error(source, output)
            policies[output_axis] = flattened
            continue
        if split_axes:
            split_axis = split_axes[0]
            if split_axis != input_axes[0] and any(
                input_shape[axis] != 1
                for axis in input_axes
                if axis < split_axis
            ):
                raise _distributed_reshape_error(source, output)
            split = source.axis_policies[split_axis]
            assert isinstance(split, SBPSplit)
            multiplier = prod(
                input_shape[axis]
                for axis in input_axes
                if axis != split_axis
            )
            scaled = scale_split_units(split, multiplier, 1)
            if scaled is None:  # pragma: no cover - multiplication is exact.
                raise _distributed_reshape_error(source, output)
            policies[output_axis] = scaled
        else:
            policies[output_axis] = SBP.broadcast()

    if any(policy is None for policy in policies):
        mapped_input_axes = set(forward)
        for input_axes in backward.values():
            mapped_input_axes.update(input_axes)
        if any(
            not isinstance(policy, SBPBroadCast)
            for axis, policy in enumerate(source.axis_policies)
            if axis not in mapped_input_axes
        ):
            raise _distributed_reshape_error(source, output)
        policies = [
            SBP.broadcast() if policy is None else policy
            for policy in policies
        ]

    typed_policies = tuple(policy for policy in policies if policy is not None)
    if len(typed_policies) != len(output_shape) or not is_distributable(
        output,
        typed_policies,
        source.placement,
    ):
        raise _distributed_reshape_error(source, output)
    return typed_policies


def _reshape_shape_map_matrix(
    input_shape: tuple[int, ...],
    output_shape: tuple[int, ...],
) -> tuple[tuple[int, ...], ...] | None:
    if 1 not in input_shape and 1 not in output_shape:
        return _shape_map_matrix(input_shape, output_shape)
    input_axes = tuple(index for index, extent in enumerate(input_shape) if extent != 1)
    output_axes = tuple(index for index, extent in enumerate(output_shape) if extent != 1)
    reduced_input = tuple(input_shape[index] for index in input_axes)
    reduced_output = tuple(output_shape[index] for index in output_axes)
    if not reduced_input and not reduced_output:
        return tuple(tuple(0 for _ in input_shape) for _ in output_shape)
    if reduced_input and reduced_output:
        reduced = _shape_map_matrix(reduced_input, reduced_output)
        if reduced is not None:
            expanded = [[0 for _ in input_shape] for _ in output_shape]
            for reduced_output_axis, output_axis in enumerate(output_axes):
                for reduced_input_axis, input_axis in enumerate(input_axes):
                    expanded[output_axis][input_axis] = reduced[reduced_output_axis][reduced_input_axis]
            return tuple(tuple(row) for row in expanded)
    return _shape_map_matrix(input_shape, output_shape)


def _shape_map_matrix(
    input_shape: tuple[int, ...],
    output_shape: tuple[int, ...],
) -> tuple[tuple[int, ...], ...] | None:
    matrix = [[0 for _ in input_shape] for _ in output_shape]

    def input_product(output_axis: int) -> int:
        return prod(
            input_shape[input_axis]
            for input_axis in range(len(input_shape))
            if matrix[output_axis][input_axis]
        )

    def output_product(input_axis: int) -> int:
        return prod(
            output_shape[output_axis]
            for output_axis in range(len(output_shape))
            if matrix[output_axis][input_axis]
        )

    output_axis = 0
    input_axis = 0
    input_start = -1
    paths: list[tuple[int, int]] = []
    while (
        0 <= output_axis < len(output_shape)
        and 0 <= input_axis < len(input_shape)
    ):
        if (output_axis, input_axis) in paths:
            return None
        matrix[output_axis][input_axis] = 1
        paths.append((output_axis, input_axis))
        difference = input_product(output_axis) - output_product(input_axis)
        if difference == 0:
            output_axis += 1
            input_axis += 1
            if output_axis >= len(output_shape) and input_axis < len(input_shape):
                output_axis -= 1
            elif input_axis >= len(input_shape) and output_axis < len(output_shape):
                input_axis -= 1
            input_start = -1
        elif difference < 0:
            input_start = input_axis if input_start == -1 else input_start
            input_axis += 1
        elif input_product(output_axis) % output_shape[output_axis] == 0:
            output_axis += 1
            if input_start != -1 and output_axis < len(output_shape):
                for previous_input_axis in range(input_start, input_axis):
                    matrix[output_axis][previous_input_axis] = 1
            input_start = -1
        else:
            matrix[output_axis][input_axis] = 0
            input_axis -= 1
            paths.pop()
            input_start = -1
    if output_axis != len(output_shape) or input_axis != len(input_shape):
        return None
    return tuple(tuple(row) for row in matrix)


def _complete_shape_map(
    matrix: tuple[tuple[int, ...], ...],
) -> tuple[dict[int, list[int]], dict[int, list[int]]]:
    forward: dict[int, list[int]] = {}
    backward: dict[int, list[int]] = {}
    output_count = len(matrix)
    input_count = len(matrix[0]) if matrix else 0
    for output_axis in range(output_count):
        for input_axis in range(input_count):
            if not matrix[output_axis][input_axis]:
                continue
            if all(
                other_input == input_axis or not matrix[output_axis][other_input]
                for other_input in range(input_count)
            ):
                forward.setdefault(input_axis, []).append(output_axis)
            if all(
                other_output == output_axis or not matrix[other_output][input_axis]
                for other_output in range(output_count)
            ):
                backward.setdefault(output_axis, []).append(input_axis)
    return forward, backward


def _flatten_block_cyclic_splits(
    source: DistributedType,
    input_axes: list[int],
    input_shape: tuple[int, ...],
) -> SBPSplit | None:
    stages: list[SplitStage] = []
    for position, input_axis in enumerate(input_axes):
        policy = source.axis_policies[input_axis]
        if not isinstance(policy, SBPSplit):
            continue
        parent_extent = input_shape[input_axis]
        for stage in policy.stages:
            if not isinstance(stage.distribution, BlockCyclicSplit):
                return None
            shard_count = prod(
                source.placement.hierarchy[axis]
                for axis in stage.hierarchy_axes
            )
            if parent_extent % (shard_count * stage.distribution.block_size):
                return None
            parent_extent //= shard_count
        trailing_extent = prod(input_shape[axis] for axis in input_axes[position + 1 :])
        scaled = scale_split_units(policy, trailing_extent, 1)
        if scaled is None:  # pragma: no cover - multiplication is exact.
            return None
        stages.extend(scaled.stages)
    if not stages:
        return None
    try:
        return SBP.split(*stages)
    except IRSchemaError:
        return None


def _split_divisor(split: SBPSplit, source: DistributedType) -> int:
    return prod(
        source.placement.hierarchy[axis]
        for stage in split.stages
        for axis in stage.hierarchy_axes
    )


def _distributed_reshape_error(source: DistributedType, output: TensorType) -> IRSchemaError:
    return IRSchemaError(
        "F.tensors.reshape cannot preserve distributed layout "
        f"from {source.tensor.shape} to {output.shape}."
    )


__all__ = ["Reshape"]
