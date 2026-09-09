# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Rounded delta-rule blocks with an explicit aliased FP32 state field.

Query/key/value and coefficients are BF16 boundaries. Each block rounds its
state operand, recalled values, value residual, new values, and decayed new
values to BF16; the persistent state remains FP32. The state field's axis order
and vector packing are explicit, independently of activation distribution.
"""

import math
from collections.abc import Mapping
from dataclasses import dataclass

from triton.flagmega.errors import EvaluationError, IRSchemaError
from triton.flagmega.ir.distributed_inference import placement_of, tensor_of
from triton.flagmega.ir.dim_expr import dim
from triton.flagmega.ir.distributed_type import SBPSplit, scale_split_units
from triton.flagmega.ir.model import DistributedType, SBP, TensorType, TupleType, effect
from triton.flagmega.ir.ops.core import OpCost, OpDefinition, attribute_parameter, input_parameter, op_definition, tensor_nbytes
from triton.flagmega.ir.ops.tensors.pack import pack_physical
from triton.flagmega.ir.ops.tensors.unpack import unpack_physical
from triton.flagmega.ir.type_pattern import has_rank, is_ref, is_tensor
from triton.flagmega.ir.types import DType, VectorType


@op_definition("nn.delta_rule_block_update", namespace="nn", functional_name="delta_rule_block_update",
               display_name="NN.DeltaRuleBlockUpdate")
class DeltaRuleBlockUpdate(OpDefinition):
    supports_broadcast_lifting = False
    query = input_parameter(is_tensor() & has_rank(3))
    key = input_parameter(is_tensor() & has_rank(3))
    value = input_parameter(is_tensor() & has_rank(3))
    coefficients = input_parameter(is_tensor() & has_rank(4))
    log_prefix = input_parameter(is_tensor() & has_rank(3))
    state = input_parameter(is_ref(), memory_effect="read_write")
    scale = attribute_parameter(default=None)
    state_field = attribute_parameter(default="matrix")
    state_layout = attribute_parameter(default=("head", "value", "key"))
    state_vector_axes = attribute_parameter(default=())

    @classmethod
    def normalize_attrs(cls, attributes):
        attrs = super().normalize_attrs(attributes)
        scale = attrs["scale"]
        if scale is not None:
            if isinstance(scale, bool) or not isinstance(scale, (int, float)) or not math.isfinite(scale) or scale <= 0:
                raise IRSchemaError("DeltaRuleBlockUpdate scale must be finite and positive.")
            attrs["scale"] = float(scale)
        if not isinstance(attrs["state_field"], str) or not attrs["state_field"] or "." in attrs["state_field"]:
            raise IRSchemaError("DeltaRuleBlockUpdate state_field must name one direct reference field.")
        for name in ("state_layout", "state_vector_axes"):
            if not isinstance(attrs[name], (tuple, list)) or not all(isinstance(axis, str) for axis in attrs[name]):
                raise IRSchemaError(f"DeltaRuleBlockUpdate {name} must be an axis-name sequence.")
            attrs[name] = tuple(attrs[name])
        layout = attrs["state_layout"]
        if len(set(layout)) != len(layout) or set(layout) not in ({"head", "value", "key"
                                                                   }, {"layer", "head", "value", "key"}):
            raise IRSchemaError("DeltaRuleBlockUpdate state_layout requires head/value/key and an optional unit layer.")
        if any(axis not in layout for axis in attrs["state_vector_axes"]):
            raise IRSchemaError("DeltaRuleBlockUpdate vector axes must occur in state_layout.")
        return attrs

    @classmethod
    def infer_type(cls, inputs, attrs):
        types = tuple(parameter.type_of(inputs) for parameter in cls.input_parameters[:-1])
        validate_block_tensors(types, attrs)
        state = cls.state.type_of(inputs)
        field_type = dict(state.fields).get(attrs["state_field"])
        state_field_layout(field_type, attrs, tensor_of(types[2]), tensor_of(types[1]))
        return TupleType((types[2], state))

    @classmethod
    def distributed_input_type_tuples(cls, choices, attrs):
        # Every tensor's head ownership is determined by the key operand.
        # Intersect those exact contracts with the available producer types;
        # do not invent a reshard or enumerate unrelated token/feature splits.
        del attrs
        available = tuple(frozenset(values) for values in choices)
        value_tensor = tensor_of(choices[cls.value.input_index][0])
        coefficient_tensor = tensor_of(choices[cls.coefficients.input_index][0])
        prefix_tensor = tensor_of(choices[cls.log_prefix.input_index][0])
        b = SBP.broadcast()
        for key in choices[cls.key.input_index]:
            if not isinstance(key, DistributedType) or key.partial is not None:
                continue
            head = key.axis_policies[1]
            if key.axis_policies != (b, head, b):
                continue
            ratio = value_tensor.shape[1].fixed_value // key.tensor.shape[1].fixed_value
            value_head = scale_split_units(head, ratio, 1) if isinstance(head, SBPSplit) else head
            if value_head is None:
                continue
            inputs = (
                key,
                key,
                DistributedType(value_tensor, (b, value_head, b), key.placement),
                DistributedType(coefficient_tensor, (b, value_head, b, b), key.placement),
                DistributedType(prefix_tensor, (b, value_head, b), key.placement),
            )
            if all(value in contracts for value, contracts in zip(inputs, available)):
                for state in choices[cls.state.input_index]:
                    yield (*inputs, state)

    @classmethod
    def infer_effect(cls, inputs, attrs):
        return effect("read_write", "delta_rule_state")

    @classmethod
    def evaluate(cls, node, arguments, context):
        query, key, value, coefficients, log_prefix, state = (parameter.read(arguments)
                                                              for parameter in cls.input_parameters)
        state_type = context.types[cls.state.read(node.inputs)]
        field_type = dict(state_type.fields)[node.attrs["state_field"]]
        spec = state_field_layout(field_type, node.attrs, tensor_of(context.types[cls.value.read(node.inputs)]),
                                  tensor_of(context.types[cls.key.read(node.inputs)]))
        field = node.attrs["state_field"]
        try:
            storage = state[field] if isinstance(state, Mapping) else getattr(state, field)
        except (KeyError, AttributeError) as error:
            raise EvaluationError(f"DeltaRuleBlockUpdate state has no field {field!r}.") from error
        unpacked = unpack_physical(storage, len(spec.axes), spec.lanes, spec.vector_axes) if spec.lanes else storage
        order = tuple(spec.axes.index(axis) for axis in ("head", "value", "key"))
        if "layer" in spec.axes:
            order = (spec.axes.index("layer"), *order)
        initial = unpacked.permute(order).reshape(value.shape[1], value.shape[2], key.shape[2])
        output, final = delta_rule_block_update(query, key, value, coefficients, log_prefix, initial,
                                                scale=node.attrs["scale"], torch=context.torch)
        canonical_axes = ("head", "value", "key")
        if "layer" in spec.axes:
            final = final.unsqueeze(0)
            canonical_axes = ("layer", *canonical_axes)
        final = final.permute(tuple(canonical_axes.index(axis) for axis in spec.axes)).contiguous()
        packed = pack_physical(final, len(spec.axes), spec.lanes, spec.vector_axes) if spec.lanes else final
        storage.copy_(packed)
        return output, state

    @classmethod
    def cost(cls, node):
        return OpCost(bytes_written=tensor_nbytes(tensor_of(node.type.fields[0])),
                      notes=("rounded-block-state-update", "fp32-reference-state"))


def validate_block_tensors(types, attrs):
    query, key, value, coefficients, prefix = tuple(tensor_of(value) for value in types)
    if any(value.dtype != DType.BFLOAT16
           for value in (query, key, value, coefficients)) or prefix.dtype != DType.FLOAT32:
        raise IRSchemaError("DeltaRuleBlockUpdate requires BF16 Q/K/V/coefficients and FP32 log-prefix.")
    if query.shape != key.shape or query.shape[0] != value.shape[0]:
        raise IRSchemaError("DeltaRuleBlockUpdate query/key and value token extents must match.")
    dimensions = (key.shape[1], value.shape[1], key.shape[2], value.shape[2])
    if any(not dim.is_fixed or dim.fixed_value <= 0 for dim in dimensions):
        raise IRSchemaError("DeltaRuleBlockUpdate requires fixed positive head counts and key/value dimensions.")
    kh, vh = key.shape[1].fixed_value, value.shape[1].fixed_value
    if vh % kh:
        raise IRSchemaError("DeltaRuleBlockUpdate value heads must be a multiple of key heads.")
    block_dim = coefficients.shape[-1]
    if not block_dim.is_fixed or block_dim.fixed_value not in (8, 16, 32, 64):
        raise IRSchemaError("DeltaRuleBlockUpdate coefficient block size must be 8/16/32/64.")
    block = block_dim.fixed_value
    blocks = (value.shape[0] + block - 1) // block
    if coefficients.shape != (blocks, value.shape[1], block_dim, block_dim) or prefix.shape != (blocks, value.shape[1],
                                                                                                block_dim):
        raise IRSchemaError("DeltaRuleBlockUpdate coefficient/log-prefix block extents must match Q/K/V.")
    placement = placement_of(*types)
    if placement is not None:
        if any(not isinstance(value, DistributedType) or value.partial is not None for value in types):
            raise IRSchemaError("DeltaRuleBlockUpdate operands must share materialized distributed ownership.")
        qtype, ktype, vtype, ctype, ptype = types
        b = SBP.broadcast()
        head = ktype.axis_policies[1]
        value_head = scale_split_units(head, vh // kh, 1) if isinstance(head, SBPSplit) else head
        if (value_head is None or qtype.axis_policies != (b, head, b) or ktype.axis_policies != (b, head, b)
                or vtype.axis_policies != (b, value_head, b) or ctype.axis_policies != (b, value_head, b, b)
                or ptype.axis_policies != (b, value_head, b)):
            raise IRSchemaError(
                "DeltaRuleBlockUpdate requires grouped head ownership and broadcast token/feature axes.")


@dataclass(frozen=True)
class StateFieldLayout:
    axes: tuple[str, ...]
    vector_axes: tuple[int, ...]
    lanes: tuple[int, ...]


def state_field_layout(field_type, attrs, value_type, key_type):
    if not isinstance(field_type, TensorType):
        raise IRSchemaError("DeltaRuleBlockUpdate state field must be a tensor.")
    layout = attrs["state_layout"]
    dtype = field_type.dtype
    lanes = dtype.lanes if isinstance(dtype, VectorType) else ()
    scalar = dtype.elem_type if isinstance(dtype, VectorType) else dtype
    axes = tuple(layout.index(axis) for axis in attrs["state_vector_axes"])
    if scalar != DType.FLOAT32 or len(axes) != len(lanes) or field_type.rank != len(layout):
        raise IRSchemaError("DeltaRuleBlockUpdate state must have FP32 elements and matching layout/vector axes.")
    shape = list(field_type.shape)
    for axis, lane in zip(axes, lanes):
        shape[axis] *= lane
    expected = {"layer": dim(1), "head": value_type.shape[1], "value": value_type.shape[2], "key": key_type.shape[2]}
    if any(size != expected[axis] for axis, size in zip(layout, shape)):
        raise IRSchemaError("DeltaRuleBlockUpdate state field extents disagree with its logical axes.")
    return StateFieldLayout(layout, axes, lanes)


def delta_rule_block_update(query, key, value, coefficients, log_prefix, initial, *, scale=None, torch):
    tokens, heads, value_dim = value.shape
    key_dim = key.shape[2]
    block_size = coefficients.shape[-1]
    scale = key_dim**-0.5 if scale is None else scale
    output = torch.empty_like(value)
    state = initial.float().clone()
    repeats = heads // key.shape[1]
    for block in range(coefficients.shape[0]):
        start = block * block_size
        valid = min(block_size, tokens - start)
        q = query.new_zeros((heads, block_size, key_dim), dtype=torch.float32)
        k = q.clone()
        v = value.new_zeros((heads, value_dim, block_size), dtype=torch.float32)
        q[:, :valid] = query[start:start + valid].repeat_interleave(repeats, dim=1).transpose(0, 1).float()
        k[:, :valid] = key[start:start + valid].repeat_interleave(repeats, dim=1).transpose(0, 1).float()
        v[:, :, :valid] = value[start:start + valid].permute(1, 2, 0).float()
        logs = log_prefix[block]
        gamma = torch.exp2(logs)
        ratio = torch.exp2(logs[:, :, None] - logs[:, None, :])
        coefficient = (coefficients[block].float() * ratio).bfloat16().float()
        score = torch.tril((q @ k.transpose(-1, -2)) * ratio * scale).bfloat16().float()
        rounded = state.bfloat16().float()
        carried = (rounded @ q.transpose(-1, -2)) * (gamma * scale)[:, None, :]
        recalled = ((rounded @ k.transpose(-1, -2)) * gamma[:, None, :]).bfloat16().float()
        residual = (v - recalled).bfloat16().float()
        new_value = (residual @ coefficient.transpose(-1, -2)).bfloat16().float()
        block_output = (new_value @ score.transpose(-1, -2) + carried).bfloat16()
        output[start:start + valid] = block_output[:, :, :valid].permute(2, 0, 1)
        end_log = logs[:, valid - 1]
        decay = torch.exp2(end_log[:, None] - logs)
        decayed = (new_value * decay[:, None, :]).bfloat16().float()
        decayed[:, :, valid:] = 0
        state = decayed @ k + state * torch.exp2(end_log)[:, None, None]
    return output, state


__all__ = ["DeltaRuleBlockUpdate"]
