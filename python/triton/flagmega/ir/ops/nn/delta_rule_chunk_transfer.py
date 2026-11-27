# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Explicit BF16 chunk transfers and ordered TF32 state composition."""

from collections.abc import Mapping

from triton.flagmega.errors import IRSchemaError
from triton.flagmega.ir.distributed_inference import tensor_of
from triton.flagmega.ir.model import DistributedType, Node, SBP, TupleType, effect, tensor_type
from triton.flagmega.ir.ops.core import OpCost, OpDefinition, attribute_parameter, input_parameter, op_definition, tensor_nbytes
from triton.flagmega.ir.ops.nn.delta_rule_block_update import (
    DeltaRuleBlockUpdate, delta_rule_block_update, state_field_layout, validate_block_tensors,
)
from triton.flagmega.ir.ops.tensors.pack import pack_physical
from triton.flagmega.ir.ops.tensors.unpack import unpack_physical
from triton.flagmega.ir.type_pattern import has_rank, is_ref, is_tensor
from triton.flagmega.ir.types import DType


def normalize_chunk_size(attrs):
    size = attrs["chunk_size"]
    if isinstance(size, bool) or not isinstance(size, int) or size <= 0 or size % 64:
        raise IRSchemaError("Delta-rule chunk_size must be a positive multiple of 64.")
    return attrs


@op_definition("nn.delta_rule_chunk_transfer", namespace="nn", functional_name="delta_rule_chunk_transfer")
class DeltaRuleChunkTransfer(OpDefinition):
    supports_broadcast_lifting = False
    key = input_parameter(is_tensor() & has_rank(3))
    value = input_parameter(is_tensor() & has_rank(3))
    coefficients = input_parameter(is_tensor() & has_rank(4))
    log_prefix = input_parameter(is_tensor() & has_rank(3))
    matrix_kind = attribute_parameter()
    chunk_size = attribute_parameter(default=512)

    @classmethod
    def normalize_attrs(cls, attributes):
        attrs = normalize_chunk_size(super().normalize_attrs(attributes))
        if attrs["matrix_kind"] not in ("transfer", "state"):
            raise IRSchemaError("Delta-rule matrix_kind must be 'transfer' or 'state'.")
        return attrs

    @classmethod
    def infer_type(cls, inputs, attrs):
        types = tuple(p.type_of(inputs) for p in cls.input_parameters)
        key, value, coefficients, _ = tuple(tensor_of(t) for t in types)
        validate_block_tensors((types[0], *types), attrs)
        if key.shape[-1].fixed_value != 128 or value.shape[-1].fixed_value != 128 or coefficients.shape[-1].fixed_value != 64:
            raise IRSchemaError("Chunk transfer requires 128-wide state and 64-token coefficient blocks.")
        output = tensor_type("float32", ((key.shape[0] + attrs["chunk_size"] - 1) // attrs["chunk_size"], value.shape[1], 128, 128))
        if not isinstance(types[1], DistributedType):
            return output
        b = SBP.broadcast()
        return DistributedType(output, (b, types[1].axis_policies[1], b, b), types[1].placement)

    @classmethod
    def distributed_input_type_tuples(cls, choices, attrs):
        # Reuse the exact grouped-head relation without a state operand.
        sentinel = tensor_type("int32", ())
        for values in DeltaRuleBlockUpdate.distributed_input_type_tuples((choices[0], *choices, (sentinel,)), attrs):
            yield values[1:-1]

    @classmethod
    def evaluate(cls, node, arguments, context):
        return chunk_transfer(*arguments, chunk_size=node.attrs["chunk_size"],
                              matrix_kind=node.attrs["matrix_kind"], torch=context.torch)

    @classmethod
    def cost(cls, node):
        return OpCost(bytes_written=tensor_nbytes(tensor_of(node.type)), notes=("ordered-bf16-chunk-transfer",))


def chunk_transfer(key, value, coefficients, log_prefix, *, chunk_size, matrix_kind, torch):
    tokens, heads, width = value.shape
    key = key.repeat_interleave(heads // key.shape[1], dim=1)
    outputs = []
    for start in range(0, tokens, chunk_size):
        acc = (torch.eye(width, device=key.device).expand(heads, width, width).clone()
               if matrix_kind == "transfer" else torch.zeros(heads, width, width, device=key.device))
        for token in range(start, min(start + chunk_size, tokens), 64):
            valid = min(64, tokens - token)
            k = key.new_zeros((heads, 64, width))
            v = value.new_zeros((heads, width, 64))
            k[:, :valid] = key[token:token + valid].transpose(0, 1)
            v[:, :, :valid] = value[token:token + valid].permute(1, 2, 0)
            block = token // 64
            x = (-(coefficients[block].transpose(-1, -2).float() @ k.float())).bfloat16()
            p = log_prefix[block]
            end = p[:, valid - 1]
            gamma = torch.exp2(end)[:, None, None]
            if matrix_kind == "transfer":
                z = k.transpose(-1, -2) if token == start else (acc.bfloat16().float() @ k.transpose(-1, -2).float()).bfloat16()
                acc = (acc + z.float() @ x.float()) * gamma
            else:
                mask = torch.arange(64, device=key.device) < valid
                decay = torch.exp2(torch.where(mask, end[:, None] - p, 0.))
                negative = torch.where(mask[None, None, :], -v.float() * decay[:, None, :], 0.)
                y = negative if token == start else torch.addcmul(negative, gamma, acc.bfloat16().float() @ k.transpose(-1, -2).float())
                acc = gamma * acc + y.bfloat16().float() @ x.float()
        outputs.append(acc)
    return torch.stack(outputs)


def _initial_state(state, state_type, attrs, value_type, key_type):
    field = attrs["state_field"]
    storage = state[field] if isinstance(state, Mapping) else getattr(state, field)
    spec = state_field_layout(dict(state_type.fields)[field], attrs, value_type, key_type)
    unpacked = unpack_physical(storage, len(spec.axes), spec.lanes, spec.vector_axes) if spec.lanes else storage
    order = tuple(spec.axes.index(axis) for axis in ("head", "value", "key"))
    if "layer" in spec.axes:
        order = (spec.axes.index("layer"), *order)
    return unpacked.permute(order).reshape(value_type.shape[1].fixed_value, 128, 128), storage, spec


@op_definition("nn.delta_rule_chunk_fixup", namespace="nn", functional_name="delta_rule_chunk_fixup")
class DeltaRuleChunkFixup(OpDefinition):
    supports_broadcast_lifting = False
    transfer = input_parameter(is_tensor() & has_rank(4))
    local_state = input_parameter(is_tensor() & has_rank(4))
    state = input_parameter(is_ref(), memory_effect="read")
    state_field = attribute_parameter(default="matrix")
    state_layout = attribute_parameter(default=("head", "value", "key"))
    state_vector_axes = attribute_parameter(default=())

    @classmethod
    def normalize_attrs(cls, attributes):
        attrs = DeltaRuleBlockUpdate.normalize_attrs(super().normalize_attrs(attributes))
        attrs.pop("scale")
        return attrs

    @classmethod
    def infer_type(cls, inputs, attrs):
        m, n = (p.type_of(inputs) for p in cls.input_parameters[:2])
        logical = tensor_of(n)
        if m != n or logical.dtype != DType.FLOAT32 or tuple(d.fixed_value for d in logical.shape[-2:]) != (128, 128):
            raise IRSchemaError("Chunk fixup requires matching FP32 chunk/head/128/128 matrices.")
        if isinstance(n, DistributedType):
            b = SBP.broadcast()
            if n.partial is not None or n.axis_policies != (b, n.axis_policies[1], b, b):
                raise IRSchemaError("Chunk fixup requires broadcast chunk/state axes and materialized head ownership.")
        tensor = tensor_type("bfloat16", (1, logical.shape[1], 128))
        state_field_layout(dict(cls.state.type_of(inputs).fields).get(attrs["state_field"]), attrs, tensor, tensor)
        return n

    @classmethod
    def distributed_input_type_tuples(cls, choices, attrs):
        del attrs
        common = set(choices[1])
        for transfer in choices[0]:
            if transfer in common:
                for state in choices[2]:
                    yield transfer, transfer, state

    @classmethod
    def infer_effect(cls, inputs, attrs):
        return effect("read", "delta_rule_state")

    @classmethod
    def evaluate(cls, node, arguments, context):
        m, n, state = arguments
        logical = tensor_of(context.types[node.inputs[1]])
        tensor = tensor_type("bfloat16", (1, logical.shape[1], 128))
        initial, _, _ = _initial_state(state, context.types[node.inputs[2]], node.attrs, tensor, tensor)
        return chunk_fixup(m, n, initial, torch=context.torch)

    @classmethod
    def cost(cls, node):
        return OpCost(bytes_written=tensor_nbytes(tensor_of(node.type)), notes=("ordered-tf32-chunk-state-composition",))


def chunk_fixup(transfer, local_state, initial, *, torch):
    state = initial.float().clone()
    outputs = []
    for m, n in zip(transfer, local_state):
        s = (state.contiguous().view(torch.int32) & -8192).view(torch.float32)
        m = (m.contiguous().view(torch.int32) & -8192).view(torch.float32)
        state = n.clone()
        for k in range(0, 128, 8):
            state = state + s[..., k:k + 8] @ m[..., k:k + 8, :]
        outputs.append(state)
    return torch.stack(outputs)


@op_definition("nn.delta_rule_chunked_block_update", namespace="nn", functional_name="delta_rule_chunked_block_update")
class DeltaRuleChunkedBlockUpdate(OpDefinition):
    supports_broadcast_lifting = False
    query = input_parameter(is_tensor() & has_rank(3))
    key = input_parameter(is_tensor() & has_rank(3))
    value = input_parameter(is_tensor() & has_rank(3))
    coefficients = input_parameter(is_tensor() & has_rank(4))
    log_prefix = input_parameter(is_tensor() & has_rank(3))
    state = input_parameter(is_ref(), memory_effect="read_write")
    fixed_state = input_parameter(is_tensor() & has_rank(4))
    scale = attribute_parameter(default=None)
    state_field = attribute_parameter(default="matrix")
    state_layout = attribute_parameter(default=("head", "value", "key"))
    state_vector_axes = attribute_parameter(default=())
    chunk_size = attribute_parameter(default=512)

    @classmethod
    def normalize_attrs(cls, attributes):
        attributes = dict(attributes)
        size = attributes.pop("chunk_size", 512)
        return normalize_chunk_size({**DeltaRuleBlockUpdate.normalize_attrs(attributes), "chunk_size": size})

    @classmethod
    def infer_type(cls, inputs, attrs):
        base = {k: v for k, v in attrs.items() if k != "chunk_size"}
        result = DeltaRuleBlockUpdate.infer_type(inputs[:6], base)
        fixed = cls.fixed_state.type_of(inputs)
        expected = DeltaRuleChunkTransfer.infer_type((inputs[1], inputs[2], inputs[3], inputs[4]),
                                                     {"chunk_size": attrs["chunk_size"], "matrix_kind": "state"})
        if fixed != expected:
            raise IRSchemaError("Chunked update requires matching materialized fixed chunk states.")
        return result

    @classmethod
    def distributed_input_type_tuples(cls, choices, attrs):
        for inputs in DeltaRuleBlockUpdate.distributed_input_type_tuples(choices[:6], attrs):
            fake = tuple(Node(str(i), "builtin.var", (), t) for i, t in enumerate(inputs))
            expected = DeltaRuleChunkTransfer.infer_type((fake[1], fake[2], fake[3], fake[4]), attrs)
            if expected in choices[6]:
                yield (*inputs, expected)

    infer_effect = classmethod(lambda cls, inputs, attrs: effect("read_write", "delta_rule_state"))

    @classmethod
    def evaluate(cls, node, arguments, context):
        q, k, v, c, p, state, fixed = arguments
        value_type, key_type = (tensor_of(context.types[node.inputs[i]]) for i in (2, 1))
        initial, storage, spec = _initial_state(state, context.types[node.inputs[5]], node.attrs, value_type, key_type)
        outputs = []
        size = node.attrs["chunk_size"]
        for chunk, begin in enumerate(range(0, len(q), size)):
            end = min(begin + size, len(q))
            output, final = delta_rule_block_update(q[begin:end], k[begin:end], v[begin:end],
                                                    c[begin // 64:(end + 63) // 64], p[begin // 64:(end + 63) // 64],
                                                    initial if chunk == 0 else fixed[chunk - 1],
                                                    scale=node.attrs["scale"], torch=context.torch)
            outputs.append(output)
        axes = ("head", "value", "key")
        if "layer" in spec.axes:
            final, axes = final.unsqueeze(0), ("layer", *axes)
        final = final.permute(tuple(axes.index(a) for a in spec.axes)).contiguous()
        storage.copy_(pack_physical(final, len(spec.axes), spec.lanes, spec.vector_axes) if spec.lanes else final)
        return context.torch.cat(outputs), state

    cost = classmethod(lambda cls, node: DeltaRuleBlockUpdate.cost(node))
