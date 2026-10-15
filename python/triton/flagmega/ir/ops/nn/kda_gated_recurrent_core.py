# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT
"""KDA recurrent step with the forget gate and the beta sigmoid computed in-kernel.

``nn.kda_recurrent_core`` consumes a fully-formed FP32 log-decay and a
post-sigmoid beta, so the published forget gate

    g = lower_bound * sigmoid(exp(A_log) * (g_raw + dt_bias))     # per channel
    beta = sigmoid(b_raw)                                         # per head

has to be built as five graph nodes plus a materialized FP32 ``[1, H, D]``
intermediate per layer.  This operator is the reference fused form instead —
upstream ``fused_recurrent_kda(use_gate_in_kernel=True,
use_beta_sigmoid_in_kernel=True, lower_bound=...)`` and
``naive_kda_lowerbound_gate`` — taking the *raw* projections and the two gate
weights and evaluating the whole gate in FP32 registers inside the recurrence.

Deliberately a separate operator rather than attributes on
``KdaRecurrentCore``: the operand list differs (raw ``g``/``beta`` plus
``decay_rate``/``dt_bias``), so one op with optional operands would make the
type contract depend on an attribute.  Decode-1 only, and — like the rest of
the KDA family — distributed operands must agree on tensor axis 1, the head
axis: either all broadcast or all carrying one common split.
"""

from collections.abc import Mapping, Sequence
from math import prod

from triton.flagmega.errors import IRSchemaError
from triton.flagmega.ir.distributed_inference import tensor_of
from triton.flagmega.ir.model import Node, tensor_type
from triton.flagmega.ir.model import effect as ir_effect
from triton.flagmega.ir.ops.core import (
    OpCost,
    OpDefinition,
    attribute_parameter,
    input_parameter,
    op_definition,
    tensor_nbytes,
)
from triton.flagmega.ir.ops.nn.kda import (
    _head_choices,
    _head_input_relation,
    _head_output_candidates,
    _head_placement,
    _head_results,
    _normalize_output_lanes,
    _recurrent_output_type,
)
from triton.flagmega.ir.type_pattern import is_tensor
from triton.flagmega.ir.types import DType


@op_definition("nn.kda_gated_recurrent_core", namespace="nn",
               functional_name="kda_gated_recurrent_core",
               display_name="NN.KdaGatedRecurrentCore")
class KdaGatedRecurrentCore(OpDefinition):
    """One KDA recurrent step that also computes its forget gate and beta."""

    const_evaluable = True
    query = input_parameter(is_tensor())
    key = input_parameter(is_tensor())
    value = input_parameter(is_tensor())
    gate = input_parameter(is_tensor())
    decay_rate = input_parameter(is_tensor())
    gate_bias = input_parameter(is_tensor())
    beta_logits = input_parameter(is_tensor())
    state = input_parameter(is_tensor())
    lower_bound = attribute_parameter()
    epsilon = attribute_parameter(default=1e-6)
    l2norm_in_kernel = attribute_parameter(default=True)
    output_lanes = attribute_parameter(default=())

    @classmethod
    def normalize_attrs(cls, attributes):
        attrs = super().normalize_attrs(attributes)
        if not isinstance(attrs["l2norm_in_kernel"], bool):
            raise IRSchemaError("KdaGatedRecurrentCore l2norm_in_kernel must be boolean.")
        epsilon = attrs["epsilon"]
        if isinstance(epsilon, bool) or not isinstance(epsilon, (int, float)) or not epsilon > 0:
            raise IRSchemaError("KdaGatedRecurrentCore epsilon must be positive.")
        bound = attrs["lower_bound"]
        if isinstance(bound, bool) or not isinstance(bound, (int, float)) or not bound < 0:
            raise IRSchemaError(
                "KdaGatedRecurrentCore lower_bound must be a negative number; the "
                "softplus gate variant is a different contract.")
        return {**attrs, "lower_bound": float(bound), "epsilon": float(epsilon),
                "output_lanes": _normalize_output_lanes(attrs["output_lanes"], "KdaGatedRecurrentCore")}

    @classmethod
    def ir_attrs(cls, attrs):
        return {key: value for key, value in attrs.items() if key != "output_lanes" or value}

    @classmethod
    def infer_type(cls, inputs, attrs):
        def extents(value_type):
            return tuple(dim.fixed_value if dim.is_fixed else None for dim in value_type.shape)

        q = tensor_of(cls.query.type_of(inputs))
        k = tensor_of(cls.key.type_of(inputs))
        v = tensor_of(cls.value.type_of(inputs))
        gate = tensor_of(cls.gate.type_of(inputs))
        decay_rate = tensor_of(cls.decay_rate.type_of(inputs))
        gate_bias = tensor_of(cls.gate_bias.type_of(inputs))
        beta = tensor_of(cls.beta_logits.type_of(inputs))
        state = tensor_of(cls.state.type_of(inputs))
        tensors = (q, k, v, gate, decay_rate, gate_bias, beta, state)
        if any(not dimension.is_fixed for tensor in tensors for dimension in tensor.shape):
            raise IRSchemaError("KdaGatedRecurrentCore decode requires fixed tensor extents.")
        if q.dtype != DType.BFLOAT16 or k.dtype != DType.BFLOAT16 or v.dtype != DType.BFLOAT16:
            raise IRSchemaError("KdaGatedRecurrentCore requires BF16 q/k/v.")
        # The raw gate and beta projections arrive in the model's activation
        # dtype; the kernel widens them once, exactly like the reference, so
        # BF16 here is the upstream form rather than a precision loss.
        if gate.dtype not in {DType.BFLOAT16, DType.FLOAT32}:
            raise IRSchemaError("KdaGatedRecurrentCore raw gate must be BF16 or FP32.")
        if beta.dtype not in {DType.BFLOAT16, DType.FLOAT32}:
            raise IRSchemaError("KdaGatedRecurrentCore raw beta must be BF16 or FP32.")
        if decay_rate.dtype != DType.FLOAT32 or gate_bias.dtype != DType.FLOAT32:
            raise IRSchemaError(
                "KdaGatedRecurrentCore requires FP32 decay_rate/gate_bias weight maps.")
        if state.dtype != DType.FLOAT32:
            raise IRSchemaError("KdaGatedRecurrentCore requires an FP32 state.")
        if q.rank != 3 or extents(k) != extents(q) or extents(v) != extents(q):
            raise IRSchemaError("KdaGatedRecurrentCore q/k/v must share [1, heads, head_k] shapes.")
        if extents(gate) != extents(q):
            raise IRSchemaError("KdaGatedRecurrentCore raw gate must be [1, heads, head_k].")
        if extents(decay_rate) != extents(q) or extents(gate_bias) != extents(q):
            raise IRSchemaError(
                "KdaGatedRecurrentCore decay_rate/gate_bias maps must be [1, heads, head_k]; "
                "broadcast exp(A_log) and dt_bias in the constant island.")
        if q.shape[0].is_fixed and q.shape[0].fixed_value != 1:
            raise IRSchemaError("KdaGatedRecurrentCore decode requires exactly one token.")
        if beta.rank != 2 or extents(beta) != extents(q)[:2]:
            raise IRSchemaError("KdaGatedRecurrentCore raw beta must be [1, heads].")
        if state.rank != 4 or not state.shape[3].is_fixed or extents(state)[:3] != (1, *extents(q)[1:]):
            raise IRSchemaError(
                "KdaGatedRecurrentCore state must be [1, heads, head_k, head_v] FP32.")
        head_v = state.shape[3].fixed_value
        out = _recurrent_output_type(q, head_v, attrs, "KdaGatedRecurrentCore")
        new_state = tensor_type(DType.FLOAT32,
                                (1, q.shape[1].fixed_value, q.shape[2].fixed_value, head_v))
        contract = _head_placement(
            tuple(parameter.type_of(inputs) for parameter in cls.input_parameters),
            "KdaGatedRecurrentCore",
        )
        return _head_results((out, new_state), contract)

    @classmethod
    def _input_relation(cls, choices):
        """All operands agree on the head policy when one is offered."""
        yield from _head_choices(choices, "KdaGatedRecurrentCore")

    @classmethod
    def distributed_input_type_tuples(cls, choices, attrs):
        yield from cls._input_relation(choices)

    @classmethod
    def distributed_output_type_candidates(cls, choices, output_type, attrs):
        del attrs
        return _head_output_candidates(choices, output_type)

    @classmethod
    def infer_distributed_input_types(cls, output_type, logical_input_types, attrs):
        del attrs
        return _head_input_relation(output_type, logical_input_types, "KdaGatedRecurrentCore")

    @classmethod
    def infer_effect(cls, inputs: Sequence[Node], attrs: Mapping[str, object]) -> None:
        del inputs, attrs
        return ir_effect("read_write", "kda_recurrent_state")

    @classmethod
    def evaluate(cls, node, arguments, context):
        q = cls.query.read(arguments).float()
        k = cls.key.read(arguments).float()
        v = cls.value.read(arguments).float()
        state = cls.state.read(arguments).float()
        epsilon = float(node.attrs["epsilon"])
        # naive_kda_lowerbound_gate: the raw projection widens first, then the
        # bias add, the exp(A_log) product and the sigmoid all stay FP32.
        gate = cls.gate.read(arguments).float() + cls.gate_bias.read(arguments).float()
        decay = float(node.attrs["lower_bound"]) * context.torch.sigmoid(
            cls.decay_rate.read(arguments).float() * gate)
        beta = context.torch.sigmoid(cls.beta_logits.read(arguments).float())
        if node.attrs["l2norm_in_kernel"]:
            q = q / context.torch.sqrt(q.pow(2).sum(-1, keepdim=True) + epsilon)
            k = k / context.torch.sqrt(k.pow(2).sum(-1, keepdim=True) + epsilon)
        state = state * decay.exp().unsqueeze(-1)
        correction = (state * k.unsqueeze(-1)).sum(-2)
        value = (v - correction) * beta.unsqueeze(-1)
        state = state + k.unsqueeze(-1) * value.unsqueeze(-2)
        out = (state * q.unsqueeze(-1)).sum(-2)
        lanes = tuple(node.attrs.get("output_lanes", ()))
        out = out.to(context.torch.bfloat16)
        return out.reshape(1, out.shape[1], out.shape[2] // prod(lanes), *lanes), state

    @classmethod
    def cost(cls, node):
        sizes = [tensor_nbytes(field) for field in node.type.fields]
        return OpCost(bytes_written=None if None in sizes else sum(sizes),
                      notes=("kda-recurrent", "kda-fused-gate"))


__all__ = ["KdaGatedRecurrentCore"]
