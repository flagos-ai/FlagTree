# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Kimi Delta Attention (KDA) decode operators for GLM-5.3-Flash.

New operators — deliberately independent of the GDN family, whose kernels use
per-head scalar decay. KDA (Kimi Linear, arXiv:2510.26692) applies
fine-grained per-channel decay:

    S_t = (I - β_t k_t k_tᵀ) Diag(g_t) S_{t-1} + β_t k_t v_tᵀ
    o_t = S_tᵀ (query_scale * q_t)

with q/k L2-normalized, ``g_t`` the per-channel log-decay emitted by the
layer's forget gate (GLM-5.3: ``lower_bound * sigmoid(exp(A_log)·(g+dt_bias))``
per channel), and ``β_t`` a per-head scalar in [0,1]. Decode-1 only: both
operators require a single token and thread their states functionally.

The recurrence is independent per head, and every KDA operand and result
carries the head count on tensor axis 1 — ``[1, H, D]`` projections, ``[1, H]``
beta, ``[1, H, Dk, Dv]`` recurrent state, ``[3, H, D, K]`` conv weight/state.
So the distributed contract is: axis 1 is either broadcast on every operand or
carries the *same* split on every operand, every other axis stays broadcast,
and no operand is partial or exclusive. Mixed head ownership is meaningless
here — one head's state, gate and projections must live on one owner.
"""

from collections.abc import Mapping, Sequence
from math import isfinite, prod

from triton.flagmega.errors import IRSchemaError
from triton.flagmega.ir.distributed_inference import tensor_of
from triton.flagmega.ir.distributed_type import SBPBroadCast, SBPSplit
from triton.flagmega.ir.model import SBP, DistributedType, Node, TensorType, TupleType, tensor_type
from triton.flagmega.ir.model import effect as ir_effect
from triton.flagmega.ir.ops.core import (
    OpCost,
    OpDefinition,
    attribute_parameter,
    input_parameter,
    op_definition,
    tensor_nbytes,
)
from triton.flagmega.ir.type_pattern import is_tensor
from triton.flagmega.ir.types import DType, VectorType

#: Every KDA operand and result carries the head count on this tensor axis.
HEAD_AXIS = 1


def _normalize_output_lanes(lanes, opname):
    if not isinstance(lanes, (tuple, list)) or any(
            isinstance(lane, bool) or not isinstance(lane, int) or lane <= 0 for lane in lanes):
        raise IRSchemaError(f"{opname} output_lanes must be a sequence of positive integers.")
    return tuple(lanes)


def _normalize_query_scale(value, opname):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not isfinite(value) or value <= 0:
        raise IRSchemaError(f"{opname} query_scale must be a positive finite number.")
    return float(value)


def _recurrent_output_type(query, head_v, attrs, opname):
    """Pack only the activation's value axis, leaving recurrent state scalar."""
    lanes = tuple(attrs.get("output_lanes", ()))
    if head_v % prod(lanes):
        raise IRSchemaError(f"{opname} head_v must be divisible by output_lanes.")
    return tensor_type(VectorType(DType.BFLOAT16, lanes) if lanes else DType.BFLOAT16,
                       (1, query.shape[1].fixed_value, head_v // prod(lanes)))


def _head_placement(types, opname):
    """Validate the KDA distributed contract and return ``(placement, head)``.

    ``head`` is the policy shared by tensor axis 1 of every operand: either
    ``SBP.broadcast()`` or one common split. Returns ``None`` when no operand is
    distributed. Only the head axis may be split because only heads are
    algebraically independent — channels and state columns are contracted
    within the step, and the head's state must sit with its own gate.
    """

    distributed = tuple(value for value in types if isinstance(value, DistributedType))
    if not distributed:
        return None
    if len(distributed) != len(types):
        raise IRSchemaError(f"{opname} requires either all plain or all distributed operands.")
    placement = distributed[0].placement
    if any(value.placement != placement for value in distributed[1:]):
        raise IRSchemaError(f"{opname} distributed operands must use one placement.")
    if any(value.partial is not None or value.exclusive is not None for value in distributed):
        raise IRSchemaError(f"{opname} cannot consume partial or exclusive operands.")
    if any(value.tensor.rank <= HEAD_AXIS for value in distributed):
        raise IRSchemaError(f"{opname} operands must carry the head count on tensor axis {HEAD_AXIS}.")
    if any(not isinstance(policy, SBPBroadCast)
           for value in distributed
           for axis, policy in enumerate(value.axis_policies) if axis != HEAD_AXIS):
        raise IRSchemaError(
            f"{opname} may split the head axis only; every other axis must stay broadcast.")
    heads = {value.axis_policies[HEAD_AXIS] for value in distributed}
    if len(heads) != 1:
        raise IRSchemaError(
            f"{opname} operands must agree on head ownership; a head's state, gate and "
            "projections cannot be split across owners.")
    return placement, heads.pop()


def _head_typed(field, placement, head):
    """Re-apply the common head policy to one inferred result tensor."""

    policies = [SBP.broadcast() for _ in field.shape]
    policies[HEAD_AXIS] = head
    return DistributedType(field, tuple(policies), placement)


def _head_results(fields, contract):
    if contract is None:
        return TupleType(tuple(fields))
    placement, head = contract
    return TupleType(tuple(_head_typed(field, placement, head) for field in fields))


def _head_choices(choices, opname):
    """Enumerate the operand tuples that satisfy the KDA contract.

    The contract pins every operand to one shared ``(placement, head policy)``
    key, so the admissible tuples are indexed by that key rather than drawn
    from the cartesian product of the per-operand choices — an eight-operand
    product is millions of tuples of which only a handful are legal.
    """

    if not choices or any(not values for values in choices):
        return
    plain = tuple(next((value for value in values if not isinstance(value, DistributedType)), None)
                  for values in choices)
    if all(value is not None for value in plain):
        yield plain
    keys: dict[tuple[object, object], None] = {}
    for values in choices:
        for value in values:
            if isinstance(value, DistributedType) and value.tensor.rank > HEAD_AXIS:
                keys[(value.placement, value.axis_policies[HEAD_AXIS])] = None
    for placement, head in keys:
        candidate = tuple(
            next((value for value in values
                  if isinstance(value, DistributedType) and value.placement == placement
                  and value.axis_policies[HEAD_AXIS] == head
                  and all(isinstance(policy, SBPBroadCast)
                          for axis, policy in enumerate(value.axis_policies) if axis != HEAD_AXIS)
                  and value.partial is None and value.exclusive is None), None)
            for values in choices)
        if any(value is None for value in candidate):
            continue
        try:
            _head_placement(candidate, opname)
        except IRSchemaError:
            continue
        yield candidate


def _head_output_candidates(choices, output_type):
    """Offer the head-split result contracts visible from the operand choices.

    ``TypeInferenceCandidateProvider`` seeds target-owned output types only for
    ops returning a single ``DistributedType``; the KDA ops return tuples, so
    without an op-owned seed the search never asks for a head-split result. The
    seed matters because some operands have no head-split layout available at
    their use edge — GLM-5.3's ``[1, heads]`` beta comes straight off a
    replicated projection — and only an output demand lets the planner insert
    that one reshard instead of dropping the whole split.
    """

    keys: dict[tuple[object, object], None] = {}
    for values in choices:
        for value in values:
            if (isinstance(value, DistributedType) and value.tensor.rank > HEAD_AXIS
                    and isinstance(value.axis_policies[HEAD_AXIS], SBPSplit)
                    and value.partial is None and value.exclusive is None
                    and all(isinstance(policy, SBPBroadCast)
                            for axis, policy in enumerate(value.axis_policies) if axis != HEAD_AXIS)):
                keys[(value.placement, value.axis_policies[HEAD_AXIS])] = None
    fields = tuple(tensor_of(field) for field in output_type.fields) if isinstance(
        output_type, TupleType) else (tensor_of(output_type), )
    results = []
    for placement, head in keys:
        try:
            results.append(TupleType(tuple(_head_typed(field, placement, head) for field in fields)))
        except IRSchemaError:
            # The split does not divide some result's head extent on this mesh.
            continue
    return tuple(results)


def _head_input_relation(output_type, logical_input_types, opname):
    """Invert a requested output contract onto the operand list.

    The relation is the identity on the head policy: a head-split result can
    only come from head-split operands, so the demand propagates unchanged.
    """

    del opname
    fields = output_type.fields if isinstance(output_type, TupleType) else (output_type,)
    if not fields or any(not isinstance(field, DistributedType) or field.partial is not None
                         or field.exclusive is not None for field in fields):
        return ()
    placement = fields[0].placement
    heads = {field.axis_policies[HEAD_AXIS] for field in fields}
    if len(heads) != 1 or any(field.placement != placement for field in fields):
        return ()
    if any(not isinstance(policy, SBPBroadCast)
           for field in fields
           for axis, policy in enumerate(field.axis_policies) if axis != HEAD_AXIS):
        return ()
    head = heads.pop()
    if any(not isinstance(value, TensorType) or value.rank <= HEAD_AXIS
           for value in logical_input_types):
        return ()
    try:
        inputs = tuple(_head_typed(value, placement, head) for value in logical_input_types)
    except IRSchemaError:
        # The split does not divide some operand's head extent on this mesh.
        return ()
    return (inputs, )


@op_definition("nn.kda_convolution", namespace="nn", functional_name="kda_convolution",
               display_name="NN.KdaConvolution")
class KdaConvolution(OpDefinition):
    """Causal depthwise short convolution update (decode): silu(conv(state, x)).

    The three published projections stay separate and head-major: q/k/v arrive
    as ``[1, heads, head_dim]`` and the weight/state carry a leading part axis
    (``[3, heads, head_dim, kernel]`` / ``[..., kernel - 1]``).  This mirrors
    upstream's three independent ``q/k/v_conv1d`` layers.  A single flat
    ``[3*heads*head_dim, kernel]`` channel axis would interleave the parts, so
    head ``h`` would own three discontiguous ranges and no SBP policy could
    describe per-owner head ownership.
    """

    const_evaluable = True
    query = input_parameter(is_tensor())
    key = input_parameter(is_tensor())
    value = input_parameter(is_tensor())
    conv_state = input_parameter(is_tensor())
    conv_weight = input_parameter(is_tensor())
    kernel_size = attribute_parameter()
    activation = attribute_parameter(default="silu")

    @classmethod
    def normalize_attrs(cls, attributes):
        attrs = super().normalize_attrs(attributes)
        if isinstance(attrs["kernel_size"], bool) or not isinstance(attrs["kernel_size"], int) or attrs["kernel_size"] < 2:
            raise IRSchemaError("KdaConvolution kernel_size must be an integer >= 2.")
        if attrs["activation"] not in {"silu"}:
            raise IRSchemaError("KdaConvolution activation must be silu.")
        return attrs

    @classmethod
    def infer_type(cls, inputs, attrs):
        parts = tuple(tensor_of(parameter.type_of(inputs))
                      for parameter in (cls.query, cls.key, cls.value))
        state = tensor_of(cls.conv_state.type_of(inputs))
        weight = tensor_of(cls.conv_weight.type_of(inputs))
        if any(tensor.dtype != DType.BFLOAT16 for tensor in (*parts, state, weight)):
            raise IRSchemaError("KdaConvolution requires BF16 q/k/v, state and weight.")
        kernel = int(attrs["kernel_size"])
        if any(tensor.rank != 3 for tensor in parts) or state.rank != 4 or weight.rank != 4:
            raise IRSchemaError(
                "KdaConvolution expects rank-3 [1, heads, head_dim] q/k/v and rank-4 state/weight.")
        if any(not dimension.is_fixed for tensor in (*parts, state, weight) for dimension in tensor.shape):
            raise IRSchemaError("KdaConvolution decode requires fixed head and state extents.")
        extents = tuple(dimension.fixed_value for dimension in parts[0].shape)
        if extents[0] != 1:
            raise IRSchemaError("KdaConvolution decode requires exactly one token.")
        if any(tuple(dimension.fixed_value for dimension in tensor.shape) != extents
               for tensor in parts[1:]):
            raise IRSchemaError("KdaConvolution q/k/v must share one [1, heads, head_dim] shape.")
        heads, head_dim = extents[1], extents[2]
        if tuple(dimension.fixed_value for dimension in weight.shape) != (3, heads, head_dim, kernel):
            raise IRSchemaError(
                "KdaConvolution weight must be [3, heads, head_dim, kernel_size] in part-major order.")
        if tuple(dimension.fixed_value for dimension in state.shape) != (3, heads, head_dim, kernel - 1):
            raise IRSchemaError(
                "KdaConvolution conv state must be [3, heads, head_dim, kernel_size - 1].")
        new_state = tensor_type(DType.BFLOAT16, (3, heads, head_dim, kernel - 1))
        out = tensor_type(DType.BFLOAT16, (1, heads, head_dim))
        contract = _head_placement(
            tuple(parameter.type_of(inputs) for parameter in cls.input_parameters), "KdaConvolution")
        return _head_results((out, out, out, new_state), contract)


    @classmethod
    def _input_relation(cls, choices):
        """Require one common all-plain or all-head-agreeing contract."""
        yield from _head_choices(choices, "KdaConvolution")

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
        return _head_input_relation(output_type, logical_input_types, "KdaConvolution")

    @classmethod
    def infer_effect(cls, inputs: Sequence[Node], attrs: Mapping[str, object]) -> None:
        del inputs, attrs
        return ir_effect("read_write", "kda_conv_state")

    @classmethod
    def evaluate(cls, node, arguments, context):
        torch = context.torch
        # [3, heads, head_dim]: the published part order is q, k, v.
        current = torch.stack([parameter.read(arguments).float().squeeze(0)
                               for parameter in (cls.query, cls.key, cls.value)])
        state = cls.conv_state.read(arguments).float()                 # [3, heads, head_dim, kernel-1]
        weight = cls.conv_weight.read(arguments).float()               # [3, heads, head_dim, kernel]
        kernel = int(node.attrs["kernel_size"])
        convolved = torch.nn.functional.silu(
            (state * weight[..., : kernel - 1]).sum(dim=-1) + weight[..., kernel - 1] * current
        ).to(torch.bfloat16)
        shifted = torch.cat([state[..., 1:], current.unsqueeze(-1)], dim=-1).to(torch.bfloat16)
        return (*(convolved[part].unsqueeze(0) for part in range(3)), shifted)

    @classmethod
    def cost(cls, node):
        return OpCost(bytes_written=sum(tensor_nbytes(field) for field in node.type.fields),
                      notes=("kda-short-conv", ))


@op_definition("nn.kda_recurrent_core", namespace="nn", functional_name="kda_recurrent_core",
               display_name="NN.KdaRecurrentCore")
class KdaRecurrentCore(OpDefinition):
    """KDA with query scaling after L2 normalization and an independent state update.

    ``query_scale=1`` retains the unscaled readout of existing IR. Model
    importers specify their attention scale explicitly.
    """

    const_evaluable = True
    query = input_parameter(is_tensor())
    key = input_parameter(is_tensor())
    value = input_parameter(is_tensor())
    decay = input_parameter(is_tensor())
    beta = input_parameter(is_tensor())
    state = input_parameter(is_tensor())
    epsilon = attribute_parameter(default=1e-6)
    l2norm_in_kernel = attribute_parameter(default=True)
    query_scale = attribute_parameter(default=1.0)
    output_lanes = attribute_parameter(default=())

    @classmethod
    def normalize_attrs(cls, attributes):
        attrs = super().normalize_attrs(attributes)
        if not isinstance(attrs["l2norm_in_kernel"], bool):
            raise IRSchemaError("KdaRecurrentCore l2norm_in_kernel must be boolean.")
        epsilon = attrs["epsilon"]
        if isinstance(epsilon, bool) or not isinstance(epsilon, (int, float)) or not epsilon > 0:
            raise IRSchemaError("KdaRecurrentCore epsilon must be positive.")
        attrs["output_lanes"] = _normalize_output_lanes(attrs["output_lanes"], "KdaRecurrentCore")
        attrs["query_scale"] = _normalize_query_scale(attrs["query_scale"], "KdaRecurrentCore")
        return attrs

    @classmethod
    def ir_attrs(cls, attrs):
        return {key: value for key, value in attrs.items()
                if (key != "output_lanes" or value) and (key != "query_scale" or value != 1.0)}

    @classmethod
    def infer_type(cls, inputs, attrs):
        def extents(value_type):
            return tuple(dim.fixed_value if dim.is_fixed else None for dim in value_type.shape)

        q = tensor_of(cls.query.type_of(inputs))
        k = tensor_of(cls.key.type_of(inputs))
        v = tensor_of(cls.value.type_of(inputs))
        decay = tensor_of(cls.decay.type_of(inputs))
        beta = tensor_of(cls.beta.type_of(inputs))
        state = tensor_of(cls.state.type_of(inputs))
        if any(not dimension.is_fixed for tensor in (q, k, v, decay, beta, state) for dimension in tensor.shape):
            raise IRSchemaError("KdaRecurrentCore decode requires fixed tensor extents.")
        if q.dtype != DType.BFLOAT16 or k.dtype != DType.BFLOAT16 or v.dtype != DType.BFLOAT16:
            raise IRSchemaError("KdaRecurrentCore requires BF16 q/k/v.")
        if decay.dtype != DType.FLOAT32 or beta.dtype != DType.FLOAT32 or state.dtype != DType.FLOAT32:
            raise IRSchemaError("KdaRecurrentCore requires FP32 decay/beta/state.")
        if q.rank != 3 or extents(k) != extents(q) or extents(v) != extents(q) or extents(decay) != extents(q):
            raise IRSchemaError("KdaRecurrentCore q/k/v/decay must share [1, heads, head_k] shapes.")
        if q.shape[0].is_fixed and q.shape[0].fixed_value != 1:
            raise IRSchemaError("KdaRecurrentCore decode requires exactly one token.")
        if beta.rank != 2 or extents(beta) != extents(q)[:2]:
            raise IRSchemaError("KdaRecurrentCore beta must be [1, heads] FP32.")
        if state.rank != 4 or not state.shape[3].is_fixed or extents(state)[:3] != (1, *extents(q)[1:]):
            raise IRSchemaError("KdaRecurrentCore state must be [1, heads, head_k, head_v] FP32.")
        head_v = state.shape[3].fixed_value
        out = _recurrent_output_type(q, head_v, attrs, "KdaRecurrentCore")
        new_state = tensor_type(DType.FLOAT32, (1, q.shape[1].fixed_value, q.shape[2].fixed_value, head_v))
        contract = _head_placement(
            tuple(parameter.type_of(inputs) for parameter in cls.input_parameters),
            "KdaRecurrentCore",
        )
        return _head_results((out, new_state), contract)

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
        return _head_input_relation(output_type, logical_input_types, "KdaRecurrentCore")

    @classmethod
    def _input_relation(cls, choices):
        """All operands agree on the head policy when one is offered."""
        yield from _head_choices(choices, "KdaRecurrentCore")

    @classmethod
    def infer_effect(cls, inputs: Sequence[Node], attrs: Mapping[str, object]) -> None:
        del inputs, attrs
        return ir_effect("read_write", "kda_recurrent_state")

    @classmethod
    def evaluate(cls, node, arguments, context):
        q = cls.query.read(arguments).float()
        k = cls.key.read(arguments).float()
        v = cls.value.read(arguments).float()
        decay = cls.decay.read(arguments).float()
        beta = cls.beta.read(arguments).float()
        state = cls.state.read(arguments).float()
        epsilon = float(node.attrs["epsilon"])
        if node.attrs["l2norm_in_kernel"]:
            q = q / context.torch.sqrt(q.pow(2).sum(-1, keepdim=True) + epsilon)
            k = k / context.torch.sqrt(k.pow(2).sum(-1, keepdim=True) + epsilon)
        q = q * float(node.attrs.get("query_scale", 1.0))
        state = state * decay.exp().unsqueeze(-1)
        correction = (state * k.unsqueeze(-1)).sum(-2)
        value = v - correction
        value = value * beta.unsqueeze(-1)
        state = state + k.unsqueeze(-1) * value.unsqueeze(-2)
        out = (state * q.unsqueeze(-1)).sum(-2)
        lanes = tuple(node.attrs.get("output_lanes", ()))
        out = out.to(context.torch.bfloat16)
        return out.reshape(1, out.shape[1], out.shape[2] // prod(lanes), *lanes), state

    @classmethod
    def cost(cls, node):
        sizes = [tensor_nbytes(field) for field in node.type.fields]
        return OpCost(bytes_written=None if None in sizes else sum(sizes), notes=("kda-recurrent", ))


__all__ = ["KdaConvolution", "KdaRecurrentCore"]
