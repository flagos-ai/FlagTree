# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Hyper-connection pre stage: coefficient generation and residual collapse.

A hyper-connection block replaces the single residual stream with ``hc_mult``
parallel streams.  Before each sublayer the streams are normalized and projected
to three coefficient groups, the mixing matrix is made doubly stochastic by
Sinkhorn-Knopp normalization, and the streams collapse to one sublayer input.

Expressing that as graph algebra costs one node per Sinkhorn iteration and
direction -- 20 iterations become well over two hundred nodes whose numerical
contract is spread across all of them, and whose intermediates each make a
round trip through memory.  Keeping it as one operation states the contract
once and lets the iteration live inside a kernel, which matches how serving
runtimes implement it (vLLM ``mhc_pre``/``MHCPreOp``, one kernel whose 4x4
matrix never leaves registers).

The stage emits ``post`` and ``comb`` for the matching :class:`MhcPost`, plus
the collapsed sublayer input; ``pre`` is consumed internally by the collapse.
"""

from __future__ import annotations

from math import prod
from typing import Mapping, Sequence

from triton.flagmega.errors import IRSchemaError
from triton.flagmega.ir.distributed_inference import tensor_of
from triton.flagmega.ir.model import SBP, DistributedType, IRType, Node, TupleType, tensor_type
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


@op_definition("nn.mhc_pre", namespace="nn", functional_name="mhc_pre",
               display_name="NN.MhcPre")
class MhcPre(OpDefinition):
    """Collapse ``hc_mult`` residual streams and emit the mixing coefficients.

    Inputs
        ``streams``   ``(tokens, hc_mult, hidden)`` residual streams.
        ``projection``  ``((2 + hc_mult) * hc_mult, hc_mult * hidden)`` coefficient
        weight, applied to the RMS-normalized flattened streams.
        ``base``   ``((2 + hc_mult) * hc_mult,)`` per-coefficient bias.
        ``scale``  ``(3,)`` per-group scale for the pre/post/comb groups.

    Outputs a tuple ``(post, comb, collapsed)`` where ``post`` is
    ``(tokens, hc_mult)``, ``comb`` is ``(tokens, hc_mult, hc_mult)`` and doubly
    stochastic, and ``collapsed`` is ``(tokens, hidden)`` in the stream dtype.
    ``output_lanes`` packs only collapsed's hidden axis; the coefficients
    retain their FP32 scalar representation.
    """

    const_evaluable = True
    streams = input_parameter(is_tensor())
    projection = input_parameter(is_tensor())
    base = input_parameter(is_tensor())
    scale = input_parameter(is_tensor())
    epsilon = attribute_parameter()
    sinkhorn_iterations = attribute_parameter()
    sinkhorn_epsilon = attribute_parameter()
    post_multiplier = attribute_parameter(default=2.0)
    output_lanes = attribute_parameter(default=())

    @classmethod
    def normalize_attrs(cls, attributes: Mapping[str, object]) -> dict[str, object]:
        attrs = super().normalize_attrs(attributes)
        iterations = attrs["sinkhorn_iterations"]
        if isinstance(iterations, bool) or not isinstance(iterations, int) or iterations <= 0:
            raise IRSchemaError("MhcPre sinkhorn_iterations must be a positive integer.")
        for name in ("epsilon", "sinkhorn_epsilon", "post_multiplier"):
            value = attrs[name]
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise IRSchemaError(f"MhcPre {name} must be a number.")
            attrs[name] = float(value)
        if attrs["epsilon"] <= 0.0 or attrs["sinkhorn_epsilon"] <= 0.0:
            raise IRSchemaError("MhcPre epsilon values must be positive.")
        attrs["sinkhorn_iterations"] = int(iterations)
        lanes = attrs["output_lanes"]
        if not isinstance(lanes, (tuple, list)) or any(
                isinstance(lane, bool) or not isinstance(lane, int) or lane <= 0 for lane in lanes):
            raise IRSchemaError("MhcPre output_lanes must be a sequence of positive integers.")
        attrs["output_lanes"] = tuple(lanes)
        return attrs

    @classmethod
    def ir_attrs(cls, attrs: Mapping[str, object]) -> Mapping[str, object]:
        # Preserve scalar checkpoints' canonical attributes and semantic hash.
        return {key: value for key, value in attrs.items() if key != "output_lanes" or value}

    @classmethod
    def infer_type(cls, inputs: Sequence[Node], attrs: Mapping[str, object]) -> IRType:
        for name, parameter in (("streams", cls.streams), ("projection", cls.projection),
                                ("base", cls.base), ("scale", cls.scale)):
            if isinstance(tensor_of(parameter.type_of(inputs)).dtype, VectorType):
                raise IRSchemaError(f"MhcPre {name} must use a scalar element type.")
        stream_source = cls.streams.type_of(inputs)
        streams = tensor_of(stream_source)
        if streams.rank != 3 or not all(dim.is_fixed for dim in streams.shape[1:]):
            raise IRSchemaError("MhcPre streams must be (tokens, hc_mult, hidden) with fixed trailing extents.")
        tokens, hc, hidden = streams.shape[0], streams.shape[1].fixed_value, streams.shape[2].fixed_value
        lanes = tuple(attrs.get("output_lanes", ()))
        if hidden % prod(lanes):
            raise IRSchemaError("MhcPre hidden must be divisible by output_lanes.")
        coefficients = (2 + hc) * hc
        projection = tensor_of(cls.projection.type_of(inputs))
        if (projection.rank != 2 or projection.shape[0].fixed_value != coefficients
                or projection.shape[1].fixed_value != hc * hidden):
            raise IRSchemaError(
                "MhcPre projection must be ((2 + hc_mult) * hc_mult, hc_mult * hidden).")
        base = tensor_of(cls.base.type_of(inputs))
        if base.rank != 1 or base.shape[0].fixed_value != coefficients:
            raise IRSchemaError("MhcPre base must carry one entry per coefficient.")
        scale = tensor_of(cls.scale.type_of(inputs))
        if scale.rank != 1 or scale.shape[0].fixed_value != 3:
            raise IRSchemaError("MhcPre scale must hold the three group scales.")
        # Coefficients stay FP32 so the Sinkhorn normalization keeps its
        # reference precision; the collapsed input returns to the stream dtype.
        post = tensor_type(DType.FLOAT32, (tokens, hc))
        comb = tensor_type(DType.FLOAT32, (tokens, hc, hc))
        collapsed = tensor_type(VectorType(streams.dtype, lanes) if lanes else streams.dtype,
                                (tokens, hidden // prod(lanes)))
        if not isinstance(stream_source, DistributedType):
            return TupleType((post, comb, collapsed))
        if stream_source.partial is not None or any(policy != SBP.broadcast()
                                                    for policy in stream_source.axis_policies):
            raise IRSchemaError("MhcPre requires broadcast streams; insert explicit Boxing.")
        placement = stream_source.placement
        for name, source in (("projection", cls.projection.type_of(inputs)),
                             ("base", cls.base.type_of(inputs)),
                             ("scale", cls.scale.type_of(inputs))):
            if not isinstance(source, DistributedType) or source.placement != placement \
                    or any(policy != SBP.broadcast() for policy in source.axis_policies):
                raise IRSchemaError(f"MhcPre {name} must be broadcast like the streams.")
        return TupleType(tuple(
            DistributedType(field, tuple(SBP.broadcast() for _ in field.shape), placement)
            for field in (post, comb, collapsed)))

    @classmethod
    def evaluate(cls, node, arguments, context):
        torch = context.torch
        streams = cls.streams.read(arguments)
        projection = cls.projection.read(arguments).float()
        base = cls.base.read(arguments).float()
        scale = cls.scale.read(arguments).float()
        attrs = node.attrs
        epsilon = float(attrs["epsilon"])
        sinkhorn_epsilon = float(attrs["sinkhorn_epsilon"])
        hc = streams.shape[-2]
        flat = streams.reshape(streams.shape[0], -1).float()
        # RMS normalization over the flattened streams, without a mean term.
        mixes = (flat @ projection.t()) * torch.rsqrt(
            flat.square().mean(-1, keepdim=True) + epsilon)
        # The floor on pre_mix is vLLM's ``hc_eps``, which this op carries as
        # sinkhorn_epsilon -- not the RMS epsilon above. GLM-5.3 sets them to
        # 1e-6 and 1e-5, so confusing the two biases every stream coefficient.
        pre = torch.sigmoid(mixes[:, :hc] * scale[0] + base[:hc]) + sinkhorn_epsilon
        post = torch.sigmoid(mixes[:, hc:2 * hc] * scale[1] + base[hc:2 * hc]) \
            * float(attrs["post_multiplier"])
        comb = mixes[:, 2 * hc:].reshape(-1, hc, hc) * scale[2] + base[2 * hc:].reshape(1, hc, hc)
        comb = torch.softmax(comb, dim=-1) + sinkhorn_epsilon
        comb = comb / (comb.sum(dim=-2, keepdim=True) + sinkhorn_epsilon)
        for _ in range(int(attrs["sinkhorn_iterations"]) - 1):
            comb = comb / (comb.sum(dim=-1, keepdim=True) + sinkhorn_epsilon)
            comb = comb / (comb.sum(dim=-2, keepdim=True) + sinkhorn_epsilon)
        collapsed = (pre.unsqueeze(-1) * streams.float()).sum(dim=1).to(streams.dtype)
        lanes = tuple(attrs.get("output_lanes", ()))
        collapsed = collapsed.reshape(streams.shape[0], streams.shape[-1] // prod(lanes), *lanes)
        return post, comb, collapsed

    @classmethod
    def cost(cls, node):
        sizes = [tensor_nbytes(field) for field in node.type.fields]
        return OpCost(bytes_written=None if None in sizes else sum(sizes),
                      notes=("mhc-pre", ))


__all__ = ["MhcPre"]
