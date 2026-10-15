# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Hyper-connection post stage: distribute a sublayer result over the streams.

The counterpart to :class:`~triton.flagmega.ir.ops.nn.mhc_pre.MhcPre`.  Each
output stream receives the doubly-stochastic mix of all input streams plus its
own share of the sublayer result:

    out[t, j] = sum_i comb[t, i, j] * residual[t, i] + post[t, j] * value[t]

As graph algebra that is ``hc_mult ** 2`` multiplies plus ``hc_mult * (hc_mult - 1)``
adds with a concat, all on hidden-sized tensors -- 28 nodes at the published
``hc_mult = 4``, twice per block.  One operation keeps the contraction in a
single pass, matching vLLM ``mhc_post``/``MHCPostOp``.
"""

from __future__ import annotations

from typing import Mapping, Sequence

from triton.flagmega.errors import IRSchemaError
from triton.flagmega.ir.distributed_inference import tensor_of
from triton.flagmega.ir.model import SBP, DistributedType, IRType, Node, tensor_type
from triton.flagmega.ir.ops.core import (
    OpCost,
    OpDefinition,
    input_parameter,
    op_definition,
    tensor_nbytes,
)
from triton.flagmega.ir.type_pattern import is_tensor
from triton.flagmega.ir.types import DType


@op_definition("nn.mhc_post", namespace="nn", functional_name="mhc_post",
               display_name="NN.MhcPost")
class MhcPost(OpDefinition):
    """Mix residual streams and add the sublayer result's per-stream share.

    Inputs
        ``value``    ``(tokens, hidden)`` sublayer result.
        ``residual`` ``(tokens, hc_mult, hidden)`` incoming streams.
        ``post``     ``(tokens, hc_mult)`` FP32 per-stream share of ``value``.
        ``comb``     ``(tokens, hc_mult, hc_mult)`` FP32 doubly-stochastic mix.

    Emits ``(tokens, hc_mult, hidden)`` in the residual dtype.
    """

    const_evaluable = True
    value = input_parameter(is_tensor())
    residual = input_parameter(is_tensor())
    post = input_parameter(is_tensor())
    comb = input_parameter(is_tensor())

    @classmethod
    def infer_type(cls, inputs: Sequence[Node], attrs: Mapping[str, object]) -> IRType:
        residual_source = cls.residual.type_of(inputs)
        residual = tensor_of(residual_source)
        if residual.rank != 3 or not all(dim.is_fixed for dim in residual.shape[1:]):
            raise IRSchemaError("MhcPost residual must be (tokens, hc_mult, hidden) with fixed trailing extents.")
        tokens, hc, hidden = residual.shape[0], residual.shape[1].fixed_value, residual.shape[2].fixed_value
        value = tensor_of(cls.value.type_of(inputs))
        if value.rank != 2 or value.shape[1].fixed_value != hidden:
            raise IRSchemaError("MhcPost value must be (tokens, hidden) matching the residual streams.")
        post = tensor_of(cls.post.type_of(inputs))
        if post.dtype != DType.FLOAT32 or post.rank != 2 or post.shape[1].fixed_value != hc:
            raise IRSchemaError("MhcPost post must be FP32 with one coefficient per stream.")
        comb = tensor_of(cls.comb.type_of(inputs))
        if comb.dtype != DType.FLOAT32 or comb.rank != 3 \
                or comb.shape[1].fixed_value != hc or comb.shape[2].fixed_value != hc:
            raise IRSchemaError("MhcPost comb must be an FP32 (tokens, hc_mult, hc_mult) mixing matrix.")
        output = tensor_type(residual.dtype, (tokens, hc, hidden))
        if not isinstance(residual_source, DistributedType):
            return output
        if residual_source.partial is not None or any(policy != SBP.broadcast()
                                                      for policy in residual_source.axis_policies):
            raise IRSchemaError("MhcPost requires broadcast residual streams; insert explicit Boxing.")
        placement = residual_source.placement
        for name, source in (("value", cls.value.type_of(inputs)), ("post", cls.post.type_of(inputs)),
                             ("comb", cls.comb.type_of(inputs))):
            if not isinstance(source, DistributedType) or source.placement != placement \
                    or any(policy != SBP.broadcast() for policy in source.axis_policies):
                raise IRSchemaError(f"MhcPost {name} must be broadcast like the residual streams.")
        return DistributedType(output, tuple(SBP.broadcast() for _ in output.shape), placement)

    @classmethod
    def evaluate(cls, node, arguments, context):
        value = cls.value.read(arguments).float()
        residual = cls.residual.read(arguments)
        post = cls.post.read(arguments).float()
        comb = cls.comb.read(arguments).float()
        # out[t, j, h] = sum_i comb[t, i, j] * residual[t, i, h] + post[t, j] * value[t, h]
        mixed = context.torch.einsum("tij,tih->tjh", comb, residual.float())
        return (mixed + post.unsqueeze(-1) * value.unsqueeze(-2)).to(residual.dtype)

    @classmethod
    def cost(cls, node):
        return OpCost(bytes_written=tensor_nbytes(node.type), notes=("mhc-post", ))


__all__ = ["MhcPost"]
