# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Rounded block-triangular coefficients for chunked delta-rule attention.

For each token block and value head, return
``round_input(inv_fp16(I + tril(beta * K @ K.T, -1)) * beta[None, :])``.
The inverse is a numerical algorithm: 8-element diagonal elimination, then
16/32/64 block composition with explicit FP16 intermediates. It is not an
algebraic matrix inverse that an optimizer may freely reassociate. Decay and
state updates are separate consumers; no runtime/model identity is encoded.
"""

from triton.flagmega.errors import IRSchemaError
from triton.flagmega.ir.distributed_inference import placement_of, tensor_of
from triton.flagmega.ir.model import DistributedType, SBP, tensor_type
from triton.flagmega.ir.distributed_type import SBPSplit, scale_split_units
from triton.flagmega.ir.ops.core import OpCost, OpDefinition, attribute_parameter, input_parameter, op_definition, tensor_nbytes
from triton.flagmega.ir.type_pattern import has_rank, is_tensor
from triton.flagmega.ir.types import DType


@op_definition("nn.delta_rule_coefficients", namespace="nn", functional_name="delta_rule_coefficients",
               display_name="NN.DeltaRuleCoefficients")
class DeltaRuleCoefficients(OpDefinition):
    supports_broadcast_lifting = False
    key = input_parameter(is_tensor() & has_rank(3))
    beta = input_parameter(is_tensor() & has_rank(2))
    block_size = attribute_parameter(default=64)

    @classmethod
    def normalize_attrs(cls, attributes):
        attrs = super().normalize_attrs(attributes)
        block = attrs["block_size"]
        if isinstance(block, bool) or not isinstance(block, int) or block not in (8, 16, 32, 64):
            raise IRSchemaError("DeltaRuleCoefficients block_size must be 8, 16, 32 or 64.")
        return attrs

    @classmethod
    def infer_type(cls, inputs, attrs):
        key_type, beta_type = cls.key.type_of(inputs), cls.beta.type_of(inputs)
        key, beta = tensor_of(key_type), tensor_of(beta_type)
        if key.dtype != DType.BFLOAT16 or beta.dtype != DType.FLOAT32:
            raise IRSchemaError("DeltaRuleCoefficients requires BF16 key and FP32 beta.")
        if key.shape[0] != beta.shape[0]:
            raise IRSchemaError("DeltaRuleCoefficients key/beta token dimensions must match.")
        if not key.shape[1].is_fixed or not beta.shape[1].is_fixed:
            raise IRSchemaError("DeltaRuleCoefficients requires fixed positive head counts.")
        key_heads, value_heads = key.shape[1].fixed_value, beta.shape[1].fixed_value
        if key_heads <= 0 or value_heads <= 0 or value_heads % key_heads:
            raise IRSchemaError("DeltaRuleCoefficients value heads must be a positive multiple of key heads.")
        if key.shape[2].is_fixed and key.shape[2].fixed_value <= 0:
            raise IRSchemaError("DeltaRuleCoefficients key dimension must be positive.")
        block = attrs["block_size"]
        output = tensor_type(key.dtype, ((key.shape[0] + block - 1) // block, value_heads, block, block))
        placement = placement_of(key_type, beta_type)
        if placement is None:
            return output
        if not isinstance(key_type, DistributedType) or not isinstance(beta_type, DistributedType):
            raise IRSchemaError("DeltaRuleCoefficients requires both operands to name their placement.")
        broadcast = SBP.broadcast()
        key_head = key_type.axis_policies[1]
        head = scale_split_units(key_head, value_heads // key_heads, 1) if isinstance(key_head, SBPSplit) else key_head
        if (key_type.partial is not None or beta_type.partial is not None
                or key_type.axis_policies != (broadcast, key_head, broadcast) or head is None
                or beta_type.axis_policies != (broadcast, head)):
            raise IRSchemaError(
                "DeltaRuleCoefficients requires group-aligned head ownership and broadcast token/key axes.")
        return DistributedType(output, (broadcast, head, broadcast, broadcast), placement)

    @classmethod
    def evaluate(cls, node, arguments, context):
        return delta_rule_coefficients(cls.key.read(arguments), cls.beta.read(arguments), node.attrs["block_size"],
                                       torch=context.torch)

    @classmethod
    def cost(cls, node):
        return OpCost(bytes_written=tensor_nbytes(tensor_of(node.type)),
                      notes=("rounded-block-triangular-inverse", "no-decay-or-state-update"))


def block_inverse_fp16(lower, *, torch):
    """Preserve intermediate casts, including the final two rounded partials."""
    block_size = lower.shape[-1]
    inverse = lower.clone()
    for begin in range(0, block_size, 8):
        diagonal = inverse[..., begin:begin + 8, begin:begin + 8]
        diagonal.diagonal(dim1=-2, dim2=-1).fill_(1)
        for pivot in range(7):
            factor = -diagonal[..., pivot + 1:, pivot].clone()
            if pivot:
                diagonal[..., pivot + 1:, :pivot] = torch.addcmul(diagonal[..., pivot + 1:, :pivot], factor[..., None],
                                                                  diagonal[..., pivot:pivot + 1, :pivot])
            diagonal[..., pivot + 1:, pivot] = factor
        diagonal.copy_(diagonal.half().float())
    for width in (16, 32, 64):
        if width > block_size:
            break
        half = width // 2
        for begin in range(0, block_size, width):
            a = inverse[..., begin:begin + half, begin:begin + half]
            d = inverse[..., begin + half:begin + width, begin + half:begin + width]
            c = inverse[..., begin + half:begin + width, begin:begin + half]
            product = (-(d @ c)).half().float()
            if width == 64:
                partial0 = (product[..., :16] @ a[..., :16, :]).half()
                partial1 = (product[..., 16:] @ a[..., 16:, :]).half()
                c.copy_((partial0 + partial1).half().float())
            else:
                c.copy_((product @ a).half().float())
    return inverse


def delta_rule_coefficients(key, beta, block_size=64, *, torch):
    tokens, key_heads, dimension = key.shape
    heads = beta.shape[1]
    blocks = (tokens + block_size - 1) // block_size
    padded = key.new_zeros((heads, blocks * block_size, dimension), dtype=torch.float32)
    padded[:, :tokens] = key.repeat_interleave(heads // key_heads, dim=1).transpose(0, 1).float()
    padded_beta = beta.new_zeros((heads, blocks * block_size))
    padded_beta[:, :tokens] = beta.transpose(0, 1)
    keys = padded.reshape(heads, blocks, block_size, dimension).transpose(0, 1)
    betas = padded_beta.reshape(heads, blocks, block_size).transpose(0, 1)
    lower = torch.tril((keys @ keys.transpose(-1, -2)) * betas[..., None], diagonal=-1).half().float()
    inverse = block_inverse_fp16(lower, torch=torch)
    result = (inverse * betas[..., None, :]).to(key.dtype)
    # Padded rows and columns are not part of the recurrence.
    positions = torch.arange(blocks * block_size, device=key.device).reshape(blocks, block_size)
    valid = positions < tokens
    return torch.where(valid[:, None, :, None] & valid[:, None, None, :], result, 0).contiguous()


__all__ = ["DeltaRuleCoefficients"]
