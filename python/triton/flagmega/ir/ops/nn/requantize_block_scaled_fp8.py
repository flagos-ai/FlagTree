# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Closed checkpoint transformation for power-of-two block-scaled FP8."""

from math import isfinite, prod
from struct import pack, unpack

from triton.flagmega.errors import IRSchemaError, IRVerificationError
from triton.flagmega.ir import DistributedType, DType, TupleType, tensor_type
from triton.flagmega.ir.ops.core import (
    OpCost,
    OpDefinition,
    attribute_parameter,
    get_definition,
    input_parameter,
    op_definition,
    tensor_nbytes,
)
from triton.flagmega.ir.type_pattern import is_tensor


@op_definition("nn.requantize_block_scaled_fp8", namespace="nn", functional_name="requantize_block_scaled_fp8",
               display_name="NN.RequantizeBlockScaledFP8")
class RequantizeBlockScaledFP8(OpDefinition):
    """Requantize codes and scales together, inside a closed constant recipe.

    Inputs are FP8 E4M3FN ``[..., N, K]`` codes and FP32 scales
    ``[..., ceil(N / block_n), ceil(K / block_k)]``. Dequantization, amax,
    division by 448, log2, ceil and pow use FP32. ``scale_division="rn"``
    performs true division; ``"reciprocal_multiply"`` instead multiplies amax
    by the FP32-rounded constant 1/448 before log2. The new scale is
    ``2 ** ceil(log2(max(amax, minimum_amax) / 448))``. Each block is then
    quantized again with that scale. Partial edge blocks are zero padded.

    Returns fresh codes and scales with the original shapes and dtypes. This
    is an offline transformation: verification requires closed constant
    inputs. FreezeConstantIslands removes it from compute before distribution;
    no runtime kernel or deferred numeric source-layout program is implied.
    """

    const_evaluable = True
    supports_broadcast_lifting = False
    weight = input_parameter(is_tensor())
    scale = input_parameter(is_tensor())
    block_n = attribute_parameter()
    block_k = attribute_parameter()
    minimum_amax = attribute_parameter(default=1e-4)
    scale_division = attribute_parameter(default="rn")

    @classmethod
    def normalize_attrs(cls, attributes):
        attrs = super().normalize_attrs(attributes)
        for name in ("block_n", "block_k"):
            value = attrs[name]
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise IRSchemaError(f"RequantizeBlockScaledFP8 {name} must be a positive integer.")
        floor = attrs["minimum_amax"]
        try:
            if isinstance(floor, bool) or not isinstance(floor, (int, float)):
                raise TypeError
            floor = unpack("f", pack("f", float(floor)))[0]
            linear_floor = unpack("f", pack("f", floor / 448.))[0]
            if not isfinite(floor) or floor <= 0 or linear_floor <= 0:
                raise ValueError
        except (TypeError, ValueError, OverflowError) as error:
            raise IRSchemaError("RequantizeBlockScaledFP8 minimum_amax must be positive and finite in FP32, "
                                "with a nonzero FP32 scale after division by 448.") from error
        attrs["minimum_amax"] = floor
        if not isinstance(attrs["scale_division"], str) or attrs["scale_division"] not in {"rn", "reciprocal_multiply"}:
            raise IRSchemaError("RequantizeBlockScaledFP8 scale_division must be rn or reciprocal_multiply.")
        return attrs

    @classmethod
    def ir_attrs(cls, attrs):
        return {key: value for key, value in attrs.items() if key != "scale_division" or value != "rn"}

    @classmethod
    def infer_type(cls, inputs, attrs):
        weight, scale = (parameter.type_of(inputs) for parameter in cls.input_parameters)
        if isinstance(weight, DistributedType) or isinstance(scale, DistributedType):
            raise IRSchemaError("RequantizeBlockScaledFP8 must run offline before distribution.")
        if weight.dtype != DType.FLOAT8_E4M3FN or scale.dtype != DType.FLOAT32:
            raise IRSchemaError("RequantizeBlockScaledFP8 requires scalar FP8 E4M3FN codes and FP32 scales.")
        if weight.rank < 2 or scale.rank != weight.rank or any(not dim.is_fixed for dim in weight.shape):
            raise IRSchemaError("RequantizeBlockScaledFP8 requires fixed weight/scale shapes of equal rank >=2.")
        bn, bk = attrs["block_n"], attrs["block_k"]
        n, k = (dim.fixed_value for dim in weight.shape[-2:])
        expected = tensor_type(DType.FLOAT32, (*weight.shape[:-2], (n + bn - 1) // bn, (k + bk - 1) // bk))
        if scale != expected:
            raise IRSchemaError(f"RequantizeBlockScaledFP8 scale shape must match {expected}.")
        return TupleType((weight, scale))

    @classmethod
    def verify(cls, node, module):
        super().verify(node, module)
        if any(isinstance(module.node_map[key].type, DistributedType) for key in node.inputs):
            raise IRVerificationError("RequantizeBlockScaledFP8 must run offline before distribution.", node_id=node.id)
        # Walk only these inputs' ancestors. Scanning the full model for each
        # weight pair would make verification quadratic in checkpoint size.
        pending, seen = list(node.inputs), set()
        while pending:
            key = pending.pop()
            if key in seen:
                continue
            seen.add(key)
            source = module.node_map[key]
            definition = get_definition(source.op)
            closed_leaf = definition.constant_source and not source.inputs
            closed_expression = definition.const_evaluable and definition.deterministic and bool(source.inputs)
            if not source.effect.is_pure or not (closed_leaf or closed_expression):
                raise IRVerificationError("RequantizeBlockScaledFP8 requires closed constant inputs; "
                                          "runtime quantization is not supported.", node_id=node.id)
            pending.extend(source.inputs)

    @classmethod
    def evaluate(cls, node, arguments, context):
        torch = context.torch
        weight, old_scale = (parameter.read(arguments) for parameter in cls.input_parameters)
        bn, bk = int(node.attrs["block_n"]), int(node.attrs["block_k"])
        n, k = weight.shape[-2:]
        ng, kg = (n + bn - 1) // bn, (k + bk - 1) // bk
        new_weight = torch.empty(weight.shape, dtype=weight.dtype, device=weight.device)
        new_scale = torch.empty(old_scale.shape, dtype=old_scale.dtype, device=old_scale.device)
        if weight.numel() == 0:
            return new_weight, new_scale
        batches = prod(weight.shape[:-2])
        source = weight.reshape(batches, n, k)
        scales = old_scale.reshape(batches, ng, kg)
        codes_out = new_weight.reshape(batches, n, k)
        scales_out = new_scale.reshape(batches, ng, kg)
        # Bound wide intermediates to approximately one million scalars even
        # for an expert bank. Output codes/scales are the only full-size data.
        block_rows = max(1, (1 << 20) // (bn * kg * bk))
        for batch in range(batches):
            for first in range(0, ng, block_rows):
                last = min(first + block_rows, ng)
                begin, end = first * bn, min(last * bn, n)
                expanded = scales[batch, first:last].repeat_interleave(bn, 0).repeat_interleave(bk, 1)
                wide = source[batch, begin:end].float() * expanded[:end - begin, :k]
                wide = torch.nn.functional.pad(wide, (0, kg * bk - k, 0, (last - first) * bn - (end - begin)))
                blocks = wide.reshape(last - first, bn, kg, bk)
                amax = blocks.abs().amax(dim=(1, 3), keepdim=True).clamp_min(node.attrs["minimum_amax"])
                if node.attrs.get("scale_division", "rn") == "reciprocal_multiply":
                    linear_scale = amax * amax.new_tensor(1. / 448.)
                else:
                    # A device tensor divisor preserves true FP32 division.
                    linear_scale = amax / amax.new_tensor(448.)
                target_scale = torch.pow(2., torch.ceil(torch.log2(linear_scale)))
                quantized = (blocks / target_scale).to(torch.float8_e4m3fn)
                codes_out[batch, begin:end].copy_(quantized.reshape((last - first) * bn, kg * bk)[:end - begin, :k])
                scales_out[batch, first:last].copy_(target_scale.reshape(last - first, kg))
        return new_weight, new_scale

    @classmethod
    def cost(cls, node):
        size = sum(tensor_nbytes(field) for field in node.type.fields)
        return OpCost(bytes_read=size, bytes_written=size, notes=("offline-fp8-requantization",))


__all__ = ["RequantizeBlockScaledFP8"]
