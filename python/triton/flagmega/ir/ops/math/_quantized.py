# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Logical matrix width for native GGUF byte tensors."""

from triton.flagmega.errors import IRSchemaError
from triton.flagmega.ir.types import DType
from triton.flagmega.ir.model import tensor_type

BLOCK_BYTES = {"IQ4_XS": 136, "Q4_K": 144}


def matrix_width(weight, encoding):
    if encoding is None:
        return weight.shape[1]
    if encoding not in BLOCK_BYTES or weight.dtype != DType.UINT8:
        raise IRSchemaError("Native GGUF weights require supported encoding and UINT8 storage.")
    width = weight.shape[1]
    block = BLOCK_BYTES[encoding]
    if not width.is_fixed or width.fixed_value % block:
        raise IRSchemaError("Native GGUF rows must contain whole 256-element quantization blocks.")
    return tensor_type(DType.UINT8, (width.fixed_value // block * 256, )).shape[0]




def verify_native_scale_type(value, weight, scale, *, encoding, block_n, block_k, opname):
    """Native scale operands retain the logical FP8 schema, not its arithmetic."""
    if encoding is None:
        return
    k = matrix_width(weight, encoding)
    if not k.is_fixed or not weight.shape[0].is_fixed:
        raise IRSchemaError(f"{opname} native dimensions must be fixed.")
    expected = tensor_type(DType.FLOAT32, ((weight.shape[0].fixed_value + block_n - 1)//block_n,
                                         (k.fixed_value + block_k - 1)//block_k))
    if scale != expected or value.shape[-1] != k:
        raise IRSchemaError(f"{opname} native placeholder scales require logical-N/K FP32 shape {expected.shape}.")


def decode_native_rows(weight, encoding, *, logical_k=None):
    """Diagnostic Torch decoding of complete native GGML rows into FP32.

    Runtime/checkpoint storage remains native UINT8. Header scales are FP16
    widened to FP32; decoded weights never acquire a BF16 rounding boundary.
    """
    from triton.flagmega.errors import EvaluationError
    from triton.flagmega.ir.ops.math._block_scaled import _torch
    torch = _torch()
    block_bytes = BLOCK_BYTES.get(encoding)
    if (block_bytes is None or weight.dtype != torch.uint8 or weight.ndim != 2
            or weight.shape[1] % block_bytes):
        raise EvaluationError("Native GGUF decoding requires supported encoding and complete UINT8 rows.")
    k = weight.shape[1] // block_bytes * 256
    if logical_k is not None and k != logical_k:
        raise EvaluationError(f"Native GGUF logical K mismatch: activation {logical_k}, weight {k}.")
    blocks = weight.contiguous().reshape(weight.shape[0], -1, block_bytes)
    d = blocks[..., :2].contiguous().view(torch.float16).float()
    group = torch.arange(8, device=weight.device)
    if encoding == "IQ4_XS":
        low = (blocks[..., 4:8].long()[..., group // 2] >> (4 * (group % 2))) & 15
        high_word = blocks[..., 2].long() | (blocks[..., 3].long() << 8)
        high = (high_word[..., None] >> (2 * group)) & 3
        scales = (low | (high << 4)) - 32
        packed = blocks[..., 8:].long().reshape(*blocks.shape[:2], 8, 16)
        codes = torch.cat((packed & 15, packed >> 4), dim=-1)
        values = torch.tensor([-127, -104, -83, -65, -49, -35, -22, -10,
                               1, 13, 25, 38, 53, 69, 89, 113], device=weight.device, dtype=torch.float32)
        decoded = (d * scales.float())[..., None] * values[codes]
    else:
        scales = blocks[..., 4:16].long()
        a, b, c = (scales[..., offset + group % 4] for offset in (0, 4, 8))
        scale = torch.where(group < 4, a & 63, (c & 15) | ((a >> 2) & 48))
        minimum = torch.where(group < 4, b & 63, (c >> 4) | ((b >> 2) & 48))
        packed = blocks[..., 16:].long().reshape(*blocks.shape[:2], 4, 32)[..., group // 2, :]
        codes = (packed >> (4 * (group % 2))[:, None]) & 15
        dmin = blocks[..., 2:4].contiguous().view(torch.float16).float()
        decoded = (d * scale.float())[..., None] * codes.float() - (dmin * minimum.float())[..., None]
    return decoded.reshape(weight.shape[0], k)


def native_linear(value, weight, encoding, *, output_dtype=None):
    from triton.flagmega.ir.ops.math._block_scaled import _torch
    torch = _torch()
    decoded = decode_native_rows(weight, encoding, logical_k=value.shape[-1])
    result = torch.matmul(value.float(), decoded.transpose(0, 1))
    return result.to(output_dtype or value.dtype)
