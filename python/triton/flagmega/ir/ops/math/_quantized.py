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
