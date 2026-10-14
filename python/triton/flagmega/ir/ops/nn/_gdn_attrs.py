# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Shared attribute contract used by the three GDN op definitions."""

from typing import Mapping

from triton.flagmega.errors import IRSchemaError

_DIMENSION_ATTRIBUTE_NAMES = (
    "num_key_heads",
    "num_value_heads",
    "key_head_dim",
    "value_head_dim",
    "conv_kernel_size",
    "epsilon",
    "weight_block_n",
    "weight_block_k",
)
_ENCODING_ATTRIBUTE_NAMES = (
    "qkv_weight_encoding",
    "z_weight_encoding",
    "output_weight_encoding",
    "b_weight_encoding",
    "a_weight_encoding",
)
GDN_ATTRIBUTE_NAMES = (*_DIMENSION_ATTRIBUTE_NAMES, *_ENCODING_ATTRIBUTE_NAMES, "value_head_order")


def normalize_gdn_attrs(attributes: Mapping[str, object]) -> dict[str, object]:
    allowed = set(GDN_ATTRIBUTE_NAMES)
    if set(attributes) - allowed:
        raise IRSchemaError(f"Gated DeltaNet has unknown attributes: {sorted(set(attributes) - allowed)}.")
    missing = set(_DIMENSION_ATTRIBUTE_NAMES) - set(attributes)
    if missing:
        raise IRSchemaError(f"Gated DeltaNet requires attributes: {sorted(missing)}.")
    result: dict[str, object] = {
        name: float(attributes[name]) if name == "epsilon" else int(attributes[name])
        for name in _DIMENSION_ATTRIBUTE_NAMES
    }
    for name in _ENCODING_ATTRIBUTE_NAMES:
        value = attributes.get(name)
        if value is not None and not isinstance(value, str):
            raise IRSchemaError(f"Gated DeltaNet {name} must be a string or None.")
        if value is not None:
            result[name] = value
    head_order = attributes.get("value_head_order", "grouped")
    if head_order not in ("grouped", "tiled"):
        raise IRSchemaError("Gated DeltaNet value_head_order must be grouped or tiled.")
    if head_order != "grouped":
        result["value_head_order"] = head_order
    integer_values = [int(result[name]) for name in _DIMENSION_ATTRIBUTE_NAMES if name != "epsilon"]
    if any(value <= 0 for value in integer_values) or float(result["epsilon"]) <= 0:
        raise IRSchemaError("Gated DeltaNet dimensions, block sizes, and epsilon must be positive.")
    if int(result["num_value_heads"]) % int(result["num_key_heads"]) != 0:
        raise IRSchemaError("Gated DeltaNet value heads must be divisible by key heads.")
    return result
