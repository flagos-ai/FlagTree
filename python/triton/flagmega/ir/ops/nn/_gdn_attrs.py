# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Shared attribute contract used by the three GDN op definitions."""

from typing import Mapping

from triton.flagmega.errors import IRSchemaError


GDN_ATTRIBUTE_NAMES = (
    "num_key_heads",
    "num_value_heads",
    "key_head_dim",
    "value_head_dim",
    "conv_kernel_size",
    "epsilon",
    "weight_block_n",
    "weight_block_k",
)


def normalize_gdn_attrs(attributes: Mapping[str, object]) -> dict[str, object]:
    if set(attributes) != set(GDN_ATTRIBUTE_NAMES):
        raise IRSchemaError(f"Gated DeltaNet attributes must be exactly {GDN_ATTRIBUTE_NAMES}.")
    result: dict[str, object] = {
        name: float(attributes[name]) if name == "epsilon" else int(attributes[name])
        for name in GDN_ATTRIBUTE_NAMES
    }
    integer_values = [int(result[name]) for name in GDN_ATTRIBUTE_NAMES if name != "epsilon"]
    if any(value <= 0 for value in integer_values) or float(result["epsilon"]) <= 0:
        raise IRSchemaError("Gated DeltaNet dimensions, block sizes, and epsilon must be positive.")
    if int(result["num_value_heads"]) % int(result["num_key_heads"]) != 0:
        raise IRSchemaError("Gated DeltaNet value heads must be divisible by key heads.")
    return result
