# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Host values passed by value through a typed TIR scalar ABI."""

from triton.flagmega.errors import RuntimeContractError


def validate_scalar_argument(name: str, value, dtype: str) -> None:
    if dtype in {"int32", "int64"}:
        bits = 32 if dtype == "int32" else 64
        valid = type(value) is int and -(1 << (bits - 1)) <= value < (1 << (bits - 1))
    elif dtype == "bool":
        valid = type(value) is bool
    elif dtype == "float32":
        valid = type(value) is float
    else:
        raise RuntimeContractError(f"Unsupported by-value scalar ABI dtype {dtype!r} for {name!r}.")
    if not valid:
        raise RuntimeContractError(f"TIR scalar {name!r} requires a host {dtype} value, got {value!r}.")
