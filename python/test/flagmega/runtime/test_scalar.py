# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT

import pytest

from triton.flagmega.errors import RuntimeContractError
from triton.flagmega.runtime.scalar import validate_scalar_argument


@pytest.mark.parametrize("dtype,value", [
    ("int32", -(1 << 31)),
    ("int32", (1 << 31) - 1),
    ("int64", -(1 << 63)),
    ("int64", (1 << 63) - 1),
    ("bool", True),
    ("bool", False),
    ("float32", 0.25),
])
def test_scalar_abi_accepts_only_typed_by_value_host_values(dtype, value):
    validate_scalar_argument("index", value, dtype)


@pytest.mark.parametrize("dtype,value", [
    ("int32", -(1 << 31) - 1),
    ("int32", 1 << 31),
    ("int32", True),
    ("int32", 1.5),
    ("int64", -(1 << 63) - 1),
    ("int64", 1 << 63),
    ("bool", 1),
    ("float32", 1),
    ("bfloat16", 1.0),
])
def test_scalar_abi_rejects_lossy_or_ambiguous_conversions(dtype, value):
    with pytest.raises(RuntimeContractError):
        validate_scalar_argument("index", value, dtype)
