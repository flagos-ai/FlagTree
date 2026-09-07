# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT

import pytest

from triton.flagmega.codegen.triton.descriptors import coalesce_dense_weight_groups
from triton.flagmega.errors import CodegenError


def _group(offset, rows, *, count=2, k=32, dtype="bfloat16"):
    member_nbytes = rows * k * 2
    return {
        "offset": offset,
        "member_nbytes": member_nbytes,
        "stride": member_nbytes,
        "count": count,
        "shape": [rows, k],
        "dtype": dtype,
    }


def test_coalesce_dense_weight_groups_preserves_role_row_origins():
    groups = {
        "left": _group(256, 16),
        "right": _group(256 + 2 * 16 * 32 * 2, 8),
    }

    descriptor = coalesce_dense_weight_groups(
        groups, ("left", "right"), argument="fused", block_shape=(8, 32)
    )

    assert descriptor["offset"] == 256
    assert descriptor["shape"] == [48, 32]
    assert descriptor["nbytes"] == 48 * 32 * 2
    assert descriptor["row_layout"] == {
        "left": {"base_row": 0, "rows_per_member": 16},
        "right": {"base_row": 32, "rows_per_member": 8},
    }


def test_coalesce_dense_weight_groups_rejects_padding_between_roles():
    groups = {
        "left": _group(0, 16),
        "right": _group(4096, 8),
    }

    with pytest.raises(CodegenError, match="not byte-adjacent"):
        coalesce_dense_weight_groups(
            groups, ("left", "right"), argument="fused", block_shape=(8, 32)
        )
