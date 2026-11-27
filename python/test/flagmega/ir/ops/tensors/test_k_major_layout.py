# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT

import torch

from triton.flagmega.ir.ops.tensors._k_major import (
    k_major_layout_name,
    parse_k_major_layout,
    unpack_k_major_weight,
)


def test_k_major_layout_identity_round_trips_target_supplied_lanes():
    name = k_major_layout_name(16, 64, mesh_interleaved=True)

    assert name == "k_major_mesh_interleaved_n16_k64"
    assert parse_k_major_layout(name) == (16, 64, True)


def test_generic_k_major_unpack_does_not_assume_sm90_n8_k16_geometry():
    logical = torch.arange(32 * 64, dtype=torch.float32).reshape(32, 64)
    packed = (
        logical.reshape(2, 16, 1, 64)
        .permute(2, 0, 1, 3)
        .reshape(1, 2, 4, 256)
        .contiguous()
    )

    actual = unpack_k_major_weight(packed, "k_major_n16_k64")

    torch.testing.assert_close(actual, logical, rtol=0, atol=0)
