# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT

from triton.flagmega import ir as fm
from triton.flagmega.passes.auto_distributed.candidate_identity import (
    distributed_candidate_id,
)


def test_candidate_identity_is_a_readable_type_relation_not_an_ordinal():
    tensor = fm.tensor_type("bfloat16", (1, 2048))
    placement = fm.Placement((8, 16), "yx", "bb")
    split_y = fm.DistributedType(
        tensor,
        (fm.SBP.broadcast(), fm.SBP.split_contiguous((0,), 256)),
        placement,
    )
    split_yx = fm.DistributedType(
        tensor,
        (fm.SBP.broadcast(), fm.SBP.split_contiguous((0, 1), 16)),
        placement,
    )

    candidate = distributed_candidate_id(
        "norm", "inferred", split_yx, (split_y,)
    )

    assert candidate == (
        "distribution.norm.inferred."
        "mesh_yx_8x16_bb."
        "in_d_b_s_c_h0_g256."
        "out_d_b_s_c_h0_1_g16"
    )
    assert "layout_" not in candidate


def test_candidate_identity_changes_with_mesh_or_split_semantics():
    tensor = fm.tensor_type("float32", (128,))
    yx = fm.Placement((8, 16), "yx", "bb")
    x = fm.Placement((128,), "x", "b")
    lhs = fm.DistributedType(
        tensor, (fm.SBP.split_contiguous((0, 1), 1),), yx
    )
    different_axes = fm.DistributedType(
        tensor, (fm.SBP.split_contiguous((0,), 16),), yx
    )
    different_mesh = fm.DistributedType(
        tensor, (fm.SBP.split_contiguous((0,), 1),), x
    )

    identities = {
        distributed_candidate_id("value", "inferred", value, (value,))
        for value in (lhs, different_axes, different_mesh)
    }

    assert len(identities) == 3
