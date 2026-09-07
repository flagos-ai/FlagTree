# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT

import pytest

from triton.flagmega import ir as fm
from triton.flagmega.rules.ntt.vectorize.utility import generate_axis_candidates, padding_for


def test_padding_uses_lane_product_for_repeated_axes():
    value_type = fm.tensor_type("bfloat16", (3, 17))
    assert padding_for(value_type, (1, 1), (2, 4)) == (0, 7)


def test_padding_accepts_provably_divisible_dynamic_dimension():
    n = fm.dim("n", minimum=1, maximum=32)
    assert padding_for(fm.tensor_type("bfloat16", (n * 8,)), (0,), (8,)) == (0,)


def test_padding_rejects_unproven_dynamic_dimension():
    n = fm.dim("n", minimum=1, maximum=32)
    assert padding_for(fm.tensor_type("bfloat16", (n,)), (0,), (8,)) is None


def test_axis_candidates_skip_unproven_dynamic_and_dynamic_static_slice_cases():
    n = fm.dim("n", minimum=1, maximum=32)
    assert generate_axis_candidates(
        fm.tensor_type("bfloat16", (n,)), rule="Rule", lane_bytes=16, max_axes=1,
    ) == ()
    # Axis 1 is fixed but needs padding; SliceToShape cannot yet encode the
    # untouched dynamic axis, so this candidate must also be rejected.
    assert generate_axis_candidates(
        fm.tensor_type("bfloat16", (n, 10)), rule="Rule", lane_bytes=16, max_axes=1,
    ) == ()


@pytest.mark.parametrize("dtype,lane", [("bfloat16", 8), ("float32", 4)])
def test_axis_candidates_derive_lane_from_target_bytes(dtype, lane):
    candidates = generate_axis_candidates(
        fm.tensor_type(dtype, (5, 10)), rule="Rule", lane_bytes=16, max_axes=2,
    )
    by_id = {candidate.id: candidate for candidate in candidates}
    assert set(by_id) == {"vectorization.axes_0", "vectorization.last_axis", "vectorization.axes_0_1"}
    assert by_id["vectorization.last_axis"].lanes == (lane,)
    assert by_id["vectorization.axes_0_1"].facts["padding"] == [(-5) % lane, (-10) % lane]


@pytest.mark.parametrize(
    "value_type,lane_bytes",
    [
        (fm.tensor_type("bfloat16", ()), 16),
        (fm.tensor_type(fm.vector_type("bfloat16", (8,)), (2,)), 16),
        (fm.tensor_type("float32", (8,)), 4),
    ],
)
def test_axis_candidates_reject_scalar_rank_existing_vectors_and_lane_one(value_type, lane_bytes):
    assert generate_axis_candidates(value_type, rule="Rule", lane_bytes=lane_bytes, max_axes=2) == ()
