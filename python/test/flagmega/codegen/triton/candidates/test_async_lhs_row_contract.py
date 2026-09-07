# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Async row copies require complete local rows, not a model-specific SBP."""

from dataclasses import replace

import pytest

from triton.flagmega import ir as fm
from triton.flagmega.codegen.triton.candidates.dense_matmul import _supports_lhs_staging
from triton.flagmega.targets.nvidia import sm90_triton_implementation_model


@pytest.mark.parametrize("shape, policy, expected", (
    ((1, 6144), fm.SBP.broadcast(), True),
    ((1, 12288), fm.SBP.split_contiguous((0,), 6144), True),
    ((1, 12288), fm.SBP.split_block_cyclic((0,), 1), True),
    ((1, 12000), fm.SBP.split_contiguous((0,), 6144), False),
    ((1, 16384), fm.SBP.broadcast(), False),
    ((2, 6144), fm.SBP.broadcast(), False),
))
def test_candidate_uses_local_capacity_and_all_owner_activity(shape, policy, expected):
    model = sm90_triton_implementation_model()
    candidate = next(value for value in model.implementations if value.id.endswith("norm_stats_lhs8192_async"))
    value = fm.DistributedType(fm.tensor_type("bfloat16", shape),
                               (fm.SBP.broadcast(), policy), fm.Placement((2, 3), "ab", "bb"))
    assert _supports_lhs_staging(candidate, value) is expected
    assert "async_copy" in candidate.requires
    # Synchronous staging may mask a final copy tile. This is a distinct
    # legal algorithm, not a fallback hidden inside the async implementation.
    if expected:
        wider_tile = replace(candidate, parameters={**candidate.parameters, "lhs_copy_tile": 4096})
        assert not _supports_lhs_staging(wider_tile, value)
