# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Repeated heads, head transitions, tails and inactive owners use valid projections."""

import pytest

from triton.flagmega import ir as fm
from python.test.flagmega.codegen.triton.kernels.gdn_recurrent.helpers import execute_recurrent
from python.test.flagmega.gdn.recurrent_helpers import recurrent_case, recurrent_reference


@pytest.mark.parametrize("dimension,value_dimension,mesh,split_axes",
                         [(16, 16, (2, 4), (0, 1)),  # Two value tiles share each projection key.
                          (12, 12, (1, 1), (0, 1)),  # Each owner crosses several head boundaries.
                          (8, 6, (1, 2), (0, 1)),  # Mixed head IDs in one four-value tile.
                          (12, 12, (2, 4), (0, 1)),  # Final partial tile has inactive lanes.
                          (8, 8, (2, 4), ()),  # Fully replicated state has only one elected writer.
                          (128, 128, (2, 4), (0, 1)),  # Many cache hits, followed by new calls.
                          ])
def test_projection_cache_respects_head_keys_and_call_lifetime(tmp_path, dimension, value_dimension, mesh, split_axes):
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (9, 0):
        pytest.skip("SM90 CUDA required")
    config, attrs, types, values = recurrent_case(dimension=dimension, value_dimension=value_dimension,
                                                  qk_norm_mode="add", qk_norm_epsilon=1e-6, round_core=True)
    # The projection source changes on every call, so retaining a cache across
    # tokens (instead of within this invocation) would corrupt the next state.
    values["projection_input"][:, 0] = torch.tensor([1., -2., .5]).bfloat16()
    expected, expected_states = recurrent_reference(values, attrs)
    actual, states = execute_recurrent(tmp_path, torch, config, attrs, types, values,
                                       placement=fm.Placement(mesh, "yx", "bb"), split_axes=split_axes)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    for actual_state, expected_state in zip(states, expected_states):
        torch.testing.assert_close(actual_state, expected_state, rtol=2e-5, atol=1e-7)
