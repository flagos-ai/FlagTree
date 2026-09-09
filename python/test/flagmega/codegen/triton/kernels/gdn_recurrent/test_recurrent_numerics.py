# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT

import pytest

from python.test.flagmega.codegen.triton.kernels.gdn_recurrent.helpers import execute_recurrent
from python.test.flagmega.gdn.recurrent_helpers import recurrent_case, recurrent_reference


@pytest.mark.parametrize("mode", ["clamp", "add"])
@pytest.mark.parametrize("round_qk,round_beta,round_core", [(False, True, False), (False, True, True),
                                                            (True, False, True), (True, False, False)])
def test_recurrent_storage_boundaries_on_device(tmp_path, mode, round_qk, round_beta, round_core):
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (9, 0):
        pytest.skip("SM90 CUDA required")
    config, attrs, types, values = recurrent_case(qk_norm_mode=mode, qk_norm_epsilon=1e-6, round_normalized_qk=round_qk,
                                                  round_beta=round_beta, round_core=round_core)
    expected, expected_states = recurrent_reference(values, attrs)
    actual, states = execute_recurrent(tmp_path, torch, config, attrs, types, values)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    for actual_state, expected_state in zip(states, expected_states):
        torch.testing.assert_close(actual_state, expected_state, rtol=2e-5, atol=1e-7)
