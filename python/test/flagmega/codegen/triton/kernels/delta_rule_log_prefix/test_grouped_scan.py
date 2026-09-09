# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT

import pytest

from triton.flagmega import ir as fm
from python.test.flagmega.codegen.triton.kernels.delta_rule_log_prefix.helpers import execute_log_prefix
from python.test.flagmega.codegen.triton.kernels.delta_rule_log_prefix.oracles import scan_oracle


def require_device():
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (9, 0):
        pytest.skip("SM90 CUDA required")
    return torch


@pytest.mark.parametrize("block,group,tokens", [(8, 4, 3), (64, 32, 31), (64, 32, 33), (64, 32, 65), (128, 16, 129)])
@pytest.mark.parametrize("mode", ["fast", "accurate"])
@pytest.mark.parametrize("head,local", [(fm.SBP.broadcast(), False), (fm.SBP.split_contiguous((0, 1)), False),
                                        (fm.SBP.split_block_cyclic((0, ), 2), True)])
def test_scan_group_boundaries_padding_and_physical_storage(tmp_path, block, group, tokens, mode, head, local):
    torch = require_device()
    generator = torch.Generator().manual_seed(339)
    alpha = torch.exp2(torch.randint(-4, 5, (tokens, 8), generator=generator).float())
    expected = scan_oracle(alpha, block, group, mode, torch=torch)
    actual = execute_log_prefix(tmp_path, alpha, head_policy=head, local_input=local, block_size=block,
                                scan_group_size=group, log2_mode=mode)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_padding_does_not_add_epsilon_decay(tmp_path):
    torch = require_device()
    alpha = torch.zeros((1, 4))
    actual = execute_log_prefix(tmp_path, alpha, block_size=8, scan_group_size=4, epsilon=2.)
    torch.testing.assert_close(actual, torch.ones((1, 4, 8)), rtol=0, atol=0)
