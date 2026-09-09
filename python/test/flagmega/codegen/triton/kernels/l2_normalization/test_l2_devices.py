# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT

import pytest

from triton.flagmega import ir as fm
from python.test.flagmega.codegen.triton.kernels.l2_normalization.helpers import execute_l2


@pytest.mark.parametrize("shape,axes,split", (
    ((3, 8, 128), (-1,), True), ((3, 8, 513), (-1,), True),
    ((3, 7, 9), (0, 2), True), ((1, 3, 128), (-1,), False),
))
@pytest.mark.parametrize("local_input,local_result", ((False, False), (True, False), (False, True), (True, True)))
@pytest.mark.parametrize("epsilon_mode,division", (("add", "reciprocal_multiply"), ("clamp", "divide")))
def test_l2_device_axes_tiles_and_physical_storage(tmp_path, shape, axes, split, local_input, local_result,
                                                   epsilon_mode, division):
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (9, 0):
        pytest.skip("SM90 CUDA required")
    generator = torch.Generator().manual_seed(137)
    value = torch.randn(shape, generator=generator).bfloat16()
    b = fm.SBP.broadcast()
    policies = (b, fm.SBP.split_block_cyclic((0,), 2), b) if split else (b, b, b)
    actual = execute_l2(tmp_path, value, policies=policies, local_input=local_input, local_result=local_result,
                         axes=axes, epsilon=1e-6, epsilon_mode=epsilon_mode, division_mode=division)
    sums = value.float().square().sum(axes, keepdim=True)
    denom = (sums + 1e-6 if epsilon_mode == "add" else sums.clamp_min(1e-6)).sqrt()
    expected = value.float() * (1. / denom) if division == "reciprocal_multiply" else value.float() / denom
    torch.testing.assert_close(actual, expected.bfloat16(), rtol=0, atol=0)
