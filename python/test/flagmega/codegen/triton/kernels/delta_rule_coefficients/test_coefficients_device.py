# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT

import pytest

from triton.flagmega.ir.ops.nn.delta_rule_coefficients import delta_rule_coefficients
from python.test.flagmega.codegen.triton.kernels.delta_rule_coefficients.helpers import execute_coefficients


@pytest.mark.parametrize("block_size,tokens,dimension", [(8, 3, 1), (16, 17, 7), (32, 33, 16), (64, 65, 16),
                                                       (64, 1, 128), (64, 65, 128)])
@pytest.mark.parametrize("split_axes,local_inputs", [((), False), ((0, 1), False), ((0, ), True)])
def test_coefficients_normal_compile_rounding_tail_and_storage(tmp_path, block_size, tokens, dimension, split_axes,
                                                               local_inputs):
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (9, 0):
        pytest.skip("SM90 CUDA required")
    generator = torch.Generator().manual_seed(1649)
    key = (torch.randint(-4, 5, (tokens, 4, dimension), generator=generator).float() / 16).bfloat16()
    beta = torch.randint(1, 8, (tokens, 8), generator=generator).float() / 8
    actual = execute_coefficients(tmp_path, key, beta, block_size=block_size, split_axes=split_axes,
                                  local_inputs=local_inputs)
    expected = delta_rule_coefficients(key, beta, block_size, torch=torch)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
