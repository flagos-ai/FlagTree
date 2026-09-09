# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT

import pytest

from triton.flagmega import ir as fm
from python.test.flagmega.codegen.triton.kernels.delta_rule_gates.helpers import execute_gates


@pytest.mark.parametrize("token,head", (
    (fm.SBP.broadcast(), fm.SBP.broadcast()),
    (fm.SBP.broadcast(), fm.SBP.split_contiguous((0, 1))),
    (fm.SBP.split_block_cyclic((0,), 2), fm.SBP.split_contiguous((1,))),
))
@pytest.mark.parametrize("local_inputs,local_results", (
    ((), (False, False)), (("a", "dt_bias"), (True, False)), (("b", "a_log"), (False, True)),
))
@pytest.mark.parametrize("mode", ("fast", "accurate"))
def test_gate_device_numerics_and_independent_storage(tmp_path, token, head, local_inputs, local_results, mode):
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (9, 0):
        pytest.skip("SM90 CUDA required")
    generator = torch.Generator().manual_seed(917)
    values = {"a": torch.randn((17, 8), generator=generator).bfloat16(),
              "b": torch.randn((17, 8), generator=generator).bfloat16(),
              "a_log": torch.randn(8, generator=generator) * .25,
              "dt_bias": torch.randn(8, generator=generator).bfloat16() * .25}
    actual, beta = execute_gates(tmp_path, torch, values, token_policy=token, head_policy=head,
                                 local_inputs=local_inputs, local_results=local_results, alpha_exp_mode=mode)
    x = values["a"].float() + values["dt_bias"].float()
    expected = (-values["a_log"].exp() * torch.nn.functional.softplus(x)).exp()
    # CPU accurate math is not a bitwise oracle for the declared fast inner
    # exp/log/sigmoid. The pinned native GPU comparison is separate.
    torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-7)
    torch.testing.assert_close(beta, values["b"].float().sigmoid(), rtol=2e-7, atol=1e-7)
