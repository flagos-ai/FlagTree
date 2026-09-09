# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Device checks for independent rounding and accumulator ordering contracts."""

import pytest

from python.test.flagmega.codegen.triton.kernels.gdn_convolution.helpers import execute_convolution


@pytest.mark.parametrize("kernel", [2, 4, 6])
@pytest.mark.parametrize("round_products,round_activation", [(True, True), (True, False), (False, True),
                                                             (False, False)])
def test_numerical_attributes_survive_python_resume_and_real_lowering(tmp_path, kernel, round_products,
                                                                      round_activation):
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (9, 0):
        pytest.skip("SM90 CUDA required")
    generator = torch.Generator().manual_seed(214)
    inputs = torch.randn((3, 64), generator=generator).bfloat16()
    weight = torch.randn((64, kernel), generator=generator).bfloat16()
    history = torch.randn((64, kernel - 1), generator=generator).bfloat16()
    actual, states = execute_convolution(tmp_path, torch, inputs, weight, history, round_products=round_products,
                                         round_before_activation=round_activation, accumulation_order="chronological")
    expected = []
    for index, current in enumerate(inputs):
        window = torch.cat((history, current[:, None]), dim=1)
        products = window.float() * weight.float()
        if round_products:
            products = products.bfloat16().float()
        value = torch.zeros(64)
        for tap in range(kernel):
            value = (value.double() + products[:, tap].double()).float()
        if round_activation:
            value = value.bfloat16().float()
        expected.append(torch.nn.functional.silu(value).bfloat16())
        history = window[:, 1:]
        torch.testing.assert_close(states[index], history, rtol=0, atol=0)
    torch.testing.assert_close(actual, torch.stack(expected), rtol=0, atol=0)


@pytest.mark.parametrize("order,accumulator", [("current_first", 1.), ("chronological", 2.)])
def test_explicit_order_is_observable_under_cancellation(tmp_path, order, accumulator):
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (9, 0):
        pytest.skip("SM90 CUDA required")
    inputs = torch.ones((1, 64), dtype=torch.bfloat16)
    weight = torch.ones((64, 4), dtype=torch.bfloat16)
    history = torch.tensor([1.e20, -1.e20, 1.], dtype=torch.bfloat16).repeat(64, 1)
    actual, _ = execute_convolution(tmp_path, torch, inputs, weight, history, accumulation_order=order)
    expected = torch.nn.functional.silu(torch.full((1, 64), accumulator)).bfloat16()
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
