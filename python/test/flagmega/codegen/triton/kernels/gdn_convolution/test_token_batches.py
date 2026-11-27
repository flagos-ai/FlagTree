# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT

import pytest

from triton.flagmega.ir.ops.nn._gdn_state import GatedDeltaNetStateConfig, create_gdn_state
from triton.flagmega.ir.ops.nn.gdn_convolution import gated_delta_net_convolution
from python.test.flagmega.codegen.triton.kernels.gdn_convolution.helpers import execute_convolution


@pytest.mark.parametrize("tokens_per_call", (2, 3, 9))
@pytest.mark.parametrize("round_products,round_activation,order",
                         ((True, False, "chronological"), (False, True, "current_first")))
def test_convolution_consumes_every_prompt_token_and_continues_from_updated_history(tmp_path, tokens_per_call,
                                                                                    round_products, round_activation,
                                                                                    order):
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (9, 0):
        pytest.skip("SM90 CUDA required")
    config = GatedDeltaNetStateConfig(1, 2, 4, 8, 8, 4, 32)
    generator = torch.Generator().manual_seed(719)
    inputs = (torch.randn((tokens_per_call * 2, config.conv_dim), generator=generator) * .5).bfloat16()
    weights = (torch.randn((config.conv_dim, 4), generator=generator) * .25).bfloat16()
    history = (torch.randn((config.conv_dim, 3), generator=generator) * .25).bfloat16()
    attributes = {
        "round_products": round_products, "round_before_activation": round_activation, "accumulation_order": order
    }
    expected_state = create_gdn_state(config)
    expected_state.update_convolution_layer(history)
    expected, states = [], []
    for start in range(0, inputs.shape[0], tokens_per_call):
        values, _ = gated_delta_net_convolution(qkv=inputs[start:start + tokens_per_call], state=expected_state,
                                                conv_weight=weights, conv_kernel_size=4, torch=torch, **attributes)
        expected.append(values)
        states.append(expected_state.convolution_layer().clone())
    actual, actual_states = execute_convolution(tmp_path, torch, inputs, weights, history,
                                                tokens_per_call=tokens_per_call, **attributes)
    torch.testing.assert_close(actual, torch.cat(expected), rtol=0, atol=0)
    for actual_state, expected in zip(actual_states, states):
        torch.testing.assert_close(actual_state, expected, rtol=0, atol=0)
