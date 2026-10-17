# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: MIT
"""The FP32 branch product must round before the mHC residual addition."""

import pytest
from triton.flagmega.runtime import load

from python.test.flagmega.codegen.triton.kernels.mhc.test_mhc_runtime import (
    HC,
    HIDDEN,
    POST_INPUTS,
    TOKENS,
    _artifact,
    _post_module,
    _run,
    _skip_without_sm90,
)


def test_post_product_rounding_survives_cancellation_before_bf16_output(tmp_path):
    torch = pytest.importorskip("torch")
    _skip_without_sm90(torch)
    value = torch.full((TOKENS, HIDDEN), 129 / 128, dtype=torch.bfloat16)
    residual = torch.zeros((TOKENS, HC, HIDDEN), dtype=torch.bfloat16)
    residual[:, 0] = -1
    post = torch.full((TOKENS, HC), 1 + 2 ** -23, dtype=torch.float32)
    diagonal = 0.9999696016311646
    comb = torch.full((TOKENS, HC, HC), (1 - diagonal) / (HC - 1), dtype=torch.float32)
    for index in range(HC):
        comb[:, index, index] = diagonal
    mixed = torch.einsum("tij,tih->tjh", comb, residual.float())
    expected = (mixed + post.unsqueeze(-1) * value.float().unsqueeze(-2)).bfloat16()
    # The separately rounded sum is exactly the midpoint; RNE selects the
    # lower BF16 value. Fusing the product into the add crosses that midpoint.
    assert torch.equal(expected[:, 0], torch.full((TOKENS, HIDDEN), 2 ** -7, dtype=torch.bfloat16))
    torch.testing.assert_close(comb.sum(-1), torch.ones((TOKENS, HC)), rtol=0, atol=2 ** -23)
    host = {"value": value, "residual": residual, "post": post, "comb": comb}
    values = {name: host[name].cuda() for name in POST_INPUTS}
    outputs = {"mixed": torch.empty_like(expected, device="cuda")}
    runtime = load(_artifact(tmp_path, "post-rne", _post_module()), device="cuda:0")
    try:
        _run(runtime, values, outputs, torch)
        torch.testing.assert_close(outputs["mixed"].cpu(), expected, rtol=0, atol=0)
        assert runtime.resource_report["spill_bytes"] == 0
    finally:
        runtime.close()
