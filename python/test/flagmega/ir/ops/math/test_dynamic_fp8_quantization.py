# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Device choice must preserve the FP8 scale and round-to-even boundary."""

import pytest
from triton.flagmega.ir.ops.math._block_scaled import dynamic_block_quant_dequant


@pytest.mark.parametrize("dtype_name", ("bfloat16", "float32"))
@pytest.mark.parametrize("sign", (-1, 1))
def test_dynamic_quantization_keeps_exact_scale_ties_on_cpu_and_cuda(dtype_name, sign):
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required to compare the evaluator's device semantics")
    value = torch.zeros((2, 129), dtype=getattr(torch, dtype_name))
    value[0, 0], value[0, 1] = 1.203125, sign * .81640625
    # 1.203125 / 448 = 11/4096 exactly. The second quotient is +/-304,
    # halfway between +/-288 and +/-320, so the even FP8 result is +/-320.
    # The second row and the padded final group cover the all-zero scale path.
    expected = torch.zeros((2, 129), dtype=torch.float32)
    expected[0, 0], expected[0, 1] = 1.203125, sign * .859375
    for device in ("cpu", "cuda"):
        actual = dynamic_block_quant_dequant(value.to(device), block_k=128).cpu()
        assert actual.shape == value.shape
        assert actual.dtype == torch.float32
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
