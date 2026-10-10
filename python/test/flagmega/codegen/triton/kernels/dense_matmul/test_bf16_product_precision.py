# Copyright 2026- FlagOS Contributors
# SPDX-License-Identifier: MIT
"""GEMV operands must be promoted before multiplying, including fused GLU."""
import importlib.util

import pytest

from triton.flagmega.codegen.triton.templates import TritonTemplateRegistry


@pytest.mark.parametrize("family", ["dense_matmul", "dense_matmul_glu"])
def test_bf16_gemv_retains_product_bits_before_reduction(tmp_path, family):
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("requires a GPU")
    helper_name = "_accumulate"
    context = {family + "_accumulate_helper": helper_name}
    helper = TritonTemplateRegistry().render(
        f"kernels/{family}/_local_accumulate.py.jinja", context,
    )
    arguments = "X, W, offsets, offsets[None, :], offsets < 16, tl.arange(0, 1) < 1"
    result = "value"
    if family == "dense_matmul_glu":
        arguments = "X, W, W, offsets, offsets[None, :], offsets[None, :], offsets < 16, tl.arange(0, 1) < 1"
        result = "value[0] + value[1]"
    source = f'''import triton
import triton.language as tl
{helper}
@triton.jit
def run(X, W, Y):
    offsets = tl.arange(0, 16)
    value = _accumulate({arguments})
    tl.store(Y + tl.arange(0, 1), {result})
'''
    path = tmp_path / "gemv_product_precision.py"
    path.write_text(source)
    spec = importlib.util.spec_from_file_location("gemv_product_precision", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    # Both operands are exactly representable BF16; their product requires
    # more mantissa bits. FP32 accumulation cannot recover a rounded product.
    x = torch.full((16,), 1.0078125, device="cuda", dtype=torch.bfloat16)
    w = x.clone()
    y = torch.empty((1,), device="cuda", dtype=torch.float32)
    module.run[(1,)](x, w, y)
    expected = (x.float() * w.float()).sum().reshape(1)
    if family == "dense_matmul_glu":
        expected *= 2
    torch.testing.assert_close(y, expected, rtol=0, atol=0)
