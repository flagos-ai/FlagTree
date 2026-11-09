# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.
import pytest
import torch
import torch_npu  # noqa: F401
import triton
import triton.language as tl


@triton.jit
def masked_load_below_base(x_ptr, o_ptr, BIG: tl.constexpr, BLOCK: tl.constexpr):
    offs = tl.arange(0, BLOCK)
    idx = offs - BIG
    mask = idx >= 0
    v = tl.load(x_ptr + idx, mask=mask, other=0.0)
    tl.store(o_ptr + offs, v)


def _expected(block: int, big: int) -> torch.Tensor:
    # Valid lanes load ones from x; masked-out lanes get other=0.
    out = torch.zeros(block, dtype=torch.float32)
    if big < block:
        out[big:] = 1.0
    return out


@pytest.mark.parametrize("big,block", [
    (2, 256),
])
def test_continuous_head_mask_subi_load(big, block):
    """
    Usage:
    PYTORCH_NPU_ALLOC_CONF=expandable_segments:True pytest test_continuous_mask_subi_load.py
    """
    x = torch.ones(100000, dtype=torch.float32, device="npu")
    o = torch.full((block, ), -1.0, dtype=torch.float32, device="npu")

    masked_load_below_base[(1, )](x, o, big, block)
    torch.npu.synchronize()

    torch.testing.assert_close(o.cpu(), _expected(block, big).cpu(), rtol=0, atol=0)
