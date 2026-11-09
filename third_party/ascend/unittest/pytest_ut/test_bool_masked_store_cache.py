# Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
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
import torch_npu
import triton
import triton.language as tl


@triton.jit
def bool_masked_store_cache_kernel(x_ptr, y_ptr, mask_ptr):
    offsets = tl.program_id(0) + tl.arange(0, 1)

    mask = tl.load(mask_ptr + offsets) != 0
    value = tl.load(x_ptr + offsets)

    tl.store(
        y_ptr + offsets,
        value,
        mask=mask,
        cache_modifier=".cg",
        eviction_policy="evict_first",
    )


@pytest.mark.functiontest
def test_bool_masked_store_with_cache_modifier_and_eviction_policy():
    x = torch.tensor([True], dtype=torch.bool).npu()
    y = torch.tensor([False], dtype=torch.bool).npu()
    mask = torch.tensor([True], dtype=torch.bool).npu()

    bool_masked_store_cache_kernel[(1, )](x, y, mask)

    torch.npu.synchronize()

    assert y.cpu().item() is True
