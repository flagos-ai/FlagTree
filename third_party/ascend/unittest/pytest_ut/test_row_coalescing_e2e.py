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
"""NPU regression for the pure-SIMT Row compatibility scheduling slot."""

import pytest
import torch
import triton
import triton.language as tl
from triton.backends.ascend.utils import is_compile_on_910_95

pytestmark = pytest.mark.skipif(
    not is_compile_on_910_95(triton.runtime.driver.active.get_current_target().arch),
    reason="Row pure-SIMT binary validation requires an Ascend 950 toolchain",
)


@triton.jit
def row_coalescing_tail_copy(src, dst, n, BLOCK: tl.constexpr):
    # Keep the canonical scalar row guard that RowCoalescing matches.  The
    # runtime ``n`` makes the final group partial, so the pass must combine its
    # generated row-tail mask with these load/store masks.
    pid = tl.program_id(0)
    if pid >= n:
        return
    offsets = pid + tl.arange(0, BLOCK)
    mask = offsets < n
    value = tl.load(src + offsets, mask=mask, other=0.0)
    tl.store(dst + offsets, value, mask=mask)


@pytest.mark.parametrize("dtype", (torch.float32, torch.float16))
def test_row_coalescing_tail_pure_simt_e2e(dtype):
    # 19 is intentionally not divisible by the expected H=8 for a 16-element
    # base tile.  The compatibility pass must use the original ceil-div launch
    # contract and suppress stores for rows past ``n``.
    n = 19
    src = torch.arange(n, dtype=dtype).npu()
    dst = torch.full_like(src, -1)

    row_coalescing_tail_copy[(n, )](
        src,
        dst,
        n,
        BLOCK=16,
        compile_mode="simt_only",
    )
    torch.npu.synchronize()

    assert torch.equal(dst.cpu(), src.cpu())
