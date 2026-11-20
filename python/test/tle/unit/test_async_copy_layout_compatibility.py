# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Required cp.async widths must match both global and shared layouts."""

import pytest
import torch
import triton
import triton.language as tl
import triton.experimental.tle.language as tle
from triton._internal_testing import is_hopper

pytestmark = pytest.mark.skipif(not is_hopper(), reason="requires NVIDIA Hopper")


@triton.jit
def _layout_copy(source, result, COLUMN_MAJOR: tl.constexpr, SHARED_LAYOUT: tl.constexpr):
    row = tl.arange(0, 4)[:, None]
    col = tl.arange(0, 32)[None, :]
    stage = tle.gpu.alloc([4, 32], dtype=source.dtype.element_ty, layout=SHARED_LAYOUT, nv_mma_shared_layout=False)
    offsets = row * 32 + col
    if COLUMN_MAJOR:
        offsets = col * 4 + row
    tle.gpu.copy(source + offsets, stage, [4, 32], is_async=True)
    tle.gpu.async_commit_group()
    tle.gpu.async_wait_group(0)
    tl.debug_barrier()
    values = tl.load(tle.gpu.local_ptr(stage, (row, col), [4, 32]))
    tl.store(result + row * 32 + col, values)


def test_column_bf16_rejects_unrepresentable_shared_layout(capfd):
    source = torch.arange(128, device="cuda", dtype=torch.bfloat16)
    result = torch.empty_like(source)
    with pytest.raises(RuntimeError, match="PassManager::run failed"):
        _layout_copy.warmup(source, result, True, None, grid=(1, ), num_warps=4)
    assert "cannot be represented by a legal NVIDIA cp.async width" in capfd.readouterr().err


@pytest.mark.parametrize("dtype,column_major,match_shared", [
    (torch.bfloat16, False, False),
    (torch.bfloat16, True, True),
    (torch.float32, True, False),
])
@pytest.mark.parametrize("num_warps", [4, 8])
def test_representable_source_shared_layouts_execute(dtype, column_major, match_shared, num_warps):
    source = torch.arange(128, device="cuda", dtype=dtype)
    result = torch.empty_like(source)
    layout = tle.gpu.swizzled_shared_layout(1, 1, 1, [0, 1], [1, 1], [1, 1], [1, 1], [0, 1]) if match_shared else None
    kernel = _layout_copy[(1, )](source, result, column_major, layout, num_warps=num_warps)
    expected = source.reshape(32, 4).T.reshape(-1) if column_major else source
    torch.testing.assert_close(result, expected, rtol=0, atol=0)
    assert "cp.async" in kernel.asm["ptx"]
