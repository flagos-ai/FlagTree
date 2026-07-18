# flagtree tle
"""
TLE tle.load end-to-end test (sunrise)
======================================

tle.load is tl.load plus a `tt.load.async` bool attribute (see
python/triton/experimental/tle/language/core.py). When is_async=True, the
add_lower_async_load TTGIR pass (wired into the sunrise pipeline) rewrites the
load into the async copy chain: local_alloc + async_copy_global_to_local +
async_commit_group + async_wait + local_load. On sunrise these lower to
async_load.* + STVM::Barrier0Op (the async_wait barrier), so the result must be
numerically identical to a plain tl.load. is_async=False is exactly tl.load.

This test loads through tle.load (both async modes), adds, and stores; result
must match a torch reference.
"""

import torch
import triton
import triton.language as tl
import triton.experimental.tle.language as tle
import pytest

DEVICE = triton.runtime.driver.active.get_active_torch_device()


@triton.jit
def tle_load_add_kernel(x_ptr, y_ptr, out_ptr, n_elements, BLOCK: tl.constexpr, IS_ASYNC: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n_elements
    # tle.load: tl.load + tt.load.async attr. IS_ASYNC=True -> async copy chain.
    x = tle.load(x_ptr + offs, mask=mask, other=0.0, is_async=IS_ASYNC)
    y = tle.load(y_ptr + offs, mask=mask, other=0.0, is_async=IS_ASYNC)
    tl.store(out_ptr + offs, x + y, mask=mask)


@pytest.mark.parametrize("is_async", [False, True])
@pytest.mark.parametrize("n_elements", [128, 256, 1000])
def test_tle_load_add(n_elements, is_async):
    torch.manual_seed(0)
    BLOCK = 128
    x = torch.randn(n_elements, device=DEVICE, dtype=torch.float32)
    y = torch.randn(n_elements, device=DEVICE, dtype=torch.float32)
    out = torch.empty_like(x)

    grid = (triton.cdiv(n_elements, BLOCK), )
    tle_load_add_kernel[grid](x, y, out, n_elements, BLOCK, is_async)

    torch.testing.assert_close(out.cpu(), (x + y).cpu(), atol=1e-5, rtol=1e-5)


@triton.jit
def tle_load_2d_kernel(x_ptr, out_ptr, M: tl.constexpr, N: tl.constexpr, IS_ASYNC: tl.constexpr):
    offs_m = tl.arange(0, M)
    offs_n = tl.arange(0, N)
    ptrs = x_ptr + offs_m[:, None] * N + offs_n[None, :]
    # 2D async load -> async copy of a tile into shared, then local_load.
    x = tle.load(ptrs, is_async=IS_ASYNC)
    tl.store(out_ptr + offs_m[:, None] * N + offs_n[None, :], x * 2.0)


@pytest.mark.parametrize("is_async", [False, True])
@pytest.mark.parametrize("M,N", [(32, 64), (64, 64)])
def test_tle_load_2d(M, N, is_async):
    torch.manual_seed(1)
    x = torch.randn(M, N, device=DEVICE, dtype=torch.float32)
    out = torch.empty_like(x)

    tle_load_2d_kernel[(1, )](x, out, M, N, is_async)

    torch.testing.assert_close(out.cpu(), (x * 2.0).cpu(), atol=1e-5, rtol=1e-5)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
