# flagtree tle
"""
TLE tle.cumsum end-to-end test (sunrise)
========================================

tle.cumsum computes an EXCLUSIVE CTA-wide prefix sum and returns
(exclusive_sum, total_sum). It is a single TLE op (ExclusiveCumsumOp), distinct
from upstream tl.cumsum (which is inclusive and tensor-local).

On sunrise the lowering uses a sunrise-local conversion
(third_party/sunrise/plugin/lib/TritonSunriseGPUToLLVM/ExclusiveCumsumOpToLLVM.cpp)
that drops the NVVM `shfl.sync.up.b32` i32 fast path and instead always uses the
generic targetInfo.shuffleUp / shuffleIdx / shared-memory cross-warp scan, which
lowers to STVM. Scratch for the scan is reserved by the sunrise TLE-aware
allocation pass.

reference: exclusive[i] = sum(x[:i]); total = sum(x).
"""

import torch
import triton
import triton.language as tl
import triton.experimental.tle.language as tle
import pytest

DEVICE = triton.runtime.driver.active.get_active_torch_device()


@triton.jit
def cumsum_kernel(x_ptr, exclusive_ptr, total_ptr, n, BLOCK: tl.constexpr, REVERSE: tl.constexpr):
    offs = tl.arange(0, BLOCK)
    mask = offs < n
    x = tl.load(x_ptr + offs, mask=mask, other=0)
    exclusive, total = tle.cumsum(x, axis=0, reverse=REVERSE)
    tl.store(exclusive_ptr + offs, exclusive, mask=mask)
    tl.store(total_ptr, total)


def _ref_exclusive(x, reverse):
    if reverse:
        # reverse-exclusive: exclusive[i] = sum(x[i+1:])
        rev = torch.flip(x, dims=[0])
        excl_rev = torch.cumsum(rev, dim=0) - rev
        return torch.flip(excl_rev, dims=[0])
    return torch.cumsum(x, dim=0) - x


@pytest.mark.parametrize("reverse", [False, True])
@pytest.mark.parametrize("n,BLOCK", [(128, 128), (256, 256), (100, 128)])
def test_cumsum_int32(n, BLOCK, reverse):
    torch.manual_seed(0)
    x = torch.randint(0, 10, (BLOCK, ), device=DEVICE, dtype=torch.int32)
    x[n:] = 0  # masked-out tail
    exclusive = torch.empty(BLOCK, device=DEVICE, dtype=torch.int32)
    total = torch.zeros(1, device=DEVICE, dtype=torch.int32)

    cumsum_kernel[(1, )](x, exclusive, total, n, BLOCK, reverse)

    xv = x[:n].to(torch.int64)
    ref_excl = _ref_exclusive(xv, reverse)
    torch.testing.assert_close(exclusive[:n].to(torch.int64).cpu(), ref_excl.cpu(), atol=0, rtol=0)
    torch.testing.assert_close(int(total.item()), int(xv.sum().item()))


@pytest.mark.parametrize("n,BLOCK", [(128, 128), (256, 256)])
def test_cumsum_float32(n, BLOCK):
    torch.manual_seed(1)
    x = torch.randn(BLOCK, device=DEVICE, dtype=torch.float32)
    x[n:] = 0.0
    exclusive = torch.empty(BLOCK, device=DEVICE, dtype=torch.float32)
    total = torch.zeros(1, device=DEVICE, dtype=torch.float32)

    cumsum_kernel[(1, )](x, exclusive, total, n, BLOCK, False)

    ref_excl = torch.cumsum(x[:n], dim=0) - x[:n]
    torch.testing.assert_close(exclusive[:n].cpu(), ref_excl.cpu(), atol=1e-4, rtol=1e-4)
    torch.testing.assert_close(total.item(), x[:n].sum().item(), atol=1e-4, rtol=1e-4)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
