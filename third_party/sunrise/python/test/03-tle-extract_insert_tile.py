# flagtree tle
"""
TLE extract_tile / insert_tile end-to-end test (sunrise)
========================================================

Exercises ``tle.extract_tile`` and ``tle.insert_tile`` on the sunrise backend,
covering BOTH lowering paths now that sunrise has its own tile-op lowering
(third_party/sunrise/plugin/lib/TritonSunriseGPUToLLVM/TileOpsToLLVM.cpp):

  * STATIC + CTA-tile aligned index -> register-permutation path (no SMEM).
  * dynamic index OR non-aligned tile -> SMEM relay path. The sunrise lowering
    routes the barrier through targetInfo.barrier() (-> STVM::Barrier0Op) and
    the thread id through getThreadId() (-> STVM), so unlike the common TLE
    lowering it does not require NVVM and works on sunrise.

SMEM relay is reached via two distinct entries, both covered here:
  - dynamic (runtime) index (test_*_dynamic_index), and
  - static index with a CTA-tile non-aligned (thin) tile shape
    (test_*_static_nonaligned).

The (M, N, tile, index) values mirror python/test/tle/unit/test_extract_tile_static_index.py.
"""

import torch
import triton
import triton.language as tl
import triton.experimental.tle.language as tle
import pytest

DEVICE = triton.runtime.driver.active.get_active_torch_device()


# ---------------------------------------------------------------------------
# Static, CTA-tile-aligned index -> register path
# ---------------------------------------------------------------------------
@triton.jit
def extract_tile_kernel(x_ptr, out_ptr, M: tl.constexpr, N: tl.constexpr, TM: tl.constexpr, TN: tl.constexpr):
    offs_m = tl.arange(0, M)
    offs_n = tl.arange(0, N)
    x = tl.load(x_ptr + offs_m[:, None] * N + offs_n[None, :])

    tile = tle.extract_tile(x, index=[1, 1], tile_shape=[TM, TN])

    out_m = tl.arange(0, TM)
    out_n = tl.arange(0, TN)
    tl.store(out_ptr + out_m[:, None] * TN + out_n[None, :], tile)


@triton.jit
def insert_tile_kernel(x_ptr, y_ptr, out_ptr, M: tl.constexpr, N: tl.constexpr, TM: tl.constexpr, TN: tl.constexpr):
    offs_m = tl.arange(0, M)
    offs_n = tl.arange(0, N)
    x = tl.load(x_ptr + offs_m[:, None] * N + offs_n[None, :])

    tile_m = tl.arange(0, TM)
    tile_n = tl.arange(0, TN)
    y = tl.load(y_ptr + tile_m[:, None] * TN + tile_n[None, :])

    z = tle.insert_tile(x, y, index=[1, 1])

    tl.store(out_ptr + offs_m[:, None] * N + offs_n[None, :], z)


# ---------------------------------------------------------------------------
# Dynamic (runtime) index -> SMEM relay path
# ---------------------------------------------------------------------------
@triton.jit
def extract_tile_dyn_kernel(x_ptr, out_ptr, idx, M: tl.constexpr, N: tl.constexpr, TM: tl.constexpr, TN: tl.constexpr):
    offs_m = tl.arange(0, M)
    offs_n = tl.arange(0, N)
    x = tl.load(x_ptr + offs_m[:, None] * N + offs_n[None, :])

    # Runtime linear tile index -> always uses the SMEM relay path.
    lin = tl.load(idx)
    tile = tle.extract_tile(x, index=lin, tile_shape=[TM, TN])

    out_m = tl.arange(0, TM)
    out_n = tl.arange(0, TN)
    tl.store(out_ptr + out_m[:, None] * TN + out_n[None, :], tile)


@triton.jit
def insert_tile_dyn_kernel(x_ptr, y_ptr, out_ptr, idx, M: tl.constexpr, N: tl.constexpr, TM: tl.constexpr,
                           TN: tl.constexpr):
    offs_m = tl.arange(0, M)
    offs_n = tl.arange(0, N)
    x = tl.load(x_ptr + offs_m[:, None] * N + offs_n[None, :])

    tile_m = tl.arange(0, TM)
    tile_n = tl.arange(0, TN)
    y = tl.load(y_ptr + tile_m[:, None] * TN + tile_n[None, :])

    lin = tl.load(idx)
    z = tle.insert_tile(x, y, index=lin)

    tl.store(out_ptr + offs_m[:, None] * N + offs_n[None, :], z)


def _lin_index(grid_n, ti, tj, device):
    """row-major linear tile index for tile coord (ti, tj)."""
    return torch.tensor([ti * grid_n + tj], device=device, dtype=torch.int32)


# ---------------------------------------------------------------------------
# Register-path tests (static aligned)
# ---------------------------------------------------------------------------
def test_extract_tile_static_aligned():
    M, N = 512, 512
    TM, TN = 128, 128

    x = torch.arange(M * N, device=DEVICE, dtype=torch.float32).reshape(M, N)
    out = torch.zeros((TM, TN), device=DEVICE, dtype=torch.float32)

    extract_tile_kernel[(1, )](x, out, M, N, TM, TN)

    expected = x[TM:2 * TM, TN:2 * TN]
    torch.testing.assert_close(out.cpu(), expected.cpu(), atol=0, rtol=0)


def test_insert_tile_static_aligned():
    M, N = 128, 128
    TM, TN = 32, 32

    x = torch.arange(M * N, device=DEVICE, dtype=torch.float32).reshape(M, N)
    y = (100000 + torch.arange(TM * TN, device=DEVICE, dtype=torch.float32)).reshape(TM, TN)
    out = torch.empty_like(x)

    insert_tile_kernel[(1, )](x, y, out, M, N, TM, TN)

    expected = x.clone()
    expected[TM:2 * TM, TN:2 * TN] = y
    torch.testing.assert_close(out.cpu(), expected.cpu(), atol=0, rtol=0)


def test_extract_then_insert_roundtrip():
    """extract a tile then insert it back at the same index -> identity."""
    M, N = 128, 128
    TM, TN = 32, 32

    x = torch.randn(M, N, device=DEVICE, dtype=torch.float32)
    tile = torch.empty((TM, TN), device=DEVICE, dtype=torch.float32)
    out = torch.empty_like(x)

    extract_tile_kernel[(1, )](x, tile, M, N, TM, TN)
    insert_tile_kernel[(1, )](x, tile, out, M, N, TM, TN)

    torch.testing.assert_close(out.cpu(), x.cpu(), atol=0, rtol=0)


# ---------------------------------------------------------------------------
# SMEM-path tests (dynamic index)
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("ti,tj", [(0, 0), (1, 1), (2, 3)])
def test_extract_tile_dynamic_index(ti, tj):
    M, N = 128, 128
    TM, TN = 32, 32
    grid_n = N // TN

    x = torch.arange(M * N, device=DEVICE, dtype=torch.float32).reshape(M, N)
    out = torch.zeros((TM, TN), device=DEVICE, dtype=torch.float32)
    idx = _lin_index(grid_n, ti, tj, DEVICE)

    extract_tile_dyn_kernel[(1, )](x, out, idx, M, N, TM, TN)

    expected = x[ti * TM:(ti + 1) * TM, tj * TN:(tj + 1) * TN]
    torch.testing.assert_close(out.cpu(), expected.cpu(), atol=0, rtol=0)


@pytest.mark.parametrize("ti,tj", [(0, 0), (1, 1), (2, 3)])
def test_insert_tile_dynamic_index(ti, tj):
    M, N = 128, 128
    TM, TN = 32, 32
    grid_n = N // TN

    x = torch.arange(M * N, device=DEVICE, dtype=torch.float32).reshape(M, N)
    y = (100000 + torch.arange(TM * TN, device=DEVICE, dtype=torch.float32)).reshape(TM, TN)
    out = torch.empty_like(x)
    idx = _lin_index(grid_n, ti, tj, DEVICE)

    insert_tile_dyn_kernel[(1, )](x, y, out, idx, M, N, TM, TN)

    expected = x.clone()
    expected[ti * TM:(ti + 1) * TM, tj * TN:(tj + 1) * TN] = y
    torch.testing.assert_close(out.cpu(), expected.cpu(), atol=0, rtol=0)


# ---------------------------------------------------------------------------
# Static but CTA-tile NON-aligned index -> SMEM relay path
#
# isCTATileAligned() returns false when tileShape[i] % ctaTile[i] != 0 (or the
# tile offset is not a multiple of ctaTile[i]); ctaTile = getShapePerCTATile is
# derived from the source blocked encoding. A very "thin" tile dimension (e.g.
# 2) is almost never a multiple of the per-CTA tile, so even with a static index
# the conversion takes the SMEM relay path (lowerExtractTileViaSMEM), exercising
# the targetInfo.barrier()/getThreadId() lowering rather than the register path.
# This is a different SMEM-path entry than the dynamic-index tests above.
# ---------------------------------------------------------------------------
@triton.jit
def extract_tile_thin_kernel(x_ptr, out_ptr, M: tl.constexpr, N: tl.constexpr, TM: tl.constexpr, TN: tl.constexpr):
    offs_m = tl.arange(0, M)
    offs_n = tl.arange(0, N)
    x = tl.load(x_ptr + offs_m[:, None] * N + offs_n[None, :])

    # static index, but thin tile -> CTA-tile non-aligned -> SMEM relay path
    tile = tle.extract_tile(x, index=[1, 0], tile_shape=[TM, TN])

    out_m = tl.arange(0, TM)
    out_n = tl.arange(0, TN)
    tl.store(out_ptr + out_m[:, None] * TN + out_n[None, :], tile)


@triton.jit
def insert_tile_thin_kernel(x_ptr, y_ptr, out_ptr, M: tl.constexpr, N: tl.constexpr, TM: tl.constexpr,
                            TN: tl.constexpr):
    offs_m = tl.arange(0, M)
    offs_n = tl.arange(0, N)
    x = tl.load(x_ptr + offs_m[:, None] * N + offs_n[None, :])

    tile_m = tl.arange(0, TM)
    tile_n = tl.arange(0, TN)
    y = tl.load(y_ptr + tile_m[:, None] * TN + tile_n[None, :])

    z = tle.insert_tile(x, y, index=[1, 0])

    tl.store(out_ptr + offs_m[:, None] * N + offs_n[None, :], z)


@pytest.mark.parametrize("M,N,TM,TN", [(32, 32, 2, 32), (32, 32, 4, 32)])
def test_extract_tile_static_nonaligned(M, N, TM, TN):
    """Static index + thin tile -> SMEM relay path (non-aligned)."""
    torch.manual_seed(3)
    x = torch.arange(M * N, device=DEVICE, dtype=torch.float32).reshape(M, N)
    out = torch.zeros((TM, TN), device=DEVICE, dtype=torch.float32)

    extract_tile_thin_kernel[(1, )](x, out, M, N, TM, TN)

    # index=[1,0] -> rows [TM:2*TM], cols [0:TN]
    expected = x[TM:2 * TM, 0:TN]
    torch.testing.assert_close(out.cpu(), expected.cpu(), atol=0, rtol=0)


@pytest.mark.parametrize("M,N,TM,TN", [(32, 32, 2, 32), (32, 32, 4, 32)])
def test_insert_tile_static_nonaligned(M, N, TM, TN):
    """Static index + thin tile insert -> SMEM relay path (non-aligned)."""
    torch.manual_seed(4)
    x = torch.arange(M * N, device=DEVICE, dtype=torch.float32).reshape(M, N)
    y = (100000 + torch.arange(TM * TN, device=DEVICE, dtype=torch.float32)).reshape(TM, TN)
    out = torch.empty_like(x)

    insert_tile_thin_kernel[(1, )](x, y, out, M, N, TM, TN)

    expected = x.clone()
    expected[TM:2 * TM, 0:TN] = y
    torch.testing.assert_close(out.cpu(), expected.cpu(), atol=0, rtol=0)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
