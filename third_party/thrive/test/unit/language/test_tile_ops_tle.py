"""
TLE tile ops (thrive): tle.extract_tile / tle.insert_tile
===================================================================================================

Adapted from the upstream TLE unit tests:
  - python/test/tle/unit/test_extract_tile_static_index.py
  - python/test/tle/unit/test_insert_tile_static_index.py
  - python/test/tle/unit/test_extract_tile_dynamic_index.py
  - python/test/tle/unit/test_insert_tile_dynamic_index.py

Both ops lower through thrive's TleToThvTile conversion:
  tle.extract_tile  -> thvtile.extract_slice
  tle.insert_tile   -> thvtile.insert_slice

The static cases pin down the tile-index arithmetic: with a 256x256 source and
128x128 tiles, index [1, 1] must land at offsets [128, 128]. The TLE frontend
linearizes a multi-dim index row-major into one tile index (1*2 + 1 = 3) and
thrive's SliceOpsToThvTile de-linearizes it back, so both sides have to agree on
the same convention -- that is what these tests verify end to end.

The source side is kept at 256x256 rather than 512x512: the whole source is
staged in SRAM, and 512x512 blows past the SRAM budget.

The dynamic cases drive the same ops behind a runtime branch on tl.program_id,
which is how the upstream tests exercise the per-program tile position.

Differences from the upstream TLE tests:
  - ``device`` comes from the thrive test-suite fixture instead of ``cuda``
  - ``num_cores`` is an extra parameterization axis (1 and 2)
  - assertions use ``torch.testing.assert_close``
  - no IR/ISA assertions
  - the dynamic cases compare every element of the affected region instead of
    its mean: same condition, stated more precisely
"""

import pytest
import torch
import triton
import triton.language as tl
import triton.experimental.tle.language as tle

# ---------------------------------------------------------------------------------------------------
# static index
# ---------------------------------------------------------------------------------------------------


@triton.jit
def extract_tile_kernel(x_ptr, out_ptr, M: tl.constexpr, N: tl.constexpr):
    offs_m = tl.arange(0, M)
    offs_n = tl.arange(0, N)
    x = tl.load(x_ptr + offs_m[:, None] * N + offs_n[None, :])

    # index [1, 1] over 128x128 tiles starts at row 128, column 128.
    tile = tle.extract_tile(x, index=[1, 1], tile_shape=[128, 128])

    out_offs_m = tl.arange(0, 128)
    out_offs_n = tl.arange(0, 128)
    tl.store(out_ptr + out_offs_m[:, None] * 128 + out_offs_n[None, :], tile)


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


@pytest.mark.parametrize("num_cores", [1, 2])
def test_extract_tile_static_index(device, num_cores):
    M, N = 256, 256

    x = torch.arange(M * N, dtype=torch.float32).reshape(M, N).to(device)
    out = torch.zeros((128, 128), dtype=torch.float32, device=device)

    extract_tile_kernel[(1, )](x, out, M, N, num_cores=num_cores)

    # index [1, 1] * tile_shape [128, 128] -> rows/cols 128:256
    expected = x.cpu()[128:256, 128:256]
    # Every value is an exactly representable integer, so require bit-exactness.
    torch.testing.assert_close(out.cpu(), expected, rtol=0, atol=0)


@pytest.mark.parametrize("num_cores", [1, 2])
def test_insert_tile_static_index(device, num_cores):
    M, N = 256, 256
    TM, TN = 128, 128

    x = torch.arange(M * N, dtype=torch.float32).reshape(M, N).to(device)
    y = (100000 + torch.arange(TM * TN, dtype=torch.float32)).reshape(TM, TN).to(device)
    out = torch.empty_like(x)

    insert_tile_kernel[(1, )](x, y, out, M, N, TM, TN, num_cores=num_cores)

    expected = x.cpu().clone()
    expected[TM:2 * TM, TN:2 * TN] = y.cpu()
    torch.testing.assert_close(out.cpu(), expected, rtol=0, atol=0)


# ----------------------------------------------------------------------------------------------
# dynamic (per-program) index
# ----------------------------------------------------------------------------------------------


@triton.jit
def simple_extract_kernel(x_ptr, out_ptr, stride_xb, stride_xm, stride_xn, stride_ob, stride_om, stride_on,
                          BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, TILE_M: tl.constexpr, TILE_N: tl.constexpr):
    pid_z = tl.program_id(0)

    offs_m = tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    x_ptrs = x_ptr + pid_z * stride_xb + offs_m[:, None] * stride_xm + offs_n[None, :] * stride_xn
    bg_tile = tl.load(x_ptrs)

    # Layer 0 extracts from top-left [0, 0], layer 1 from bottom-right [1, 1].
    # The branch is runtime; each arm still uses a compile-time tile index.
    if pid_z % 2 == 0:
        extracted_tile = tle.extract_tile(bg_tile, index=[0, 0], tile_shape=[TILE_M, TILE_N])
    else:
        extracted_tile = tle.extract_tile(bg_tile, index=[1, 1], tile_shape=[TILE_M, TILE_N])

    offs_tm = tl.arange(0, TILE_M)
    offs_tn = tl.arange(0, TILE_N)
    out_ptrs = out_ptr + pid_z * stride_ob + offs_tm[:, None] * stride_om + offs_tn[None, :] * stride_on
    tl.store(out_ptrs, extracted_tile)


@triton.jit
def simple_insert_kernel(x_ptr, y_ptr, stride_xb, stride_xm, stride_xn, stride_ym, stride_yn, BLOCK_M: tl.constexpr,
                         BLOCK_N: tl.constexpr, TILE_M: tl.constexpr, TILE_N: tl.constexpr):
    pid_z = tl.program_id(0)
    pid_m = tl.program_id(1)
    pid_n = tl.program_id(2)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    x_ptrs = x_ptr + pid_z * stride_xb + offs_m[:, None] * stride_xm + offs_n[None, :] * stride_xn
    bg_tile = tl.load(x_ptrs)

    offs_tm = tl.arange(0, TILE_M)
    offs_tn = tl.arange(0, TILE_N)
    y_ptrs = y_ptr + offs_tm[:, None] * stride_ym + offs_tn[None, :] * stride_yn
    small_tile = tl.load(y_ptrs)

    # Layer 0 inserts at top-left [0, 0], layer 1 at bottom-right [1, 1].
    if pid_z % 2 == 0:
        res_tile = tle.insert_tile(bg_tile, small_tile, index=[0, 0])
    else:
        res_tile = tle.insert_tile(bg_tile, small_tile, index=[1, 1])

    tl.store(x_ptrs, res_tile)


@pytest.mark.parametrize("num_cores", [1, 2])
def test_simple_extract_kernel_dynamic_index(device, num_cores):
    B = 2
    M, N = 32, 32
    TM, TN = 16, 16

    x = torch.zeros((B, M, N), dtype=torch.float32)
    x[0, 0:16, 0:16] = 88.0
    x[1, 16:32, 16:32] = 99.0
    x = x.to(device)
    out = torch.zeros((B, TM, TN), dtype=torch.float32, device=device)

    simple_extract_kernel[(B, 1, 1)](
        x,
        out,
        x.stride(0),
        x.stride(1),
        x.stride(2),
        out.stride(0),
        out.stride(1),
        out.stride(2),
        BLOCK_M=32,
        BLOCK_N=32,
        TILE_M=16,
        TILE_N=16,
        num_cores=num_cores,
    )

    out = out.cpu()
    ones = torch.ones((TM, TN), dtype=torch.float32)
    torch.testing.assert_close(out[0], 88.0 * ones, rtol=0, atol=1e-5)
    torch.testing.assert_close(out[1], 99.0 * ones, rtol=0, atol=1e-5)


@pytest.mark.parametrize("num_cores", [1, 2])
def test_simple_insert_kernel_dynamic_index(device, num_cores):
    B = 2
    M, N = 32, 32
    TM, TN = 16, 16

    x = torch.zeros((B, M, N), dtype=torch.float32)
    y = torch.ones((TM, TN), dtype=torch.float32) * 99.0
    x = x.to(device)
    y = y.to(device)

    simple_insert_kernel[(B, 1, 1)](
        x,
        y,
        x.stride(0),
        x.stride(1),
        x.stride(2),
        y.stride(0),
        y.stride(1),
        BLOCK_M=32,
        BLOCK_N=32,
        TILE_M=16,
        TILE_N=16,
        num_cores=num_cores,
    )

    x = x.cpu()
    # Layer 0 (z=0): tile written at top-left [0:16, 0:16].
    # Layer 1 (z=1): tile written at bottom-right [16:32, 16:32].
    torch.testing.assert_close(x[0, 0:16, 0:16], torch.full((TM, TN), 99.0), rtol=0, atol=1e-5)
    torch.testing.assert_close(x[0, 16:32, 16:32], torch.zeros((TM, TN)), rtol=0, atol=1e-5)
    torch.testing.assert_close(x[1, 0:16, 0:16], torch.zeros((TM, TN)), rtol=0, atol=1e-5)
    torch.testing.assert_close(x[1, 16:32, 16:32], torch.full((TM, TN), 99.0), rtol=0, atol=1e-5)
