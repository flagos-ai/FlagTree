import torch
import triton
import triton.language as tl
import triton.experimental.tle.language as tle
import pytest


@triton.jit
def simple_insert_dynamic_index_kernel(
    x_ptr,
    y_ptr,
    index_ptr,
    stride_xb,
    stride_xm,
    stride_xn,
    stride_ym,
    stride_yn,
    stride_ib,
    stride_ic,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    TILE_M: tl.constexpr,
    TILE_N: tl.constexpr,
):
    # 1. Get 3D coordinates: z (layer/batch), m (row block), n (col block)
    pid_z = tl.program_id(0)
    pid_m = tl.program_id(1)
    pid_n = tl.program_id(2)

    # 2. Load the background slice of x for the current z layer
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    x_ptrs = x_ptr + pid_z * stride_xb + offs_m[:, None] * stride_xm + offs_n[None, :] * stride_xn
    bg_tile = tl.load(x_ptrs)

    # 3. Load the small y tile (2D, shared across all z layers)
    offs_tm = tl.arange(0, TILE_M)
    offs_tn = tl.arange(0, TILE_N)
    y_ptrs = y_ptr + offs_tm[:, None] * stride_ym + offs_tn[None, :] * stride_yn
    small_tile = tl.load(y_ptrs)

    # 4. Runtime index per layer: index[pid_z] = [idx_m, idx_n]
    idx_m = tl.load(index_ptr + pid_z * stride_ib + 0 * stride_ic)
    idx_n = tl.load(index_ptr + pid_z * stride_ib + 1 * stride_ic)
    res_tile = tle.insert_tile(bg_tile, small_tile, index=[idx_m, idx_n])

    # 5. Store the resulting tile back to memory
    tl.store(x_ptrs, res_tile)


@triton.jit
def simple_insert_dynamic_index_stride_kernel(
    x_ptr,
    y_ptr,
    index_ptr,
    stride_xb,
    stride_xm,
    stride_xn,
    stride_ym,
    stride_yn,
    stride_ib,
    stride_ic,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    TILE_M: tl.constexpr,
    TILE_N: tl.constexpr,
    STRIDE_M: tl.constexpr,
    STRIDE_N: tl.constexpr,
):
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

    idx_m = tl.load(index_ptr + pid_z * stride_ib + 0 * stride_ic)
    idx_n = tl.load(index_ptr + pid_z * stride_ib + 1 * stride_ic)
    res_tile = tle.insert_tile(bg_tile, small_tile, index=[idx_m, idx_n], strides=(STRIDE_M, STRIDE_N))

    # 5. Store the resulting tile back to memory
    tl.store(x_ptrs, res_tile)


# ------------------------------------------------------------
# Minimal Test
# ------------------------------------------------------------


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_simple_insert_kernel_with_dynamic_index():
    B = 2  # 2 layers (Z dimension)
    M, N = 32, 32  # 32x32 size per layer
    TM, TN = 16, 16  # The inserted small tile is 16x16

    # x is an all-zero 3D background tensor
    x = torch.zeros((B, M, N), device="cuda", dtype=torch.float32)
    # y is an all-99.0 2D small tile
    y = torch.ones((TM, TN), device="cuda", dtype=torch.float32) * 99.0

    # Dynamic insertion indices per layer (no stride argument):
    # Layer 0 -> [0, 0] => start at [0, 0]
    # Layer 1 -> [1, 1] => start at [16, 16] (tile size is 16x16)
    index = torch.tensor([[0, 0], [1, 1]], device="cuda", dtype=torch.int32)

    # Launch Kernel: B layers, each needs exactly 1x1 block (since M=32 and BLOCK_M=32)
    grid = (B, 1, 1)

    simple_insert_dynamic_index_kernel[grid](
        x,
        y,
        index,
        x.stride(0),
        x.stride(1),
        x.stride(2),
        y.stride(0),
        y.stride(1),
        index.stride(0),
        index.stride(1),
        BLOCK_M=32,
        BLOCK_N=32,
        TILE_M=16,
        TILE_N=16,
    )

    # --- Verification ---
    expected = torch.zeros_like(x)
    expected[0, 0:16, 0:16] = 99.0  # [idx_m, idx_n] = [0, 0]
    expected[1, 16:32, 16:32] = 99.0  # [idx_m, idx_n] = [1, 1]

    assert torch.allclose(x, expected)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
@pytest.mark.parametrize(
    "SM,SN,idx_m1,idx_n1",
    [
        (4, 4, 3, 2),
        (8, 8, 2, 1),
        (16, 8, 1, 1),
    ],
)
def test_simple_insert_kernel_with_dynamic_index_and_stride(SM, SN, idx_m1, idx_n1):
    B = 2  # 2 layers (Z dimension)
    M, N = 32, 32  # 32x32 size per layer
    TM, TN = 16, 16  # The inserted small tile is 16x16

    x = torch.zeros((B, M, N), device="cuda", dtype=torch.float32)
    y = torch.ones((TM, TN), device="cuda", dtype=torch.float32) * 99.0

    # Layer 0 is fixed at top-left; Layer 1 uses parameterized dynamic index.
    index = torch.tensor([[0, 0], [idx_m1, idx_n1]], device="cuda", dtype=torch.int32)

    grid = (B, 1, 1)

    simple_insert_dynamic_index_stride_kernel[grid](
        x,
        y,
        index,
        x.stride(0),
        x.stride(1),
        x.stride(2),
        y.stride(0),
        y.stride(1),
        index.stride(0),
        index.stride(1),
        BLOCK_M=32,
        BLOCK_N=32,
        TILE_M=16,
        TILE_N=16,
        STRIDE_M=SM,
        STRIDE_N=SN,
    )

    expected = torch.zeros_like(x)
    expected[0, 0:16, 0:16] = 99.0
    start_m = idx_m1 * SM
    start_n = idx_n1 * SN
    expected[1, start_m:start_m + TM, start_n:start_n + TN] = 99.0

    assert torch.allclose(x, expected)
