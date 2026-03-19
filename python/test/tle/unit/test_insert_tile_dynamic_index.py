import torch
import triton
import triton.language as tl
import triton.experimental.tle as tle
import pytest


@triton.jit
def simple_insert_kernel(x_ptr, y_ptr, stride_xb, stride_xm, stride_xn, stride_ym, stride_yn, BLOCK_M: tl.constexpr,
                         BLOCK_N: tl.constexpr, TILE_M: tl.constexpr, TILE_N: tl.constexpr):
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

    # 4. Determine insertion position:
    # Layer 0 inserts at top-left [0, 0], Layer 1 inserts at bottom-right [1, 1]
    # Note: tle.insert_tile 'index' usually must be a constant or determined by static logic
    if pid_z % 2 == 0:
        res_tile = tle.insert_tile(bg_tile, small_tile, index=[0, 0])
    else:
        res_tile = tle.insert_tile(bg_tile, small_tile, index=[1, 1])

    # 5. Store the resulting tile back to memory
    tl.store(x_ptrs, res_tile)


# ------------------------------------------------------------
# Minimal Test
# ------------------------------------------------------------


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_simple_insert_kernel_inserts_tiles_correctly():
    B = 2  # 2 layers (Z dimension)
    M, N = 32, 32  # 32x32 size per layer
    TM, TN = 16, 16  # The inserted small tile is 16x16

    # x is an all-zero 3D background tensor
    x = torch.zeros((B, M, N), device="cuda", dtype=torch.float32)
    # y is an all-99.0 2D small tile
    y = torch.ones((TM, TN), device="cuda", dtype=torch.float32) * 99.0

    # Launch Kernel: B layers, each needs exactly 1x1 block (since M=32 and BLOCK_M=32)
    grid = (B, 1, 1)

    simple_insert_kernel[grid](
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
    )

    # --- Verification ---
    # Layer 0 (Z=0): tile inserted at top-left [0:16, 0:16]
    layer0_tl_mean = x[0, 0:16, 0:16].mean().item()
    layer0_br_mean = x[0, 16:32, 16:32].mean().item()

    # Layer 1 (Z=1): tile inserted at bottom-right [16:32, 16:32]
    layer1_tl_mean = x[1, 0:16, 0:16].mean().item()
    layer1_br_mean = x[1, 16:32, 16:32].mean().item()

    assert layer0_tl_mean == pytest.approx(99.0)
    assert layer0_br_mean == pytest.approx(0.0)
    assert layer1_tl_mean == pytest.approx(0.0)
    assert layer1_br_mean == pytest.approx(99.0)
