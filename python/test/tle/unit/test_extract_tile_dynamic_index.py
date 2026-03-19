import torch
import triton
import triton.language as tl
import triton.experimental.tle as tle
import pytest


@triton.jit
def simple_extract_kernel(x_ptr, out_ptr, stride_xb, stride_xm, stride_xn, stride_ob, stride_om, stride_on,
                          BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, TILE_M: tl.constexpr, TILE_N: tl.constexpr):
    # 1. Get 3D coordinates: z (layer/batch)
    pid_z = tl.program_id(0)

    # 2. Read the full background slice of x for this layer (e.g., 32x32)
    offs_m = tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    x_ptrs = x_ptr + pid_z * stride_xb + offs_m[:, None] * stride_xm + offs_n[None, :] * stride_xn
    bg_tile = tl.load(x_ptrs)

    # 3. Perform extraction based on the layer index
    # Layer 0 extracts from top-left [0, 0], Layer 1 extracts from bottom-right [1, 1]
    # We use a conditional check because 'index' usually requires constant values for TLE hardware
    if pid_z % 2 == 0:
        extracted_tile = tle.extract_tile(bg_tile, index=[0, 0], tile_shape=[TILE_M, TILE_N])
    else:
        extracted_tile = tle.extract_tile(bg_tile, index=[1, 1], tile_shape=[TILE_M, TILE_N])

    # 4. Store the extracted small tile into the corresponding Z layer of the output tensor
    offs_tm = tl.arange(0, TILE_M)
    offs_tn = tl.arange(0, TILE_N)
    out_ptrs = out_ptr + pid_z * stride_ob + offs_tm[:, None] * stride_om + offs_tn[None, :] * stride_on
    tl.store(out_ptrs, extracted_tile)


# ------------------------------------------------------------
# Minimal Test
# ------------------------------------------------------------


def test_simple_extract_kernel_cuda():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available; skipping Triton CUDA test.")

    B = 2  # 2 layers (Z dimension)
    M, N = 32, 32  # 32x32 size per layer
    TM, TN = 16, 16  # Extracted tile size 16x16

    # Initialize x background tensor
    x = torch.zeros((B, M, N), device="cuda", dtype=torch.float32)

    # Preset values for verification:
    # Layer 0: Set 88.0 in the top-left quadrant
    x[0, 0:16, 0:16] = 88.0
    # Layer 1: Set 99.0 in the bottom-right quadrant
    x[1, 16:32, 16:32] = 99.0

    # Output buffer for extracted 16x16 tiles
    out = torch.zeros((B, TM, TN), device="cuda", dtype=torch.float32)

    # Launch Kernel: B layers, 1x1 grid per layer
    grid = (B, 1, 1)

    simple_extract_kernel[grid](
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
    )

    # Verification: Means should match the preset values
    assert torch.allclose(out[0].mean(), torch.tensor(88.0, device="cuda"), atol=1e-5)
    assert torch.allclose(out[1].mean(), torch.tensor(99.0, device="cuda"), atol=1e-5)
