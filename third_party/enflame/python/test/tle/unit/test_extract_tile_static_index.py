import torch
import triton
import triton.language as tl
import triton.experimental.tle.language as tle
import pytest
import os

_DEVICE = 'gcu' if os.getenv('TRITON_TEST_DEVICE', 'gcu') == 'gcu' else 'cuda'
if _DEVICE == 'gcu':
    import torch_gcu


@triton.jit
def extract_tile_kernel(x_ptr, out_ptr, M: tl.constexpr, N: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    TM: tl.constexpr = 128  # type: ignore[assignment]
    TN: tl.constexpr = 128  # type: ignore[assignment]

    gbl_tile_m = TM
    gbl_tile_n = TN

    blk_start_m = pid_m * BLOCK_SIZE
    blk_start_n = pid_n * BLOCK_SIZE

    loc_tile_m = gbl_tile_m - blk_start_m
    loc_tile_n = gbl_tile_n - blk_start_n

    if loc_tile_m >= 0 and loc_tile_m + TM <= BLOCK_SIZE and \
       loc_tile_n >= 0 and loc_tile_n + TN <= BLOCK_SIZE:
        offs_m = pid_m * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        offs_n = pid_n * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        x = tl.load(x_ptr + offs_m[:, None] * N + offs_n[None, :])

        tile_idx_m = loc_tile_m // TM
        tile_idx_n = loc_tile_n // TN
        tile = tle.extract_tile(x, index=[tile_idx_m, tile_idx_n], tile_shape=[TM, TN])  # type: ignore[arg-type]

        out_offs_m = tl.arange(0, TM)
        out_offs_n = tl.arange(0, TN)
        tl.store(out_ptr + out_offs_m[:, None] * TN + out_offs_n[None, :], tile)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required for this test")
def test_extract_tile_kernel():
    # Set matrix dimensions
    M, N = 512, 512
    BLOCK_SIZE = 128
    # Create input tensor with sequential values
    x = torch.arange(M * N, device='cuda', dtype=torch.float32).reshape(M, N)
    # Prepare output buffer for a 128x128 result
    out = torch.zeros((128, 128), device='cuda', dtype=torch.float32)

    # Launch kernel with grid to split work across blocks
    grid = (M // BLOCK_SIZE, N // BLOCK_SIZE)
    extract_tile_kernel[grid](x, out, M, N, BLOCK_SIZE)

    # Verification:
    # Since index=[1, 1] and tile_shape=[128, 128], the extraction starts at
    # row 1 * 128 and column 1 * 128.
    expected = x[128:256, 128:256]

    assert torch.allclose(out, expected), "The extracted tile does not match the expected slice!"
