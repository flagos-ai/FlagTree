import torch
import triton
import triton.language as tl
import triton.experimental.tle.language as tle
import pytest


@triton.jit
def extract_tile_kernel(x_ptr, out_ptr, M: tl.constexpr, N: tl.constexpr):
    # Set M, N as input matrix dimensions
    offs_m = tl.arange(0, M)
    offs_n = tl.arange(0, N)
    x = tl.load(x_ptr + offs_m[:, None] * N + offs_n[None, :])

    # Extract a 128x128 tile starting from index [1, 1]
    # Note: index refers to the tile position (e.g., index [1, 1] for 128x128 tiles
    # starts at row 128 and column 128)
    tile = tle.extract_tile(x, index=[1, 1], tile_shape=[128, 128])

    # Store the 128x128 extracted tile into the output pointer
    out_offs_m = tl.arange(0, 128)
    out_offs_n = tl.arange(0, 128)
    tl.store(out_ptr + out_offs_m[:, None] * 128 + out_offs_n[None, :], tile)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required for this test")
@pytest.mark.require_tle("extract_tile")
def test_extract_tile_kernel():
    # Set matrix dimensions
    M, N = 512, 512
    # Create input tensor with sequential values
    x = torch.arange(M * N, device='cuda', dtype=torch.float32).reshape(M, N)
    # Prepare output buffer for a 128x128 result
    out = torch.zeros((128, 128), device='cuda', dtype=torch.float32)

    # Launch kernel with a single program (grid size 1)
    extract_tile_kernel[(1, )](x, out, M, N)

    # Verification:
    # Since index=[1, 1] and tile_shape=[128, 128], the extraction starts at
    # row 1 * 128 and column 1 * 128.
    expected = x[128:256, 128:256]

    assert torch.allclose(out, expected), "The extracted tile does not match the expected slice!"


@triton.jit
def _extract_then_change_layout(x_ptr, out_ptr, SRC: tl.constexpr, DST: tl.constexpr, TILE: tl.constexpr,
                                INDEX: tl.constexpr):
    offsets = tle.gpu.set_layout(tl.reshape(tl.arange(0, 1024), (32, 32)), SRC)
    values = tle.gpu.set_layout(tl.load(x_ptr + offsets), SRC)
    tile = tle.extract_tile(values, index=[INDEX, INDEX], tile_shape=[TILE, TILE])
    tile = tle.gpu.set_layout(tile, DST)
    out_offsets = tle.gpu.set_layout(tl.reshape(tl.arange(0, TILE * TILE), (TILE, TILE)), DST)
    tl.store(out_ptr + out_offsets, tile)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required for this test")
@pytest.mark.parametrize("tile_size,tile_index", [(32, 0), (16, 1)])
@pytest.mark.parametrize("rlc_enhance", [False, True])
def test_extract_tile_then_change_lane_layout(tile_size, tile_index, rlc_enhance, monkeypatch):
    if triton.runtime.driver.active.get_current_target().backend != "cuda":
        pytest.skip("requires NVIDIA warp layouts")
    monkeypatch.setenv("FLAGTREE_RLC_ENHANCE", "1" if rlc_enhance else "0")
    monkeypatch.setenv("TRITON_ALWAYS_COMPILE", "1")
    # Both layouts use four warps, but elements move between lanes. Extraction
    # must not replace the set_layout conversion with a register reinterpretation.
    src = tle.gpu.BlockEncoding([1, 1], [4, 8], [4, 1], [1, 0])
    dst = tle.gpu.BlockEncoding([1, 1], [8, 4], [4, 1], [1, 0])
    x = torch.arange(1024, device="cuda", dtype=torch.float32).reshape(32, 32)
    out = torch.empty((tile_size, tile_size), device="cuda", dtype=torch.float32)
    _extract_then_change_layout[(1, )](x, out, src, dst, tile_size, tile_index, num_warps=4)
    start = tile_index * tile_size
    torch.testing.assert_close(out, x[start:start + tile_size, start:start + tile_size], rtol=0, atol=0)
