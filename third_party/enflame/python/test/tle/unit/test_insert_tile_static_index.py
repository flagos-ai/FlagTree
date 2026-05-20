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
def insert_tile_kernel(
    x_ptr,
    y_ptr,
    out_ptr,
    M: tl.constexpr,
    N: tl.constexpr,
    TM: tl.constexpr,
    TN: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    offs_n = pid_n * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    x = tl.load(x_ptr + offs_m[:, None] * N + offs_n[None, :])

    gbl_tile_m = TM
    gbl_tile_n = TN

    blk_start_m = pid_m * BLOCK_SIZE
    blk_start_n = pid_n * BLOCK_SIZE

    loc_tile_m = gbl_tile_m - blk_start_m
    loc_tile_n = gbl_tile_n - blk_start_n

    z = x
    if loc_tile_m >= 0 and loc_tile_m + TM <= BLOCK_SIZE and \
       loc_tile_n >= 0 and loc_tile_n + TN <= BLOCK_SIZE:
        tile_m = tl.arange(0, TM)
        tile_n = tl.arange(0, TN)
        y = tl.load(y_ptr + tile_m[:, None] * TN + tile_n[None, :])
        tile_idx_m = loc_tile_m // TM
        tile_idx_n = loc_tile_n // TN
        z = tle.insert_tile(x, y, index=[tile_idx_m, tile_idx_n])

    tl.store(out_ptr + offs_m[:, None] * N + offs_n[None, :], z)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required for this test")
def test_insert_tile_static_index():
    M, N = 512, 512
    TM, TN = 128, 128
    BLOCK_SIZE = TM

    x = torch.arange(M * N, device="cuda", dtype=torch.float32).reshape(M, N)
    y = (100000 + torch.arange(TM * TN, device="cuda", dtype=torch.float32)).reshape(TM, TN)
    out = torch.empty_like(x)

    grid = (M // BLOCK_SIZE, N // BLOCK_SIZE)
    print(
        f"Running insert_tile kernel with x={M}x{N}, tile={TM}x{TN}, BLOCK_SIZE={BLOCK_SIZE}, grid={grid}, index=[1, 1]..."
    )
    insert_tile_kernel[grid](x, y, out, M, N, TM, TN, BLOCK_SIZE)
    print("Kernel executed.\n")

    expected = x.clone()
    expected[TM:2 * TM, TN:2 * TN] = y

    max_abs_diff = (out - expected).abs().max().item()
    print(f"max_abs_diff = {max_abs_diff}")

    if torch.allclose(out, expected):
        print("Test passed: insert_tile updated the target tile correctly.")
    else:
        print("Test failed: output does not match expected result.")

    print("\nSample check:")
    print("original x[128:132, 128:132]:")
    print(x[128:132, 128:132].cpu().int())
    print("tile y[0:4, 0:4]:")
    print(y[0:4, 0:4].cpu().int())
    print("output out[128:132, 128:132]:")
    print(out[128:132, 128:132].cpu().int())

    assert torch.allclose(out, expected)
