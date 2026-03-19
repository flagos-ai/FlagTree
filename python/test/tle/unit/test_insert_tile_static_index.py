import torch
import triton
import triton.language as tl
import triton.experimental.tle as tle
import pytest


@triton.jit
def insert_tile_kernel(
    x_ptr,
    y_ptr,
    out_ptr,
    M: tl.constexpr,
    N: tl.constexpr,
    TM: tl.constexpr,
    TN: tl.constexpr,
):
    offs_m = tl.arange(0, M)
    offs_n = tl.arange(0, N)
    x = tl.load(x_ptr + offs_m[:, None] * N + offs_n[None, :])

    tile_m = tl.arange(0, TM)
    tile_n = tl.arange(0, TN)
    y = tl.load(y_ptr + tile_m[:, None] * TN + tile_n[None, :])

    z = tle.insert_tile(x, y, index=[1, 1])

    tl.store(out_ptr + offs_m[:, None] * N + offs_n[None, :], z)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required for this test")
def test_insert_tile_static_index():
    M, N = 512, 512
    TM, TN = 128, 128

    x = torch.arange(M * N, device="cuda", dtype=torch.float32).reshape(M, N)
    y = (100000 + torch.arange(TM * TN, device="cuda", dtype=torch.float32)).reshape(TM, TN)
    out = torch.empty_like(x)

    print(f"Running insert_tile kernel with x={M}x{N}, tile={TM}x{TN}, index=[1, 1]...")
    insert_tile_kernel[(1, )](x, y, out, M, N, TM, TN)
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
