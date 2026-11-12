# Copyright 2026 FlagOS Contributors
# Copyright 2026- XCoreSigma Technology Co., Ltd
"""Correctness and benchmark tests for the vectorized flip custom op."""
import torch
import torch_npu  # noqa: F401
import triton
import triton.language as tl
import triton.experimental.tle.language.raw as tle_raw
from triton.backends.ascend.testing import do_bench_npu

_TL_DTYPE = {
    torch.float16: tl.float16,
    torch.bfloat16: tl.bfloat16,
    torch.float32: tl.float32,
    torch.int16: tl.int16,
    torch.uint16: tl.uint16,
    torch.int32: tl.int32,
    torch.uint32: tl.uint32,
    torch.int8: tl.int8,
    torch.uint8: tl.uint8,
}


@triton.jit
def flip_kernel(X, Y, M: tl.constexpr, N: tl.constexpr, BLOCK_M: tl.constexpr, DTYPE: tl.constexpr,
                TMP_N: tl.constexpr):
    pid = tl.program_id(0)
    rows = pid * BLOCK_M + tl.arange(0, BLOCK_M)
    cols = tl.arange(0, N)
    mask = rows[:, None] < M
    x = tl.load(X + rows[:, None] * N + cols[None, :], mask=mask, other=0.0)
    t = tl.full((1, TMP_N), 0, DTYPE)
    y = tl.full((BLOCK_M, N), 0, DTYPE)
    y = tle_raw.call("ascend_flip", x, t, out=y)
    tl.store(Y + rows[:, None] * N + cols[None, :], y, mask=mask)


def flip_custom(x0):
    assert x0.dim() == 2
    out = torch.empty_like(x0)
    M, N = x0.shape
    BLOCK_N = triton.next_power_of_2(N)
    ub_size = 125 * 1024 // 2
    BLOCK_M = max(1, min(M, ub_size // (2 * x0.element_size() * BLOCK_N)))
    grid = ((M - 1) // BLOCK_M + 1, )
    flip_kernel[grid](x0, out, M, BLOCK_N, BLOCK_M, _TL_DTYPE[x0.dtype], max(256, 512 // x0.element_size()))
    return out


def test_op(shape, dtype):
    if dtype == torch.int8:
        x0 = torch.randint(-128, 128, shape, dtype=dtype).npu()
    elif dtype in (torch.int16, torch.int32):
        x0 = torch.randint(-3000, 3000, shape, dtype=dtype).npu()
    else:
        x0 = torch.randn(shape, dtype=dtype).npu()
    ref = torch.flip(x0, dims=[-1])
    out = flip_custom(x0)
    torch.testing.assert_close(out, ref, rtol=0, atol=0, equal_nan=True)
    print("PASSED!")

    triton_time = do_bench_npu(lambda: flip_custom(x0), clear_l2_cache=True)
    print(f"Triton time: {triton_time:.3f} ms")
    torch_time = do_bench_npu(lambda: torch.flip(x0, dims=[-1]), clear_l2_cache=True)
    print(f"Torch time: {torch_time:.3f} ms, speedup: {torch_time / triton_time:.1f}x")


shapes = [
    (64, 512),
    (20669, 256),
    (2048, 8192),
]
dtypes = [
    torch.int8,
    torch.float16,
    torch.bfloat16,
    torch.float32,
    torch.int32,
]
test_cases = [(shape, dtype) for shape in shapes for dtype in dtypes]

if __name__ == "__main__":
    for i, (shape, dtype) in enumerate(test_cases, start=1):
        print(f"====================第{i}次测试=================, shape={shape}, dtype={dtype}")
        test_op(shape, dtype)
