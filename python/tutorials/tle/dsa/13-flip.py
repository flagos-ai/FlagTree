# Copyright 2026- FlagOS Contributors
# Copyright 2026- XCoreSigma Technology Co.

import torch
import torch_npu
import triton
import triton.language as tl
import triton.experimental.tle as tle
from triton.backends.ascend.testing import do_bench_npu

torch.manual_seed(20)
DEVICE = "npu"
DEVICE_ID = 0
torch_npu.npu.set_device(int(DEVICE_ID))
torch.set_printoptions(sci_mode=False, precision=4, linewidth=300)
NUM_VEC_CORES = 48


@triton.jit
def flip_tle_kernel(in_ptr, out_ptr, N: tl.constexpr, M: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):
    """tle.dsa.ascend.flip: plain load, in-UB flip, plain store."""
    pid = tl.program_id(0)
    rows = pid * BLOCK_M + tl.arange(0, BLOCK_M)
    cols = tl.arange(0, BLOCK_N)
    row_mask = rows < M

    x = tl.load(in_ptr + rows[:, None] * N + cols[None, :], row_mask[:, None])
    y = tle.dsa.ascend.flip(x, dim=-1)
    tl.store(out_ptr + rows[:, None] * N + cols[None, :], y, row_mask[:, None])


def flip(x0, dim):
    assert dim in (1, -1) and x0.dim() == 2
    out = torch.empty_like(x0)

    M, N = x0.shape[-2], x0.shape[-1]
    # flip dim must be fully covered by one tile
    BLOCK_N = triton.next_power_of_2(N)

    # UB budget: double-buffered, in + out tiles
    ub_size = 125 * 1024 // 2
    BLOCK_M = ub_size // (2 * x0.element_size() * BLOCK_N)

    grid = ((M - 1) // BLOCK_M + 1, )
    flip_tle_kernel[grid](x0, out, N, M, BLOCK_M, BLOCK_N)
    return out


def test_op(src_shape, dim, dtype):
    x0 = torch.randn(src_shape, dtype=dtype).npu()

    torch_ref = torch.flip(x0, dims=[dim])
    tle_out = flip(x0, dim)

    torch.testing.assert_close(tle_out, torch_ref, rtol=0, atol=0, equal_nan=True)
    print(f"PASSED! src={src_shape}, dim={dim}, dtype={dtype}")

    tle_time = do_bench_npu(lambda: flip(x0, dim), clear_l2_cache=True)
    torch_time = do_bench_npu(lambda: torch.flip(x0, dims=[dim]), clear_l2_cache=True)
    print(f"tle.dsa.ascend.flip time: {tle_time:.2f} us")
    print(f"torch.flip time: {torch_time:.2f} us")


# (src_shape, dim, dtype)
test_cases = [
    ((1024, 64), -1, torch.float32),
    ((4096, 128), -1, torch.float32),
    ((20669, 256), -1, torch.float32),
    ((1024, 4096), -1, torch.float16),
    ((2048, 8192), -1, torch.float16),
]

if __name__ == "__main__":
    for i, (src_shape, dim, dtype) in enumerate(test_cases, start=1):
        print(f"====================第{i}次测试=================, src={src_shape}, dim={dim}, dtype={dtype}")
        test_op(src_shape, dim, dtype)
