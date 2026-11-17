# Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
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
def isfinite_tle_kernel(in_ptr, out_ptr, N: tl.constexpr, BLOCK: tl.constexpr):
    """tle.dsa.ascend.isfinite: NaN / Inf classification in UB."""
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N

    x = tl.load(in_ptr + offs, mask=mask)
    y = tle.dsa.ascend.isfinite(x)
    tl.store(out_ptr + offs, y, mask=mask)


def isfinite(x0):
    N = x0.numel()
    out = torch.empty(N, dtype=torch.bool).npu()

    # UB budget: double-buffered, in tile (x0) + out tile (bool). Round the
    # block DOWN to a power of two: upward rounding overshoots the budget and
    # triggers a bishengir "ub overflow", while too-small blocks leave per-
    # program overhead unamortized.
    ub_size = 125 * 1024 // 2
    BLOCK = min(triton.next_power_of_2(ub_size // (x0.element_size() + 1)) // 2, triton.next_power_of_2(N))

    grid = ((N - 1) // BLOCK + 1, )
    isfinite_tle_kernel[grid](x0, out, N, BLOCK)
    return out


def make_input(N, dtype):
    x0 = torch.randn(N, dtype=dtype)
    # randn never produces inf/nan: inject them at fixed slots
    x0[0] = float("inf")
    x0[1] = float("-inf")
    x0[2] = float("nan")
    x0[3] = 0.0
    return x0.npu()


def test_op(src_numel, dtype):
    x0 = make_input(src_numel, dtype)

    torch_ref = torch.isfinite(x0)
    tle_out = isfinite(x0)

    torch.testing.assert_close(tle_out, torch_ref, rtol=0, atol=0)
    print(f"PASSED! numel={src_numel}, dtype={dtype}")

    tle_time = do_bench_npu(lambda: isfinite(x0), clear_l2_cache=True)
    torch_time = do_bench_npu(lambda: torch.isfinite(x0), clear_l2_cache=True)
    print(f"tle.dsa.ascend.isfinite time: {tle_time:.4f} ms")
    print(f"torch.isfinite time: {torch_time:.4f} ms")


# (src_numel, dtype)
test_cases = [
    (1024 * 1024, torch.float32),
    (20669 * 128, torch.float32),
    (4096 * 4096, torch.float16),
    (1024 * 1024, torch.bfloat16),
    (8192 * 4096, torch.float16),
]

if __name__ == "__main__":
    for i, (src_numel, dtype) in enumerate(test_cases, start=1):
        print(f"====================第{i}次测试=================, numel={src_numel}, dtype={dtype}")
        test_op(src_numel, dtype)
