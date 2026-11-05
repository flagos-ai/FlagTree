# Copyright (c) Huawei Technologies Co., Ltd. 2025.
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
def index_select_baseline_kernel(in_ptr, indices_ptr, out_ptr, dim: tl.constexpr, other_numel: tl.constexpr,
                                 g_stride: tl.constexpr, indice_length: tl.constexpr, g_block: tl.constexpr,
                                 g_block_sub: tl.constexpr, other_block: tl.constexpr):
    """Baseline: plain tl.load gather with computed offsets."""
    g_begin = tl.program_id(0) * g_block
    for goffs in range(0, g_block, g_block_sub):
        g_idx = tl.arange(0, g_block_sub) + g_begin + goffs
        g_mask = g_idx < indice_length
        indices = tl.load(indices_ptr + g_idx, g_mask, other=0)

        for other_offset in range(0, g_stride, other_block):
            other_idx = tl.arange(0, other_block) + other_offset
            other_mask = other_idx < g_stride

            src_offsets = indices[:, None] * g_stride + other_idx[None, :]
            tmp_buf = tl.load(in_ptr + src_offsets, g_mask[:, None] & other_mask[None, :])

            tl.store(out_ptr + g_idx[:, None] * g_stride + other_idx[None, :], tmp_buf,
                     g_mask[:, None] & other_mask[None, :])


@triton.jit
def index_select_tle_kernel(in_ptr, indices_ptr, out_ptr, dim: tl.constexpr, other_numel: tl.constexpr,
                            g_stride: tl.constexpr, indice_length: tl.constexpr, g_block: tl.constexpr,
                            g_block_sub: tl.constexpr, other_block: tl.constexpr):
    """tle.dsa.ascend.index_select_simd: GM -> UB SIMD gather."""
    g_begin = tl.program_id(0) * g_block
    for goffs in range(0, g_block, g_block_sub):
        g_idx = tl.arange(0, g_block_sub) + g_begin + goffs
        g_mask = g_idx < indice_length
        indices = tl.load(indices_ptr + g_idx, g_mask, other=0)

        for other_offset in range(0, g_stride, other_block):
            other_idx = tl.arange(0, other_block) + other_offset
            other_mask = other_idx < g_stride

            tmp_buf = tle.dsa.ascend.index_select_simd(src=in_ptr, dim=dim, index=indices,
                                                       src_shape=(other_numel, g_stride), src_offset=(-1, 0),
                                                       read_shape=(-1, other_block))

            tl.store(out_ptr + g_idx[:, None] * g_stride + other_idx[None, :], tmp_buf,
                     g_mask[:, None] & other_mask[None, :])


def index_select(x0, dim, indices, impl="tle"):
    out_shape = list(x0.shape)
    out_shape[dim] = indices.numel()
    out = torch.empty(tuple(out_shape), dtype=x0.dtype).npu()

    g_stride = x0.stride(dim)
    indice_length = indices.numel()
    g_block = (indice_length - 1) // NUM_VEC_CORES + 1

    # UB budget: double-buffered, index row + gathered tile (+ mask overhead for baseline)
    ub_size = 125 * 1024 // 2
    other_block = g_stride
    g_block_sub = ub_size // (x0.element_size() * g_stride * 3 + indices.element_size())
    if g_block_sub < 1:
        other_block = (ub_size - indices.element_size()) // x0.element_size()
        g_block_sub = 1

    kernel = index_select_tle_kernel if impl == "tle" else index_select_baseline_kernel
    kernel[NUM_VEC_CORES, 1,
           1](x0, indices, out, dim, other_numel=x0.shape[0], g_stride=g_stride, indice_length=indice_length,
              g_block=g_block, g_block_sub=g_block_sub, other_block=other_block)
    return out


def test_op(src_shape, dim, indice_shape, dtype):
    x0 = torch.randn(src_shape, dtype=dtype).npu()
    indices = torch.randint(0, src_shape[dim], size=indice_shape, dtype=torch.int32).npu()

    torch_ref = torch.index_select(x0, dim, indices)
    baseline_out = index_select(x0, dim, indices, impl="baseline")
    tle_out = index_select(x0, dim, indices, impl="tle")

    torch.testing.assert_close(baseline_out, torch_ref, rtol=0, atol=0, equal_nan=True)
    torch.testing.assert_close(tle_out, torch_ref, rtol=0, atol=0, equal_nan=True)
    print(f"PASSED! src={src_shape}, dim={dim}, indices={indice_shape}, dtype={dtype}")

    baseline_time = do_bench_npu(lambda: index_select(x0, dim, indices, impl="baseline"), clear_l2_cache=True)
    tle_time = do_bench_npu(lambda: index_select(x0, dim, indices, impl="tle"), clear_l2_cache=True)
    torch_time = do_bench_npu(lambda: torch.index_select(x0, dim, indices), clear_l2_cache=True)
    print(f"Baseline tl.load time: {baseline_time:.2f} us")
    print(f"tle.dsa.ascend.index_select_simd time: {tle_time:.2f} us")
    print(f"torch.index_select time: {torch_time:.2f} us")


# (src_shape, dim, indice_shape, dtype)
test_cases = [
    ((10, 16), 0, (1024, ), torch.float32),
    ((20669, 8), 0, (2048, ), torch.float32),
    ((1034, 16), 0, (48000, ), torch.float32),
    ((270610, 32), 0, (2928000, ), torch.float32),
    ((3200, 16), 0, (1971940, ), torch.float16),
]

if __name__ == "__main__":
    for i, (src_shape, dim, indice_shape, dtype) in enumerate(test_cases, start=1):
        print(f"====================第{i}次测试=================, src={src_shape}, indices={indice_shape}, "
              f"dtype={dtype}")
        test_op(src_shape, dim, indice_shape, dtype)
