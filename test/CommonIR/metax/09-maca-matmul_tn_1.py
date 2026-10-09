"""TLE/CommonIR port of gluon/09-maca-matmul_tn_1.py.

Keeps the four shared slots, sixteen 32x32 dot accumulators and explicit
MetaX copy/wait/scheduling sequence. Inputs use complete 128x128x128 tiles;
copy masks are uniform over a whole tile, avoiding element-mask limitations.

    MACA_VISIBLE_DEVICES=6 python test/CommonIR/metax/09-maca-matmul_tn_1.py
"""

import argparse

import pytest
import torch
import triton

import triton.language as tl
import triton.experimental.tle.language as tle
from triton._common_ir import ENABLED

BENCHMARK_SIZES = [128 * i for i in range(2, 33)]
BENCHMARK_SHAPES = [(size, size, size) for size in BENCHMARK_SIZES]


def is_maca():
    try:
        target = triton.runtime.driver.active.get_current_target()
    except RuntimeError:
        return False
    return target.backend == "maca"


@triton.jit(autolayout=True)
def matmul_kernel(a_ptr, b_ptr, c_ptr, M, N, K, stride_am, stride_ak, stride_bk, stride_bn, stride_cm, stride_cn,
                  BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr, GROUP_M: tl.constexpr):
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    num_pid_in_group = GROUP_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    a_offsets = offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
    b_offsets = offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn
    a_ptrs = a_ptr + a_offsets
    b_ptrs = b_ptr + b_offsets

    tl.static_assert(BLOCK_M == 128 and BLOCK_N == 128 and BLOCK_K == 128)
    a_smem = tle.gpu.alloc((4, 32, BLOCK_K), a_ptr.dtype.element_ty)
    b_smem = tle.gpu.alloc((4, BLOCK_K, 32), b_ptr.dtype.element_ty)
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    num_k_tiles = tl.cdiv(K, BLOCK_K)
    has_first = num_k_tiles > 0
    a_load_mask = tl.full((BLOCK_M, BLOCK_K), has_first, tl.int1)
    b_load_mask = tl.full((BLOCK_K, BLOCK_N), has_first, tl.int1)

    a_ptrs0_ptrs = tle.extract_tile(a_ptrs, (0, 0), [32, BLOCK_K])
    a_ptrs0_mask = tle.extract_tile(a_load_mask, (0, 0), [32, BLOCK_K])
    tle.gpu.metax.async_copy_global_to_local(a_ptrs0_ptrs, a_smem.slot(0), mask=a_ptrs0_mask)
    b_ptrs0_ptrs = tle.extract_tile(b_ptrs, (0, 0), [BLOCK_K, 32])
    b_ptrs0_mask = tle.extract_tile(b_load_mask, (0, 0), [BLOCK_K, 32])
    tle.gpu.metax.async_copy_global_to_local(b_ptrs0_ptrs, b_smem.slot(0), mask=b_ptrs0_mask)
    a_ptrs1_ptrs = tle.extract_tile(a_ptrs, (1, 0), [32, BLOCK_K])
    a_ptrs1_mask = tle.extract_tile(a_load_mask, (1, 0), [32, BLOCK_K])
    tle.gpu.metax.async_copy_global_to_local(a_ptrs1_ptrs, a_smem.slot(1), mask=a_ptrs1_mask)
    b_ptrs1_ptrs = tle.extract_tile(b_ptrs, (0, 1), [BLOCK_K, 32])
    b_ptrs1_mask = tle.extract_tile(b_load_mask, (0, 1), [BLOCK_K, 32])
    tle.gpu.metax.async_copy_global_to_local(b_ptrs1_ptrs, b_smem.slot(1), mask=b_ptrs1_mask)
    a_ptrs2_ptrs = tle.extract_tile(a_ptrs, (2, 0), [32, BLOCK_K])
    a_ptrs2_mask = tle.extract_tile(a_load_mask, (2, 0), [32, BLOCK_K])
    tle.gpu.metax.async_copy_global_to_local(a_ptrs2_ptrs, a_smem.slot(2), mask=a_ptrs2_mask)
    b_ptrs2_ptrs = tle.extract_tile(b_ptrs, (0, 2), [BLOCK_K, 32])
    b_ptrs2_mask = tle.extract_tile(b_load_mask, (0, 2), [BLOCK_K, 32])
    tle.gpu.metax.async_copy_global_to_local(b_ptrs2_ptrs, b_smem.slot(2), mask=b_ptrs2_mask)
    a_ptrs3_ptrs = tle.extract_tile(a_ptrs, (3, 0), [32, BLOCK_K])
    a_ptrs3_mask = tle.extract_tile(a_load_mask, (3, 0), [32, BLOCK_K])
    tle.gpu.metax.async_copy_global_to_local(a_ptrs3_ptrs, a_smem.slot(3), mask=a_ptrs3_mask)
    b_ptrs3_ptrs = tle.extract_tile(b_ptrs, (0, 3), [BLOCK_K, 32])
    b_ptrs3_mask = tle.extract_tile(b_load_mask, (0, 3), [BLOCK_K, 32])
    tle.gpu.metax.async_copy_global_to_local(b_ptrs3_ptrs, b_smem.slot(3), mask=b_ptrs3_mask)
    tle.gpu.metax.gvm_arrive(12)
    tle.gpu.metax.barrier()

    a_step = tl.full((BLOCK_M, BLOCK_K), BLOCK_K * stride_ak, tl.int32)
    b_step = tl.full((BLOCK_K, BLOCK_N), BLOCK_K * stride_bk, tl.int32)
    a0 = tle.gpu.metax.local_load(a_smem.slot(0), intrinsic=True, is_constant_offs=True)
    b0 = tle.gpu.metax.local_load(b_smem.slot(0), intrinsic=True, is_constant_offs=True)
    tle.gpu.metax.gvm_arrive(8)
    tle.gpu.metax.barrier_shared()
    a1 = tle.gpu.metax.local_load(a_smem.slot(1), intrinsic=True, is_constant_offs=True)
    b1 = tle.gpu.metax.local_load(b_smem.slot(1), intrinsic=True, is_constant_offs=True)

    a_next_offsets = a_offsets + a_step
    b_next_offsets = b_offsets + b_step
    has_second = 1 < num_k_tiles
    first_next_a_mask = tl.full((BLOCK_M, BLOCK_K), has_second, tl.int1)
    a_next_ptrs0_ptrs = tle.extract_tile(a_ptr + a_next_offsets, (0, 0), [32, BLOCK_K])
    a_next_ptrs0_mask = tle.extract_tile(first_next_a_mask, (0, 0), [32, BLOCK_K])
    tle.gpu.metax.async_copy_global_to_local(a_next_ptrs0_ptrs, a_smem.slot(0), mask=a_next_ptrs0_mask)

    loop_acc = acc
    loop_a0 = a0
    loop_b0 = b0
    loop_a1 = a1
    loop_b1 = b1
    loop_a_next_offsets = a_next_offsets
    loop_b_next_offsets = b_next_offsets
    loop_counter = 0

    for _ in range(0, num_k_tiles):
        next_k = loop_counter + 1
        has_next = next_k < num_k_tiles

        c00 = tle.extract_tile(loop_acc, (0, 0), [32, 32])
        c01 = tle.extract_tile(loop_acc, (0, 1), [32, 32])
        c02 = tle.extract_tile(loop_acc, (0, 2), [32, 32])
        c03 = tle.extract_tile(loop_acc, (0, 3), [32, 32])
        c10 = tle.extract_tile(loop_acc, (1, 0), [32, 32])
        c11 = tle.extract_tile(loop_acc, (1, 1), [32, 32])
        c12 = tle.extract_tile(loop_acc, (1, 2), [32, 32])
        c13 = tle.extract_tile(loop_acc, (1, 3), [32, 32])
        c20 = tle.extract_tile(loop_acc, (2, 0), [32, 32])
        c21 = tle.extract_tile(loop_acc, (2, 1), [32, 32])
        c22 = tle.extract_tile(loop_acc, (2, 2), [32, 32])
        c23 = tle.extract_tile(loop_acc, (2, 3), [32, 32])
        c30 = tle.extract_tile(loop_acc, (3, 0), [32, 32])
        c31 = tle.extract_tile(loop_acc, (3, 1), [32, 32])
        c32 = tle.extract_tile(loop_acc, (3, 2), [32, 32])
        c33 = tle.extract_tile(loop_acc, (3, 3), [32, 32])

        a_next_ptrs = a_ptr + loop_a_next_offsets
        b_next_ptrs = b_ptr + loop_b_next_offsets
        a_next_next_offsets = loop_a_next_offsets + a_step
        b_next_next_offsets = loop_b_next_offsets + b_step
        next_a_mask = tl.full((BLOCK_M, BLOCK_K), has_next, tl.int1)
        next_b_mask = tl.full((BLOCK_K, BLOCK_N), has_next, tl.int1)

        b_next_ptrs0_ptrs = tle.extract_tile(b_next_ptrs, (0, 0), [BLOCK_K, 32])
        b_next_ptrs0_mask = tle.extract_tile(next_b_mask, (0, 0), [BLOCK_K, 32])
        tle.gpu.metax.async_copy_global_to_local(b_next_ptrs0_ptrs, b_smem.slot(0), mask=b_next_ptrs0_mask)

        c00 = tl.dot(loop_a0, loop_b0, c00, input_precision="tf32")
        tle.gpu.metax.iglp(config_2=2, config_5=1, config_6=2)
        tle.gpu.metax.gvm_arrive(8)
        tle.gpu.metax.barrier_shared()

        b2 = tle.gpu.metax.local_load(b_smem.slot(2), intrinsic=True, is_constant_offs=True)
        a2 = tle.gpu.metax.local_load(a_smem.slot(2), intrinsic=True, is_constant_offs=True)

        a_next_ptrs1_ptrs = tle.extract_tile(a_next_ptrs, (1, 0), [32, BLOCK_K])
        a_next_ptrs1_mask = tle.extract_tile(next_a_mask, (1, 0), [32, BLOCK_K])
        tle.gpu.metax.async_copy_global_to_local(a_next_ptrs1_ptrs, a_smem.slot(1), mask=a_next_ptrs1_mask)
        b_next_ptrs1_ptrs = tle.extract_tile(b_next_ptrs, (0, 1), [BLOCK_K, 32])
        b_next_ptrs1_mask = tle.extract_tile(next_b_mask, (0, 1), [BLOCK_K, 32])
        tle.gpu.metax.async_copy_global_to_local(b_next_ptrs1_ptrs, b_smem.slot(1), mask=b_next_ptrs1_mask)

        c10 = tl.dot(loop_a1, loop_b0, c10, input_precision="tf32")
        c01 = tl.dot(loop_a0, loop_b1, c01, input_precision="tf32")
        c11 = tl.dot(loop_a1, loop_b1, c11, input_precision="tf32")
        tle.gpu.metax.iglp(config_0=2, config_2=2, config_5=1, config_6=7, config_7=8)
        tle.gpu.metax.gvm_arrive(8)
        tle.gpu.metax.barrier_shared()

        a3 = tle.gpu.metax.local_load(a_smem.slot(3), intrinsic=True, is_constant_offs=True)
        b3 = tle.gpu.metax.local_load(b_smem.slot(3), intrinsic=True, is_constant_offs=True)

        a_next_ptrs2_ptrs = tle.extract_tile(a_next_ptrs, (2, 0), [32, BLOCK_K])
        a_next_ptrs2_mask = tle.extract_tile(next_a_mask, (2, 0), [32, BLOCK_K])
        tle.gpu.metax.async_copy_global_to_local(a_next_ptrs2_ptrs, a_smem.slot(2), mask=a_next_ptrs2_mask)
        b_next_ptrs2_ptrs = tle.extract_tile(b_next_ptrs, (0, 2), [BLOCK_K, 32])
        b_next_ptrs2_mask = tle.extract_tile(next_b_mask, (0, 2), [BLOCK_K, 32])
        tle.gpu.metax.async_copy_global_to_local(b_next_ptrs2_ptrs, b_smem.slot(2), mask=b_next_ptrs2_mask)

        c20 = tl.dot(a2, loop_b0, c20, input_precision="tf32")
        c21 = tl.dot(a2, loop_b1, c21, input_precision="tf32")
        c02 = tl.dot(loop_a0, b2, c02, input_precision="tf32")
        c12 = tl.dot(loop_a1, b2, c12, input_precision="tf32")
        c22 = tl.dot(a2, b2, c22, input_precision="tf32")
        tle.gpu.metax.iglp(config_2=2, config_5=1, config_6=5)
        tle.gpu.metax.gvm_arrive(8)
        tle.gpu.metax.barrier_shared()
        tle.gpu.metax.iglp(config_2=2, config_5=1, config_6=5)

        a_next_ptrs3_ptrs = tle.extract_tile(a_next_ptrs, (3, 0), [32, BLOCK_K])
        a_next_ptrs3_mask = tle.extract_tile(next_a_mask, (3, 0), [32, BLOCK_K])
        tle.gpu.metax.async_copy_global_to_local(a_next_ptrs3_ptrs, a_smem.slot(3), mask=a_next_ptrs3_mask)
        b_next_ptrs3_ptrs = tle.extract_tile(b_next_ptrs, (0, 3), [BLOCK_K, 32])

        c30 = tl.dot(a3, loop_b0, c30, input_precision="tf32")
        c03 = tl.dot(loop_a0, b3, c03, input_precision="tf32")
        tle.gpu.metax.sched_bound()
        a0_next = tle.gpu.metax.local_load(a_smem.slot(0), intrinsic=True, is_constant_offs=True)
        b0_next = tle.gpu.metax.local_load(b_smem.slot(0), intrinsic=True, is_constant_offs=True)
        c31 = tl.dot(a3, loop_b1, c31, input_precision="tf32")
        c13 = tl.dot(loop_a1, b3, c13, input_precision="tf32")
        tle.gpu.metax.iglp(config_2=2, config_5=1, config_6=5)
        b_next_ptrs3_mask = tle.extract_tile(next_b_mask, (0, 3), [BLOCK_K, 32])
        tle.gpu.metax.async_copy_global_to_local(b_next_ptrs3_ptrs, b_smem.slot(3), mask=b_next_ptrs3_mask)

        c32 = tl.dot(a3, b2, c32, input_precision="tf32")
        tle.gpu.metax.gvm_arrive(8)
        tle.gpu.metax.barrier_shared()
        a1_next = tle.gpu.metax.local_load(a_smem.slot(1), intrinsic=True, is_constant_offs=True)
        b1_next = tle.gpu.metax.local_load(b_smem.slot(1), intrinsic=True, is_constant_offs=True)

        next_next_k = loop_counter + 2
        has_next_next = next_next_k < num_k_tiles
        next_next_a_mask = tl.full((BLOCK_M, BLOCK_K), has_next_next, tl.int1)
        a_next_next_ptrs0_ptrs = tle.extract_tile(a_ptr + a_next_next_offsets, (0, 0), [32, BLOCK_K])
        a_next_next_ptrs0_mask = tle.extract_tile(next_next_a_mask, (0, 0), [32, BLOCK_K])
        tle.gpu.metax.async_copy_global_to_local(a_next_next_ptrs0_ptrs, a_smem.slot(0), mask=a_next_next_ptrs0_mask)

        c23 = tl.dot(a2, b3, c23, input_precision="tf32")
        tle.gpu.metax.iglp(config_0=2, config_2=2, config_5=1, config_6=5, config_7=0)
        c33 = tl.dot(a3, b3, c33, input_precision="tf32")

        loop_acc = tle.insert_tile(loop_acc, c00, (0, 0))
        loop_acc = tle.insert_tile(loop_acc, c01, (0, 1))
        loop_acc = tle.insert_tile(loop_acc, c02, (0, 2))
        loop_acc = tle.insert_tile(loop_acc, c03, (0, 3))
        loop_acc = tle.insert_tile(loop_acc, c10, (1, 0))
        loop_acc = tle.insert_tile(loop_acc, c11, (1, 1))
        loop_acc = tle.insert_tile(loop_acc, c12, (1, 2))
        loop_acc = tle.insert_tile(loop_acc, c13, (1, 3))
        loop_acc = tle.insert_tile(loop_acc, c20, (2, 0))
        loop_acc = tle.insert_tile(loop_acc, c21, (2, 1))
        loop_acc = tle.insert_tile(loop_acc, c22, (2, 2))
        loop_acc = tle.insert_tile(loop_acc, c23, (2, 3))
        loop_acc = tle.insert_tile(loop_acc, c30, (3, 0))
        loop_acc = tle.insert_tile(loop_acc, c31, (3, 1))
        loop_acc = tle.insert_tile(loop_acc, c32, (3, 2))
        loop_acc = tle.insert_tile(loop_acc, c33, (3, 3))

        loop_a0 = a0_next
        loop_b0 = b0_next
        loop_a1 = a1_next
        loop_b1 = b1_next
        loop_a_next_offsets = a_next_next_offsets
        loop_b_next_offsets = b_next_next_offsets
        loop_counter = next_k

    tle.gpu.metax.gvm_arrive(0)
    tle.gpu.metax.barrier_shared()
    c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    c_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    c = loop_acc.to(c_ptr.dtype.element_ty)
    tl.store(c_ptrs, c, mask=c_mask)


def matmul(a, b, c, BLOCK_M=128, BLOCK_N=128, BLOCK_K=128, GROUP_M=1):
    M, K = a.shape
    Kb, N = b.shape
    assert ENABLED, "Requires a MetaX FLAGTREE_COMMON_IR build"
    assert K == Kb
    assert (BLOCK_M, BLOCK_N, BLOCK_K) == (128, 128, 128)
    assert M > 0 and N > 0 and K > 0 and M % 128 == N % 128 == K % 128 == 0
    assert a.stride(1) == b.stride(0) == 1, "TN pipeline requires contiguous K in A and B"
    assert a.dtype == b.dtype == c.dtype and a.dtype in (torch.float16, torch.bfloat16)
    assert tuple(c.shape) == (M, N)
    grid = (triton.cdiv(M, BLOCK_M) * triton.cdiv(N, BLOCK_N), )
    matmul_kernel[grid](a, b, c, M, N, K, a.stride(0), a.stride(1), b.stride(0), b.stride(1), c.stride(0), c.stride(1),
                        BLOCK_M, BLOCK_N, BLOCK_K, GROUP_M, num_warps=4, num_stages=4, pipeline="cpasync")
    return c


@pytest.mark.skipif(not ENABLED or not is_maca(), reason="Requires MetaX/MACA target")
@pytest.mark.parametrize(
    "M, N, K",
    [(128, 128, 128), (256, 256, 256), (128, 256, 128), (256, 128, 256)],
)
def test_maca_matmul(M, N, K):
    torch.manual_seed(0)
    a = torch.randn((M, K), device="cuda", dtype=torch.float16)
    # Expose logical (K, N) from contiguous physical (N, K), so K has stride 1.
    b = torch.randn((N, K), device="cuda", dtype=torch.float16).transpose(0, 1)
    c = torch.empty((M, N), device="cuda", dtype=torch.float16)
    matmul(a, b, c)
    torch_output = torch.matmul(a, b).to(torch.float16)
    torch.testing.assert_close(c, torch_output, atol=1e-2, rtol=1e-2)


def _tflops(M, N, K, ms):
    return 2.0 * M * N * K * 1e-12 / (ms * 1e-3)


def _shape_name(M, N, K):
    return f"{M}x{N}x{K}"


def _assert_benchmark_shapes():
    assert BENCHMARK_SIZES == [128 * i for i in range(2, 33)]
    for M, N, K in BENCHMARK_SHAPES:
        assert M == N == K
        assert M % 128 == 0 and N % 128 == 0 and K % 128 == 0


def _make_inputs(M, N, K):
    torch.manual_seed(0)
    a = torch.randn((M, K), device="cuda", dtype=torch.float16)
    # Expose logical (K, N) from contiguous physical (N, K), so K has stride 1.
    b = torch.randn((N, K), device="cuda", dtype=torch.float16).transpose(0, 1)
    c = torch.empty((M, N), device="cuda", dtype=torch.float16)
    return a, b, c


def _measure_accuracy(a, b, c):
    matmul(a, b, c)
    torch_output = torch.matmul(a, b).to(torch.float16)
    diff = (c - torch_output).abs()
    rel = diff / torch_output.abs().clamp_min(1e-6)
    return {
        "max_abs": diff.max().item(),
        "mean_abs": diff.mean().item(),
        "max_rel": rel.max().item(),
        "allclose": torch.allclose(c, torch_output, atol=1e-2, rtol=1e-2),
    }


def _print_markdown_table(headers, rows):
    print("| " + " | ".join(headers) + " |")
    print("| " + " | ".join(["---"] * len(headers)) + " |")
    for row in rows:
        print("| " + " | ".join(str(item) for item in row) + " |")


def run_accuracy_cases(shapes=((128, 128, 128), (256, 256, 256))):
    rows = []
    for M, N, K in shapes:
        a, b, c = _make_inputs(M, N, K)
        result = _measure_accuracy(a, b, c)
        rows.append([
            _shape_name(M, N, K),
            f"{result['max_abs']:.6g}",
            f"{result['mean_abs']:.6g}",
            f"{result['max_rel']:.6g}",
            result["allclose"],
        ])
    _print_markdown_table(["shape", "max_abs", "mean_abs", "max_rel", "allclose"], rows)


def run_benchmark(warmup=25, rep=100, check_correctness=True):
    _assert_benchmark_shapes()
    rows = []
    for M, N, K in BENCHMARK_SHAPES:
        a, b, c = _make_inputs(M, N, K)
        accuracy = _measure_accuracy(a, b, c)
        if check_correctness and not accuracy["allclose"]:
            torch_output = torch.matmul(a, b).to(torch.float16)
            torch.testing.assert_close(c, torch_output, atol=1e-2, rtol=1e-2)

        torch_ms = triton.testing.do_bench(lambda: torch.matmul(a, b), warmup=warmup, rep=rep)
        tle_ms = triton.testing.do_bench(lambda: matmul(a, b, c), warmup=warmup, rep=rep)
        torch_tflops = _tflops(M, N, K, torch_ms)
        tle_tflops = _tflops(M, N, K, tle_ms)
        rows.append([
            _shape_name(M, N, K),
            f"{torch_ms:.4f}",
            f"{tle_ms:.4f}",
            f"{torch_tflops:.2f}",
            f"{tle_tflops:.2f}",
            f"{torch_ms / tle_ms:.3f}",
            f"{accuracy['max_abs']:.6g}",
            accuracy["allclose"],
        ])
    _print_markdown_table(
        ["shape", "torch_ms", "tle_ms", "torch_tflops", "tle_tflops", "speedup", "max_abs", "allclose"],
        rows,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark", action="store_true", help="benchmark square matmul shapes from 256 to 4096")
    parser.add_argument("--warmup", type=int, default=25)
    parser.add_argument("--rep", type=int, default=100)
    parser.add_argument("--no-check", action="store_true", help="skip correctness check during benchmark")
    args = parser.parse_args()

    if args.benchmark:
        run_benchmark(warmup=args.warmup, rep=args.rep, check_correctness=not args.no_check)
    else:
        run_accuracy_cases()
