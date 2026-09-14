import argparse

import pytest
import torch
import triton

from triton.experimental import gluon
from triton.experimental.gluon import language as gl


BENCHMARK_SIZES = [128 * i for i in range(2, 33)]
BENCHMARK_SHAPES = [(size, size, size) for size in BENCHMARK_SIZES]


def is_maca():
    try:
        target = triton.runtime.driver.active.get_current_target()
    except RuntimeError:
        return False
    return target.backend == "maca"


@gluon.jit
def matmul_kernel(a_ptr, b_ptr, c_ptr, M, N, K, stride_am, stride_ak, stride_bk, stride_bn, stride_cm, stride_cn,
                  BLOCK_M: gl.constexpr, BLOCK_N: gl.constexpr, BLOCK_K: gl.constexpr, GROUP_M: gl.constexpr):
    pid = gl.program_id(axis=0)
    num_pid_m = gl.cdiv(M, BLOCK_M)
    num_pid_n = gl.cdiv(N, BLOCK_N)
    num_pid_in_group = GROUP_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    offs_m = pid_m * BLOCK_M + gl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + gl.arange(0, BLOCK_N)
    offs_k = gl.arange(0, BLOCK_K)

    a_offsets = offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
    b_offsets = offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn
    a_ptrs = a_ptr + a_offsets
    b_ptrs = b_ptr + b_offsets

    a_smem = gl.local_alloc(a_ptr.dtype.element_ty, [32, BLOCK_K], num_buffers=4)
    b_smem = gl.local_alloc(b_ptr.dtype.element_ty, [BLOCK_K, 32], num_buffers=4)
    acc = gl.zeros((BLOCK_M, BLOCK_N), dtype=gl.float32)

    num_k_tiles = gl.cdiv(K, BLOCK_K)
    has_first = num_k_tiles > 0
    a_load_mask = gl.full((BLOCK_M, BLOCK_K), has_first, gl.int1)
    b_load_mask = gl.full((BLOCK_K, BLOCK_N), has_first, gl.int1)

    a_ptrs0_ptrs = gl.metax.slice(a_ptrs, [32, BLOCK_K], [0 * 32, 0])
    a_ptrs0_mask = gl.metax.slice(a_load_mask, [32, BLOCK_K], [0 * 32, 0])
    gl.metax.async_copy_global_to_shared(a_smem.index(0), a_ptrs0_ptrs, mask=a_ptrs0_mask, other=0.0, intrinsic=True)
    b_ptrs0_ptrs = gl.metax.slice(b_ptrs, [BLOCK_K, 32], [0, 0 * 32])
    b_ptrs0_mask = gl.metax.slice(b_load_mask, [BLOCK_K, 32], [0, 0 * 32])
    gl.metax.async_copy_global_to_shared(b_smem.index(0), b_ptrs0_ptrs, mask=b_ptrs0_mask, other=0.0, intrinsic=True)
    a_ptrs1_ptrs = gl.metax.slice(a_ptrs, [32, BLOCK_K], [1 * 32, 0])
    a_ptrs1_mask = gl.metax.slice(a_load_mask, [32, BLOCK_K], [1 * 32, 0])
    gl.metax.async_copy_global_to_shared(a_smem.index(1), a_ptrs1_ptrs, mask=a_ptrs1_mask, other=0.0, intrinsic=True)
    b_ptrs1_ptrs = gl.metax.slice(b_ptrs, [BLOCK_K, 32], [0, 1 * 32])
    b_ptrs1_mask = gl.metax.slice(b_load_mask, [BLOCK_K, 32], [0, 1 * 32])
    gl.metax.async_copy_global_to_shared(b_smem.index(1), b_ptrs1_ptrs, mask=b_ptrs1_mask, other=0.0, intrinsic=True)
    a_ptrs2_ptrs = gl.metax.slice(a_ptrs, [32, BLOCK_K], [2 * 32, 0])
    a_ptrs2_mask = gl.metax.slice(a_load_mask, [32, BLOCK_K], [2 * 32, 0])
    gl.metax.async_copy_global_to_shared(a_smem.index(2), a_ptrs2_ptrs, mask=a_ptrs2_mask, other=0.0, intrinsic=True)
    b_ptrs2_ptrs = gl.metax.slice(b_ptrs, [BLOCK_K, 32], [0, 2 * 32])
    b_ptrs2_mask = gl.metax.slice(b_load_mask, [BLOCK_K, 32], [0, 2 * 32])
    gl.metax.async_copy_global_to_shared(b_smem.index(2), b_ptrs2_ptrs, mask=b_ptrs2_mask, other=0.0, intrinsic=True)
    a_ptrs3_ptrs = gl.metax.slice(a_ptrs, [32, BLOCK_K], [3 * 32, 0])
    a_ptrs3_mask = gl.metax.slice(a_load_mask, [32, BLOCK_K], [3 * 32, 0])
    gl.metax.async_copy_global_to_shared(a_smem.index(3), a_ptrs3_ptrs, mask=a_ptrs3_mask, other=0.0, intrinsic=True)
    b_ptrs3_ptrs = gl.metax.slice(b_ptrs, [BLOCK_K, 32], [0, 3 * 32])
    b_ptrs3_mask = gl.metax.slice(b_load_mask, [BLOCK_K, 32], [0, 3 * 32])
    gl.metax.async_copy_global_to_shared(b_smem.index(3), b_ptrs3_ptrs, mask=b_ptrs3_mask, other=0.0, intrinsic=True)
    gl.metax.gvm_arrive(12)
    gl.metax.barrier()

    a_step = gl.full((BLOCK_M, BLOCK_K), BLOCK_K * stride_ak, gl.int32)
    b_step = gl.full((BLOCK_K, BLOCK_N), BLOCK_K * stride_bk, gl.int32)
    a0 = a_smem.index(0).load(intrinsic=True, is_constant_offs=True)
    b0 = b_smem.index(0).load(intrinsic=True, is_constant_offs=True)
    gl.metax.gvm_arrive(8)
    gl.metax.barrier_shared()
    a1 = a_smem.index(1).load(intrinsic=True, is_constant_offs=True)
    b1 = b_smem.index(1).load(intrinsic=True, is_constant_offs=True)

    a_next_offsets = a_offsets + a_step
    b_next_offsets = b_offsets + b_step
    has_second = 1 < num_k_tiles
    first_next_a_mask = gl.full((BLOCK_M, BLOCK_K), has_second, gl.int1)
    a_next_ptrs0_ptrs = gl.metax.slice(a_ptr + a_next_offsets, [32, BLOCK_K], [0 * 32, 0])
    a_next_ptrs0_mask = gl.metax.slice(first_next_a_mask, [32, BLOCK_K], [0 * 32, 0])
    gl.metax.async_copy_global_to_shared(a_smem.index(0), a_next_ptrs0_ptrs, mask=a_next_ptrs0_mask, other=0.0, intrinsic=True)

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

        c00 = gl.metax.slice(loop_acc, [32, 32], [0 * 32, 0 * 32])
        c01 = gl.metax.slice(loop_acc, [32, 32], [0 * 32, 1 * 32])
        c02 = gl.metax.slice(loop_acc, [32, 32], [0 * 32, 2 * 32])
        c03 = gl.metax.slice(loop_acc, [32, 32], [0 * 32, 3 * 32])
        c10 = gl.metax.slice(loop_acc, [32, 32], [1 * 32, 0 * 32])
        c11 = gl.metax.slice(loop_acc, [32, 32], [1 * 32, 1 * 32])
        c12 = gl.metax.slice(loop_acc, [32, 32], [1 * 32, 2 * 32])
        c13 = gl.metax.slice(loop_acc, [32, 32], [1 * 32, 3 * 32])
        c20 = gl.metax.slice(loop_acc, [32, 32], [2 * 32, 0 * 32])
        c21 = gl.metax.slice(loop_acc, [32, 32], [2 * 32, 1 * 32])
        c22 = gl.metax.slice(loop_acc, [32, 32], [2 * 32, 2 * 32])
        c23 = gl.metax.slice(loop_acc, [32, 32], [2 * 32, 3 * 32])
        c30 = gl.metax.slice(loop_acc, [32, 32], [3 * 32, 0 * 32])
        c31 = gl.metax.slice(loop_acc, [32, 32], [3 * 32, 1 * 32])
        c32 = gl.metax.slice(loop_acc, [32, 32], [3 * 32, 2 * 32])
        c33 = gl.metax.slice(loop_acc, [32, 32], [3 * 32, 3 * 32])

        a_next_ptrs = a_ptr + loop_a_next_offsets
        b_next_ptrs = b_ptr + loop_b_next_offsets
        a_next_next_offsets = loop_a_next_offsets + a_step
        b_next_next_offsets = loop_b_next_offsets + b_step
        next_a_mask = gl.full((BLOCK_M, BLOCK_K), has_next, gl.int1)
        next_b_mask = gl.full((BLOCK_K, BLOCK_N), has_next, gl.int1)

        b_next_ptrs0_ptrs = gl.metax.slice(b_next_ptrs, [BLOCK_K, 32], [0, 0 * 32])
        b_next_ptrs0_mask = gl.metax.slice(next_b_mask, [BLOCK_K, 32], [0, 0 * 32])
        gl.metax.async_copy_global_to_shared(b_smem.index(0), b_next_ptrs0_ptrs, mask=b_next_ptrs0_mask, other=0.0, intrinsic=True)

        c00 = gl.dot(loop_a0, loop_b0, c00, input_precision="tf32")
        gl.metax.iglp(config_2=2, config_5=1, config_6=2)
        gl.metax.gvm_arrive(8)
        gl.metax.barrier_shared()

        b2 = b_smem.index(2).load(intrinsic=True, is_constant_offs=True)
        a2 = a_smem.index(2).load(intrinsic=True, is_constant_offs=True)

        a_next_ptrs1_ptrs = gl.metax.slice(a_next_ptrs, [32, BLOCK_K], [1 * 32, 0])
        a_next_ptrs1_mask = gl.metax.slice(next_a_mask, [32, BLOCK_K], [1 * 32, 0])
        gl.metax.async_copy_global_to_shared(a_smem.index(1), a_next_ptrs1_ptrs, mask=a_next_ptrs1_mask, other=0.0, intrinsic=True)
        b_next_ptrs1_ptrs = gl.metax.slice(b_next_ptrs, [BLOCK_K, 32], [0, 1 * 32])
        b_next_ptrs1_mask = gl.metax.slice(next_b_mask, [BLOCK_K, 32], [0, 1 * 32])
        gl.metax.async_copy_global_to_shared(b_smem.index(1), b_next_ptrs1_ptrs, mask=b_next_ptrs1_mask, other=0.0, intrinsic=True)

        c10 = gl.dot(loop_a1, loop_b0, c10, input_precision="tf32")
        c01 = gl.dot(loop_a0, loop_b1, c01, input_precision="tf32")
        c11 = gl.dot(loop_a1, loop_b1, c11, input_precision="tf32")
        gl.metax.iglp(config_0=2, config_2=2, config_5=1, config_6=7, config_7=8)
        gl.metax.gvm_arrive(8)
        gl.metax.barrier_shared()

        a3 = a_smem.index(3).load(intrinsic=True, is_constant_offs=True)
        b3 = b_smem.index(3).load(intrinsic=True, is_constant_offs=True)

        a_next_ptrs2_ptrs = gl.metax.slice(a_next_ptrs, [32, BLOCK_K], [2 * 32, 0])
        a_next_ptrs2_mask = gl.metax.slice(next_a_mask, [32, BLOCK_K], [2 * 32, 0])
        gl.metax.async_copy_global_to_shared(a_smem.index(2), a_next_ptrs2_ptrs, mask=a_next_ptrs2_mask, other=0.0, intrinsic=True)
        b_next_ptrs2_ptrs = gl.metax.slice(b_next_ptrs, [BLOCK_K, 32], [0, 2 * 32])
        b_next_ptrs2_mask = gl.metax.slice(next_b_mask, [BLOCK_K, 32], [0, 2 * 32])
        gl.metax.async_copy_global_to_shared(b_smem.index(2), b_next_ptrs2_ptrs, mask=b_next_ptrs2_mask, other=0.0, intrinsic=True)

        c20 = gl.dot(a2, loop_b0, c20, input_precision="tf32")
        c21 = gl.dot(a2, loop_b1, c21, input_precision="tf32")
        c02 = gl.dot(loop_a0, b2, c02, input_precision="tf32")
        c12 = gl.dot(loop_a1, b2, c12, input_precision="tf32")
        c22 = gl.dot(a2, b2, c22, input_precision="tf32")
        gl.metax.iglp(config_2=2, config_5=1, config_6=5)
        gl.metax.gvm_arrive(8)
        gl.metax.barrier_shared()
        gl.metax.iglp(config_2=2, config_5=1, config_6=5)

        a_next_ptrs3_ptrs = gl.metax.slice(a_next_ptrs, [32, BLOCK_K], [3 * 32, 0])
        a_next_ptrs3_mask = gl.metax.slice(next_a_mask, [32, BLOCK_K], [3 * 32, 0])
        gl.metax.async_copy_global_to_shared(a_smem.index(3), a_next_ptrs3_ptrs, mask=a_next_ptrs3_mask, other=0.0, intrinsic=True)
        b_next_ptrs3_ptrs = gl.metax.slice(b_next_ptrs, [BLOCK_K, 32], [0, 3 * 32])

        c30 = gl.dot(a3, loop_b0, c30, input_precision="tf32")
        c03 = gl.dot(loop_a0, b3, c03, input_precision="tf32")
        gl.metax.sched_bound()
        a0_next = a_smem.index(0).load(intrinsic=True, is_constant_offs=True)
        b0_next = b_smem.index(0).load(intrinsic=True, is_constant_offs=True)
        c31 = gl.dot(a3, loop_b1, c31, input_precision="tf32")
        c13 = gl.dot(loop_a1, b3, c13, input_precision="tf32")
        gl.metax.iglp(config_2=2, config_5=1, config_6=5)
        b_next_ptrs3_mask = gl.metax.slice(next_b_mask, [BLOCK_K, 32], [0, 3 * 32])
        gl.metax.async_copy_global_to_shared(b_smem.index(3), b_next_ptrs3_ptrs, mask=b_next_ptrs3_mask, other=0.0, intrinsic=True)

        c32 = gl.dot(a3, b2, c32, input_precision="tf32")
        gl.metax.gvm_arrive(8)
        gl.metax.barrier_shared()
        a1_next = a_smem.index(1).load(intrinsic=True, is_constant_offs=True)
        b1_next = b_smem.index(1).load(intrinsic=True, is_constant_offs=True)

        next_next_k = loop_counter + 2
        has_next_next = next_next_k < num_k_tiles
        next_next_a_mask = gl.full((BLOCK_M, BLOCK_K), has_next_next, gl.int1)
        a_next_next_ptrs0_ptrs = gl.metax.slice(a_ptr + a_next_next_offsets, [32, BLOCK_K], [0 * 32, 0])
        a_next_next_ptrs0_mask = gl.metax.slice(next_next_a_mask, [32, BLOCK_K], [0 * 32, 0])
        gl.metax.async_copy_global_to_shared(a_smem.index(0), a_next_next_ptrs0_ptrs, mask=a_next_next_ptrs0_mask, other=0.0, intrinsic=True)

        c23 = gl.dot(a2, b3, c23, input_precision="tf32")
        gl.metax.iglp(config_0=2, config_2=2, config_5=1, config_6=5, config_7=0)
        c33 = gl.dot(a3, b3, c33, input_precision="tf32")

        loop_acc = gl.metax.slice_update(loop_acc, c00, [0 * 32, 0 * 32])
        loop_acc = gl.metax.slice_update(loop_acc, c01, [0 * 32, 1 * 32])
        loop_acc = gl.metax.slice_update(loop_acc, c02, [0 * 32, 2 * 32])
        loop_acc = gl.metax.slice_update(loop_acc, c03, [0 * 32, 3 * 32])
        loop_acc = gl.metax.slice_update(loop_acc, c10, [1 * 32, 0 * 32])
        loop_acc = gl.metax.slice_update(loop_acc, c11, [1 * 32, 1 * 32])
        loop_acc = gl.metax.slice_update(loop_acc, c12, [1 * 32, 2 * 32])
        loop_acc = gl.metax.slice_update(loop_acc, c13, [1 * 32, 3 * 32])
        loop_acc = gl.metax.slice_update(loop_acc, c20, [2 * 32, 0 * 32])
        loop_acc = gl.metax.slice_update(loop_acc, c21, [2 * 32, 1 * 32])
        loop_acc = gl.metax.slice_update(loop_acc, c22, [2 * 32, 2 * 32])
        loop_acc = gl.metax.slice_update(loop_acc, c23, [2 * 32, 3 * 32])
        loop_acc = gl.metax.slice_update(loop_acc, c30, [3 * 32, 0 * 32])
        loop_acc = gl.metax.slice_update(loop_acc, c31, [3 * 32, 1 * 32])
        loop_acc = gl.metax.slice_update(loop_acc, c32, [3 * 32, 2 * 32])
        loop_acc = gl.metax.slice_update(loop_acc, c33, [3 * 32, 3 * 32])

        loop_a0 = a0_next
        loop_b0 = b0_next
        loop_a1 = a1_next
        loop_b1 = b1_next
        loop_a_next_offsets = a_next_next_offsets
        loop_b_next_offsets = b_next_next_offsets
        loop_counter = next_k

    gl.metax.gvm_arrive(0)
    gl.metax.barrier_shared()
    c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    c_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    c = loop_acc.to(c_ptr.dtype.element_ty)
    gl.store(c_ptrs, c, mask=c_mask)


def matmul(a, b, c, BLOCK_M=128, BLOCK_N=128, BLOCK_K=128, GROUP_M=8):
    M, K = a.shape
    Kb, N = b.shape
    assert K == Kb
    grid = (triton.cdiv(M, BLOCK_M) * triton.cdiv(N, BLOCK_N), )
    matmul_kernel[grid](a, b, c, M, N, K, a.stride(0), a.stride(1), b.stride(0), b.stride(1), c.stride(0),
                        c.stride(1), BLOCK_M, BLOCK_N, BLOCK_K, GROUP_M, num_warps=4, num_stages=4,
                        pipeline="cpasync", scenario="")
    return c


@pytest.mark.skipif(not is_maca(), reason="Requires MetaX/MACA target")
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
        gluon_ms = triton.testing.do_bench(lambda: matmul(a, b, c), warmup=warmup, rep=rep)
        torch_tflops = _tflops(M, N, K, torch_ms)
        gluon_tflops = _tflops(M, N, K, gluon_ms)
        rows.append([
            _shape_name(M, N, K),
            f"{torch_ms:.4f}",
            f"{gluon_ms:.4f}",
            f"{torch_tflops:.2f}",
            f"{gluon_tflops:.2f}",
            f"{torch_ms / gluon_ms:.3f}",
            f"{accuracy['max_abs']:.6g}",
            accuracy["allclose"],
        ])
    _print_markdown_table(
        ["shape", "torch_ms", "gluon_ms", "torch_tflops", "gluon_tflops", "speedup", "max_abs", "allclose"],
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
