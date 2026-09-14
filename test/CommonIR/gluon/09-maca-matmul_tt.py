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
                  BLOCK_M: gl.constexpr, BLOCK_N: gl.constexpr, BLOCK_K: gl.constexpr):
    pid = gl.program_id(axis=0)
    num_pid_m = gl.cdiv(M, BLOCK_M)
    pid_m = pid % num_pid_m
    pid_n = pid // num_pid_m

    offs_m = pid_m * BLOCK_M + gl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + gl.arange(0, BLOCK_N)
    offs_k_a = gl.arange(0, BLOCK_K)
    offs_k_b = gl.arange(0, BLOCK_K)

    a_offsets = offs_m[:, None] * stride_am + offs_k_a[None, :] * stride_ak
    b_offsets = offs_k_b[:, None] * stride_bk + offs_n[None, :] * stride_bn
    a_ptrs = a_ptr + a_offsets
    b_ptrs = b_ptr + b_offsets

    a_smem = gl.local_alloc(a_ptr.dtype.element_ty, [32, BLOCK_K], num_buffers=4)
    b_smem = gl.local_alloc(b_ptr.dtype.element_ty, [32, BLOCK_N], num_buffers=4)
    acc = gl.zeros((BLOCK_M, BLOCK_N), dtype=gl.float32)


    num_k_tiles = gl.cdiv(K, BLOCK_K)
    has_first = num_k_tiles > 0
    a_load_mask = gl.full((BLOCK_M, BLOCK_K), has_first, gl.int1)
    b_load_mask = gl.full((BLOCK_K, BLOCK_N), has_first, gl.int1)

    a_ptrs0 = gl.metax.slice(a_ptrs, [32, BLOCK_K], [0 * 32, 0])
    a_mask0 = gl.metax.slice(a_load_mask, [32, BLOCK_K], [0 * 32, 0])
    gl.metax.async_copy_global_to_shared(a_smem.index(0), a_ptrs0, mask=a_mask0, other=0.0, intrinsic=True)
    b_ptrs0 = gl.metax.slice(b_ptrs, [32, BLOCK_N], [0 * 32, 0])
    b_mask0 = gl.metax.slice(b_load_mask, [32, BLOCK_N], [0 * 32, 0])
    gl.metax.async_copy_global_to_shared(b_smem.index(0), b_ptrs0, mask=b_mask0, other=0.0, intrinsic=True)

    a_ptrs1 = gl.metax.slice(a_ptrs, [32, BLOCK_K], [1 * 32, 0])
    a_mask1 = gl.metax.slice(a_load_mask, [32, BLOCK_K], [1 * 32, 0])
    gl.metax.async_copy_global_to_shared(a_smem.index(1), a_ptrs1, mask=a_mask1, other=0.0, intrinsic=True)
    b_ptrs1 = gl.metax.slice(b_ptrs, [32, BLOCK_N], [1 * 32, 0])
    b_mask1 = gl.metax.slice(b_load_mask, [32, BLOCK_N], [1 * 32, 0])
    gl.metax.async_copy_global_to_shared(b_smem.index(1), b_ptrs1, mask=b_mask1, other=0.0, intrinsic=True)

    a_ptrs2 = gl.metax.slice(a_ptrs, [32, BLOCK_K], [2 * 32, 0])
    a_mask2 = gl.metax.slice(a_load_mask, [32, BLOCK_K], [2 * 32, 0])
    gl.metax.async_copy_global_to_shared(a_smem.index(2), a_ptrs2, mask=a_mask2, other=0.0, intrinsic=True)
    b_ptrs2 = gl.metax.slice(b_ptrs, [32, BLOCK_N], [2 * 32, 0])
    b_mask2 = gl.metax.slice(b_load_mask, [32, BLOCK_N], [2 * 32, 0])
    gl.metax.async_copy_global_to_shared(b_smem.index(2), b_ptrs2, mask=b_mask2, other=0.0, intrinsic=True)

    a_ptrs3 = gl.metax.slice(a_ptrs, [32, BLOCK_K], [3 * 32, 0])
    a_mask3 = gl.metax.slice(a_load_mask, [32, BLOCK_K], [3 * 32, 0])
    gl.metax.async_copy_global_to_shared(a_smem.index(3), a_ptrs3, mask=a_mask3, other=0.0, intrinsic=True)
    b_ptrs3 = gl.metax.slice(b_ptrs, [32, BLOCK_N], [3 * 32, 0])
    b_mask3 = gl.metax.slice(b_load_mask, [32, BLOCK_N], [3 * 32, 0])
    gl.metax.async_copy_global_to_shared(b_smem.index(3), b_ptrs3, mask=b_mask3, other=0.0, intrinsic=True)
    gl.metax.gvm_arrive(12)
    gl.metax.barrier()

    a0 = a_smem.index(0).load(intrinsic=True, is_constant_offs=True)
    b0_raw = b_smem.index(0).load(dtype=gl.int32, intrinsic=True, is_constant_offs=True, mma_mode=2)
    b0 = gl.metax.bsm_perm(b0_raw, b_ptr.dtype.element_ty)
    gl.metax.gvm_arrive(8)
    gl.metax.barrier()
    a1 = a_smem.index(1).load(intrinsic=True, is_constant_offs=True)
    b1_raw = b_smem.index(1).load(dtype=gl.int32, intrinsic=True, is_constant_offs=True, mma_mode=2)

    a_step = gl.full((BLOCK_M, BLOCK_K), BLOCK_K * stride_ak, gl.int32)
    b_step = gl.full((BLOCK_K, BLOCK_N), BLOCK_K * stride_bk, gl.int32)

    for k in range(0, num_k_tiles):
        next_k = k + 1
        has_next = next_k < num_k_tiles
        next_a_mask = gl.full((BLOCK_M, BLOCK_K), has_next, gl.int1)
        next_b_mask = gl.full((BLOCK_K, BLOCK_N), has_next, gl.int1)

        c0 = gl.metax.slice(acc, [32, BLOCK_N], [0 * 32, 0])
        c1 = gl.metax.slice(acc, [32, BLOCK_N], [1 * 32, 0])
        c2 = gl.metax.slice(acc, [32, BLOCK_N], [2 * 32, 0])
        c3 = gl.metax.slice(acc, [32, BLOCK_N], [3 * 32, 0])

        a_next_offsets = a_offsets
        b_next_offsets = b_offsets

        a_off0 = gl.metax.slice(a_offsets, [32, BLOCK_K], [0 * 32, 0])
        a_inc0 = gl.metax.slice(a_step, [32, BLOCK_K], [0 * 32, 0])
        a_off0 = a_off0 + a_inc0
        a_next_offsets = gl.metax.slice_update(a_next_offsets, a_off0, [0 * 32, 0 * 32])
        a_next_ptrs0 = gl.metax.slice(a_ptr + a_next_offsets, [32, BLOCK_K], [0 * 32, 0])
        a_next_mask0 = gl.metax.slice(next_a_mask, [32, BLOCK_K], [0 * 32, 0])
        gl.metax.async_copy_global_to_shared(a_smem.index(0), a_next_ptrs0, mask=a_next_mask0, other=0.0,
                                             intrinsic=True)

        b_off0 = gl.metax.slice(b_offsets, [32, BLOCK_N], [0 * 32, 0])
        b_inc0 = gl.metax.slice(b_step, [32, BLOCK_N], [0 * 32, 0])
        b_off0 = b_off0 + b_inc0
        b_next_offsets = gl.metax.slice_update(b_next_offsets, b_off0, [0 * 32, 0 * 32])
        b_next_ptrs0 = gl.metax.slice(b_ptr + b_next_offsets, [32, BLOCK_N], [0 * 32, 0])
        b_next_mask0 = gl.metax.slice(next_b_mask, [32, BLOCK_N], [0 * 32, 0])
        gl.metax.async_copy_global_to_shared(b_smem.index(0), b_next_ptrs0, mask=b_next_mask0, other=0.0,
                                             intrinsic=True)

        a0k0 = gl.metax.slice(a0, [32, 32], [0 * 32, 0 * 32])
        c0 = gl.dot(a0k0, b0, c0, input_precision="tf32")
        gl.metax.iglp(config_2=2, config_5=1, config_6=2)
        gl.metax.gvm_arrive(8)
        gl.metax.barrier_shared()

        b2_raw = b_smem.index(2).load(dtype=gl.int32, intrinsic=True, is_constant_offs=True, mma_mode=2)
        a2 = a_smem.index(2).load(intrinsic=True, is_constant_offs=True)

        a_off1 = gl.metax.slice(a_offsets, [32, BLOCK_K], [1 * 32, 0])
        a_inc1 = gl.metax.slice(a_step, [32, BLOCK_K], [1 * 32, 0])
        a_off1 = a_off1 + a_inc1
        a_next_offsets = gl.metax.slice_update(a_next_offsets, a_off1, [1 * 32, 0 * 32])
        a_next_ptrs1 = gl.metax.slice(a_ptr + a_next_offsets, [32, BLOCK_K], [1 * 32, 0])
        a_next_mask1 = gl.metax.slice(next_a_mask, [32, BLOCK_K], [1 * 32, 0])
        gl.metax.async_copy_global_to_shared(a_smem.index(1), a_next_ptrs1, mask=a_next_mask1, other=0.0,
                                             intrinsic=True)

        b_off1 = gl.metax.slice(b_offsets, [32, BLOCK_N], [1 * 32, 0])
        b_inc1 = gl.metax.slice(b_step, [32, BLOCK_N], [1 * 32, 0])
        b_off1 = b_off1 + b_inc1
        b_next_offsets = gl.metax.slice_update(b_next_offsets, b_off1, [1 * 32, 0 * 32])
        b_next_ptrs1 = gl.metax.slice(b_ptr + b_next_offsets, [32, BLOCK_N], [1 * 32, 0])
        b_next_mask1 = gl.metax.slice(next_b_mask, [32, BLOCK_N], [1 * 32, 0])
        gl.metax.async_copy_global_to_shared(b_smem.index(1), b_next_ptrs1, mask=b_next_mask1, other=0.0,
                                             intrinsic=True)

        a1k0 = gl.metax.slice(a1, [32, 32], [0 * 32, 0 * 32])
        c1 = gl.dot(a1k0, b0, c1, input_precision="tf32")
        b1 = gl.metax.bsm_perm(b1_raw, b_ptr.dtype.element_ty)
        a0k1 = gl.metax.slice(a0, [32, 32], [0 * 32, 1 * 32])
        c0 = gl.dot(a0k1, b1, c0, input_precision="tf32")
        a1k1 = gl.metax.slice(a1, [32, 32], [0 * 32, 1 * 32])
        c1 = gl.dot(a1k1, b1, c1, input_precision="tf32")
        gl.metax.iglp(config_0=2, config_2=2, config_5=1, config_6=7, config_7=8)
        gl.metax.gvm_arrive(8)
        gl.metax.barrier_shared()

        a3 = a_smem.index(3).load(intrinsic=True, is_constant_offs=True)
        b3_raw = b_smem.index(3).load(dtype=gl.int32, intrinsic=True, is_constant_offs=True, mma_mode=2)

        b_off2 = gl.metax.slice(b_offsets, [32, BLOCK_N], [2 * 32, 0])
        b_inc2 = gl.metax.slice(b_step, [32, BLOCK_N], [2 * 32, 0])
        b_off2 = b_off2 + b_inc2
        b_next_offsets = gl.metax.slice_update(b_next_offsets, b_off2, [2 * 32, 0 * 32])
        b_next_ptrs2 = gl.metax.slice(b_ptr + b_next_offsets, [32, BLOCK_N], [2 * 32, 0])
        b_next_mask2 = gl.metax.slice(next_b_mask, [32, BLOCK_N], [2 * 32, 0])
        gl.metax.async_copy_global_to_shared(b_smem.index(2), b_next_ptrs2, mask=b_next_mask2, other=0.0,
                                             intrinsic=True)

        a_off2 = gl.metax.slice(a_offsets, [32, BLOCK_K], [2 * 32, 0])
        a_inc2 = gl.metax.slice(a_step, [32, BLOCK_K], [2 * 32, 0])
        a_off2 = a_off2 + a_inc2
        a_next_offsets = gl.metax.slice_update(a_next_offsets, a_off2, [2 * 32, 0 * 32])
        a_next_ptrs2 = gl.metax.slice(a_ptr + a_next_offsets, [32, BLOCK_K], [2 * 32, 0])
        a_next_mask2 = gl.metax.slice(next_a_mask, [32, BLOCK_K], [2 * 32, 0])
        gl.metax.async_copy_global_to_shared(a_smem.index(2), a_next_ptrs2, mask=a_next_mask2, other=0.0,
                                             intrinsic=True)

        a2k0 = gl.metax.slice(a2, [32, 32], [0 * 32, 0 * 32])
        c2 = gl.dot(a2k0, b0, c2, input_precision="tf32")
        b2 = gl.metax.bsm_perm(b2_raw, b_ptr.dtype.element_ty)
        a0k2 = gl.metax.slice(a0, [32, 32], [0 * 32, 2 * 32])
        c0 = gl.dot(a0k2, b2, c0, input_precision="tf32")
        a2k1 = gl.metax.slice(a2, [32, 32], [0 * 32, 1 * 32])
        c2 = gl.dot(a2k1, b1, c2, input_precision="tf32")
        a1k2 = gl.metax.slice(a1, [32, 32], [0 * 32, 2 * 32])
        c1 = gl.dot(a1k2, b2, c1, input_precision="tf32")
        a2k2 = gl.metax.slice(a2, [32, 32], [0 * 32, 2 * 32])
        c2 = gl.dot(a2k2, b2, c2, input_precision="tf32")
        gl.metax.iglp(config_2=2, config_5=1, config_6=5)
        gl.metax.gvm_arrive(8)
        gl.metax.barrier_shared()
        gl.metax.iglp(config_2=2, config_5=1, config_6=5)

        a_off3 = gl.metax.slice(a_offsets, [32, BLOCK_K], [3 * 32, 0])
        a_inc3 = gl.metax.slice(a_step, [32, BLOCK_K], [3 * 32, 0])
        a_off3 = a_off3 + a_inc3
        a_next_offsets = gl.metax.slice_update(a_next_offsets, a_off3, [3 * 32, 0 * 32])
        a_next_ptrs3 = gl.metax.slice(a_ptr + a_next_offsets, [32, BLOCK_K], [3 * 32, 0])
        a_next_mask3 = gl.metax.slice(next_a_mask, [32, BLOCK_K], [3 * 32, 0])
        gl.metax.async_copy_global_to_shared(a_smem.index(3), a_next_ptrs3, mask=a_next_mask3, other=0.0,
                                             intrinsic=True)

        b_off3 = gl.metax.slice(b_offsets, [32, BLOCK_N], [3 * 32, 0])
        b_inc3 = gl.metax.slice(b_step, [32, BLOCK_N], [3 * 32, 0])
        b_off3 = b_off3 + b_inc3
        b_next_offsets = gl.metax.slice_update(b_next_offsets, b_off3, [3 * 32, 0 * 32])
        b_next_ptrs3 = gl.metax.slice(b_ptr + b_next_offsets, [32, BLOCK_N], [3 * 32, 0])

        a0k3 = gl.metax.slice(a0, [32, 32], [0 * 32, 3 * 32])
        a3k0 = gl.metax.slice(a3, [32, 32], [0 * 32, 0 * 32])
        c3 = gl.dot(a3k0, b0, c3, input_precision="tf32")
        b3 = gl.metax.bsm_perm(b3_raw, b_ptr.dtype.element_ty)
        c0 = gl.dot(a0k3, b3, c0, input_precision="tf32")
        gl.metax.sched_bound()
        b0_next_raw = b_smem.index(0).load(dtype=gl.int32, intrinsic=True, is_constant_offs=True, mma_mode=2)
        a0_next = a_smem.index(0).load(intrinsic=True, is_constant_offs=True)
        a1k3 = gl.metax.slice(a1, [32, 32], [0 * 32, 3 * 32])
        a3k1 = gl.metax.slice(a3, [32, 32], [0 * 32, 1 * 32])
        c3 = gl.dot(a3k1, b1, c3, input_precision="tf32")
        c1 = gl.dot(a1k3, b3, c1, input_precision="tf32")
        gl.metax.iglp(config_2=2, config_5=1, config_6=5)
        b_next_mask3 = gl.metax.slice(next_b_mask, [32, BLOCK_N], [3 * 32, 0])
        gl.metax.async_copy_global_to_shared(b_smem.index(3), b_next_ptrs3, mask=b_next_mask3, other=0.0,
                                             intrinsic=True)

        a3k2 = gl.metax.slice(a3, [32, 32], [0 * 32, 2 * 32])
        c3 = gl.dot(a3k2, b2, c3, input_precision="tf32")
        gl.metax.gvm_arrive(8)
        gl.metax.barrier_shared()
        a1_next = a_smem.index(1).load(intrinsic=True, is_constant_offs=True)
        b1_next_raw = b_smem.index(1).load(dtype=gl.int32, intrinsic=True, is_constant_offs=True, mma_mode=2)
        a2k3 = gl.metax.slice(a2, [32, 32], [0 * 32, 3 * 32])
        c2 = gl.dot(a2k3, b3, c2, input_precision="tf32")
        b0_next = gl.metax.bsm_perm(b0_next_raw, b_ptr.dtype.element_ty)
        a3k3 = gl.metax.slice(a3, [32, 32], [0 * 32, 3 * 32])
        c3 = gl.dot(a3k3, b3, c3, input_precision="tf32")
        gl.metax.iglp(config_2=2, config_5=1, config_6=5)

        acc = gl.metax.slice_update(acc, c0, [0 * 32, 0 * 32])
        acc = gl.metax.slice_update(acc, c1, [1 * 32, 0 * 32])
        acc = gl.metax.slice_update(acc, c2, [2 * 32, 0 * 32])
        acc = gl.metax.slice_update(acc, c3, [3 * 32, 0 * 32])

        a0 = a0_next
        b0 = b0_next
        a1 = a1_next
        b1_raw = b1_next_raw
        a_offsets = a_next_offsets
        b_offsets = b_next_offsets

    gl.metax.gvm_arrive(0)
    gl.metax.barrier_shared()
    c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    c_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    c = acc.to(c_ptr.dtype.element_ty)
    gl.store(c_ptrs, c, mask=c_mask)


def matmul(a, b, c, BLOCK_M=128, BLOCK_N=128, BLOCK_K=128):
    M, K = a.shape
    Kb, N = b.shape
    assert K == Kb
    grid = (triton.cdiv(M, BLOCK_M) * triton.cdiv(N, BLOCK_N), )
    matmul_kernel[grid](a, b, c, M, N, K, a.stride(0), a.stride(1), b.stride(0), b.stride(1), c.stride(0),
                        c.stride(1), BLOCK_M, BLOCK_N, BLOCK_K, num_warps=4)
    return c


@pytest.mark.skipif(not is_maca(), reason="Requires MetaX/MACA target")
@pytest.mark.parametrize("M, N, K", [(128, 128, 128), (256, 256, 256)])
def test_maca_matmul(M, N, K):
    torch.manual_seed(0)
    a = torch.randn((M, K), device="cuda", dtype=torch.float16)
    b = torch.randn((K, N), device="cuda", dtype=torch.float16)
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
    b = torch.randn((K, N), device="cuda", dtype=torch.float16)
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
