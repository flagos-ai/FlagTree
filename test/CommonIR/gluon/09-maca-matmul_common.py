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
def matmul_kernel(a_ptr, b_ptr, c_ptr, M, N, K, stride_am, stride_bk, stride_cm, BLOCK_M: gl.constexpr,
                  BLOCK_N: gl.constexpr, BLOCK_K: gl.constexpr, GROUP_SIZE_M: gl.constexpr):
    pid = gl.program_id(axis=0)
    num_pid_m = gl.cdiv(M, BLOCK_M)
    num_pid_n = gl.cdiv(N, BLOCK_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = gl.minimum(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    pid_m_base = pid_m * BLOCK_M
    pid_n_base = pid_n * BLOCK_N
    offs_m_a = pid_m_base + gl.arange(0, BLOCK_M)
    offs_m_c = pid_m_base + gl.arange(0, BLOCK_M)
    offs_n = pid_n_base + gl.arange(0, BLOCK_N)
    offs_k_a = gl.arange(0, BLOCK_K)
    offs_k_b = gl.arange(0, BLOCK_K)

    a_offsets = offs_m_a[:, None] * stride_am + offs_k_a[None, :]
    b_offsets = offs_k_b[:, None] * stride_bk + offs_n[None, :]
    a_ptrs = a_ptr + a_offsets
    b_ptrs = b_ptr + b_offsets

    a_smem = gl.local_alloc(a_ptr.dtype.element_ty, [BLOCK_M, BLOCK_K], num_buffers=2)
    b_smem = gl.local_alloc(b_ptr.dtype.element_ty, [BLOCK_K, BLOCK_N], num_buffers=2)
    acc = gl.zeros((BLOCK_M, BLOCK_N), dtype=gl.float32)

    num_k_tiles = gl.cdiv(K, BLOCK_K)
    has_first = num_k_tiles > 0
    a_load_mask = gl.full((BLOCK_M, BLOCK_K), has_first, gl.int1)
    b_load_mask = gl.full((BLOCK_K, BLOCK_N), has_first, gl.int1)

    a_tile = gl.load(a_ptrs, mask=a_load_mask)
    b_tile = gl.load(b_ptrs, mask=b_load_mask)
    a_stage = gl.metax.slice(a_smem, [BLOCK_M, BLOCK_K], [0, 0, 0])
    b_stage = gl.metax.slice(b_smem, [BLOCK_K, BLOCK_N], [0, 0, 0])
    a_stage.store(a_tile)
    b_stage.store(b_tile)

    a_step = gl.full((BLOCK_M, BLOCK_K), BLOCK_K, gl.int32)
    b_step = gl.full((BLOCK_K, BLOCK_N), BLOCK_K * stride_bk, gl.int32)
    loop_acc = acc
    loop_a_stage = a_stage
    loop_b_stage = b_stage
    loop_a_ptrs = a_ptrs
    loop_b_ptrs = b_ptrs
    loop_counter = 0
    write_stage = 1

    for _ in range(0, num_k_tiles):
        next_k = loop_counter + 1
        has_next = next_k < num_k_tiles
        next_stage = write_stage % 2

        next_a_ptrs = loop_a_ptrs + a_step
        next_b_ptrs = loop_b_ptrs + b_step
        next_a_mask = gl.full((BLOCK_M, BLOCK_K), has_next, gl.int1)
        next_b_mask = gl.full((BLOCK_K, BLOCK_N), has_next, gl.int1)
        next_a_tile = gl.load(next_a_ptrs, mask=next_a_mask)
        next_b_tile = gl.load(next_b_ptrs, mask=next_b_mask)

        gl.metax.barrier_shared()

        a_k0 = gl.metax.slice(loop_a_stage, [BLOCK_M, 16], [0, 0]).load(intrinsic=True)
        b_k0 = gl.metax.slice(loop_b_stage, [16, BLOCK_N], [0, 0]).load(intrinsic=True)
        loop_acc = gl.dot(a_k0, b_k0, loop_acc, input_precision="tf32")

        a_k1 = gl.metax.slice(loop_a_stage, [BLOCK_M, 16], [0, 16]).load(intrinsic=True)
        b_k1 = gl.metax.slice(loop_b_stage, [16, BLOCK_N], [16, 0]).load(intrinsic=True)
        gl.metax.sched_bound()
        loop_acc = gl.dot(a_k1, b_k1, loop_acc, input_precision="tf32")

        next_a_stage = gl.metax.slice(a_smem, [BLOCK_M, BLOCK_K], [next_stage, 0, 0])
        next_b_stage = gl.metax.slice(b_smem, [BLOCK_K, BLOCK_N], [next_stage, 0, 0])
        next_a_stage.store(next_a_tile)
        next_b_stage.store(next_b_tile)
        loop_a_stage = next_a_stage
        loop_b_stage = next_b_stage
        loop_a_ptrs = next_a_ptrs
        loop_b_ptrs = next_b_ptrs
        loop_counter = next_k
        write_stage = write_stage + 1

    gl.metax.barrier_shared()
    a_smem._keep_alive()
    b_smem._keep_alive()
    c_ptrs = c_ptr + offs_m_c[:, None] * stride_cm + offs_n[None, :]
    c_mask = (offs_m_c[:, None] < M) & (offs_n[None, :] < N)
    c = loop_acc.to(c_ptr.dtype.element_ty)
    gl.store(c_ptrs, c, mask=c_mask)


def matmul(a, b, c, BLOCK_M=256, BLOCK_N=256, BLOCK_K=32, GROUP_SIZE_M=8):
    M, K = a.shape
    Kb, N = b.shape
    assert K == Kb
    grid = (triton.cdiv(M, BLOCK_M) * triton.cdiv(N, BLOCK_N), )
    matmul_kernel[grid](a, b, c, M, N, K, a.stride(0), b.stride(0), c.stride(0), BLOCK_M, BLOCK_N, BLOCK_K,
                        GROUP_SIZE_M, num_warps=8)
    return c


@pytest.mark.skipif(not is_maca(), reason="Requires MetaX/MACA target")
@pytest.mark.parametrize("M, N, K", [(256, 256, 256), (512, 512, 512)])
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


# matmul-performance-fp16(triton 3.0 数据):
#          M       N       K      mcBLAS      Triton
# 0    256.0   256.0   256.0    1.820444    0.732246
# 1    384.0   384.0   384.0    6.059835    2.096531
# 2    512.0   512.0   512.0   12.945383    4.177594
# 3    640.0   640.0   640.0   20.897960    6.918919
# 4    768.0   768.0   768.0   34.695530   10.595640
# 5    896.0   896.0   896.0   51.556991   15.147472
# 6   1024.0  1024.0  1024.0   62.601551   20.410239
# 7   1152.0  1152.0  1152.0   81.807778   26.483228
# 8   1280.0  1280.0  1280.0  104.356686   33.233264
# 9   1408.0  1408.0  1408.0   92.013097   40.990799
# 10  1536.0  1536.0  1536.0  104.086586   49.495721
# 11  1664.0  1664.0  1664.0  122.019144   58.912681
# 12  1792.0  1792.0  1792.0  142.722839   69.272259
# 13  1920.0  1920.0  1920.0  122.066229   80.139129
# 14  2048.0  2048.0  2048.0  139.810137   91.929947
# 15  2176.0  2176.0  2176.0  160.029014  104.132718
# 16  2304.0  2304.0  2304.0  154.115302  117.097414
# 17  2432.0  2432.0  2432.0  172.623432  131.129353
# 18  2560.0  2560.0  2560.0  190.511628  145.797557
# 19  2688.0  2688.0  2688.0  158.715715   84.530485
# 20  2816.0  2816.0  2816.0  177.293521   92.944504
# 21  2944.0  2944.0  2944.0  164.475349  102.070728
# 22  3072.0  3072.0  3072.0  198.503440  111.407970
# 23  3200.0  3200.0  3200.0  171.122997  121.269540
# 24  3328.0  3328.0  3328.0  180.542431  131.370982
# 25  3456.0  3456.0  3456.0  189.474891  141.441354
# 26  3584.0  3584.0  3584.0  203.543621  152.528229
# 27  3712.0  3712.0  3712.0  182.878424  111.058751
# 28  3840.0  3840.0  3840.0  194.362030  118.724639
# 29  3968.0  3968.0  3968.0  189.037848  127.141380
# 30  4096.0  4096.0  4096.0  220.752852  135.573467
