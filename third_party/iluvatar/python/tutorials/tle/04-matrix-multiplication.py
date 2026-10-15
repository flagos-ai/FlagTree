"""
TLE GEMM benchmark
==================

A TLE shared-memory GEMM with ``nv_mma_shared_layout=True`` so Iluvatar can
use TCU shared encoding and SME global-to-shared copies.

Default: fp16, M=N=K=4096, row-major. Set ``TLE_GEMM_FULL=1`` for bf16 / fp32
/ int8, col-major A/B, and the 256..4096 square sweep from
``performance_test/03-matrix-multiplication-autotune.py``.
"""

import os

import torch
import triton
import triton.language as tl
import triton.experimental.tle.language as tle
from triton.experimental.tle.language.gpu.iluvatar import layout as iluvatar_layout
from triton.ops import matmul as triton_mm
from triton.ops.matmul import get_configs_compute_bound
from triton.ops.matmul_perf_model import early_config_prune, estimate_matmul_time

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

TLE_GEMM_FULL = os.getenv("TLE_GEMM_FULL", "0") == "1"
SQUARE_SHAPES = [128 * i for i in range(2, 33)] if TLE_GEMM_FULL else [4096]

# name, torch_in, torch_out, tl_in, tl_acc, tl_out, use_ieee
_DTYPE_SPECS = [
    ("float16", torch.float16, torch.float16, tl.float16, tl.float32, tl.float16, False),
    ("bfloat16", torch.bfloat16, torch.bfloat16, tl.bfloat16, tl.float32, tl.bfloat16, False),
    ("float32", torch.float32, torch.float32, tl.float32, tl.float32, tl.float32, True),
    ("int8", torch.int8, torch.int8, tl.int8, tl.int32, tl.int8, False),
]
_LAYOUTS = [
    (False, False, "row"),
    (True, False, "col_a"),
    (False, True, "col_b"),
    (True, True, "col_ab"),
]


def _autotune_configs(tl_dtype, col_a, col_b):
    configs = []
    for cfg in get_configs_compute_bound():
        kw = dict(cfg.kwargs)
        a_layout = (iluvatar_layout.make_tcu_swizzled_layout([kw['BLOCK_M'], kw['BLOCK_K']], tl_dtype, True)
                    if col_a else None)
        b_layout = (iluvatar_layout.make_tcu_swizzled_layout([kw['BLOCK_K'], kw['BLOCK_N']], tl_dtype, True)
                    if col_b else None)
        configs.append(
            triton.Config({**kw, 'A_LAYOUT': a_layout, 'B_LAYOUT': b_layout}, num_warps=cfg.num_warps,
                          num_stages=cfg.num_stages, num_ctas=cfg.num_ctas))
    return configs


def _make_tle_gemm_kernel(tl_dtype, acc_tl_dtype, out_tl_dtype, use_ieee_flag, col_a, col_b):
    use_ieee = tl.constexpr(use_ieee_flag)

    @triton.autotune(
        configs=_autotune_configs(tl_dtype, col_a, col_b),
        key=['M', 'N', 'K'],
        prune_configs_by={
            'early_config_prune': early_config_prune,
            'perf_model': estimate_matmul_time,
            'top_k': 15,
        },
    )
    @triton.jit
    def _kernel(
        A,
        B,
        C,
        M,
        N,
        K,
        stride_am,
        stride_ak,
        stride_bk,
        stride_bn,
        stride_cm,
        stride_cn,
        A_LAYOUT: tl.constexpr,
        B_LAYOUT: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_K: tl.constexpr,
        SPLIT_K: tl.constexpr,
    ):
        pid = tl.program_id(axis=0)
        grid_n = tl.cdiv(N, BLOCK_N)
        pid_m = pid // grid_n
        pid_n = pid % grid_n

        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

        accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=acc_tl_dtype)
        a_smem = tle.gpu.alloc([BLOCK_M, BLOCK_K], dtype=tl_dtype, layout=A_LAYOUT, scope=tle.gpu.smem,
                               nv_mma_shared_layout=True)
        b_smem = tle.gpu.alloc([BLOCK_K, BLOCK_N], dtype=tl_dtype, layout=B_LAYOUT, scope=tle.gpu.smem,
                               nv_mma_shared_layout=True)

        a_row_ids = tl.broadcast_to(tl.arange(0, BLOCK_M)[:, None], (BLOCK_M, BLOCK_K))
        a_col_ids = tl.broadcast_to(tl.arange(0, BLOCK_K)[None, :], (BLOCK_M, BLOCK_K))
        b_row_ids = tl.broadcast_to(tl.arange(0, BLOCK_K)[:, None], (BLOCK_K, BLOCK_N))
        b_col_ids = tl.broadcast_to(tl.arange(0, BLOCK_N)[None, :], (BLOCK_K, BLOCK_N))
        a_smem_ptrs = tle.gpu.local_ptr(a_smem, (a_row_ids, a_col_ids))
        b_smem_ptrs = tle.gpu.local_ptr(b_smem, (b_row_ids, b_col_ids))

        for k_start in range(0, K, BLOCK_K):
            k_offs = k_start + tl.arange(0, BLOCK_K)
            As = A + offs_m[:, None] * stride_am + k_offs[None, :] * stride_ak
            Bs = B + k_offs[:, None] * stride_bk + offs_n[None, :] * stride_bn
            tle.gpu.copy(As, a_smem, [BLOCK_M, BLOCK_K])
            tle.gpu.copy(Bs, b_smem, [BLOCK_K, BLOCK_N])
            a_tile = tl.load(a_smem_ptrs)
            b_tile = tl.load(b_smem_ptrs)
            if use_ieee:
                accumulator = tl.dot(a_tile, b_tile, accumulator, input_precision="ieee")
            else:
                accumulator = tl.dot(a_tile, b_tile, accumulator, out_dtype=acc_tl_dtype)

        Cs = C + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
        tl.store(Cs, accumulator.to(out_tl_dtype), mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))

    return _kernel


_GEMM_KERNELS = {(name, col_a, col_b): _make_tle_gemm_kernel(tl_in, tl_acc, tl_out, use_ieee, col_a, col_b)
                 for name, _, _, tl_in, tl_acc, tl_out, use_ieee in _DTYPE_SPECS
                 for col_a, col_b, _ in _LAYOUTS}


def _init_operand(m, n, torch_dtype):
    if torch_dtype == torch.int8:
        return torch.randint(-8, 8, (m, n), device="cuda", dtype=torch.int8).contiguous()
    return torch.randn((m, n), device="cuda", dtype=torch_dtype).contiguous()


def _make_operands(M, N, K, torch_in, col_a, col_b):
    a = _init_operand(K, M, torch_in).T if col_a else _init_operand(M, K, torch_in)
    b = _init_operand(N, K, torch_in).T if col_b else _init_operand(K, N, torch_in)
    return a, b


def _atol_rtol(torch_out):
    if torch_out == torch.float32:
        return 1e-4, 1e-4
    if torch_out in (torch.float16, torch.bfloat16):
        return 1e-2, 1e-2
    return 0, 0


def tle_gemm(a, b, c, dtype_name, col_a, col_b):
    assert a.shape[1] == b.shape[0], "Incompatible dimensions"
    M, K = a.shape
    Kb, N = b.shape
    assert K == Kb and c.shape == (M, N)
    grid = lambda META: (triton.cdiv(M, META['BLOCK_M']) * triton.cdiv(N, META['BLOCK_N']), )
    _GEMM_KERNELS[(dtype_name, col_a, col_b)][grid](a, b, c, M, N, K, a.stride(0), a.stride(1), b.stride(0),
                                                    b.stride(1), c.stride(0), c.stride(1))
    return c


def _cases():
    if TLE_GEMM_FULL:
        return [(spec, col_a, col_b, tag) for spec in _DTYPE_SPECS for col_a, col_b, tag in _LAYOUTS]
    spec = _DTYPE_SPECS[0]
    return [(spec, False, False, "row")]


def _check_accuracy(spec, col_a, col_b, n):
    name, torch_in, torch_out, *_ = spec
    a, b = _make_operands(n, n, n, torch_in, col_a, col_b)
    c = torch.empty((n, n), device="cuda", dtype=torch_out)
    tle_gemm(a, b, c, name, col_a, col_b)
    ref = torch.matmul(a.to(torch.float32), b.to(torch.float32)).to(torch_out)
    atol, rtol = _atol_rtol(torch_out)
    torch.testing.assert_close(c, ref, atol=atol, rtol=rtol)
    tt = triton_mm(a, b)
    assert tt.dtype == c.dtype, (tt.dtype, c.dtype)
    torch.testing.assert_close(c, tt, atol=atol, rtol=rtol)


def _make_benchmark(spec, col_a, col_b, tag):
    name, torch_in, torch_out, *_ = spec

    @triton.testing.perf_report(
        triton.testing.Benchmark(
            x_names=['M', 'N', 'K'],
            x_vals=SQUARE_SHAPES,
            line_arg='provider',
            line_vals=['ixblas', 'triton', 'tle'],
            line_names=["ixBLAS", "Triton", "TLE"],
            styles=[('green', '-'), ('blue', '-'), ('orange', '-')],
            ylabel="TFLOPS",
            plot_name=f"tle-square-shapes-matmul-{name}-{tag}",
            args={},
        ))
    def bench(M, N, K, provider):
        a, b = _make_operands(M, N, K, torch_in, col_a, col_b)
        quantiles = [0.5, 0.2, 0.8]
        if provider == 'ixblas':
            if torch_in == torch.int8:
                a_ref, b_ref = a.to(torch.float32), b.to(torch.float32)
                ms, min_ms, max_ms = triton.testing.do_bench(lambda: torch.matmul(a_ref, b_ref), quantiles=quantiles)
            else:
                ms, min_ms, max_ms = triton.testing.do_bench(lambda: torch.matmul(a, b), quantiles=quantiles)
        if provider == 'triton':
            ms, min_ms, max_ms = triton.testing.do_bench(lambda: triton_mm(a, b), quantiles=quantiles)
        if provider == 'tle':
            c = torch.empty((M, N), device='cuda', dtype=torch_out)
            ms, min_ms, max_ms = triton.testing.do_bench(lambda: tle_gemm(a, b, c, name, col_a, col_b),
                                                         quantiles=quantiles)
        perf = lambda ms: 2 * M * N * K * 1e-12 / (ms * 1e-3)
        return perf(ms), perf(max_ms), perf(min_ms)

    return bench


if __name__ == "__main__":
    mode = "full" if TLE_GEMM_FULL else "fp16-4096-row"
    print(f"==================== TLE GEMM benchmark ({mode}) ====================")
    torch.manual_seed(0)
    n = SQUARE_SHAPES[0]
    for spec, col_a, col_b, tag in _cases():
        print(f"---- {spec[0]} {tag} ----")
        _check_accuracy(spec, col_a, col_b, n)
        _make_benchmark(spec, col_a, col_b, tag).run(show_plots=True, print_data=True, save_path='.')
