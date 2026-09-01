import os

import triton
import torch
from triton.ops import matmul as triton_mm
from triton.ops import bmm as triton_bmm
from utils import PERF_MM_SHAPES, PERF_BMM_SHAPES, IXBLAS_SHAPES_FP32, IXBLAS_SHAPES_FP16

DTYPE = torch.float16
TRITON_PERF_WITH_FULL_MODE = (os.getenv("TRITON_PERF_WITH_FULL_MODE", default='0') == '1')

print(f"==================== 1. square shapes matmul performance ====================")


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=['M', 'N', 'K'],  # Argument names to use as an x-axis for the plot
        x_vals=[128 * i for i in range(2, 33)],  # Different possible values for `x_name`
        line_arg='provider',  # Argument name whose value corresponds to a different line in the plot
        # Possible values for `line_arg`
        line_vals=['ixblas', 'triton'],
        # Label name for the lines
        line_names=["ixBLAS", "Triton"],
        # Line styles
        styles=[('green', '-'), ('blue', '-')],
        ylabel="TFLOPS",  # Label name for the y-axis
        plot_name="square-shapes-matmul-performance",  # Name for the plot, used also as a file name for saving the plot.
        args={},
    ))
def benchmark_square_shapes_mm_fp16(M, N, K, provider):
    a = torch.randn((M, K), device='cuda', dtype=DTYPE)
    b = torch.randn((K, N), device='cuda', dtype=DTYPE)
    quantiles = [0.5, 0.2, 0.8]
    if provider == 'ixblas':
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: torch.matmul(a, b), quantiles=quantiles)
    if provider == 'triton':
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: triton_mm(a, b), quantiles=quantiles)
    perf = lambda ms: 2 * M * N * K * 1e-12 / (ms * 1e-3)
    return perf(ms), perf(max_ms), perf(min_ms)


benchmark_square_shapes_mm_fp16.run(show_plots=True, print_data=True, save_path='.')

print(f"==================== 2. model shapes matmul performance ====================")
MODEL_SHAPES = sorted(set(PERF_MM_SHAPES + IXBLAS_SHAPES_FP16))
if not TRITON_PERF_WITH_FULL_MODE:
    import random
    random.seed(123)
    perf_nums = 20
    random_numbers = random.sample(range(0, len(MODEL_SHAPES)), perf_nums)
    MODEL_SHAPES = sorted([MODEL_SHAPES[idx] for idx in random_numbers])


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=['M', 'N', 'K'],  # Argument names to use as an x-axis for the plot
        x_vals=[(m, n, k) for m, n, k in MODEL_SHAPES],  # Different possible values for `x_name`
        line_arg='provider',  # Argument name whose value corresponds to a different line in the plot
        # Possible values for `line_arg`
        line_vals=['ixblas', 'triton'],
        # Label name for the lines
        line_names=["ixBLAS", "Triton"],
        # Line styles
        styles=[('green', '-'), ('blue', '-')],
        ylabel="TFLOPS",  # Label name for the y-axis
        plot_name="model-shapes-matmul-performance",  # Name for the plot, used also as a file name for saving the plot.
        args={},
    ))
def benchmark_model_shapes_mm_fp16(M, N, K, provider):
    a = torch.randn((M, K), device='cuda', dtype=DTYPE)
    b = torch.randn((K, N), device='cuda', dtype=DTYPE)
    quantiles = [0.5, 0.2, 0.8]
    if provider == 'ixblas':
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: torch.matmul(a, b), quantiles=quantiles)
    if provider == 'triton':
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: triton_mm(a, b), quantiles=quantiles)
    perf = lambda ms: 2 * M * N * K * 1e-12 / (ms * 1e-3)
    return perf(ms), perf(max_ms), perf(min_ms)


benchmark_model_shapes_mm_fp16.run(show_plots=True, print_data=True, save_path='.')

print(f"==================== 3. model shapes bmm performance ====================")


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=['B', 'M', 'N', 'K'],  # Argument names to use as an x-axis for the plot
        x_vals=[(b, m, n, k) for b, m, n, k in sorted(set(PERF_BMM_SHAPES))],  # Different possible values for `x_name`
        line_arg='provider',  # Argument name whose value corresponds to a different line in the plot
        # Possible values for `line_arg`
        line_vals=['ixblas', 'triton'],
        # Label name for the lines
        line_names=["ixBLAS", "Triton"],
        # Line styles
        styles=[('green', '-'), ('blue', '-')],
        ylabel="TFLOPS",  # Label name for the y-axis
        plot_name="model-shapes-bmm-performance",  # Name for the plot, used also as a file name for saving the plot.
        args={},
    ))
def benchmark_model_shapes_bmm_fp16(B, M, N, K, provider):
    a = torch.randn((B, M, K), device='cuda', dtype=DTYPE)
    b = torch.randn((B, K, N), device='cuda', dtype=DTYPE)
    quantiles = [0.5, 0.2, 0.8]
    if provider == 'ixblas':
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: torch.bmm(a, b), quantiles=quantiles)
    if provider == 'triton':
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: triton_bmm(a, b), quantiles=quantiles)
    perf = lambda ms: 2 * M * N * K * B * 1e-12 / (ms * 1e-3)
    return perf(ms), perf(max_ms), perf(min_ms)


benchmark_model_shapes_bmm_fp16.run(show_plots=True, print_data=True, save_path='.')
