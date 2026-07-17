import pytest
import torch
import triton
import argparse
from triton_kernels.matmul_ampere_gluon import matmul
from triton_kernels.matmul_ampere_gluon import is_ampere_or_newer

BENCHMARK_SIZES = [128 * i for i in range(2, 33)]
BENCHMARK_SHAPES = [(size, size, size) for size in BENCHMARK_SIZES]

@pytest.mark.skipif(not is_ampere_or_newer(), reason="Requires NVIDIA Ampere-or-newer CUDA target")
@pytest.mark.parametrize("M, N, K", [(128, 128, 128), (256, 256, 256)])
def test_ampere_matmul(M, N, K):
    torch.manual_seed(0)
    a = torch.randn((M, K), device="cuda", dtype=torch.float16)
    b = torch.randn((K, N), device="cuda", dtype=torch.float16).transpose(0, 1)
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
    b = torch.randn((K, N), device="cuda", dtype=torch.float16).transpose(0, 1)
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
