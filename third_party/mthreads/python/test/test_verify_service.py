"""
Direct kernel verification for mthreads/MUSA.

Examples:
    python test_verify_service.py --chip moore --kernel gelu
    python test_verify_service.py --chip moore --all
    python test_verify_service.py --chip moore --kernel gelu --benchmark
"""

import argparse
import os
import sys
import time
from dataclasses import dataclass
from typing import Callable, Iterable

import torch
import torch.nn.functional as F
import triton
import triton.language as tl
import triton.testing as triton_testing

__test__ = False

CHIP_CONFIG = {
    "nvidia": {"device_keyword": "cuda", "desc": "NVIDIA CUDA"},
    "haiguang": {"device_keyword": "cuda", "desc": "Haiguang DCU (CUDA-compatible)"},
    "tianshu": {"device_keyword": "cuda", "desc": "Tianshu (CUDA-compatible)"},
    "muxi": {"device_keyword": "cuda", "desc": "Muxi (CUDA-compatible)"},
    "pingtouge": {"device_keyword": "cuda", "desc": "Pingtouge (CUDA-compatible)"},
    "huawei": {"device_keyword": "npu", "desc": "Huawei Ascend"},
    "moore": {"device_keyword": "musa", "desc": "Moore Threads MUSA"},
}


@triton.jit
def gelu_kernel(x_ptr, y_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    sqrt_2_over_pi = 0.7978845608028654
    coeff = 0.044715
    x3 = x * x * x
    inner = sqrt_2_over_pi * (x + coeff * x3)
    exp2x = tl.math.exp(2.0 * inner)
    tanh_inner = (exp2x - 1.0) / (exp2x + 1.0)
    y = 0.5 * x * (1.0 + tanh_inner)
    tl.store(y_ptr + offsets, y, mask=mask)


def gelu(x: torch.Tensor) -> torch.Tensor:
    n_elements = x.numel()
    y = torch.empty_like(x)
    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]), )
    gelu_kernel[grid](x, y, n_elements, BLOCK_SIZE=1024)
    return y


@triton.jit
def matmul_kernel(
    A_ptr,
    B_ptr,
    C_ptr,
    M,
    N,
    K,
    stride_am,
    stride_ak,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    rk = tl.arange(0, BLOCK_K)
    A = A_ptr + rm[:, None] * stride_am + rk[None, :] * stride_ak
    B = B_ptr + rk[:, None] * stride_bk + rn[None, :] * stride_bn
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(0, K, BLOCK_K):
        a = tl.load(A, mask=(rm[:, None] < M) & (rk[None, :] < K - k), other=0.0)
        b = tl.load(B, mask=(rk[:, None] < K - k) & (rn[None, :] < N), other=0.0)
        acc += tl.dot(a, b)
        A += BLOCK_K * stride_ak
        B += BLOCK_K * stride_bk
    C = C_ptr + rm[:, None] * stride_cm + rn[None, :] * stride_cn
    tl.store(C, acc, mask=(rm[:, None] < M) & (rn[None, :] < N))


def matmul(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    M, K1 = A.shape
    K2, N = B.shape
    if K1 != K2:
        raise ValueError(f"incompatible matmul shapes: {A.shape} and {B.shape}")
    C = torch.empty((M, N), device=A.device, dtype=A.dtype)
    grid = lambda meta: (triton.cdiv(M, meta["BLOCK_M"]), triton.cdiv(N, meta["BLOCK_N"]))
    matmul_kernel[grid](
        A,
        B,
        C,
        M,
        N,
        K1,
        A.stride(0),
        A.stride(1),
        B.stride(0),
        B.stride(1),
        C.stride(0),
        C.stride(1),
        BLOCK_M=64,
        BLOCK_N=64,
        BLOCK_K=32,
    )
    return C


@triton.jit
def add_kernel(x_ptr, y_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    tl.store(out_ptr + offsets, x + y, mask=mask)


def add(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    n_elements = x.numel()
    out = torch.empty_like(x)
    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]), )
    add_kernel[grid](x, y, out, n_elements, BLOCK_SIZE=1024)
    return out


@dataclass(frozen=True)
class KernelSpec:
    name: str
    desc: str
    runner: Callable[[str, str], int]
    benchmark: Callable[[str, str, int, int], None] | None = None


def _import_backend_extension(device_type: str) -> None:
    if device_type == "musa":
        try:
            import torch_musa  # noqa: F401
        except ImportError:
            pass
    elif device_type == "npu":
        try:
            import torch_npu  # noqa: F401
        except ImportError:
            pass


def _configure_device(chip: str, gpu_id: int) -> tuple[str, str]:
    device_type = CHIP_CONFIG[chip]["device_keyword"]
    if device_type == "musa":
        os.environ.setdefault("TRITON_DEFAULT_BACKEND", "mthreads")

    _import_backend_extension(device_type)
    device_interface = getattr(torch, device_type, None)
    if device_interface is None:
        raise RuntimeError(f"torch.{device_type} is not available")

    is_available = getattr(device_interface, "is_available", None)
    if callable(is_available) and not is_available():
        raise RuntimeError(f"{device_type} device is not available")

    if gpu_id >= 0:
        set_device = getattr(device_interface, "set_device", None)
        if callable(set_device):
            set_device(gpu_id)
        return device_type, f"{device_type}:{gpu_id}"
    return device_type, device_type


def _synchronize(device_type: str) -> None:
    device_interface = getattr(torch, device_type)
    synchronize = getattr(device_interface, "synchronize", None)
    if callable(synchronize):
        synchronize()


def _assert_close(
    label: str,
    actual: torch.Tensor,
    expected: torch.Tensor,
    *,
    rtol: float,
    atol: float,
) -> None:
    if torch.allclose(actual, expected, rtol=rtol, atol=atol):
        return
    diff = (actual.float() - expected.float()).abs().max().item()
    raise AssertionError(f"{label}: max_diff={diff:.6e}, rtol={rtol}, atol={atol}")


def _run_cases(
    name: str,
    cases: Iterable,
    run_case: Callable[[object], None],
) -> int:
    passed = 0
    for case in cases:
        t0 = time.time()
        run_case(case)
        elapsed = time.time() - t0
        print(f"    PASS {name} {case} ({elapsed:.2f}s)")
        passed += 1
    return passed


def run_gelu(device_type: str, device: str) -> int:
    shapes = [
        (1024, ),
        (4096, ),
        (16384, ),
        (65536, ),
        (262144, ),
        (1024, 1024),
        (4096, 4096),
    ]

    def case(shape):
        x = torch.randn(*shape, device=device, dtype=torch.float32)
        actual = gelu(x)
        expected = F.gelu(x, approximate="tanh")
        _synchronize(device_type)
        _assert_close(f"gelu shape={shape}", actual, expected, rtol=1e-3, atol=1e-5)

    return _run_cases("gelu", shapes, case)


def benchmark_gelu(device_type: str, device: str, warmup_ms: int, rep_ms: int) -> None:
    cases = [
        (torch.float16, (1024, )),
        (torch.float16, (256, 256)),
        (torch.float16, (4, 512, 512)),
        (torch.float16, (8, 16, 64, 64)),
        (torch.float32, (1024, )),
        (torch.float32, (256, 256)),
        (torch.float32, (4, 512, 512)),
        (torch.float32, (8, 16, 64, 64)),
        (torch.bfloat16, (1024, )),
        (torch.bfloat16, (256, 256)),
        (torch.bfloat16, (4, 512, 512)),
        (torch.bfloat16, (8, 16, 64, 64)),
    ]

    print("    dtype        shape              triton_ms  torch_ms  speedup")
    for dtype, shape in cases:
        x = torch.randn(*shape, device=device, dtype=dtype)
        triton_ms = triton_testing.do_bench(
            lambda: gelu(x),
            warmup=warmup_ms,
            rep=rep_ms,
            device_type=device_type,
        )
        torch_ms = triton_testing.do_bench(
            lambda: F.gelu(x, approximate="tanh"),
            warmup=warmup_ms,
            rep=rep_ms,
            device_type=device_type,
        )
        speedup = torch_ms / triton_ms if triton_ms > 0 else 0.0
        print(f"    {str(dtype):<12} {str(shape):<18} {triton_ms:9.4f} {torch_ms:8.4f} {speedup:7.3f}x")


def run_matmul(device_type: str, device: str) -> int:
    configs = [(128, 256, 128), (256, 512, 256), (512, 1024, 512)]

    def case(config):
        M, N, K = config
        torch.manual_seed(42)
        A = torch.randn(M, K, device=device, dtype=torch.float32)
        B = torch.randn(K, N, device=device, dtype=torch.float32)
        actual = matmul(A, B)
        expected = A @ B
        _synchronize(device_type)
        diff = (actual.float() - expected.float()).abs()
        max_diff = diff.max().item()
        ok = torch.allclose(actual, expected, rtol=1e-2, atol=1e-2)
        print(f"    ({M},{N},{K}): max_diff={max_diff:.6e} {'PASS' if ok else 'FAIL'}")
        if not ok:
            tile = diff[:64, :64]
            for row in range(0, 64, 16):
                print(f"      Row {row:2d} max diff: {tile[row].max().item():.4e}")
            raise AssertionError(f"matmul shape=({M},{N},{K}): max_diff={max_diff:.6e}, rtol=1e-2, atol=1e-2")

    return _run_cases("matmul", configs, case)


def run_add(device_type: str, device: str) -> int:
    sizes = [4096 * 4096, 8192 * 8192, 16384 * 4096]

    def case(size):
        x = torch.randn(size, device=device, dtype=torch.float32)
        y = torch.randn(size, device=device, dtype=torch.float32)
        actual = add(x, y)
        expected = x + y
        _synchronize(device_type)
        _assert_close(f"add size={size}", actual, expected, rtol=1e-5, atol=1e-8)

    return _run_cases("add", sizes, case)


KERNELS = {
    "gelu": KernelSpec(
        name="gelu",
        desc="GELU activation (Triton direct run)",
        runner=run_gelu,
        benchmark=benchmark_gelu,
    ),
    "matmul": KernelSpec(
        name="matmul",
        desc="FP32 tl.dot loop accumulation correctness",
        runner=run_matmul,
    ),
    "add": KernelSpec(
        name="add",
        desc="Vector addition correctness",
        runner=run_add,
    ),
}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run Triton kernels directly on the selected backend.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""Examples:
  python test_verify_service.py --chip moore --kernel gelu
  python test_verify_service.py --chip moore --all
  python test_verify_service.py --chip moore --kernel gelu --benchmark""",
    )
    parser.add_argument("--chip", default="moore", choices=list(CHIP_CONFIG.keys()), help="chip type")
    parser.add_argument("--gpu-id", type=int, default=-1, help="GPU ID")
    parser.add_argument("--all", action="store_true", help="run all kernels")
    parser.add_argument("--kernel", default="gelu", choices=list(KERNELS.keys()), help="kernel to run")
    parser.add_argument("--benchmark", action="store_true", help="also run benchmark for kernels that support it")
    parser.add_argument("--benchmark-warmup-ms", type=int, default=25, help="benchmark warmup window in ms")
    parser.add_argument("--benchmark-rep-ms", type=int, default=100, help="benchmark repeat window in ms")
    args = parser.parse_args()

    cfg = CHIP_CONFIG[args.chip]
    device_type, device = _configure_device(args.chip, args.gpu_id)
    kernels_to_run = list(KERNELS) if args.all else [args.kernel]

    print("=" * 72)
    print("  Direct Triton Kernel Test")
    print("=" * 72)
    print(f"  chip:   {args.chip} ({cfg['desc']})")
    print(f"  device: {device}")
    print()

    total_passed = 0
    try:
        for name in kernels_to_run:
            spec = KERNELS[name]
            print(f"{'-' * 72}")
            print(f"  [{spec.name}] {spec.desc}")
            print(f"{'-' * 72}")
            passed = spec.runner(device_type, device)
            total_passed += passed
            if args.benchmark and spec.benchmark is not None:
                print("    benchmark:")
                spec.benchmark(device_type, device, args.benchmark_warmup_ms, args.benchmark_rep_ms)
            elif args.benchmark:
                print("    benchmark: skipped (not available)")
            print()
    except Exception as exc:
        print(f"FAILED: {exc}", file=sys.stderr)
        sys.exit(1)

    print("=" * 72)
    print(f"  PASS: {total_passed} case(s)")
    print("=" * 72)


if __name__ == "__main__":
    main()
