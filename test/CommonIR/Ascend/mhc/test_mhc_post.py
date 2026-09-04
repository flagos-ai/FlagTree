"""Accuracy and performance tests for mhc_post TLE kernels.

Usage:
    # Run all tests
    python test_mhc_post.py

    # Run only accuracy tests
    python test_mhc_post.py --accuracy

    # Run only performance tests
    python test_mhc_post.py --perf

    # Specify device
    python test_mhc_post.py --device npu:0
"""

from __future__ import annotations

import argparse
import time
from typing import Tuple

import torch

import importlib
import sys

# Import old version from /data/yuansheng/tle_version/mhc_post.py
sys.path.insert(0, "/data/yuansheng/tle_version")
_old_mod = importlib.import_module("mhc_post")
mhc_post_old = _old_mod.mhc_post
sys.path.pop(0)

# Import new version from current directory (ensure it takes priority)
if "mhc_post" in sys.modules:
    del sys.modules["mhc_post"]
from mhc_post import mhc_post, mhc_post_ref  # noqa: E402


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _generate_inputs(
    T: int,
    N: int,
    D: int,
    dtype: torch.dtype = torch.bfloat16,
    device: str = "npu",
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Generate random inputs matching the mhc_post signature."""
    x = torch.randn(T, N, D, dtype=dtype, device=device)
    h_res = torch.randn(T, N, N, dtype=torch.float32, device=device)
    h_out = torch.randn(T, D, dtype=dtype, device=device)
    h_post = torch.randn(T, N, dtype=torch.float32, device=device)
    return x, h_res, h_out, h_post


def _generate_inputs_4d(
    B: int,
    S: int,
    N: int,
    D: int,
    dtype: torch.dtype = torch.bfloat16,
    device: str = "npu",
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Generate random 4D (B, S, N, D) inputs."""
    x = torch.randn(B, S, N, D, dtype=dtype, device=device)
    h_res = torch.randn(B, S, N, N, dtype=torch.float32, device=device)
    h_out = torch.randn(B, S, D, dtype=dtype, device=device)
    h_post = torch.randn(B, S, N, dtype=torch.float32, device=device)
    return x, h_res, h_out, h_post


# ---------------------------------------------------------------------------
# Accuracy tests
# ---------------------------------------------------------------------------
class AccuracyTests:
    """Numerical accuracy tests comparing TLE kernel vs PyTorch reference."""

    def __init__(self, device: str = "npu"):
        self.device = device
        self.passed = 0
        self.failed = 0

    def _check(
        self,
        name: str,
        out: torch.Tensor,
        ref: torch.Tensor,
        atol: float,
        rtol: float,
    ):
        """Check closeness and report."""
        max_diff = (out.float() - ref.float()).abs().max().item()
        cos_sim = torch.nn.functional.cosine_similarity(
            out.float().reshape(-1).unsqueeze(0),
            ref.float().reshape(-1).unsqueeze(0),
        ).item()
        ok = torch.allclose(out.float(), ref.float(), atol=atol, rtol=rtol)
        status = "PASS" if ok else "FAIL"
        if ok:
            self.passed += 1
        else:
            self.failed += 1
        print(f"  [{status}] {name:40s} "
              f"max_diff={max_diff:.6e}  cos_sim={cos_sim:.8f}")
        return ok

    def test_basic_shapes(self):
        """Test various (T, D) combinations."""
        print("\n=== Accuracy: Basic shapes (bf16) ===")
        cases = [
            (1, 4, 64),
            (1, 4, 128),
            (4, 4, 256),
            (16, 4, 512),
            (32, 4, 1024),
            (64, 4, 2048),
            (128, 4, 4096),
        ]
        for T, N, D in cases:
            x, h_res, h_out, h_post = _generate_inputs(T, N, D, torch.bfloat16, self.device)
            out = mhc_post(x, h_res, h_out, h_post)
            ref = mhc_post_ref(x, h_res, h_out, h_post)
            self._check(f"T={T:4d} N={N} D={D:5d} bf16", out, ref, atol=1e-2, rtol=1e-2)

    def test_fp16(self):
        """Test with float16 dtype."""
        print("\n=== Accuracy: fp16 ===")
        cases = [
            (8, 4, 256),
            (32, 4, 1024),
            (64, 4, 2048),
        ]
        for T, N, D in cases:
            x, h_res, h_out, h_post = _generate_inputs(T, N, D, torch.float16, self.device)
            out = mhc_post(x, h_res, h_out, h_post)
            ref = mhc_post_ref(x, h_res, h_out, h_post)
            self._check(f"T={T:4d} N={N} D={D:5d} fp16", out, ref, atol=1e-2, rtol=1e-2)

    def test_non_power_of_2_D(self):
        """Test with D that is not a power of 2 (tail masking)."""
        print("\n=== Accuracy: Non-power-of-2 D (tail masking) ===")
        cases = [
            (4, 4, 100),
            (8, 4, 200),
            (16, 4, 300),
            (32, 4, 500),
            (64, 4, 768),
            (32, 4, 1000),
            (16, 4, 1536),
            (8, 4, 3000),
        ]
        for T, N, D in cases:
            x, h_res, h_out, h_post = _generate_inputs(T, N, D, torch.bfloat16, self.device)
            out = mhc_post(x, h_res, h_out, h_post)
            ref = mhc_post_ref(x, h_res, h_out, h_post)
            self._check(f"T={T:4d} N={N} D={D:5d} bf16", out, ref, atol=1e-2, rtol=1e-2)

    def test_4d_input(self):
        """Test with 4D (B, S, N, D) input."""
        print("\n=== Accuracy: 4D input (B, S, N, D) ===")
        cases = [
            (1, 16, 4, 512),
            (2, 8, 4, 1024),
            (4, 32, 4, 2048),
        ]
        for B, S, N, D in cases:
            x, h_res, h_out, h_post = _generate_inputs_4d(B, S, N, D, torch.bfloat16, self.device)
            out = mhc_post(x, h_res, h_out, h_post)
            ref = mhc_post_ref(x, h_res, h_out, h_post)
            self._check(
                f"B={B} S={S:3d} N={N} D={D:5d} bf16",
                out,
                ref,
                atol=1e-2,
                rtol=1e-2,
            )

    def test_large_shapes(self):
        """Test with large typical shapes (B, S, N=4, D=3584)."""
        print("\n=== Accuracy: Large shapes (B, S, 4, 3584) bf16 ===")
        cases = [
            (1, 64, 4, 3584),
            (1, 256, 4, 3584),
            (2, 1024, 4, 3584),
            (1, 4096, 4, 3584),
            (1, 8192, 4, 3584),
        ]
        for B, S, N, D in cases:
            x, h_res, h_out, h_post = _generate_inputs_4d(B, S, N, D, torch.bfloat16, self.device)
            out = mhc_post(x, h_res, h_out, h_post)
            ref = mhc_post_ref(x, h_res, h_out, h_post)
            self._check(
                f"B={B} S={S:5d} N={N} D={D} bf16",
                out,
                ref,
                atol=1e-2,
                rtol=1e-2,
            )

    def test_large_T(self):
        """Test with large batch to check grid dimension handling."""
        print("\n=== Accuracy: Large T ===")
        cases = [
            (256, 4, 1024),
            (512, 4, 512),
            (1024, 4, 256),
        ]
        for T, N, D in cases:
            x, h_res, h_out, h_post = _generate_inputs(T, N, D, torch.bfloat16, self.device)
            out = mhc_post(x, h_res, h_out, h_post)
            ref = mhc_post_ref(x, h_res, h_out, h_post)
            self._check(f"T={T:4d} N={N} D={D:5d} bf16", out, ref, atol=1e-2, rtol=1e-2)

    def test_numerical_edge_cases(self):
        """Test with edge-case values (zeros, large values, etc.)."""
        print("\n=== Accuracy: Numerical edge cases ===")
        T, N, D = 16, 4, 512

        # All zeros
        x = torch.zeros(T, N, D, dtype=torch.bfloat16, device=self.device)
        h_res = torch.zeros(T, N, N, dtype=torch.float32, device=self.device)
        h_out = torch.zeros(T, D, dtype=torch.bfloat16, device=self.device)
        h_post = torch.zeros(T, N, dtype=torch.float32, device=self.device)
        out = mhc_post(x, h_res, h_out, h_post)
        ref = mhc_post_ref(x, h_res, h_out, h_post)
        self._check("all_zeros", out, ref, atol=1e-6, rtol=0)

        # Identity-like h_res (diagonal = 1)
        x, h_res, h_out, h_post = _generate_inputs(T, N, D, torch.bfloat16, self.device)
        h_res = torch.zeros(T, N, N, dtype=torch.float32, device=self.device)
        for j in range(N):
            h_res[:, j, j] = 1.0
        h_post.zero_()
        out = mhc_post(x, h_res, h_out, h_post)
        ref = mhc_post_ref(x, h_res, h_out, h_post)
        self._check("identity_hres_zero_hpost", out, ref, atol=1e-2, rtol=1e-2)

        # Large values
        x = torch.randn(T, N, D, dtype=torch.bfloat16, device=self.device) * 100
        h_res = torch.randn(T, N, N, dtype=torch.float32, device=self.device) * 10
        h_out = torch.randn(T, D, dtype=torch.bfloat16, device=self.device) * 100
        h_post = torch.randn(T, N, dtype=torch.float32, device=self.device) * 10
        out = mhc_post(x, h_res, h_out, h_post)
        ref = mhc_post_ref(x, h_res, h_out, h_post)
        self._check("large_values", out, ref, atol=5e-0, rtol=5e-2)

    def run_all(self):
        """Run all accuracy tests."""
        self.test_basic_shapes()
        self.test_fp16()
        self.test_non_power_of_2_D()
        self.test_4d_input()
        self.test_large_shapes()
        self.test_large_T()
        self.test_numerical_edge_cases()
        print(f"\n--- Accuracy Summary: {self.passed} passed, {self.failed} failed ---")
        return self.failed == 0


# ---------------------------------------------------------------------------
# Performance tests
# ---------------------------------------------------------------------------
class PerfTests:
    """Performance benchmarks for mhc_post kernel."""

    def __init__(self, device: str = "npu", warmup: int = 10, repeat: int = 100):
        self.device = device
        self.warmup = warmup
        self.repeat = repeat

    def _bench(self, fn, *args) -> float:
        """Benchmark a function, return avg time in ms."""
        # Warmup
        for _ in range(self.warmup):
            fn(*args)
        if self.device.startswith("npu"):
            torch.npu.synchronize()
        elif self.device.startswith("cuda"):
            torch.cuda.synchronize()

        # Timed runs
        start = time.perf_counter()
        for _ in range(self.repeat):
            fn(*args)
        if self.device.startswith("npu"):
            torch.npu.synchronize()
        elif self.device.startswith("cuda"):
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - start

        return elapsed / self.repeat * 1000  # ms

    def _bandwidth(self, T: int, N: int, D: int, dtype: torch.dtype, time_ms: float) -> float:
        """Compute effective bandwidth in GB/s.

        Data moved: read x(T,4,D) + h_res(T,4,4)*4B + h_out(T,D) + h_post(T,4)*4B
                  + write out(T,4,D)
        """
        elem_bytes = torch.finfo(dtype).bits // 8
        read_bytes = (
            T * N * D * elem_bytes  # x
            + T * N * N * 4  # h_res (fp32)
            + T * D * elem_bytes  # h_out
            + T * N * 4  # h_post (fp32)
        )
        write_bytes = T * N * D * elem_bytes  # out
        total_bytes = read_bytes + write_bytes
        return total_bytes / (time_ms * 1e-3) / 1e9  # GB/s

    def test_sweep_D(self):
        """Sweep D dimension with fixed T."""
        print("\n=== Performance: Sweep D (T=128, N=4, bf16) ===")
        print(f"{'D':>6s} {'kernel(ms)':>12s} {'ref(ms)':>12s} {'speedup':>8s} {'BW(GB/s)':>10s}")
        print("-" * 52)
        T, N = 128, 4
        for D in [128, 256, 512, 1024, 2048, 4096]:
            x, h_res, h_out, h_post = _generate_inputs(T, N, D, torch.bfloat16, self.device)
            t_kernel = self._bench(mhc_post, x, h_res, h_out, h_post)
            t_ref = self._bench(mhc_post_ref, x, h_res, h_out, h_post)
            bw = self._bandwidth(T, N, D, torch.bfloat16, t_kernel)
            speedup = t_ref / t_kernel if t_kernel > 0 else float("inf")
            print(f"{D:>6d} {t_kernel:>12.4f} {t_ref:>12.4f} {speedup:>7.2f}x {bw:>9.2f}")

    def test_sweep_T(self):
        """Sweep T (batch) dimension with fixed D."""
        print("\n=== Performance: Sweep T (D=1024, N=4, bf16) ===")
        print(f"{'T':>6s} {'kernel(ms)':>12s} {'ref(ms)':>12s} {'speedup':>8s} {'BW(GB/s)':>10s}")
        print("-" * 52)
        N, D = 4, 1024
        for T in [1, 4, 16, 64, 128, 256, 512]:
            x, h_res, h_out, h_post = _generate_inputs(T, N, D, torch.bfloat16, self.device)
            t_kernel = self._bench(mhc_post, x, h_res, h_out, h_post)
            t_ref = self._bench(mhc_post_ref, x, h_res, h_out, h_post)
            bw = self._bandwidth(T, N, D, torch.bfloat16, t_kernel)
            speedup = t_ref / t_kernel if t_kernel > 0 else float("inf")
            print(f"{T:>6d} {t_kernel:>12.4f} {t_ref:>12.4f} {speedup:>7.2f}x {bw:>9.2f}")

    def test_typical_workloads(self):
        """Benchmark typical inference/training shapes."""
        print("\n=== Performance: Typical workloads (bf16) ===")
        print(f"{'shape':>24s} {'kernel(ms)':>12s} {'ref(ms)':>12s} "
              f"{'speedup':>8s} {'BW(GB/s)':>10s}")
        print("-" * 70)
        workloads = [
            # (T, N, D) — typical transformer shapes
            (32, 4, 768),  # small model
            (64, 4, 1024),  # medium
            (128, 4, 2048),  # large
            (256, 4, 4096),  # XL
            (512, 4, 1024),  # long sequence, medium dim
            (1024, 4, 512),  # very long sequence
        ]
        for T, N, D in workloads:
            x, h_res, h_out, h_post = _generate_inputs(T, N, D, torch.bfloat16, self.device)
            t_kernel = self._bench(mhc_post, x, h_res, h_out, h_post)
            t_ref = self._bench(mhc_post_ref, x, h_res, h_out, h_post)
            bw = self._bandwidth(T, N, D, torch.bfloat16, t_kernel)
            speedup = t_ref / t_kernel if t_kernel > 0 else float("inf")
            shape_str = f"({T},{N},{D})"
            print(f"{shape_str:>24s} {t_kernel:>12.4f} {t_ref:>12.4f} "
                  f"{speedup:>7.2f}x {bw:>9.2f}")

    def test_dtype_comparison(self):
        """Compare bf16 vs fp16 performance."""
        print("\n=== Performance: bf16 vs fp16 (T=128, N=4, D=2048) ===")
        print(f"{'dtype':>8s} {'kernel(ms)':>12s} {'BW(GB/s)':>10s}")
        print("-" * 34)
        T, N, D = 128, 4, 2048
        for dtype in [torch.bfloat16, torch.float16]:
            x, h_res, h_out, h_post = _generate_inputs(T, N, D, dtype, self.device)
            t_kernel = self._bench(mhc_post, x, h_res, h_out, h_post)
            bw = self._bandwidth(T, N, D, dtype, t_kernel)
            dtype_str = "bf16" if dtype == torch.bfloat16 else "fp16"
            print(f"{dtype_str:>8s} {t_kernel:>12.4f} {bw:>9.2f}")

    def test_large_shapes(self):
        """Benchmark large typical shapes (B, S, 4, 3584)."""
        print("\n=== Performance: Large shapes (B, S, 4, 3584) bf16 ===")
        print(f"{'(B,S,N,D)':>24s} {'kernel(ms)':>12s} {'ref(ms)':>12s} "
              f"{'speedup':>8s} {'BW(GB/s)':>10s}")
        print("-" * 70)
        workloads = [
            (1, 64, 4, 3584),
            (1, 256, 4, 3584),
            (2, 1024, 4, 3584),
            (1, 4096, 4, 3584),
            (1, 8192, 4, 3584),
        ]
        for B, S, N, D in workloads:
            T = B * S
            x, h_res, h_out, h_post = _generate_inputs_4d(B, S, N, D, torch.bfloat16, self.device)
            t_kernel = self._bench(mhc_post, x, h_res, h_out, h_post)
            t_ref = self._bench(mhc_post_ref, x, h_res, h_out, h_post)
            bw = self._bandwidth(T, N, D, torch.bfloat16, t_kernel)
            speedup = t_ref / t_kernel if t_kernel > 0 else float("inf")
            shape_str = f"({B},{S},{N},{D})"
            print(f"{shape_str:>24s} {t_kernel:>12.4f} {t_ref:>12.4f} "
                  f"{speedup:>7.2f}x {bw:>9.2f}")

    def test_pipeline_vs_nopipeline(self):
        """Compare pipeline kernel vs non-pipeline kernel on large shapes."""
        print("\n=== Performance: Pipeline vs Non-Pipeline (B, S, 4, 3584) bf16 ===")
        print(f"{'(B,S,N,D)':>24s} {'pipeline(ms)':>14s} {'no-pipe(ms)':>14s} "
              f"{'pipe_speedup':>13s} {'pipe BW(GB/s)':>14s}")
        print("-" * 83)
        workloads = [
            (1, 64, 4, 3584),
            (1, 256, 4, 3584),
            (2, 1024, 4, 3584),
            (1, 4096, 4, 3584),
            (1, 8192, 4, 3584),
        ]
        for B, S, N, D in workloads:
            T = B * S
            x, h_res, h_out, h_post = _generate_inputs_4d(B, S, N, D, torch.bfloat16, self.device)
            # Pipeline version (use_pipeline=True, default)
            t_pipe = self._bench(
                lambda a, b, c, d: mhc_post(a, b, c, d, use_pipeline=True),
                x,
                h_res,
                h_out,
                h_post,
            )
            # Non-pipeline version (use_pipeline=False)
            t_nopipe = self._bench(
                lambda a, b, c, d: mhc_post(a, b, c, d, use_pipeline=False),
                x,
                h_res,
                h_out,
                h_post,
            )
            bw_pipe = self._bandwidth(T, N, D, torch.bfloat16, t_pipe)
            speedup = t_nopipe / t_pipe if t_pipe > 0 else float("inf")
            shape_str = f"({B},{S},{N},{D})"
            print(f"{shape_str:>24s} {t_pipe:>14.4f} {t_nopipe:>14.4f} "
                  f"{speedup:>12.2f}x {bw_pipe:>13.2f}")

    def test_new_vs_old(self):
        """Compare new (op/tle_version) vs old (tle_version) mhc_post."""
        print("\n=== Performance: New vs Old mhc_post (B, S, 4, 3584) bf16 ===")
        print(f"{'(B,S,N,D)':>24s} {'new(ms)':>12s} {'old(ms)':>12s} "
              f"{'speedup':>10s} {'new BW(GB/s)':>14s}")
        print("-" * 76)
        workloads = [
            (1, 64, 4, 3584),
            (1, 256, 4, 3584),
            (2, 1024, 4, 3584),
            (1, 4096, 4, 3584),
            (1, 8192, 4, 3584),
        ]
        for B, S, N, D in workloads:
            T = B * S
            x, h_res, h_out, h_post = _generate_inputs_4d(B, S, N, D, torch.bfloat16, self.device)
            t_new = self._bench(mhc_post, x, h_res, h_out, h_post)
            t_old = self._bench(mhc_post_old, x, h_res, h_out, h_post)
            bw_new = self._bandwidth(T, N, D, torch.bfloat16, t_new)
            speedup = t_old / t_new if t_new > 0 else float("inf")
            shape_str = f"({B},{S},{N},{D})"
            print(f"{shape_str:>24s} {t_new:>12.4f} {t_old:>12.4f} "
                  f"{speedup:>9.2f}x {bw_new:>13.2f}")

    def run_all(self):
        """Run all performance tests."""
        self.test_sweep_D()
        self.test_sweep_T()
        self.test_typical_workloads()
        self.test_dtype_comparison()
        self.test_large_shapes()
        self.test_pipeline_vs_nopipeline()
        self.test_new_vs_old()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="mhc_post accuracy & performance tests")
    parser.add_argument("--device", type=str, default="npu", help="Device to run on (default: npu)")
    parser.add_argument("--accuracy", action="store_true", help="Run only accuracy tests")
    parser.add_argument("--perf", action="store_true", help="Run only performance tests")
    parser.add_argument("--warmup", type=int, default=10, help="Warmup iterations for perf (default: 10)")
    parser.add_argument("--repeat", type=int, default=100, help="Repeat iterations for perf (default: 100)")
    args = parser.parse_args()

    run_both = not args.accuracy and not args.perf

    print(f"Device: {args.device}")
    print("=" * 70)

    if args.accuracy or run_both:
        acc = AccuracyTests(device=args.device)
        acc_ok = acc.run_all()

    if args.perf or run_both:
        perf = PerfTests(device=args.device, warmup=args.warmup, repeat=args.repeat)
        perf.run_all()

    if (args.accuracy or run_both) and not acc_ok:
        print("\n*** ACCURACY TESTS FAILED ***")
        exit(1)

    print("\nDone.")


if __name__ == "__main__":
    main()
