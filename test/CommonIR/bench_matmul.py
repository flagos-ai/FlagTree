"""Benchmark: native_matmul vs matmul_double_buffer.

Usage:
    python bench_matmul.py [--M 1024] [--N 1024] [--K 1024]
                           [--warmup 5] [--rep 20]
                           [--no-check]
"""

import argparse
import sys
import time

import torch

# ---------------------------------------------------------------------------
# Import the two kernels under distinct names
# ---------------------------------------------------------------------------
# Both files live next to this script, so we add the directory to sys.path.
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import native_matmul as _native
import matmul_double_buffer as _db

# ---------------------------------------------------------------------------
# Shared defaults
# ---------------------------------------------------------------------------
_DEFAULT_M = 1024
_DEFAULT_N = 1024
_DEFAULT_K = 1024
_DEFAULT_WARMUP = 5
_DEFAULT_REP = 20


def get_number_cores():
    """Return the number of AI cores available, falling back to 24."""
    try:
        import torch_npu  # noqa: F401
        return torch.npu.get_device_properties(0).ai_core_num
    except Exception:
        return 24


# ---------------------------------------------------------------------------
# Timing helpers
# ---------------------------------------------------------------------------

class _env:
    """Context manager to temporarily set / unset an environment variable."""

    def __init__(self, key: str, value: str | None):
        self._key = key
        self._value = value      # None → remove the variable
        self._saved = None
        self._was_set = False

    def __enter__(self):
        self._was_set = self._key in os.environ
        self._saved = os.environ.get(self._key)
        if self._value is None:
            os.environ.pop(self._key, None)
        else:
            os.environ[self._key] = self._value

    def __exit__(self, *_):
        if self._was_set:
            os.environ[self._key] = self._saved
        else:
            os.environ.pop(self._key, None)


def _sync(device: str):
    """Synchronise the device so elapsed wall-time is accurate."""
    if device == "npu":
        torch.npu.synchronize()
    elif device == "cuda":
        torch.cuda.synchronize()


def _bench(fn, mat_a, mat_b, num_cores, device, warmup, rep):
    """Run *fn(mat_a, mat_b, num_cores)* and return median latency in ms."""
    # Warm-up: let the JIT compile and cache the kernel.
    for _ in range(warmup):
        _ = fn(mat_a, mat_b, num_cores)
    _sync(device)

    latencies = []
    for _ in range(rep):
        _sync(device)
        t0 = time.perf_counter()
        _ = fn(mat_a, mat_b, num_cores)
        _sync(device)
        t1 = time.perf_counter()
        latencies.append((t1 - t0) * 1e3)  # → ms

    latencies.sort()
    median = latencies[len(latencies) // 2]
    mean = sum(latencies) / len(latencies)
    minimum = latencies[0]
    maximum = latencies[-1]
    return median, mean, minimum, maximum


def _tflops(M, N, K, latency_ms):
    """Compute TFLOPS given matrix dimensions and latency in ms."""
    flops = 2.0 * M * N * K          # multiply-add counts as 2 ops
    return flops / (latency_ms * 1e-3) / 1e12


# ---------------------------------------------------------------------------
# Main benchmark
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Benchmark native_matmul vs matmul_double_buffer")
    parser.add_argument("--M", type=int, default=_DEFAULT_M)
    parser.add_argument("--N", type=int, default=_DEFAULT_N)
    parser.add_argument("--K", type=int, default=_DEFAULT_K)
    parser.add_argument("--num-cores", type=int, default=None)
    parser.add_argument("--warmup", type=int, default=_DEFAULT_WARMUP,
                        help="Number of warm-up iterations (default: %(default)s)")
    parser.add_argument("--rep", type=int, default=_DEFAULT_REP,
                        help="Number of timed iterations (default: %(default)s)")
    parser.add_argument("--no-check", action="store_true",
                        help="Skip correctness check against torch.matmul")
    args = parser.parse_args()

    M, N, K = args.M, args.N, args.K
    num_cores = args.num_cores or get_number_cores()

    device = "npu" if hasattr(torch, "npu") and torch.npu.is_available() else "cuda"
    torch.manual_seed(0)
    mat_a = torch.randn((M, K), dtype=torch.float16, device=device)
    mat_b = torch.randn((K, N), dtype=torch.float16, device=device)

    print(f"\n{'='*60}")
    print(f"  Matmul benchmark  M={M}  N={N}  K={K}")
    print(f"  device={device}  num_cores={num_cores}")
    print(f"  warmup={args.warmup}  rep={args.rep}")
    print(f"{'='*60}\n")

    # ---- optional correctness check ----------------------------------------
    if not args.no_check:
        ref = torch.matmul(mat_a.float(), mat_b.float()).to(torch.float16)
        # native: no COMMONIR_SKIP_CSE
        with _env("COMMONIR_SKIP_CSE", None):
            out_native = _native.call(mat_a, mat_b, num_cores)
        # double_buffer: requires COMMONIR_SKIP_CSE=1
        #with _env("COMMONIR_SKIP_CSE", "1"):
        with _env("USE_CUSTOM_COMPILE_OPT", "1"):
            out_db = _db.call(mat_a, mat_b, num_cores)
        for label, out in [("native", out_native), ("double_buffer", out_db)]:
            try:
                torch.testing.assert_close(ref, out, rtol=1e-2, atol=1e-2)
                print(f"  [{label}] correctness check PASSED")
            except AssertionError as e:
                print(f"  [{label}] correctness check FAILED: {e}")
        print()

    # ---- timing -------------------------------------------------------------
    # native runs without COMMONIR_SKIP_CSE; double_buffer requires it set to 1.
    _kernel_envs = [
        ("native_matmul      ", _native.call, None),
        ("double_buffer_matmul", _db.call,    "1"),
    ]
    results = {}
    for label, fn, skip_cse in _kernel_envs:
        with _env("COMMONIR_SKIP_CSE", skip_cse):
            median, mean, mn, mx = _bench(
                fn, mat_a, mat_b, num_cores, device,
                warmup=args.warmup, rep=args.rep)
        results[label] = median
        tfl = _tflops(M, N, K, median)
        print(f"  {label}  median={median:7.3f} ms  mean={mean:7.3f} ms  "
              f"min={mn:7.3f} ms  max={mx:7.3f} ms  |  {tfl:.3f} TFLOPS")

    # ---- summary ------------------------------------------------------------
    native_ms = results["native_matmul      "]
    db_ms = results["double_buffer_matmul"]
    speedup = native_ms / db_ms if db_ms > 0 else float("inf")
    print(f"\n  Speedup (double_buffer / native): {speedup:.3f}x")
    if speedup > 1:
        print(f"  double_buffer is {speedup:.3f}x faster than native")
    elif speedup < 1:
        print(f"  native is {1/speedup:.3f}x faster than double_buffer")
    else:
        print("  Both kernels have the same latency")
    print()


if __name__ == "__main__":
    main()
