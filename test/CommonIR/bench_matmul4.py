"""Benchmark: native_matmul vs matmul_double_buffer vs PyTorch.

Two timing modes
----------------
wall   (default)
    sync -> start -> kernel -> sync -> stop.
    Measures end-to-end latency including Python dispatch and kernel-launch
    overhead.  Uses time.perf_counter + device synchronize.

kernel
    Uses device events (NPU/CUDA) bracketing only the kernel execution,
    matching what triton.testing.do_bench does.  Eliminates host-side
    overhead so the numbers reflect pure hardware throughput.

Usage
-----
    python bench_matmul.py [--M 1024] [--N 1024] [--K 1024]
                           [--warmup 5] [--rep 20]
                           [--mode wall|kernel]
                           [--no-check]
"""

import argparse
import os
import sys
import time

import torch

# ---------------------------------------------------------------------------
# Import the custom kernels
# ---------------------------------------------------------------------------
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import native_matmul as _native          # noqa: E402
import matmul_double_buffer as _db       # noqa: E402
import flaggems_matmul as _flaggems      # noqa: E402

# ---------------------------------------------------------------------------
# Defaults
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
# Environment-variable context manager
# ---------------------------------------------------------------------------

class _env:
    """Temporarily set or remove an environment variable."""

    def __init__(self, key: str, value: str | None):
        self._key = key
        self._value = value   # None -> remove the variable
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


# ---------------------------------------------------------------------------
# Device helpers
# ---------------------------------------------------------------------------

def _sync(device: str):
    """Block until all pending device work is complete."""
    if device == "npu":
        torch.npu.synchronize()
    elif device == "cuda":
        torch.cuda.synchronize()


# ---------------------------------------------------------------------------
# Timing backends
# ---------------------------------------------------------------------------

def _stats(latencies):
    """Return (median, mean, min, max) from a list of ms samples."""
    s = sorted(latencies)
    median = s[len(s) // 2]
    mean = sum(s) / len(s)
    return median, mean, s[0], s[-1]


def _bench_wall(fn, mat_a, mat_b, num_cores, device, warmup, rep):
    """Wall-clock timing: includes Python dispatch and kernel-launch overhead.

    Measurement: sync -> perf_counter -> fn() -> sync -> perf_counter.
    """
    for _ in range(warmup):
        fn(mat_a, mat_b, num_cores)
    _sync(device)

    latencies = []
    for _ in range(rep):
        _sync(device)
        t0 = time.perf_counter()
        fn(mat_a, mat_b, num_cores)
        _sync(device)
        latencies.append((time.perf_counter() - t0) * 1e3)

    return _stats(latencies)


def _bench_kernel(fn, mat_a, mat_b, num_cores, warmup, rep):
    """Pure kernel timing via triton.testing.do_bench.

    Uses device events internally; eliminates Python dispatch and launch
    overhead.  Returns a single median latency in milliseconds.
    """
    import triton.testing
    return triton.testing.do_bench(
        lambda: fn(mat_a, mat_b, num_cores),
        warmup=warmup,
        rep=rep,
        return_mode="median",
    )


def _tflops(M, N, K, latency_ms):
    """Compute TFLOPS from matrix dimensions and latency in ms."""
    return 2.0 * M * N * K / (latency_ms * 1e-3) / 1e12


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Benchmark native_matmul vs matmul_double_buffer vs PyTorch",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--M", type=int, default=_DEFAULT_M)
    parser.add_argument("--N", type=int, default=_DEFAULT_N)
    parser.add_argument("--K", type=int, default=_DEFAULT_K)
    parser.add_argument("--num-cores", type=int, default=None)
    parser.add_argument("--warmup", type=int, default=_DEFAULT_WARMUP,
                        help="Number of warm-up iterations (default: %(default)s)")
    parser.add_argument("--rep", type=int, default=_DEFAULT_REP,
                        help="Number of timed iterations (default: %(default)s)")
    parser.add_argument(
        "--mode", choices=["wall", "kernel"], default="wall",
        help=(
            "wall  : perf_counter + synchronize (includes dispatch overhead). "
            "kernel: device events bracketing only kernel execution "
            "(matches triton.testing.do_bench). "
            "Default: wall"
        ),
    )
    parser.add_argument("--no-check", action="store_true",
                        help="Skip correctness check against torch.matmul")
    args = parser.parse_args()

    M, N, K = args.M, args.N, args.K
    num_cores = args.num_cores or get_number_cores()

    device = "npu" if hasattr(torch, "npu") and torch.npu.is_available() else "cuda"
    bench_fn = _bench_kernel if args.mode == "kernel" else _bench_wall

    torch.manual_seed(0)
    mat_a = torch.randn((M, K), dtype=torch.float16, device=device)
    mat_b = torch.randn((K, N), dtype=torch.float16, device=device)

    print(f"\n{'='*62}")
    print(f"  Matmul benchmark  M={M}  N={N}  K={K}")
    print(f"  device={device}  num_cores={num_cores}")
    print(f"  mode={args.mode}  warmup={args.warmup}  rep={args.rep}")
    print(f"{'='*62}\n")

    # ---- optional correctness check ----------------------------------------
    if not args.no_check:
        ref = torch.matmul(mat_a.float(), mat_b.float()).to(torch.float16)
        # native: no COMMONIR_SKIP_CSE
        with _env("COMMONIR_SKIP_CSE", None):
            out_native = _native.call(mat_a, mat_b, num_cores)
        # double_buffer: requires COMMONIR_SKIP_CSE=1
        with _env("COMMONIR_SKIP_CSE", "1"):
            out_db = _db.call(mat_a, mat_b, num_cores)
        # flaggems: no special env var needed
        out_flaggems = _flaggems.call(mat_a, mat_b, num_cores)
        for label, out in [("pytorch", ref), ("native", out_native),
                           ("double_buffer", out_db), ("flaggems", out_flaggems)]:
            try:
                torch.testing.assert_close(ref, out, rtol=1e-2, atol=1e-2)
                print(f"  [{label}] correctness check PASSED")
            except AssertionError as e:
                print(f"  [{label}] correctness check FAILED: {e}")
        print()

    # ---- timing -------------------------------------------------------------
    # torch_call adapts torch.matmul to the (a, b, num_cores) signature used
    # by the custom kernels so all three fit the same _bench_* interface.
    def torch_call(a, b, _num_cores):
        return torch.matmul(a, b)

    # (label, call_fn, COMMONIR_SKIP_CSE value)
    # None means the variable should be absent from the environment.
    _kernel_envs = [
        ("pytorch             ", torch_call,        None),
        ("flaggems_matmul     ", _flaggems.call,    None),
        ("native_matmul       ", _native.call,      None),
        ("double_buffer_matmul", _db.call,          "1"),
    ]

    results = {}
    for label, fn, skip_cse in _kernel_envs:
        with _env("COMMONIR_SKIP_CSE", skip_cse):
            if args.mode == "kernel":
                median = bench_fn(
                    fn, mat_a, mat_b, num_cores,
                    warmup=args.warmup, rep=args.rep)
                tfl = _tflops(M, N, K, median)
                print(f"  {label}  median={median:7.3f} ms  |  {tfl:.3f} TFLOPS")
            else:
                median, mean, mn, mx = bench_fn(
                    fn, mat_a, mat_b, num_cores, device,
                    warmup=args.warmup, rep=args.rep)
                tfl = _tflops(M, N, K, median)
                print(f"  {label}  median={median:7.3f} ms  mean={mean:7.3f} ms  "
                      f"min={mn:7.3f} ms  max={mx:7.3f} ms  |  {tfl:.3f} TFLOPS")
        results[label] = median

    # ---- summary ------------------------------------------------------------
    pt_ms     = results["pytorch             "]
    fg_ms     = results["flaggems_matmul     "]
    native_ms = results["native_matmul       "]
    db_ms     = results["double_buffer_matmul"]

    print()
    for label, ms in [("flaggems_matmul", fg_ms),
                      ("native_matmul",   native_ms),
                      ("double_buffer",   db_ms)]:
        ratio = pt_ms / ms if ms > 0 else float("inf")
        direction = "faster" if ratio > 1 else "slower"
        print(f"  {label} vs pytorch: {ratio:.3f}x {direction}")

    print()
    for label, ms in [("flaggems_matmul", fg_ms),
                      ("double_buffer",   db_ms)]:
        ratio = native_ms / ms if ms > 0 else float("inf")
        direction = "faster" if ratio > 1 else "slower"
        print(f"  {label} vs native_matmul: {ratio:.3f}x {direction}")

    db_vs_fg = fg_ms / db_ms if db_ms > 0 else float("inf")
    direction = "faster" if db_vs_fg > 1 else "slower"
    print(f"  double_buffer vs flaggems_matmul: {db_vs_fg:.3f}x {direction}")
    print()


if __name__ == "__main__":
    main()
