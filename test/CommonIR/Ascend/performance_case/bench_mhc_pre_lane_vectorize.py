"""Benchmark the mhc TLE kernels.

The ``triton-lane-vectorize`` TTIR pass (see
``third_party/flir/lib/Transforms/LaneVectorize/LaneVectorize.cpp``) is always
enabled in the Ascend TTIR pipeline, so this script simply measures the mhc
kernels with it on.

Supported kernels (``--kernel``):
    mhc_pre_clamp_sinkhorn  (default) the Sinkhorn normalization pre-kernel
    mhc_post                the fused post-kernel
    all                     run every kernel above in one invocation

Timing modes (same as bench_fa_triton_arch.py)
----------------------------------------------
wall    (default) end-to-end latency: sync -> perf_counter -> kernel -> sync.
kernel  kernel-only via ``do_bench_npu`` from testing.py (mspti -> profiler).

Metrics
-------
- Latency (ms)
- Bandwidth (GB/s): read the inputs, write the outputs (per kernel; see
  ``_bytes_io_for``).

Usage
-----
    # default kernel and shape
    python bench_mhc_pre_lane_vectorize.py

    # run every supported kernel
    python bench_mhc_pre_lane_vectorize.py --kernel all

    # benchmark mhc_post instead
    python bench_mhc_pre_lane_vectorize.py --kernel mhc_post --B 2 --S 1024 --D 3584

    # specific shape / sinkhorn settings / timing mode
    python bench_mhc_pre_lane_vectorize.py --B 2 --S 1024 --D 3584 \
        --iter-times 20 --clamp-min 0 --clamp-max 1 --mode kernel

    # choose the Sinkhorn IR shape (constexpr kernel knobs)
    python bench_mhc_pre_lane_vectorize.py --sinkhorn-loop range --eps off --norm-order col_first

    # sweep a set of shapes, verifying outputs
    python bench_mhc_pre_lane_vectorize.py --sweep --check
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import sys
import time

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
_HERE = os.path.dirname(os.path.abspath(__file__))
_MHC_DIR = os.path.abspath(os.path.join(_HERE, "..", "mhc"))
sys.path.insert(0, _MHC_DIR)  # mhc_pre_clamp_sinkhorn.py, mhc_post.py

# The Ascend backend ships the maintained do_bench_npu; load it by explicit
# path (the module name `testing` is generic and easily shadowed on sys.path).
_ASCEND_TESTING = os.path.abspath(
    os.path.join(_HERE, "..", "..", "..", "..", "third_party", "ascend", "backend", "testing.py"))


def _load_do_bench_npu():
    """Load ``do_bench_npu`` from third_party/ascend/backend/testing.py."""
    if not os.path.exists(_ASCEND_TESTING):
        raise ImportError(f"cannot find {_ASCEND_TESTING}")
    spec = importlib.util.spec_from_file_location("_mhc_bench_testing", _ASCEND_TESTING)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load testing.py from {_ASCEND_TESTING}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.do_bench_npu


# ---------------------------------------------------------------------------
# Configurations
# ---------------------------------------------------------------------------
_DEFAULT_SHAPE = (1, 256, 4, 3584)  # (B, S, N, D); N must be 4

_SWEEP_SHAPES = [
    (1, 64, 4, 3584),
    (1, 256, 4, 3584),
    (2, 1024, 4, 3584),
    (1, 4096, 4, 3584),
    (1, 1024, 4, 512),
    (1, 1024, 4, 1024),
    (1, 1024, 4, 2048),
]


def _resolve_shapes(args):
    if args.sweep:
        return list(_SWEEP_SHAPES)
    return [(args.B, args.S, args.N, args.D)]


def _kernel_kwargs(args):
    """Forward kwargs of mhc_pre_clamp_sinkhorn (not of its reference)."""
    return dict(
        norm_eps=args.norm_eps,
        hc_eps=args.hc_eps,
        clamp_min=args.clamp_min,
        clamp_max=args.clamp_max,
        iter_times=args.iter_times,
        need_backward=args.need_backward,
        use_static_range=(args.sinkhorn_loop == "static_range"),
        apply_eps=(args.eps == "on"),
        norm_order=(0 if args.norm_order == "row_first" else 1),
    )


def _ref_kwargs(args):
    """Forward kwargs of mhc_pre_clamp_sinkhorn_ref (no need_backward / loop form)."""
    return dict(
        norm_eps=args.norm_eps,
        hc_eps=args.hc_eps,
        clamp_min=args.clamp_min,
        clamp_max=args.clamp_max,
        iter_times=args.iter_times,
        apply_eps=(args.eps == "on"),
        norm_order=(0 if args.norm_order == "row_first" else 1),
    )


def _device(torch):
    if hasattr(torch, "npu") and torch.npu.is_available():
        return "npu"
    return "cuda"


def _sync(device: str, torch) -> None:
    if device == "npu":
        torch.npu.synchronize()
    elif device == "cuda":
        torch.cuda.synchronize()


def _make_inputs(torch, B, S, N, D, dtype, device):
    """Match test_mhc_pre_clamp_sinkhorn._make_inputs."""
    hc_mix = N * (N + 2)  # 24 for N == 4
    hc_d = N * D
    x = torch.randn(B, S, N, D, dtype=dtype, device=device)
    phi = torch.randn(hc_mix, hc_d, dtype=torch.float32, device=device)
    alpha = torch.randn(3, dtype=torch.float32, device=device)
    base = torch.randn(hc_mix, dtype=torch.float32, device=device)
    return x, phi, alpha, base


def _bytes_io(B, S, N, D, dtype):
    """Read x + phi + alpha + base, write y + post_out + comb_frag."""
    elem = dtype.itemsize
    hc_mix = N * (N + 2)
    hc_d = N * D
    T = B * S
    read = (
        T * N * D * elem  # x
        + hc_mix * hc_d * 4  # phi (fp32)
        + 3 * 4  # alpha
        + hc_mix * 4)  # base
    write = (
        T * D * elem  # y
        + T * N * 4  # post_out
        + T * N * N * 4)  # comb_frag
    return read + write


def _bench_wall(fn, device, torch, warmup, rep):
    for _ in range(warmup):
        fn()
    _sync(device, torch)
    times = []
    for _ in range(rep):
        _sync(device, torch)
        t0 = time.perf_counter()
        fn()
        _sync(device, torch)
        times.append((time.perf_counter() - t0) * 1e3)
    s = sorted(times)
    return dict(ms=s[len(s) // 2], min=s[0], max=s[-1], mean=sum(s) / len(s))


def _bench_kernel(fn, warmup, rep, do_bench_npu):
    avg = do_bench_npu([fn], warmup=warmup, active=rep, clear_l2_cache=True, keep_res=False)
    if isinstance(avg, list):
        avg = avg[0]
    return dict(ms=avg, min=avg, max=avg, mean=avg)


def _check_outputs(torch, out, ref):
    """Compare y / post_out / comb_frag (tolerances from test_mhc_pre_clamp_sinkhorn)."""
    y_ok = bool(torch.allclose(out["y"].float(), ref["y"].float(), atol=1e-2, rtol=1e-2))
    post_ok = bool(torch.allclose(out["post_out"].float(), ref["post_out"].float(), atol=1e-4, rtol=1e-4))
    comb_ok = bool(torch.allclose(out["comb_frag"].float(), ref["comb_frag"].float(), atol=1e-4, rtol=1e-4))
    return y_ok and post_ok and comb_ok, dict(y=y_ok, post_out=post_ok, comb_frag=comb_ok)


# ---------------------------------------------------------------------------
# Kernel registry: per-kernel inputs, callable kwargs, bytes/IO and checks
# ---------------------------------------------------------------------------

_KERNELS = ("mhc_pre_clamp_sinkhorn", "mhc_post")


def _make_inputs_for(args, kernel, torch, B, S, N, D, dtype, device):
    """Build the positional inputs of the selected kernel."""
    if kernel == "mhc_post":
        x = torch.randn(B, S, N, D, dtype=dtype, device=device)
        h_res = torch.randn(B, S, N, N, dtype=torch.float32, device=device)
        h_out = torch.randn(B, S, D, dtype=dtype, device=device)
        h_post = torch.randn(B, S, N, dtype=torch.float32, device=device)
        return (x, h_res, h_out, h_post)
    return _make_inputs(torch, B, S, N, D, dtype, device)


def _kernel_kwargs_for(args, kernel):
    if kernel == "mhc_post":
        return dict(use_pipeline=(args.post_pipeline == "on"), use_concat_reduce=args.post_concat_reduce)
    return _kernel_kwargs(args)


def _ref_kwargs_for(args, kernel):
    if kernel == "mhc_post":
        return {}
    return _ref_kwargs(args)


def _bytes_io_for(args, kernel, B, S, N, D, dtype):
    """Bytes read + written by the selected kernel (see the module docstring)."""
    if kernel == "mhc_post":
        elem = dtype.itemsize
        T = B * S
        read = (
            T * N * D * elem  # x
            + T * N * N * 4  # h_res (fp32)
            + T * D * elem  # h_out
            + T * N * 4)  # h_post (fp32)
        write = T * N * D * elem  # out
        return read + write
    return _bytes_io(B, S, N, D, dtype)


def _check_outputs_for(args, kernel, torch, out, ref):
    if kernel == "mhc_post":
        ok = bool(torch.allclose(out.float(), ref.float(), atol=1e-2, rtol=1e-2))
        return ok, dict(out=ok)
    return _check_outputs(torch, out, ref)


def _load_kernel(kernel):
    """Import the kernel and its reference from the mhc directory."""
    if kernel == "mhc_post":
        from mhc_post import mhc_post as kernel_fn, mhc_post_ref as ref_fn
        return kernel_fn, ref_fn
    from mhc_pre_clamp_sinkhorn import (  # noqa: E402
        mhc_pre_clamp_sinkhorn as kernel_fn, mhc_pre_clamp_sinkhorn_ref as ref_fn,
    )
    return kernel_fn, ref_fn


def _run_kernel(args, kernel) -> bool:
    import torch  # noqa: E402  (imported late so --help does not need torch)

    kernel_fn, ref_fn = _load_kernel(kernel)
    do_bench_npu = _load_do_bench_npu()
    device = _device(torch)
    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16}[args.dtype]
    kwargs = _kernel_kwargs_for(args, kernel)
    ref_kwargs = _ref_kwargs_for(args, kernel)

    print()
    print(f"Kernel : {kernel}")
    print(f"Device : {device}   timing={args.mode}   dtype={args.dtype}")
    if kernel == "mhc_post":
        print(f"Options : use_pipeline={args.post_pipeline}   use_concat_reduce={args.post_concat_reduce}")
    else:
        print(f"Options : iter_times={args.iter_times}   clamp=({args.clamp_min},{args.clamp_max})   "
              f"need_backward={args.need_backward}")
        print(f"Sinkhorn : loop={args.sinkhorn_loop}   eps={args.eps}   norm_order={args.norm_order}")
    print(f"Warmup : {args.warmup}   Rep/Active: {args.rep}")
    print()

    hdr = f"{'config':>22} | {'latency(ms)':>11} | {'bandwidth (GB/s)':>16} | {'check':>5}"
    print(hdr)
    print("-" * len(hdr))

    ok = True
    for (B, S, N, D) in _resolve_shapes(args):
        label = f"B{B}_S{S}_N{N}_D{D}"
        torch.manual_seed(0)
        inputs = _make_inputs_for(args, kernel, torch, B, S, N, D, dtype, device)

        def fn(inputs=inputs):
            return kernel_fn(*inputs, **kwargs)

        check = None
        if args.check:
            ref = ref_fn(*inputs, **ref_kwargs)
            out = fn()
            check, detail = _check_outputs_for(args, kernel, torch, out, ref)
            if not check:
                print(f"  {label}: check FAIL {detail}", file=sys.stderr)
                ok = False

        if args.mode == "kernel":
            t = _bench_kernel(fn, args.warmup, args.rep, do_bench_npu)
        else:
            t = _bench_wall(fn, device, torch, args.warmup, args.rep)

        bandwidth = _bytes_io_for(args, kernel, B, S, N, D, dtype) / (t["ms"] * 1e-3) / 1e9
        check_col = "" if check is None else ("ok" if check else "FAIL")
        print(f"{label:>22} | {t['ms']:>11.4f} | {bandwidth:>16.2f} | {check_col:>5}")

    return ok


def _run(args) -> int:
    kernels = list(_KERNELS) if args.kernel == "all" else [args.kernel]
    ok = True
    for kernel in kernels:
        if not _run_kernel(args, kernel):
            ok = False
    print()
    return 0 if ok else 1


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args():
    p = argparse.ArgumentParser(description="Benchmark the mhc TLE kernels")
    p.add_argument("--kernel", choices=[*_KERNELS, "all"], default=_KERNELS[0],
                   help="mhc kernel to benchmark; 'all' runs every supported kernel")
    p.add_argument("--B", type=int, default=_DEFAULT_SHAPE[0], help="batch size")
    p.add_argument("--S", type=int, default=_DEFAULT_SHAPE[1], help="sequence length")
    p.add_argument("--N", type=int, default=_DEFAULT_SHAPE[2], help="head multiplier (must be 4)")
    p.add_argument("--D", type=int, default=_DEFAULT_SHAPE[3], help="hidden dim per head")
    p.add_argument("--dtype", choices=["bf16", "fp16"], default="bf16", help="input dtype for x/y")
    p.add_argument("--mode", choices=["wall", "kernel"], default="wall",
                   help="wall: perf_counter+sync; kernel: do_bench_npu (mspti/profiler)")
    p.add_argument("--iter-times", type=int, default=20, help="sinkhorn iteration count")
    p.add_argument("--sinkhorn-loop", choices=["static_range", "range"], default="static_range",
                   help="unrolled tl.static_range vs looped tl.range Sinkhorn body")
    p.add_argument("--eps", choices=["on", "off"], default="on", help="emit/omit the Sinkhorn HC_EPS adds")
    p.add_argument("--norm-order", choices=["row_first", "col_first"], default="row_first",
                   help="row-first (0) vs col-first (1) initial norm and iteration order")
    p.add_argument("--norm-eps", type=float, default=1e-6, help="RMSNorm epsilon")
    p.add_argument("--hc-eps", type=float, default=1e-6, help="sinkhorn epsilon")
    p.add_argument("--clamp-min", type=float, default=0.0, help="logits clamp min (0 = disabled)")
    p.add_argument("--clamp-max", type=float, default=0.0, help="logits clamp max (0 = disabled)")
    p.add_argument("--need-backward", action="store_true", help="request the extra saved intermediates")
    p.add_argument("--warmup", type=int, default=10, help="warmup iterations")
    p.add_argument("--rep", type=int, default=50, help="measurement iterations (active runs for kernel mode)")
    p.add_argument("--check", action="store_true", help="verify output against the kernel reference")
    p.add_argument("--post-pipeline", choices=["on", "off"], default="on",
                   help="mhc_post: use the software-pipelined kernel (default) or the 2D-grid one")
    p.add_argument("--post-concat-reduce", action="store_true",
                   help="mhc_post: use the concat+reduce variant (takes precedence over --post-pipeline)")
    p.add_argument("--sweep", action="store_true", help="benchmark a set of shapes instead of one")
    return p.parse_args()


def main() -> int:
    return _run(_parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
