"""Minimal runner for mhc_pre_clamp_sinkhorn.

Usage:
    python test_mhc_pre_clamp_sinkhorn.py
    python test_mhc_pre_clamp_sinkhorn.py --device npu --B 1 --S 16 --D 128
    python test_mhc_pre_clamp_sinkhorn.py --check
    MLIR_ENABLE_DUMP=1 python test_mhc_pre_clamp_sinkhorn.py --dump

This script constructs one valid input set and runs mhc_pre_clamp_sinkhorn
across the requested Sinkhorn variants, optionally comparing each result
against the matching PyTorch reference.

The variants exercise the structural matcher in the LaneVectorize pass:

    --sinkhorn-loop {static_range,range,all}   unrolled vs looped   (default all)
    --eps           {on,off,all}               sinkhorn HC_EPS     (default all)
    --norm-order    {row_first,col_first,all}  normalisation order (default all)

"w/o EPS" toggles only the Sinkhorn HC_EPS (the pre-head + HC_EPS is
unchanged). static_range vs range is numerically a no-op (unrolled vs looped)
but changes the IR the pass sees.
"""

from __future__ import annotations

import argparse
import os

import torch

from mhc_pre_clamp_sinkhorn import mhc_pre_clamp_sinkhorn, mhc_pre_clamp_sinkhorn_ref


def _resolve_device(device: str) -> str:
    if device == "auto":
        if hasattr(torch, "npu") and torch.npu.is_available():
            return "npu"
        if torch.cuda.is_available():
            return "cuda"
        return "cpu"
    return device


def _sync(device: str) -> None:
    if device.startswith("npu") and hasattr(torch, "npu"):
        torch.npu.synchronize()
    elif device.startswith("cuda"):
        torch.cuda.synchronize()


def _make_inputs(B: int, S: int, N: int, D: int, dtype: torch.dtype, device: str):
    assert N == 4, "mhc_pre_clamp_sinkhorn currently requires N=4"
    hc_mix = N * (N + 2)
    hc_d = N * D

    x = torch.randn((B, S, N, D), dtype=dtype, device=device)
    phi = torch.randn((hc_mix, hc_d), dtype=torch.float32, device=device)
    alpha = torch.randn((3, ), dtype=torch.float32, device=device)
    base = torch.randn((hc_mix, ), dtype=torch.float32, device=device)
    return x, phi, alpha, base


_LOOP_VARIANTS = {"static_range": True, "range": False}
_EPS_VARIANTS = {"on": True, "off": False}
_ORDER_VARIANTS = {"row_first": 0, "col_first": 1}


def _pick(selection: str, mapping: dict):
    if selection == "all":
        return list(mapping.items())
    return [(selection, mapping[selection])]


def _variants(args):
    """Expand the selectors into a list of (label, kernel_kwargs, ref_kwargs)."""
    out = []
    for loop_name, use_static_range in _pick(args.sinkhorn_loop, _LOOP_VARIANTS):
        for eps_name, apply_eps in _pick(args.eps, _EPS_VARIANTS):
            for order_name, norm_order in _pick(args.norm_order, _ORDER_VARIANTS):
                out.append((
                    f"loop={loop_name:<12} eps={eps_name:<3} order={order_name}",
                    dict(use_static_range=use_static_range, apply_eps=apply_eps, norm_order=norm_order),
                    dict(apply_eps=apply_eps, norm_order=norm_order),
                ))
    return out


def _check(out, ref):
    y_ok = torch.allclose(out["y"].float(), ref["y"].float(), atol=1e-2, rtol=1e-2)
    post_ok = torch.allclose(out["post_out"].float(), ref["post_out"].float(), atol=1e-4, rtol=1e-4)
    comb_ok = torch.allclose(out["comb_frag"].float(), ref["comb_frag"].float(), atol=1e-4, rtol=1e-4)
    return bool(y_ok and post_ok and comb_ok), (bool(y_ok), bool(post_ok), bool(comb_ok))


def main() -> None:
    parser = argparse.ArgumentParser(description="mhc_pre_clamp_sinkhorn variant runner")
    parser.add_argument("--device", type=str, default="auto", help="Device to run on: auto/npu/cuda/cpu")
    parser.add_argument("--B", type=int, default=1, help="Batch size")
    parser.add_argument("--S", type=int, default=8, help="Sequence length")
    parser.add_argument("--N", type=int, default=4, help="Head multiplier; must be 4")
    parser.add_argument("--D", type=int, default=128, help="Hidden dimension per head")
    parser.add_argument("--dtype", type=str, default="bf16", choices=["bf16", "fp16", "fp32"], help="Input dtype for x")
    parser.add_argument("--iter-times", type=int, default=20, help="Sinkhorn iteration count")
    parser.add_argument("--norm-eps", type=float, default=1e-6)
    parser.add_argument("--hc-eps", type=float, default=1e-6)
    parser.add_argument("--clamp-min", type=float, default=0.0)
    parser.add_argument("--clamp-max", type=float, default=0.0)
    parser.add_argument("--need-backward", action="store_true", help="Request extra saved tensors")
    parser.add_argument("--sinkhorn-loop", choices=["static_range", "range", "all"], default="all",
                        help="Sinkhorn iteration form (default: all)")
    parser.add_argument("--eps", choices=["on", "off", "all"], default="all",
                        help="Sinkhorn HC_EPS on/off (default: all)")
    parser.add_argument("--norm-order", choices=["row_first", "col_first", "all"], default="all",
                        help="Normalisation order (default: all)")
    parser.add_argument("--check", action="store_true", help="Compare each variant to the matching reference")
    parser.add_argument("--dump", action="store_true", help="Print current MLIR_ENABLE_DUMP setting as a reminder")
    args = parser.parse_args()

    device = _resolve_device(args.device)
    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[args.dtype]

    if args.dump:
        print(f"MLIR_ENABLE_DUMP={os.environ.get('MLIR_ENABLE_DUMP', '<unset>')}")

    variants = _variants(args)
    print(f"device={device} shape=({args.B}, {args.S}, {args.N}, {args.D}) dtype={args.dtype} "
          f"variants={len(variants)} check={args.check}")

    x, phi, alpha, base = _make_inputs(args.B, args.S, args.N, args.D, dtype, device)

    common = dict(
        norm_eps=args.norm_eps,
        hc_eps=args.hc_eps,
        clamp_min=args.clamp_min,
        clamp_max=args.clamp_max,
        iter_times=args.iter_times,
    )

    failed = 0
    for label, kernel_kw, ref_kw in variants:
        try:
            out = mhc_pre_clamp_sinkhorn(
                x,
                phi,
                alpha,
                base,
                need_backward=args.need_backward,
                **common,
                **kernel_kw,
            )
            _sync(device)
        except Exception as exc:  # noqa: BLE001
            print(f"[ERROR] {label}  {type(exc).__name__}: {exc}")
            failed += 1
            continue

        if args.check:
            ref = mhc_pre_clamp_sinkhorn_ref(x, phi, alpha, base, **common, **ref_kw)
            ok, detail = _check(out, ref)
            if not ok:
                failed += 1
            print(f"[{'PASS' if ok else 'FAIL'}] {label}  y={detail[0]} post={detail[1]} comb={detail[2]}")
        else:
            print(f"[ok]   {label}  y{tuple(out['y'].shape)} "
                  f"post{tuple(out['post_out'].shape)} comb{tuple(out['comb_frag'].shape)}")
            if args.need_backward:
                for key in ["inv_rms", "x_scaled", "mixes", "h_res_logits", "pre"]:
                    print(f"       {key}{tuple(out[key].shape)} dtype={out[key].dtype}")

    print(f"--- {len(variants) - failed}/{len(variants)} ok ---")
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
