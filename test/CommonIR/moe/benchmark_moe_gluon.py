#!/usr/bin/env python3
"""Benchmark: Gluon Fused MoE vs PyTorch per-expert MoE on Metax.

Fair end-to-end comparison:
  - Gluon:   fused_moe() from moe_gluon.py (custom Gluon kernels)
  - PyTorch: per-expert batched torch.mm loop (good PyTorch baseline)

Usage:
    python benchmark_moe_gluon.py
    python benchmark_moe_gluon.py --dtype bfloat16
    python benchmark_moe_gluon.py --warmup 10 --rep 50
    python benchmark_moe_gluon.py --apply-router-weight-on-input
"""

import argparse
import csv
import sys
from pathlib import Path

import torch
import triton
from fused_moe.moe_gluon import fused_moe

# ---------------------------------------------------------------------------
# Import Gluon fused MoE (as a package, to support relative imports)
# ---------------------------------------------------------------------------
_this_dir = Path(__file__).resolve().parent  # .../metax/fused_moe/
_metax_dir = _this_dir.parent  # .../metax/
if str(_metax_dir) not in sys.path:
    sys.path.insert(0, str(_metax_dir))

# ---------------------------------------------------------------------------
# MoE shapes: (M, E, topk, K, intermediate)
#
# M:           number of tokens (decode: 1-64, prefill: 128+)
# E:           number of experts
# topk:        active experts per token
# K:           hidden dimension
# intermediate: FFN intermediate size
# ---------------------------------------------------------------------------
MOE_SIZES = [
    # ── Decode (small M, memory-bound) ──
    (1, 8, 2, 1024, 512),  # single-token decode
    (8, 8, 2, 1024, 512),
    (16, 8, 2, 1024, 512),
    (32, 8, 2, 1024, 512),
    (64, 8, 2, 1024, 512),
    # ── Prefill (large M, compute-bound) ──
    (128, 8, 2, 1024, 512),
    (256, 8, 2, 1024, 512),
    (512, 8, 2, 1024, 512),
    (1024, 8, 2, 1024, 512),
    (2048, 8, 2, 1024, 512),
    # ── Larger hidden dim (medium model) ──
    (64, 8, 2, 2048, 1024),
    (128, 8, 2, 2048, 1024),
    (512, 8, 2, 2048, 1024),
    # ── Larger hidden dim (large model, half-Mixtral) ──
    (64, 8, 2, 4096, 2048),
    (128, 8, 2, 4096, 2048),
    (256, 8, 2, 4096, 2048),
    # ── More experts / higher topk ──
    (128, 4, 2, 1024, 512),  # fewer experts
    (128, 16, 4, 1024, 512),  # more experts, higher topk
    (128, 64, 6, 1024, 512),  # DeepSeek-like many experts
    (128, 32, 4, 2048, 1024),  # medium, many experts
    # ── Non-power-of-2 M (padding stress) ──
    (48, 8, 2, 1024, 512),
    (192, 8, 2, 1024, 512),
    (768, 8, 2, 1024, 512),
    (1536, 8, 2, 1024, 512),
]

# ---------------------------------------------------------------------------
# PyTorch per-expert batched reference
# ---------------------------------------------------------------------------

def fused_moe_pytorch(
    hidden_states: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    apply_router_weight_on_input: bool = False,
) -> torch.Tensor:
    """Per-expert batched PyTorch MoE (no custom kernels).

    For each expert, gathers the tokens routed to it and performs
    batched matmul.  This represents a good PyTorch baseline —
    each expert's computation is a single torch.mm call on the
    gathered batch, avoiding per-token overhead.
    """
    M, K = hidden_states.shape
    E, N, _ = w1.shape
    topk = topk_ids.shape[1]
    intermediate = N // 2

    output = torch.zeros_like(hidden_states)

    for e in range(E):
        for k in range(topk):
            # Find tokens routed to expert e in topk slot k
            token_mask = (topk_ids[:, k] == e)
            token_indices = token_mask.nonzero(as_tuple=True)[0]
            if len(token_indices) == 0:
                continue

            x = hidden_states[token_indices]  # [n, K]
     
            # GEMM1: x @ w1[e]^T
            gate_up = x @ w1[e].t()  # [n, N]
     
            if apply_router_weight_on_input:
                w = topk_weights[token_indices, k].unsqueeze(1)

            # Activation: silu(gate) * up
            gate = gate_up[:, :intermediate]
            up = gate_up[:, intermediate:]
            act = torch.nn.functional.silu(gate) * up  # [n, intermediate]

            # GEMM2: act @ w2[e]^T
            y = act @ w2[e].t()  # [n, K]

            if not apply_router_weight_on_input:
                w = topk_weights[token_indices, k].unsqueeze(1)
                y = w * y

            # Scatter add to output
            output.index_add_(0, token_indices, y)

    return output


# ---------------------------------------------------------------------------
# MoE FLOPs helper
# ---------------------------------------------------------------------------

def moe_flops(M, E, topk, K, intermediate):
    """Approximate FLOPs for a full MoE forward pass.

    MoE computation:
        GEMM1: gather(A, sorted) @ w1[expert]^T  → 2 * M * topk * K * N
        GEMM2: cache2 @ w2[expert]^T             → 2 * M * topk * intermediate * K
    where N = 2 * intermediate, so:
        Total ≈ 2 * M * topk * K * (2*intermediate + intermediate)
             = 6 * M * topk * K * intermediate

    Activation (SiLU-and-mul) is elementwise and negligible vs GEMMs.
    """
    return 6.0 * M * topk * K * intermediate


# ---------------------------------------------------------------------------
# Input data factory
# ---------------------------------------------------------------------------

def _make_inputs(M, E, topk, K, intermediate, dtype, device="cuda"):
    """Create MoE inputs with fixed seed for reproducibility."""
    N = 2 * intermediate
    torch.manual_seed(42)

    hidden_states = torch.randn(M, K, device=device, dtype=dtype)
    w1 = torch.randn(E, N, K, device=device, dtype=dtype) * 0.1
    w2 = torch.randn(E, K, intermediate, device=device, dtype=dtype) * 0.1
    topk_ids = torch.randint(0, E, (M, topk), device=device, dtype=torch.int32)
    topk_weights = torch.rand(M, topk, device=device, dtype=dtype).contiguous()

    return hidden_states, w1, w2, topk_ids, topk_weights


# ---------------------------------------------------------------------------
# Gluon Fused MoE benchmark
# ---------------------------------------------------------------------------

def benchmark_gluon(M, E, topk, K, intermediate, dtype, apply_router_weight_on_input=False, warmup=25, rep=100):
    hidden_states, w1, w2, topk_ids, topk_weights = \
        _make_inputs(M, E, topk, K, intermediate, dtype)

    def run():
        fused_moe(hidden_states, w1, w2, topk_ids, topk_weights,
                  apply_router_weight_on_input=apply_router_weight_on_input)

    ms = triton.testing.do_bench(run, warmup=warmup, rep=rep)
    flops = moe_flops(M, E, topk, K, intermediate)
    tflops = flops / (ms * 1e-3) / 1e12
    return ms, tflops


# ---------------------------------------------------------------------------
# PyTorch per-expert benchmark
# ---------------------------------------------------------------------------

def benchmark_pytorch(M, E, topk, K, intermediate, dtype,
                      apply_router_weight_on_input=False,
                      warmup=25, rep=100):
    hidden_states, w1, w2, topk_ids, topk_weights = \
        _make_inputs(M, E, topk, K, intermediate, dtype)

    def run():
        fused_moe_pytorch(hidden_states, w1, w2, topk_ids, topk_weights,
                          apply_router_weight_on_input=apply_router_weight_on_input)

    ms = triton.testing.do_bench(run, warmup=warmup, rep=rep)
    flops = moe_flops(M, E, topk, K, intermediate)
    tflops = flops / (ms * 1e-3) / 1e12
    return ms, tflops


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def safe_ratio(a, b):
    """a / b, returning nan if either is nan or b <= 0."""
    return (a / b) if (a == a and b == b and b > 0) else float("nan")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(description="Gluon Fused MoE vs PyTorch per-expert MoE (Metax) benchmark")
    p.add_argument(
        "--dtype",
        choices=["float16", "bfloat16"],
        default="bfloat16",
        help="Data type (default: bfloat16)",
    )
    p.add_argument("--warmup", type=int, default=25, help="Warmup iterations")
    p.add_argument("--rep", type=int, default=100, help="Repeat iterations")
    p.add_argument(
        "--apply-router-weight-on-input",
        action="store_true",
        default=False,
        help="Apply router weight before activation (decode-optimized path)",
    )
    p.add_argument("--output", default="benchmark_moe_gluon_results.csv")
    args = p.parse_args()

    dtype = torch.float16 if args.dtype == "float16" else torch.bfloat16
    apply_rw = args.apply_router_weight_on_input

    SEP_WIDTH = 130
    rw_label = "weight-on-input" if apply_rw else "weight-on-output"
    print("=" * SEP_WIDTH)
    print(f"  Gluon Fused MoE vs PyTorch per-expert  |  Metax {args.dtype}  |  {rw_label}")
    try:
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
    except Exception:
        print("  GPU: (Metax device)")
    print("=" * SEP_WIDTH)

    rows = []
    for M, E, topk, K, intermediate in MOE_SIZES:
        print(f"\n--- M={M} E={E} topk={topk} K={K} inter={intermediate} ---")

        # Gluon
        try:
            gluon_ms, gluon_tf = benchmark_gluon(
                M,
                E, 
                topk,
                K, 
                intermediate, 
                dtype, 
                apply_rw,
                args.warmup, 
                args.rep,
            )
            print(f"  Gluon  : {gluon_ms:8.3f} ms  {gluon_tf:8.2f} TFLOPS")
        except Exception as e:
            print(f"  Gluon  : ERROR {e}")
            gluon_ms, gluon_tf = float("nan"), float("nan")

        # PyTorch
        try:
            torch_ms, torch_tf = benchmark_pytorch(
                M, 
                E,
                topk,
                K, 
                intermediate,
                dtype, 
                apply_rw,
                args.warmup, 
                args.rep,
            )
            print(f"  PyTorch: {torch_ms:8.3f} ms  {torch_tf:8.2f} TFLOPS")
        except Exception as e:
            print(f"  PyTorch: ERROR {e}")
            torch_ms, torch_tf = float("nan"), float("nan")

        rows.append({
            "M": M, 
            "E": E, 
            "topk": topk,
            "K": K, 
            "intermediate": intermediate,
            "dtype": args.dtype,
            "gluon_ms": gluon_ms, 
            "gluon_tflops": gluon_tf,
            "torch_ms": torch_ms, 
            "torch_tflops": torch_tf,
            # 加速比 vs PyTorch (Gluon 比 PyTorch 快几倍)
            "speedup_gluon_vs_torch": safe_ratio(gluon_tf, torch_tf),
            # 时间比 (PyTorch 时间 / Gluon 时间，> 1 表示 Gluon 更快)
            "time_ratio_torch_vs_gluon": safe_ratio(torch_ms, gluon_ms),
        })

    # Summary table
    print("\n" + "=" * SEP_WIDTH)
    print(f"{'M':>5} {'E':>3} {'k':>2} {'K':>5} {'I':>5} | "
          f"{'G ms':>8} {'T ms':>8} | "
          f"{'G TF':>8} {'T TF':>8} | "
          f"{'G/T':>6} {'T/G ms':>7}")
    print("-" * SEP_WIDTH)
    for r in rows:
        print(f"{r['M']:>5} {r['E']:>3} {r['topk']:>2} {r['K']:>5} {r['intermediate']:>5} | "
              f"{r['gluon_ms']:>8.3f} {r['torch_ms']:>8.3f} | "
              f"{r['gluon_tflops']:>8.2f} {r['torch_tflops']:>8.2f} | "
              f"{r['speedup_gluon_vs_torch']:>6.3f} {r['time_ratio_torch_vs_gluon']:>7.3f}")

    # Legend
    print("\n  G = Gluon Fused MoE  |  T = PyTorch per-expert")
    print("  G/T = Gluon TFLOPS / PyTorch TFLOPS  (加速比, >1 表示 Gluon 更快)")
    print("  T/G ms = PyTorch ms / Gluon ms  (时间倍数, >1 表示 Gluon 更快)")
    
    # Averages
    ratio_keys = [
        ("speedup_gluon_vs_torch", "G/T (TFLOPS)"),
        ("time_ratio_torch_vs_gluon", "T/G (ms)"),
    ]
    print()
    for key, label in ratio_keys:
        vals = [r[key] for r in rows if r[key] == r[key]]  # filter nan
        if vals:
            avg = sum(vals) / len(vals)
            print(f"  Avg {label} = {avg:.3f}")

    # CSV output
    with open(args.output, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"\nResults -> {args.output}")


if __name__ == "__main__":
    main()
