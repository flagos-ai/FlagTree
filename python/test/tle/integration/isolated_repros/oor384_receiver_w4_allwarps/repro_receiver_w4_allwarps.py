"""Minimal entry point for the TLE WS ``Required: 384`` thread OOR.

This script intentionally does not run the full MegaMoE smoke.  It imports the
JIT kernel from ``triton_userhopper_single_kernel_l1_tldot_smoke.py``, allocates
ordinary CUDA tensors with matching shapes, and launches the smallest historical
H=256 expert-wave L1+L2 case that fails during kernel handle initialization:

  triton.runtime.errors.OutOfResources:
  out of resource: threads, Required: 384, Hardware limit: 256.

The launch is expected to fail before correctness-sensitive NVSHMEM execution,
so no MPI/NVSHMEM PE setup is required.
"""

from __future__ import annotations

import os


cuda_home = os.environ.get("CUDA_HOME", "/usr/local/cuda-12.8")
os.environ.setdefault("CUDA_HOME", cuda_home)
os.environ["CPATH"] = (
    f"{cuda_home}/targets/x86_64-linux/include:" + os.environ.get("CPATH", "")
)
os.environ["LD_LIBRARY_PATH"] = (
    f"{cuda_home}/lib64:" + os.environ.get("LD_LIBRARY_PATH", "")
)

# Fixed smallest historical OOR384 case:
# 8 ranks / 8 experts / epr=1 / topk=1 / tokens=1 / H=256 / I=128.
os.environ["USERHOPPER_WS_NUM_RANKS"] = "8"
os.environ["USERHOPPER_WS_NUM_EXPERTS"] = "8"
os.environ["USERHOPPER_WS_NUM_TOPK"] = "1"
os.environ["USERHOPPER_WS_NUM_TOKENS"] = "1"
os.environ["USERHOPPER_WS_HIDDEN"] = "256"
os.environ["USERHOPPER_WS_INTERMEDIATE_HIDDEN"] = "128"
os.environ["USERHOPPER_WS_NUM_WARPS"] = "4"
os.environ["USERHOPPER_WS_NUM_DISPATCH_WARPS"] = "1"
os.environ["USERHOPPER_WS_EXPERT_WAVE_SINGLE_KERNEL"] = "1"
os.environ["USERHOPPER_WS_EXPERT_WAVE_COMPUTE_WARPS"] = "4"
os.environ["USERHOPPER_WS_L1_I_TILES"] = "2"
os.environ["USERHOPPER_WS_L2_H_TILES"] = "1"
os.environ["USERHOPPER_WS_ALLOW_EXPERIMENTAL_H256_TLDOT"] = "1"
os.environ["USERHOPPER_WS_CLEANUP"] = "0"
os.environ["USERHOPPER_WS_REPEAT_LAUNCHES"] = "1"

import triton_userhopper_single_kernel_l1_tldot_smoke as smoke
import triton_tle_ws_userhopper_dispatch_receiver_smoke as uh
import torch


def main() -> None:
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required")

    torch.cuda.set_device(0)
    device = torch.device("cuda")

    whole = torch.empty((uh.LAYOUT["total_bytes"],), device=device, dtype=torch.uint8)
    l1_acts = torch.empty(
        (uh.LAYOUT["num_max_pool_tokens"], uh.HIDDEN),
        device=device,
        dtype=torch.float8_e4m3fn,
    )
    l1_acts_sf = torch.empty(
        (uh.LAYOUT["num_max_padded_sf_pool_tokens"], uh.HIDDEN // 128),
        device=device,
        dtype=torch.float32,
    )
    l1_topk_weights = torch.empty(
        (uh.LAYOUT["num_max_pool_tokens"],),
        device=device,
        dtype=torch.float32,
    )
    l1_weights = torch.empty(
        (uh.NUM_EXPERTS_PER_RANK, 2 * uh.INTERMEDIATE_HIDDEN, uh.HIDDEN),
        device=device,
        dtype=torch.float8_e4m3fn,
    )
    l1_weights_sf = torch.empty(
        (uh.NUM_EXPERTS_PER_RANK, 2 * uh.INTERMEDIATE_HIDDEN // 128, uh.HIDDEN // 128),
        device=device,
        dtype=torch.float32,
    )
    l2_acts = torch.empty(
        (uh.LAYOUT["num_max_pool_tokens"], uh.INTERMEDIATE_HIDDEN),
        device=device,
        dtype=torch.float8_e4m3fn,
    )
    l2_acts_sf = torch.empty(
        (uh.LAYOUT["num_max_padded_sf_pool_tokens"], uh.INTERMEDIATE_HIDDEN // 64),
        device=device,
        dtype=torch.float32,
    )
    l2_weights = torch.empty(
        (uh.NUM_EXPERTS_PER_RANK, uh.HIDDEN, uh.INTERMEDIATE_HIDDEN),
        device=device,
        dtype=torch.float8_e4m3fn,
    )
    l2_weights_sf = torch.empty(
        (uh.NUM_EXPERTS_PER_RANK, uh.HIDDEN // 128, uh.INTERMEDIATE_HIDDEN // 128),
        device=device,
        dtype=torch.float32,
    )
    l2_out = torch.empty(
        (uh.LAYOUT["num_max_pool_tokens"], uh.HIDDEN),
        device=device,
        dtype=torch.float32,
    )
    y = torch.empty((uh.NUM_MAX_TOKENS_PER_RANK, uh.HIDDEN), device=device, dtype=torch.float32)
    marker = torch.empty((1,), device=device, dtype=torch.int32)

    per_expert_counts = uh._expected_counts_for_rank(0)
    expert1_count = per_expert_counts[1] if uh.NUM_EXPERTS_PER_RANK > 1 else 0
    expert1_pool_base = uh._align(per_expert_counts[0], 64)

    smoke._single_kernel_dispatch_receiver_l1_l2_expert_wave_tldot_kernel[(1,)](
        whole,
        l1_acts,
        l1_acts_sf,
        l1_topk_weights,
        l1_weights,
        l1_weights_sf,
        l2_acts,
        l2_acts_sf,
        l2_weights,
        l2_weights_sf,
        l2_out,
        y.view(torch.uint8),
        marker,
        uh.NUM_TOKENS,
        uh.EXPECTED_LOCAL_RECV_TOKENS,
        uh.NUM_RANKS,
        uh.NUM_EXPERTS,
        uh.NUM_MAX_TOKENS_PER_RANK,
        uh.NUM_TOPK,
        uh.HIDDEN,
        uh.INTERMEDIATE_HIDDEN,
        uh.LAYOUT["num_max_padded_sf_pool_tokens"],
        uh.NUM_DISPATCH_WARPS,
        uh.NUM_EXPERTS_PER_RANK,
        per_expert_counts[0],
        expert1_count,
        expert1_pool_base,
        0,
        num_warps=uh.NUM_WARPS,
        maxnreg=uh.MAXNREG,
    )

    torch.cuda.synchronize()
    raise SystemExit("unexpected PASS: OOR384 did not reproduce")


if __name__ == "__main__":
    main()
