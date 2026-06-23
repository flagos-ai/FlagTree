"""Minimal no-NVSHMEM TLE WS 4+4+4 thread-budget probe.

This file intentionally avoids raw dialect externs and NVSHMEM.  It checks
whether a TLE warp-specialized kernel with:

  default partition: 4 warps
  worker0 partition: 4 warps
  worker1 partition: 4 warps

can reproduce the same thread-budget OOR seen in the MegaMoE-derived repros.
"""

from __future__ import annotations

import os
import site
from pathlib import Path


cuda_home = os.environ.get("CUDA_HOME", "/usr/local/cuda-12.8")
os.environ.setdefault("CUDA_HOME", cuda_home)
os.environ["CPATH"] = (
    f"{cuda_home}/targets/x86_64-linux/include:" + os.environ.get("CPATH", "")
)
os.environ["LD_LIBRARY_PATH"] = (
    f"{cuda_home}/lib64:" + os.environ.get("LD_LIBRARY_PATH", "")
)

import triton
import triton.language as tl
import triton.experimental.tle.language as tle

torch_site_packages = os.environ.get("MEGAMOE_TORCH_SITE_PACKAGES")
if torch_site_packages and Path(torch_site_packages).exists():
    site.addsitedir(torch_site_packages)

import torch


@triton.jit
def _default_partition(out, VALUE: tl.constexpr):
    tl.store(out + 0, VALUE)


@triton.jit
def _worker0_partition(out, BLOCK: tl.constexpr):
    offs = tl.arange(0, BLOCK)
    vals = offs + 10
    tl.store(out + 1 + offs, vals)


@triton.jit
def _worker1_partition(out, BLOCK: tl.constexpr):
    offs = tl.arange(0, BLOCK)
    vals = tl.load(out + 1 + offs, mask=offs < BLOCK, other=0)
    tl.store(out + 1 + BLOCK + offs, vals + 20)


@triton.jit
def _ws_444_no_nvshmem_kernel(out, BLOCK: tl.constexpr):
    tle.gpu.warp_specialize(
        [
            (_default_partition, (out, 0x444)),
            (_worker0_partition, (out, BLOCK)),
            (_worker1_partition, (out, BLOCK)),
        ],
        [4, 4],
        [80, 180],
    )


def main() -> None:
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required")

    torch.cuda.set_device(0)
    out = torch.empty((257,), device="cuda", dtype=torch.int32)
    out.zero_()
    _ws_444_no_nvshmem_kernel[(1,)](out, 128, num_warps=4, maxnreg=240)
    torch.cuda.synchronize()
    got = out[:5].detach().cpu().tolist()
    print(f"unexpected PASS: no-NVSHMEM WS 4+4+4 launched successfully, out[:5]={got}")


if __name__ == "__main__":
    main()
