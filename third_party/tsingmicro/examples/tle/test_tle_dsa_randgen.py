"""
Smoke example: FlagGems-style torch.randn via tle.dsa.randn (hardware randgen).

Requires a rebuilt triton with the DSA randgen pipeline wired
(dsa.randgen → mk.randgen → tx.randgen → __RandGen).

Tensors must live on the active TXDA device — CPU tensors are not updated
by TX81 kernels.
"""

import torch
import torch_txda  # noqa: F401

import triton
import triton.experimental.tle.language as tle
import triton.language as tl


@triton.jit
def randn_dsa_kernel(
    out_ptr,
    seed0_ptr,
    seed1_ptr,
    N,
    BLOCK: tl.constexpr,
):
    """
    BLOCK must be a multiple of 32 (tle.dsa.randn alignment).
    seed0/seed1 are length-16 int64 seed vectors in global memory.
    Each program uses the same seed vector; use grid=1 or distinct seeds
    per block for independent streams.
    """
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N

    seed0 = tl.load(seed0_ptr + tl.arange(0, 16))
    seed1 = tl.load(seed1_ptr + tl.arange(0, 16))
    vals, seed0_out, seed1_out = tle.dsa.randn(seed0, seed1, BLOCK)
    # Persist updated seeds (no-op on hardware until peri writes advanced state).
    tl.store(seed0_ptr + tl.arange(0, 16), seed0_out)
    tl.store(seed1_ptr + tl.arange(0, 16), seed1_out)
    tl.store(out_ptr + offs, vals, mask=mask)


def dsa_randn(size, *, dtype=None, device=None, seed=0):
    if dtype is None:
        dtype = torch.get_default_dtype()
    if device is None:
        device = triton.runtime.driver.active.get_active_torch_device()
    out = torch.empty(size, device=device, dtype=dtype)
    n = out.numel()
    # Single block so all samples share one independent randgen stream.
    block = 1
    while block < n:
        block *= 2
    if block < 32:
        block = 32
    assert block % 32 == 0
    grid = (1, )

    g = torch.Generator()
    g.manual_seed(int(seed))
    seed0 = torch.randint(-(1 << 63), (1 << 63) - 1, (16,),
                         dtype=torch.int64, generator=g, device="cpu").to(device)
    seed1 = torch.randint(-(1 << 63), (1 << 63) - 1, (16,),
                         dtype=torch.int64, generator=g, device="cpu").to(device)
    # Avoid all-zero xorshift state.
    seed0[0] = seed0[0] | 1
    seed1[0] = seed1[0] | 1

    randn_dsa_kernel[grid](out.view(-1), seed0, seed1, n, BLOCK=block)
    return out


if __name__ == "__main__":
    x = dsa_randn((4096,), dtype=torch.float32)
    x_cpu = x.cpu()
    print("device", x.device)
    print("mean", float(x_cpu.mean()), "std", float(x_cpu.std()))
    print("finite", bool(torch.isfinite(x_cpu).all()))
