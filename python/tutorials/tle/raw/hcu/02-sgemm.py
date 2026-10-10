"""HCU tle.raw SGEMM. The kernel only calls the HIP function."""

from pathlib import Path

import torch
import triton
from triton.experimental.tle.raw import dialect
import triton.experimental.tle.language.raw as tle_raw

DEVICE = triton.runtime.driver.active.get_active_torch_device()
TILE = 16


# deferred=True is the fast path: clang bitcode is linked with its target attributes.
@dialect(name="hcu", file=Path(__file__).parent / "02-sgemm.hip", extern_func_name="hcu_sgemm", deferred=True)
def edsl(*args, **kwargs):
    ...


@triton.jit
def sgemm_kernel(c_ptr, a_ptr, b_ptr, m, n, k):
    tle_raw.call(edsl, [c_ptr, a_ptr, b_ptr, m, n, k], output_indices=[])


def sgemm(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    m, k = a.shape
    k2, n = b.shape
    if k != k2:
        raise ValueError(f"K mismatch: {k} vs {k2}")
    c = torch.empty((m, n), device=a.device, dtype=a.dtype)
    grid = (triton.cdiv(n, TILE), triton.cdiv(m, TILE))
    sgemm_kernel[grid](c, a, b, m, n, k, num_warps=4)
    return c


if __name__ == "__main__":
    torch.manual_seed(0)
    m = n = k = 128
    a = torch.randn((m, k), device=DEVICE)
    b = torch.randn((k, n), device=DEVICE)
    got = sgemm(a, b)
    torch.cuda.synchronize()
    ref = a @ b
    ok = torch.allclose(got, ref, rtol=1e-3, atol=1e-3)
    err = (got - ref).abs().max().item()
    print("allclose", bool(ok))
    print("max_abs_err", err)
    raise SystemExit(0 if ok else 1)
