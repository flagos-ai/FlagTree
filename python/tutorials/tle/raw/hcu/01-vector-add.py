"""HCU tle.raw vector add. The kernel only calls the HIP function."""

from pathlib import Path

import torch
import triton
from triton.experimental.tle.raw import dialect
import triton.experimental.tle.language.raw as tle_raw

DEVICE = triton.runtime.driver.active.get_active_torch_device()
BLOCK = 256


# deferred=True is the fast path: clang bitcode is linked with its target attributes.
@dialect(name="hcu", file=Path(__file__).parent / "01-vector-add.hip", extern_func_name="hcu_vector_add",
         deferred=True)
def edsl(*args, **kwargs):
    ...


@triton.jit
def add_kernel(out_ptr, x_ptr, y_ptr, n_elements):
    tle_raw.call(edsl, [out_ptr, x_ptr, y_ptr, n_elements], output_indices=[])


def add(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(x)
    n_elements = out.numel()
    grid = (triton.cdiv(n_elements, BLOCK), )
    add_kernel[grid](out, x, y, n_elements, num_warps=4)
    return out


if __name__ == "__main__":
    torch.manual_seed(0)
    x = torch.randn(98432, device=DEVICE)
    y = torch.randn(98432, device=DEVICE)
    got = add(x, y)
    torch.cuda.synchronize()
    ref = x + y
    ok = torch.allclose(got, ref)
    err = (got - ref).abs().max().item()
    print("allclose", bool(ok))
    print("max_abs_err", err)
    raise SystemExit(0 if ok else 1)
