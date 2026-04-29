from pathlib import Path

import torch
import triton
import triton.language as tl
from triton.experimental.tle.raw import dialect
import triton.experimental.tle.language.raw as tle_raw

DEVICE = triton.runtime.driver.active.get_active_torch_device()


@dialect(name="cuda", file=Path(__file__).parent / "01-vector-add-half2.cu")
def edsl_half2(*args, **kwargs):
    ...


@triton.jit
def add_half2_kernel(
    x_ptr,
    y_ptr,
    output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    tle_raw.call(edsl_half2, [output_ptr, x_ptr, y_ptr, n_elements])


def add_half2(x: torch.Tensor, y: torch.Tensor):
    output = torch.empty_like(x)
    assert x.device == DEVICE and y.device == DEVICE and output.device == DEVICE
    n_elements = output.numel()
    grid = lambda meta: (triton.cdiv(n_elements // 2, meta["BLOCK_SIZE"]), )
    add_half2_kernel[grid](x, y, output, n_elements, BLOCK_SIZE=1024)
    return output


if __name__ == "__main__":
    x = torch.randn(16384 * 256, device=DEVICE, dtype=torch.float16)
    y = torch.randn(16384 * 256, device=DEVICE, dtype=torch.float16)
    z_half2 = add_half2(x, y)

    assert torch.allclose(x + y, z_half2), (x + y, z_half2)
