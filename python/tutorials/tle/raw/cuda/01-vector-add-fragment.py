from pathlib import Path

import torch
import triton
import triton.language as tl

from triton.experimental.tle.raw import dialect
import triton.experimental.tle.language.gpu as tle_gpu
import triton.experimental.tle.language.raw as tle_raw


@dialect(name="cuda", file=Path(__file__).parent / "01-vector-add-fragment.cu", extern_func_name='add_fragment')
def cuda_add(*args, **kwargs):
    ...


@triton.jit
def add_kernel(
    x_ptr,
    y_ptr,
    output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
    num_warps: tl.constexpr = 4,
):
    layout: tl.constexpr = tle_gpu.BlockEncoding(
        size_per_thread=[4],
        threads_per_warp=[32],
        warps_per_cta=[num_warps],
        order=[0],
    )
    offset = tl.program_id(0) * BLOCK_SIZE
    offsets = offset + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    # CUDA adds per-thread items, so the layout should be same.
    x = tle_gpu.set_layout(tl.load(x_ptr + offsets, mask=mask), layout)
    y = tle_gpu.set_layout(tl.load(y_ptr + offsets, mask=mask), layout)
    sum = tle_raw.call(cuda_add, [
        tle_raw.fragment(x),
        tle_raw.fragment(y),
    ], output_indices=[0])
    tl.store(output_ptr + offsets, sum, mask=mask)


DEVICE = triton.runtime.driver.active.get_active_torch_device()


def add(x: torch.Tensor, y: torch.Tensor):
    output = torch.empty_like(x)
    assert x.device == DEVICE and y.device == DEVICE
    assert output.device == DEVICE
    n_elements = output.numel()
    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]), )
    add_kernel[grid](x, y, output, n_elements, BLOCK_SIZE=1024)
    return output


if __name__ == "__main__":
    x = torch.randn(2048, device=DEVICE)
    y = torch.randn(2048, device=DEVICE)
    z = add(x, y)
    assert torch.allclose(x + y, z), (x + y, z)
