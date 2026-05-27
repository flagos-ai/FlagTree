from pathlib import Path

import torch
import triton
import triton.language as tl
from triton.experimental.tle.raw import dialect
import triton.experimental.tle.language.raw as tle_raw
from triton.language.extra.cuda import libnvshmem_device

DEVICE = triton.runtime.driver.active.get_active_torch_device()


@dialect(name="cuda",
         compiler="nvcc",
         file=Path(__file__).parent / "vector-add-template.cu", 
         extern=Path(__file__).parent / "vector-add-template-extern-call.py",
         extern_func_name="vector_add",
         macros={"VECTOR_ELEM_TYPE": "float"})
def edsl(*args, **kwargs):
    ...


@triton.jit
def add_kernel(
    x_ptr,
    y_ptr,
    output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    tle_raw.call(edsl, [output_ptr, x_ptr, y_ptr, n_elements])


def add(x: torch.Tensor, y: torch.Tensor):
    output = torch.empty_like(x)
    assert x.device == DEVICE and y.device == DEVICE and output.device == DEVICE
    n_elements = output.numel()
    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]), )
    add_kernel[grid](x, y, output, n_elements, BLOCK_SIZE=1024)
    return output


if __name__ == "__main__":
    dtype = torch.float32
    x = torch.randn(2048, dtype=dtype, device=DEVICE)
    y = torch.randn(2048, dtype=dtype, device=DEVICE)
    
    # dtype = torch.int32
    # x = torch.randint(low=0, high=10, size=(2048, ), dtype=dtype, device=DEVICE)
    # y = torch.randint(low=0, high=10, size=(2048, ), dtype=dtype, device=DEVICE)
    
    z = add(x, y)
    print(z)
    assert torch.allclose(x + y, z), (x + y, z)
