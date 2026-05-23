from pathlib import Path

import torch
import triton
import triton.language as tl
from triton.experimental.tle.raw import dialect
import triton.experimental.tle.language.raw as tle_raw
from triton.language.extra.cuda import libnvshmem_device

DEVICE = triton.runtime.driver.active.get_active_torch_device()


@dialect(
    name="cuda", 
    file=Path(__file__).parent / "vector-add-return.cu", 
    extern=Path(__file__).parent / "vector-add-return-extern-call.py",
    extern_func_name="vector_add_return",
    library={"nvshmem": "/home/zyl/zyuli/envs/nvshmem/lib/python3.12/site-packages/nvidia/nvshmem"}
)
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
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    
    ret = tle_raw.call(edsl, [output_ptr, x_ptr, y_ptr, n_elements])
    base = ret.to(tl.pointer_type(tl.float32), bitcast=True)
    x = tl.load(base + offsets)
    y = tl.full((BLOCK_SIZE, ), 1.0, tl.float32)
    output = x + y
    tl.store(output_ptr + offsets, output)



def add(x: torch.Tensor, y: torch.Tensor):
    output = torch.empty_like(x)
    assert x.device == DEVICE and y.device == DEVICE and output.device == DEVICE
    n_elements = output.numel()
    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]), )
    add_kernel[grid](x, y, output, n_elements, BLOCK_SIZE=1024)
    return output


if __name__ == "__main__":
    x = torch.randn(2048, device=DEVICE)
    y = torch.randn(2048, device=DEVICE)
    extra = torch.ones(2048, device=DEVICE)
    z = add(x, y)
    print(z)
    assert torch.allclose(x + y + extra, z), (x + y + extra, z)
