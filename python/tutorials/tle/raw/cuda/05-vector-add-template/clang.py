from pathlib import Path

import torch
import triton
import triton.language as tl
from triton.experimental.tle.raw import dialect
import triton.experimental.tle.language.raw as tle_raw

DEVICE = triton.runtime.driver.active.get_active_torch_device()


@dialect(name="cuda", file=Path(__file__).parent / "01-vector-add.cu")
def edsl(*args, **kwargs):
    ...


@triton.jit
def add_kernel_tle(
    x_ptr,
    y_ptr,
    output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    tle_raw.call(edsl, [output_ptr, x_ptr, y_ptr, n_elements])


def add_tle(x: torch.Tensor, y: torch.Tensor):
    output = torch.empty_like(x)
    assert x.device == DEVICE and y.device == DEVICE and output.device == DEVICE
    n_elements = output.numel()
    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]), )
    add_kernel_tle[grid](x, y, output, n_elements, BLOCK_SIZE=1024)
    return output

@triton.jit
def add_kernel_triton(x_ptr,  # *Pointer* to first input vector.
               y_ptr,  # *Pointer* to second input vector.
               output_ptr,  # *Pointer* to output vector.
               n_elements,  # Size of the vector.
               BLOCK_SIZE: tl.constexpr,  # Number of elements each program should process.
               ):
    pid = tl.program_id(axis=0)  # We use a 1D launch grid so axis is 0.
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    output = x + y
    tl.store(output_ptr + offsets, output, mask=mask)

def add_triton(x: torch.Tensor, y: torch.Tensor):
    output = torch.empty_like(x)
    assert x.device == DEVICE and y.device == DEVICE and output.device == DEVICE
    n_elements = output.numel()
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']), )
    add_kernel_triton[grid](x, y, output, n_elements, BLOCK_SIZE=1024)
    return output


if __name__ == "__main__":
    x = torch.randn((16384, 32768), device=DEVICE, dtype=torch.float32)
    y = torch.randn((16384, 32768), device=DEVICE, dtype=torch.float32)
    output_tle = add_tle(x, y)
    # mean_ms_tle = triton.testing.do_bench(lambda: add_tle(x, y))
    # print(f"TLE time: {mean_ms_tle:.4f} ms")
    output_triton = add_triton(x, y)

    if torch.allclose(output_triton, output_tle, atol=0.125, rtol=0):
        print("✅  output Triton and TLE match")
    else:
        print("❌  output Triton and TLE differ")


    # perf
    dtype = torch.float32
    for i in range(27, 31, 1):
        x = torch.randn(2 ** i, device=DEVICE, dtype=dtype)
        y = torch.randn(2 ** i, device=DEVICE, dtype=dtype)
        mean_ms_tle = triton.testing.do_bench(lambda: add_tle(x, y))
        mean_ms_triton = triton.testing.do_bench(lambda: add_triton(x, y))

        print(f"\n========  x: 2 **{i}    type: {dtype}  =========")
        print(f"Triton Time: {mean_ms_triton:.3f} ms")
        print(f"TLE Time: {mean_ms_tle:.3f} ms")

    shapes = [(1024, 2048), (2048, 4096), (4096, 8192), (16384, 32768)]
    dtypes = [torch.float32]
    for shape in shapes:
        for dtype in dtypes:
            x = torch.rand(shape, device=DEVICE, dtype=dtype)
            y = torch.rand(shape, device=DEVICE, dtype=dtype)
            mean_ms_triton = triton.testing.do_bench(lambda: add_triton(x, y))
            mean_ms_tle = triton.testing.do_bench(lambda: add_tle(x, y))

            print(f"\n=========  Shape: {shape}   Type: {dtype}  =========")
            print(f"Triton time: {mean_ms_triton:.4f} ms")
            print(f"TLE time: {mean_ms_tle:.4f} ms")
