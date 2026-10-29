from pathlib import Path

import torch
import triton
from triton import knobs
import triton.language as tl
import triton.experimental.tle.language.raw as tle_raw
from triton.experimental.tle.raw import dialect


@dialect(
    name="cuda",
    file=Path(__file__).with_suffix(".cu"),
    extern_func_name="SoftmaxKernel",
)
def cuda_softmax(*args, **kwargs):
    ...


@triton.autotune(configs=[triton.Config({}, num_warps=n) for n in (1, 2, 4, 8)], key=["n_rows", "n_cols"])
@triton.jit
def softmax_kernel(
    output_ptr,
    output_row_stride,
    input_ptr,
    input_row_stride,
    n_rows,
    n_cols,
    BLOCK_SIZE: tl.constexpr,
    num_warps: tl.constexpr = 4,
):
    NUM_THREADS: tl.constexpr = num_warps * 32
    row_idx = tl.program_id(0)
    columns = tl.arange(0, BLOCK_SIZE)
    mask = columns < n_cols
    row = tl.load(input_ptr + row_idx * input_row_stride + columns, mask=mask, other=-float("inf")).to(tl.float32)
    output = tl.zeros_like(row)
    result = tle_raw.call(cuda_softmax, [output, row, NUM_THREADS], output_indices=[0])
    tl.store(output_ptr + row_idx * output_row_stride + columns, result, mask=mask)


def softmax(x):
    if x.ndim != 2 or x.stride(1) != 1:
        raise ValueError("softmax expects a 2D tensor with contiguous rows")
    if x.dtype not in (torch.float16, torch.bfloat16, torch.float32):
        raise TypeError("softmax supports float16, bfloat16, and float32")
    if x.device.type != "cuda":
        raise ValueError("softmax expects a CUDA tensor")
    n_rows, n_cols = x.shape
    y = torch.empty_like(x)
    if n_rows == 0 or n_cols == 0:
        return y
    with torch.cuda.device(x.device), knobs.autotuning.scope():
        # Disable AABS for this kernel; restore the caller's settings on exit.
        # A bug in the current AABS mechanism will cause problems in this case.
        knobs.autotuning.adjust_block_size = False
        softmax_kernel[(n_rows, )](
            y,
            y.stride(0),
            x,
            x.stride(0),
            n_rows,
            n_cols,
            BLOCK_SIZE=triton.next_power_of_2(n_cols),
        )
    return y


if __name__ == "__main__":
    torch.manual_seed(0)
    device = triton.runtime.driver.active.get_active_torch_device()
    # Exercise sub-warp rows, padding, multiple warps, and multiple items/thread.
    for dtype in (torch.float32, torch.float16, torch.bfloat16):
        for n_cols in (1, 4, 31, 32, 33, 127, 128, 129, 512, 1023, 4096):
            x = torch.randn((8, n_cols), device=device, dtype=dtype)
            expected = torch.softmax(x.float(), dim=1).to(dtype)
            # Check every candidate, not just the autotuner's chosen config.
            for num_warps in (1, 2, 4, 8):
                actual = torch.empty_like(x)
                softmax_kernel.fn[(x.shape[0], )](
                    actual,
                    actual.stride(0),
                    x,
                    x.stride(0),
                    *x.shape,
                    BLOCK_SIZE=triton.next_power_of_2(n_cols),
                    num_warps=num_warps,
                )
                torch.testing.assert_close(actual, expected)
            torch.testing.assert_close(softmax(x), expected)
    print("Softmax correctness checks passed.")
