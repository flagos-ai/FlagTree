from math import gcd
from pathlib import Path

import torch
import triton
import triton.language as tl
import triton.experimental.tle.language.gpu as tle_gpu
import triton.experimental.tle.language.raw as tle_raw
from triton.experimental.tle.raw import dialect


@dialect(
    name="cuda",
    file=Path(__file__).with_suffix(".cu"),
    extern_func_name="SoftmaxFragment",
)
def cuda_softmax(*args, **kwargs):
    ...


@triton.autotune(
    configs=[triton.Config({}, num_warps=n) for n in (1, 2, 4, 8)],
    key=["n_rows", "n_cols", "VECTOR_SIZE"],
)
@triton.jit
def softmax_kernel(
    output_ptr,
    output_row_stride,
    input_ptr,
    input_row_stride,
    n_rows,
    n_cols,
    BLOCK_SIZE: tl.constexpr,
    VECTOR_SIZE: tl.constexpr = 1,
    num_warps: tl.constexpr = 4,
):
    NUM_THREADS: tl.constexpr = num_warps * 32

    # Every physical element has a unique logical position, including padding.
    PADDED_SIZE: tl.constexpr = max(BLOCK_SIZE, NUM_THREADS)

    # VECTOR_SIZE accounts for both pointers, row strides, and the tail mask.
    tl.static_assert(VECTOR_SIZE > 0 and (VECTOR_SIZE & (VECTOR_SIZE - 1)) == 0)
    SIZE_PER_THREAD: tl.constexpr = min(VECTOR_SIZE, PADDED_SIZE // NUM_THREADS)

    layout: tl.constexpr = tle_gpu.BlockEncoding(
        size_per_thread=[SIZE_PER_THREAD],
        threads_per_warp=[32],
        warps_per_cta=[num_warps],
        order=[0],
    )
    row_idx = tl.program_id(0)
    columns = tl.arange(0, PADDED_SIZE)

    # Hint derived values: this compiler does not retain multiple_of on function arguments.
    # Each vector starts aligned and has a uniform mask.
    mask = tl.max_constancy(columns < n_cols, VECTOR_SIZE)
    input_row = tl.multiple_of(input_ptr + row_idx * input_row_stride,
                               VECTOR_SIZE * (input_ptr.dtype.element_ty.primitive_bitwidth // 8))
    output_row = tl.multiple_of(output_ptr + row_idx * output_row_stride,
                                VECTOR_SIZE * (output_ptr.dtype.element_ty.primitive_bitwidth // 8))

    row = tl.load(input_row + columns, mask=mask, other=-float("inf")).to(tl.float32)
    row = tle_gpu.set_layout(row, layout)
    result = tle_raw.call(cuda_softmax, [tle_raw.fragment(row)], output_indices=[0])
    tl.store(output_row + columns, result, mask=mask)


def softmax(x):
    if x.ndim != 2 or x.stride(1) != 1:
        raise ValueError("softmax expects a 2D tensor with contiguous rows")
    if x.dtype not in (torch.float16, torch.bfloat16, torch.float32):
        raise TypeError("softmax supports float16, bfloat16, and float32")
    if x.device.type != "cuda":
        raise ValueError("softmax expects a CUDA tensor")
    n_rows, n_cols = x.shape
    if n_rows == 0 or n_cols == 0:
        return torch.empty_like(x)

    y = torch.empty_like(x)
    with torch.cuda.device(x.device):
        softmax_kernel[(n_rows, )](
            y,
            y.stride(0),
            x,
            x.stride(0),
            n_rows,
            n_cols,
            BLOCK_SIZE=triton.next_power_of_2(n_cols),
            VECTOR_SIZE=get_vector_size(x, y),
        )
    return y


def get_vector_size(x, y):
    # Work in elements; every row start and every mask boundary must align.
    # NVIDIA global loads/stores support at most 128 bits per thread.
    alignment = gcd(x.shape[1], x.stride(0), y.stride(0),
                    x.data_ptr() // x.element_size(),
                    y.data_ptr() // y.element_size())
    return min(alignment & -alignment, 16 // x.element_size(), 16 // y.element_size())


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
                    VECTOR_SIZE=get_vector_size(x, actual),
                )
                torch.testing.assert_close(actual, expected)
            torch.testing.assert_close(softmax(x), expected)
    print("Softmax correctness checks passed.")
