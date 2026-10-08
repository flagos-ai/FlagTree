from pathlib import Path

import torch
import triton
import triton.language as tl
import triton.experimental.tle.language.raw as tle_raw
from triton.experimental.tle.raw import dialect

DEVICE = triton.runtime.driver.active.get_active_torch_device()
CUDA_SOURCE = Path(__file__).parent / "04-vector-add-template.cu"


@dialect(
    name="cuda",
    file=CUDA_SOURCE,
    extern_func_name="VectorAdd_half",
    defines={"VECTOR_ELEM_TYPE": "half"},
)
def vector_add_fp16(*args, **kwargs):
    ...


@dialect(
    name="cuda",
    file=CUDA_SOURCE,
    extern_func_name="VectorAdd_half",
    defines={"VECTOR_ELEM_TYPE": "half"},
    deferred=True,
)
def vector_add_fp16_deferred(*args, **kwargs):
    ...


@dialect(
    name="cuda",
    file=CUDA_SOURCE,
    extern_func_name="VectorAdd_float",
    defines={"VECTOR_ELEM_TYPE": "float"},
)
def vector_add_fp32(*args, **kwargs):
    ...


@dialect(
    name="cuda",
    file=CUDA_SOURCE,
    extern_func_name="VectorAdd_float",
    defines={"VECTOR_ELEM_TYPE": "float"},
    deferred=True,
)
def vector_add_fp32_deferred(*args, **kwargs):
    ...


@triton.jit
def add_kernel(
    x_ptr,
    y_ptr,
    output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
    FP16: tl.constexpr,
    DEFERRED: tl.constexpr,
):
    if FP16:
        if DEFERRED:
            tle_raw.call(
                vector_add_fp16_deferred,
                [output_ptr, x_ptr, y_ptr, n_elements],
                output_indices=[0],
            )
        else:
            tle_raw.call(vector_add_fp16, [output_ptr, x_ptr, y_ptr, n_elements])
    else:
        if DEFERRED:
            tle_raw.call(
                vector_add_fp32_deferred,
                [output_ptr, x_ptr, y_ptr, n_elements],
                output_indices=[0],
            )
        else:
            tle_raw.call(vector_add_fp32, [output_ptr, x_ptr, y_ptr, n_elements])


@triton.jit
def add_two_dtypes_kernel(
    x_fp16_ptr,
    y_fp16_ptr,
    output_fp16_ptr,
    x_fp32_ptr,
    y_fp32_ptr,
    output_fp32_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
    DEFERRED: tl.constexpr,
):
    if DEFERRED:
        tle_raw.call(
            vector_add_fp16_deferred,
            [output_fp16_ptr, x_fp16_ptr, y_fp16_ptr, n_elements],
            output_indices=[0],
        )
        tle_raw.call(
            vector_add_fp32_deferred,
            [output_fp32_ptr, x_fp32_ptr, y_fp32_ptr, n_elements],
            output_indices=[0],
        )
    else:
        tle_raw.call(
            vector_add_fp16,
            [output_fp16_ptr, x_fp16_ptr, y_fp16_ptr, n_elements],
        )
        tle_raw.call(
            vector_add_fp32,
            [output_fp32_ptr, x_fp32_ptr, y_fp32_ptr, n_elements],
        )


def add(x: torch.Tensor, y: torch.Tensor, *, deferred: bool) -> torch.Tensor:
    assert x.shape == y.shape
    assert x.dtype == y.dtype
    assert x.dtype in (torch.float16, torch.float32)
    assert x.device == DEVICE and y.device == DEVICE
    output = torch.empty_like(x)
    n_elements = output.numel()
    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]), )
    add_kernel[grid](
        x,
        y,
        output,
        n_elements,
        BLOCK_SIZE=1024,
        FP16=x.dtype == torch.float16,
        DEFERRED=deferred,
    )
    return output


def run_case(dtype: torch.dtype, *, deferred: bool) -> None:
    x = torch.randn(2053, device=DEVICE, dtype=dtype)
    y = torch.randn(2053, device=DEVICE, dtype=dtype)
    actual = add(x, y, deferred=deferred)
    torch.testing.assert_close(actual, x + y, rtol=0, atol=0)
    mode = "deferred" if deferred else "standard"
    print(f"{mode} {dtype}: PASSED")


def run_two_dtypes_case(*, deferred: bool) -> None:
    n_elements = 2053
    x_fp16 = torch.randn(n_elements, device=DEVICE, dtype=torch.float16)
    y_fp16 = torch.randn(n_elements, device=DEVICE, dtype=torch.float16)
    x_fp32 = torch.randn(n_elements, device=DEVICE, dtype=torch.float32)
    y_fp32 = torch.randn(n_elements, device=DEVICE, dtype=torch.float32)
    output_fp16 = torch.empty_like(x_fp16)
    output_fp32 = torch.empty_like(x_fp32)
    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]), )
    add_two_dtypes_kernel[grid](
        x_fp16,
        y_fp16,
        output_fp16,
        x_fp32,
        y_fp32,
        output_fp32,
        n_elements,
        BLOCK_SIZE=1024,
        DEFERRED=deferred,
    )
    torch.testing.assert_close(output_fp16, x_fp16 + y_fp16, rtol=0, atol=0)
    torch.testing.assert_close(output_fp32, x_fp32 + y_fp32, rtol=0, atol=0)
    mode = "deferred" if deferred else "standard"
    print(f"{mode} mixed fp16/fp32: PASSED")


if __name__ == "__main__":
    torch.manual_seed(0)
    for dtype in (torch.float16, torch.float32):
        for deferred in (False, True):
            run_case(dtype, deferred=deferred)
    for deferred in (False, True):
        run_two_dtypes_case(deferred=deferred)
