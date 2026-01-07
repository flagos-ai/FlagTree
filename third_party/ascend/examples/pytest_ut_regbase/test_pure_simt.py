import triton
import triton.language as tl
import numpy as np
import torch
import pytest
import test_common
import math


def torch_pointwise(x0, x1):
    res = x0 + x1
    return res


@triton.jit
def add_kernel(x_ptr,  # *Pointer* to first input vector.
               y_ptr,  # *Pointer* to second input vector.
               output_ptr,  # *Pointer* to output vector.
               n_elements,  # Size of the vector.
               BLOCK_SIZE: tl.constexpr,  # Number of elements each program should process.
               # NOTE: `constexpr` so it can be used as a shape value.
               ):
    for xoffset in range(0, n_elements, BLOCK_SIZE):
        offsets = xoffset + tl.arange(0, BLOCK_SIZE)[:, None]
        # Create a mask to guard memory operations against out-of-bounds accesses.
        mask = offsets < n_elements
        # Load x and y from DRAM, masking out any extra elements in case the input is not a
        # multiple of the block size.
        x = tl.load(x_ptr + offsets, mask=mask)
        y = tl.load(y_ptr + offsets, mask=mask)
        output = x + y
        # Write x + y back to DRAM.
        tl.store(output_ptr + offsets, output, mask=mask)

@triton.jit
def add_kernel_any_grid(x_ptr,  # *Pointer* to first input vector.
                        y_ptr,  # *Pointer* to second input vector.
                        output_ptr,  # *Pointer* to output vector.
                        n_elements,  # Size of the vector.
                        BLOCK_SIZE: tl.constexpr,  # Number of elements each program should process.
                        # NOTE: `constexpr` so it can be used as a shape value.
                        ):
    pid_id = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    output = x + y
    tl.store(output_ptr + offsets, output, mask=mask)


@pytest.mark.parametrize('param_list',
                         [
                             ['float32', (98432,), 1024],
                             ['float16', (98432,), 1024],
                         ]
                         )
def test_case(param_list, monkeypatch):
    monkeypatch.setenv("TRITON_DISABLE_FFTS", "1")
    dtype, shape, xblock = param_list
    x0 = test_common.generate_tensor(shape, dtype).npu()
    x1 = test_common.generate_tensor(shape, dtype).npu()
    n_elements = math.prod(shape)
    y_ref = torch_pointwise(x0, x1)
    y_cal = torch.zeros(shape, dtype=eval('torch.' + dtype)).npu()
    # TODO: support grid > 1 after we pass grid size as input arguments
    add_kernel[(1, )](x0, x1, y_cal, n_elements, BLOCK_SIZE=xblock, force_simt_only=True)
    test_common.validate_cmp(dtype, y_cal, y_ref)
    monkeypatch.delenv("TRITON_DISABLE_FFTS")


@pytest.mark.parametrize('param_list',
                         [
                             ['float32', (98432,), 1024],
                             ['float16', (98432,), 1024],
                         ]
                         )
def test_any_grid(param_list, monkeypatch):
    monkeypatch.setenv("TRITON_DISABLE_FFTS", "1")
    dtype, shape, xblock = param_list
    x0 = test_common.generate_tensor(shape, dtype).npu()
    x1 = test_common.generate_tensor(shape, dtype).npu()
    n_elements = math.prod(shape)
    y_ref = torch_pointwise(x0, x1)
    y_cal = torch.zeros(shape, dtype=eval('torch.' + dtype)).npu()
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']), )
    add_kernel[grid](x0, x1, y_cal, n_elements, BLOCK_SIZE=xblock, force_simt_only=True)
    test_common.validate_cmp(dtype, y_cal, y_ref)
    monkeypatch.delenv("TRITON_DISABLE_FFTS")