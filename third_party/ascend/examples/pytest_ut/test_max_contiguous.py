import torch
import triton
import triton.language as tl
import pytest

@triton.jit
def compile_hint_kernel(
    input_ptr,
    output_ptr,
    n_elements:tl.constexpr,
    SIZE: tl.constexpr
):
    offsets = tl.arange(0, n_elements)
    val = tl.load(input_ptr + offsets)
    val = tl.max_contiguous(val, SIZE)
    tl.store(output_ptr + offsets, val)

@pytest.mark.parametrize('sigtype', ['int32',
                                     #'int64', 'int16', 'int8',
                                     #'uint8', 'uint16', 'uint32', 'uint64',
                                     #'float32', 'float16', 'bfloat16', 'bool'
                                     ])
def test_compile_hint(sigtype):
    n_elements = 10
    dtype = eval(f"torch.{sigtype}")
    x = torch.ones((n_elements,), dtype=dtype).npu()
    y = torch.zeros((n_elements,), dtype=dtype).npu()
    compile_hint_kernel[(1,)](
        x,
        y,
        n_elements,
        1
    )