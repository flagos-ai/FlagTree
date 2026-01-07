import triton
import triton.language as tl
import numpy as np
import torch
import pytest
import test_common

def torch_add(x, y):
    res = x + y
    return res

@triton.jit
def triton_asm_add(x_ptr, 
               y_ptr,
               output_ptr,  
               n_elements,  
               BLOCK_SIZE: tl.constexpr,  
               ):
    pid = tl.program_id(axis=0)  
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    output = tl.inline_asm_elementwise(
        asm="""
        ADD.s64 $0, $1, $2
        """,
        constraints=(
            "=l,l,l"
        ),
        args=[x, y],
        dtype=tl.int64,
        is_pure=True,
        pack=1, 
    )
    tl.store(output_ptr + offsets, output, mask=mask)


@pytest.mark.parametrize('param_list',
                         [
                             ['int64', 4096, 1024],
                         ]
                         )

def test_case(param_list):
    dtype, length, block_size = param_list
    ncore = length // block_size
    x = test_common.generate_tensor((length,), dtype).npu()
    y = test_common.generate_tensor((length,), dtype).npu()
    res_ref = torch_add(x, y)
    res_cal = torch.zeros((length,), dtype = eval('torch.' + dtype)).npu()
    triton_asm_add[(ncore,)](x, y, res_cal, length, BLOCK_SIZE=block_size)
    test_common.validate_cmp(dtype, res_cal, res_ref)
