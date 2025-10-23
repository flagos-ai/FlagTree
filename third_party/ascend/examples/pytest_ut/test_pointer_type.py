import triton
import triton.language as tl

import torch
import torch_npu
import pytest
import test_common

@triton.jit
def kernel(ans_ptr, x_ptr):
    val = tl.load(x_ptr)
    output_ptr = tl.load(ans_ptr)
    output_ptr = output_ptr.to(tl.pointer_type(val.dtype))
    tl.store(output_ptr, val)

@pytest.mark.parametrize("literal, dtype_str",[[0, eval('torch.int8')], [0, eval('torch.int16')],
                                               [0, eval('torch.int32')], [0, eval('torch.int64')],
                                               [0, eval('torch.float16')], [0, eval('torch.float32')]])
def test_pointer_type(literal, dtype_str):
    x = torch.randint(low=0, high=5, size=(1,), dtype=dtype_str).npu()
    output = torch.zeros((1,), dtype=dtype_str).npu()
    ans = []
    ans.append(output.data_ptr())
    ans_tensor = torch.tensor(ans).npu()
    kernel[(1,)](ans_tensor, x)
    assert torch.isclose(x, output)
    print("Pointer type convert successful")