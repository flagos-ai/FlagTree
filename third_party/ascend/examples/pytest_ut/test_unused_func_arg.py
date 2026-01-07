# Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.


import triton
import triton.language as tl
import test_common
import torch
import torch_npu
import pytest
import math

def expand_to_next_power_of_two(a):
    if a <= 0:
        raise ValueError("must >0")
    if (math.log2(a)).is_integer():
        return a
    return 2 ** math.ceil(math.log2(a))

@triton.jit
def triton_unused_func_arg_kernel(
      output_ptr, x_ptr,
      X : tl.constexpr, Y : tl.constexpr, Z : tl.constexpr,
      XNUMEL : tl.constexpr, YNUMEL : tl.constexpr, ZNUMEL : tl.constexpr):
    xidx = tl.arange(0,XNUMEL)
    yidx = tl.arange(0,YNUMEL)
    zidx = tl.arange(0,ZNUMEL)
    Xmask = xidx<X
    Ymask = yidx<Y
    Zmask = zidx<Z
    oidx = xidx[:,None,None]*Y*Z + yidx[None,:,None]*Z + zidx[None,None,:]
    mask = (Xmask[:,None,None]) & (Ymask[None,:,None]) & (Zmask[None,None,:])
    abc = tl.load(x_ptr + oidx, mask = mask)
    ret = tl.zeros_like(abc)
    tl.store(output_ptr + oidx, ret, mask = mask)

testlist = [
    (triton_unused_func_arg_kernel,'int8',torch.int8,2,255,9),
    (triton_unused_func_arg_kernel,'int16',torch.int16,3,5,3),
    (triton_unused_func_arg_kernel,'int32',torch.int32,2,255,9),
    (triton_unused_func_arg_kernel,'int64',torch.int64,2,5,3),
    (triton_unused_func_arg_kernel,'float16',torch.float16,55,5,16),
    (triton_unused_func_arg_kernel,'float16',torch.float16,4,5,17),
    (triton_unused_func_arg_kernel,'float16',torch.float16,6,5,15),
    (triton_unused_func_arg_kernel,'float16',torch.float16,2,1928,3),
    (triton_unused_func_arg_kernel,'float32',torch.float32,2,255,9),
    (triton_unused_func_arg_kernel,'bfloat16',torch.bfloat16,3,5,3),
    (triton_unused_func_arg_kernel,'bool',torch.bool,3,5,3),
]

@pytest.mark.parametrize('testfunc, sigtype, dtype, X, Y, Z', testlist)
def test_npu(testfunc, sigtype, dtype, X, Y, Z):
    XNUMEL=expand_to_next_power_of_two(X)
    YNUMEL=expand_to_next_power_of_two(Y)
    ZNUMEL=expand_to_next_power_of_two(Z)
    x = torch.full((X,Y,Z),10,dtype=dtype).npu()
    y = torch.full((X,Y,Z),0,dtype=dtype).npu()
    output = torch.full((X,Y,Z),5,dtype=dtype).npu()
    testfunc[1,1,1](output, x, X, Y, Z, XNUMEL, YNUMEL, ZNUMEL)
    test_common.validate_cmp(sigtype,output,y)

if __name__ == "__main__":
    test_npu(triton_unused_func_arg_kernel, "bool", torch.bool, 3, 5, 3)
