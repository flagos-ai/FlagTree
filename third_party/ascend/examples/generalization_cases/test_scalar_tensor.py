import triton
import triton.language as tl
import numpy as np
import torch
import pytest
import test_common

def torch_(x0, x1, op_type):
    if op_type == 'mul':
        return torch.tensor(x0 * x1)
    elif op_type == 'lshift':
        return torch.tensor(x0 << x1)
    elif op_type == 'eq':
        return torch.tensor(x0 == x1)
    else :
        raise TypeError('Invalid op_type')


@triton.jit
def scalar_mul(out_ptr0, val0: tl.constexpr, val1: tl.constexpr):
    scalar0 = tl.core.tensor(val0, tl.core.block_type(tl.float32, []))
    scalar1 = tl.core.tensor(val1, tl.core.block_type(tl.float32, []))
    ret = scalar0 * scalar1
    tl.store(out_ptr0, ret)

@triton.jit
def scalar_lshift(out_ptr0, val0: tl.constexpr, val1: tl.constexpr):
    scalar0 = tl.core.tensor(val0, tl.core.block_type(tl.int32, []))
    scalar1 = tl.core.tensor(val1, tl.core.block_type(tl.int32, []))
    ret = scalar0 << scalar1
    tl.store(out_ptr0, ret)

@triton.jit
def scalar_eq(out_ptr0, val0: tl.constexpr, val1: tl.constexpr):
    scalar0 = tl.core.tensor(val0, tl.core.block_type(tl.int16, []))
    scalar1 = tl.core.tensor(val1, tl.core.block_type(tl.int16, []))
    ret = scalar0 == scalar1
    tl.store(out_ptr0, ret)

@pytest.mark.parametrize('param_list',
                         [
                             ['float32', 'mul', (1,), 3.14, 6.66],
                             ['int32', 'lshift', (1,), 6, 7],
                             ['bool', 'eq', (1,), 5, 5],
                         ]
                         )
@test_common.raises_with_match(triton.compiler.errors.CompilationError, "0d block_type is forbidden")
def test_case(param_list):
    dtype, op_type, shape, lval, rval = param_list
    ans = torch_(lval, rval, op_type)
    ret = torch.zeros(shape, dtype = eval('torch.' + dtype)).npu()

    if op_type == 'mul':
        scalar_mul[1, 1, 1](ret, lval, rval)
    elif op_type == 'lshift':
        scalar_lshift[1, 1, 1](ret, lval, rval)
    elif op_type == 'eq':
        scalar_eq[1, 1, 1](ret, lval, rval)
    
    test_common.validate_cmp(dtype, ans, ret)