import pytest

import triton
import triton.language as tl
import test_common

import torch
import torch_npu

from triton.language import math


def standard_unary(x0, y0, dtype):
    res = torch.atan2(y0, x0)
    return res


def standard_binary(x0, y0, dtype):
    res = x0 + y0
    return res


@triton.jit
def triton_elementwise_unary(in_ptr0, in_ptr1, out_ptr0, N: tl.constexpr, NUMEL: tl.constexpr):
    idx_block = tl.arange(0, NUMEL)
    x = tl.load(in_ptr0 + idx_block, mask=idx_block < N)
    y = tl.load(in_ptr1 + idx_block, mask=idx_block < N)
    ret = math.atan2(y, x)
    tl.store(out_ptr0 + idx_block, ret, mask=idx_block < N)


@triton.jit
def triton_elementwise_binary(in_ptr0, in_ptr1, out_ptr0, N: tl.constexpr, NUMEL: tl.constexpr):
    idx_block = tl.arange(0, NUMEL)
    x = tl.load(in_ptr0 + idx_block, mask=idx_block < N)
    y = tl.load(in_ptr1 + idx_block, mask=idx_block < N)
    ret = x + y
    tl.store(out_ptr0 + idx_block, ret, mask=idx_block < N)


types = [
    (torch.float32, 'float32'),
    (torch.float16, 'float16'),
]

shapes = [
    (3, 32),
    (-32, 32),
    (37, 64),
    (-256, 256),
    (781, 1024),
]

map_for_64_t = {37: 31}


@pytest.mark.parametrize('dtype,sigtype', types)
@pytest.mark.parametrize('N,NUMEL', shapes)
def test_atan2_common(dtype, sigtype, N, NUMEL):
    if N == 0:
        return
    N = (-N) // torch.tensor(0, dtype=dtype).element_size() if N < 0 else N

    if sigtype == "int64":
        N = map_for_64_t[N] if N in map_for_64_t else N

    print(f"elementwise : ({N},) {dtype} {sigtype}")

    x0 = test_common.generate_tensor(shape=(N,), dtype=sigtype)
    y0 = test_common.generate_tensor(shape=(N,), dtype=sigtype)

    ans = standard_unary(x0, y0, dtype)
    x0 = x0.npu()
    y0 = y0.npu()

    out = torch.zeros((N,), dtype=dtype).npu()
    triton_elementwise_unary[1, 1, 1](x0, y0, out, N=N, NUMEL=NUMEL, debug=True)

    test_common.validate_cmp(sigtype, out, ans)

input_vals = [
    (0.0, 1.0),
    (0.0, -1.0),
]


@pytest.mark.parametrize('dtype,sigtype', types)
@pytest.mark.parametrize('N,NUMEL', shapes[0:2])
@pytest.mark.parametrize('X,Y', input_vals)
def test_atan2_special(dtype, sigtype, N, NUMEL, X, Y):
    N = (-N) // torch.tensor(0, dtype=dtype).element_size() if N < 0 else N

    if sigtype == "int64":
        N = map_for_64_t[N] if N in map_for_64_t else N

    print(f"elementwise : ({N},) {dtype} {sigtype}")
    
    x0 = torch.full((N,), X, dtype=eval(f'torch.{sigtype}'))
    y0 = torch.full((N,), Y, dtype=eval(f'torch.{sigtype}'))

    ans = standard_unary(x0, y0, dtype)
    x0 = x0.npu()
    y0 = y0.npu()

    out = torch.zeros((N,), dtype=dtype).npu()
    triton_elementwise_unary[1, 1, 1](x0, y0, out, N=N, NUMEL=NUMEL, debug=True)

    test_common.validate_cmp(sigtype, out, ans)