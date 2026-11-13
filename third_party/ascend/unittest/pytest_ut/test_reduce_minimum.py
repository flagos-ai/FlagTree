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
import torch
import torch_npu
import pytest
import test_common


@triton.jit
def minimum(a, b):
    ret = tl.minimum(a, b, tl.PropagateNan.ALL)
    # 经过测试发现，tl.minimum仅在输入类型为bfloat16时，输出的结果会转变为float32，从而导致编译报错。在GPU上测试发现，和NPU上错误的现象一致。
    # 因此此处针对输入类型为bfloat16的情况，对输出进行了类型转换来规避该错误引起的编译报错。
    if a.dtype == tl.bfloat16:
        ret = ret.to(tl.bfloat16)
    return ret


@triton.jit
def triton_min_5d_dim024(in_ptr0, out_ptr0, L: tl.constexpr, M: tl.constexpr, N: tl.constexpr, K: tl.constexpr,
                         Z: tl.constexpr):
    lblk_idx = tl.arange(0, L)
    mblk_idx = tl.arange(0, M)
    nblk_idx = tl.arange(0, N)
    kblk_idx = tl.arange(0, K)
    zblk_idx = tl.arange(0, Z)
    idx = lblk_idx[:, None, None, None, None] * Z * K * N * M + mblk_idx[None, :, None, None, None] * Z * K * N + \
          nblk_idx[None, None, :, None, None] * Z * K + kblk_idx[None, None, None, :, None] * Z + zblk_idx[
              None, None, None, None, :]
    odx = mblk_idx[:, None] * K + kblk_idx[None, :]
    x = tl.load(in_ptr0 + idx)
    ret = tl.reduce(x, 4, minimum)
    ret1 = tl.reduce(ret, 2, minimum)
    ret2 = tl.reduce(ret1, 0, minimum)
    tl.store(out_ptr0 + odx, ret2)


@triton.jit
def triton_min_5d_dim13(in_ptr0, out_ptr0, L: tl.constexpr, M: tl.constexpr, N: tl.constexpr, K: tl.constexpr,
                        Z: tl.constexpr):
    lblk_idx = tl.arange(0, L)
    mblk_idx = tl.arange(0, M)
    nblk_idx = tl.arange(0, N)
    kblk_idx = tl.arange(0, K)
    zblk_idx = tl.arange(0, Z)

    idx = (lblk_idx[:, None, None, None, None] * Z * K * N * M + mblk_idx[None, :, None, None, None] * Z * K * N +
           nblk_idx[None, None, :, None, None] * Z * K + kblk_idx[None, None, None, :, None] * Z +
           zblk_idx[None, None, None, None, :])

    odx = (lblk_idx[:, None, None] * N * Z + nblk_idx[None, :, None] * Z + zblk_idx[None, None, :])

    x = tl.load(in_ptr0 + idx)
    ret_k = tl.reduce(x, 3, minimum)  # [L, M, N, Z]
    ret_m = tl.reduce(ret_k, 1, minimum)  # [L, N, Z]
    tl.store(out_ptr0 + odx, ret_m)


@triton.jit
def triton_min_5d_dim0(in_ptr0, out_ptr0, L: tl.constexpr, M: tl.constexpr, N: tl.constexpr, K: tl.constexpr,
                       Z: tl.constexpr):
    lblk_idx = tl.arange(0, L)
    mblk_idx = tl.arange(0, M)
    nblk_idx = tl.arange(0, N)
    kblk_idx = tl.arange(0, K)
    zblk_idx = tl.arange(0, Z)

    idx = (lblk_idx[:, None, None, None, None] * Z * K * N * M + mblk_idx[None, :, None, None, None] * Z * K * N +
           nblk_idx[None, None, :, None, None] * Z * K + kblk_idx[None, None, None, :, None] * Z +
           zblk_idx[None, None, None, None, :])

    odx = (mblk_idx[:, None, None, None] * N * K * Z + nblk_idx[None, :, None, None] * K * Z +
           kblk_idx[None, None, :, None] * Z + zblk_idx[None, None, None, :])

    x = tl.load(in_ptr0 + idx)
    ret = tl.reduce(x, 0, minimum)
    tl.store(out_ptr0 + odx, ret)


@triton.jit
def triton_min_5d_dim1(in_ptr0, out_ptr0, L: tl.constexpr, M: tl.constexpr, N: tl.constexpr, K: tl.constexpr,
                       Z: tl.constexpr):
    lblk_idx = tl.arange(0, L)
    mblk_idx = tl.arange(0, M)
    nblk_idx = tl.arange(0, N)
    kblk_idx = tl.arange(0, K)
    zblk_idx = tl.arange(0, Z)

    idx = (lblk_idx[:, None, None, None, None] * Z * K * N * M + mblk_idx[None, :, None, None, None] * Z * K * N +
           nblk_idx[None, None, :, None, None] * Z * K + kblk_idx[None, None, None, :, None] * Z +
           zblk_idx[None, None, None, None, :])

    odx = (lblk_idx[:, None, None, None] * N * K * Z + nblk_idx[None, :, None, None] * K * Z +
           kblk_idx[None, None, :, None] * Z + zblk_idx[None, None, None, :])

    x = tl.load(in_ptr0 + idx)
    ret = tl.reduce(x, 1, minimum)
    tl.store(out_ptr0 + odx, ret)


@triton.jit
def triton_min_5d_dim2(in_ptr0, out_ptr0, L: tl.constexpr, M: tl.constexpr, N: tl.constexpr, K: tl.constexpr,
                       Z: tl.constexpr):
    lblk_idx = tl.arange(0, L)
    mblk_idx = tl.arange(0, M)
    nblk_idx = tl.arange(0, N)
    kblk_idx = tl.arange(0, K)
    zblk_idx = tl.arange(0, Z)

    idx = (lblk_idx[:, None, None, None, None] * Z * K * N * M + mblk_idx[None, :, None, None, None] * Z * K * N +
           nblk_idx[None, None, :, None, None] * Z * K + kblk_idx[None, None, None, :, None] * Z +
           zblk_idx[None, None, None, None, :])

    odx = (lblk_idx[:, None, None, None] * M * K * Z + mblk_idx[None, :, None, None] * K * Z +
           kblk_idx[None, None, :, None] * Z + zblk_idx[None, None, None, :])

    x = tl.load(in_ptr0 + idx)
    ret = tl.reduce(x, 2, minimum)
    tl.store(out_ptr0 + odx, ret)


@triton.jit
def triton_min_5d_dim3(in_ptr0, out_ptr0, L: tl.constexpr, M: tl.constexpr, N: tl.constexpr, K: tl.constexpr,
                       Z: tl.constexpr):
    lblk_idx = tl.arange(0, L)
    mblk_idx = tl.arange(0, M)
    nblk_idx = tl.arange(0, N)
    kblk_idx = tl.arange(0, K)
    zblk_idx = tl.arange(0, Z)

    idx = (lblk_idx[:, None, None, None, None] * Z * K * N * M + mblk_idx[None, :, None, None, None] * Z * K * N +
           nblk_idx[None, None, :, None, None] * Z * K + kblk_idx[None, None, None, :, None] * Z +
           zblk_idx[None, None, None, None, :])

    odx = (lblk_idx[:, None, None, None] * M * N * Z + mblk_idx[None, :, None, None] * N * Z +
           nblk_idx[None, None, :, None] * Z + zblk_idx[None, None, None, :])

    x = tl.load(in_ptr0 + idx)
    ret = tl.reduce(x, 3, minimum)
    tl.store(out_ptr0 + odx, ret)


@triton.jit
def triton_min_5d_dim4(in_ptr0, out_ptr0, L: tl.constexpr, M: tl.constexpr, N: tl.constexpr, K: tl.constexpr,
                       Z: tl.constexpr):
    lblk_idx = tl.arange(0, L)
    mblk_idx = tl.arange(0, M)
    nblk_idx = tl.arange(0, N)
    kblk_idx = tl.arange(0, K)
    zblk_idx = tl.arange(0, Z)

    idx = (lblk_idx[:, None, None, None, None] * Z * K * N * M + mblk_idx[None, :, None, None, None] * Z * K * N +
           nblk_idx[None, None, :, None, None] * Z * K + kblk_idx[None, None, None, :, None] * Z +
           zblk_idx[None, None, None, None, :])

    odx = (lblk_idx[:, None, None, None] * M * N * K + mblk_idx[None, :, None, None] * N * K +
           nblk_idx[None, None, :, None] * K + kblk_idx[None, None, None, :])

    x = tl.load(in_ptr0 + idx)
    ret = tl.reduce(x, 4, minimum)
    tl.store(out_ptr0 + odx, ret)


@triton.jit
def triton_min_5d_all(in_ptr0, out_ptr0, L: tl.constexpr, M: tl.constexpr, N: tl.constexpr, K: tl.constexpr,
                      Z: tl.constexpr):
    lblk_idx = tl.arange(0, L)
    mblk_idx = tl.arange(0, M)
    nblk_idx = tl.arange(0, N)
    kblk_idx = tl.arange(0, K)
    zblk_idx = tl.arange(0, Z)

    idx = (lblk_idx[:, None, None, None, None] * Z * K * N * M + mblk_idx[None, :, None, None, None] * Z * K * N +
           nblk_idx[None, None, :, None, None] * Z * K + kblk_idx[None, None, None, :, None] * Z +
           zblk_idx[None, None, None, None, :])

    x = tl.load(in_ptr0 + idx)
    ret1 = tl.reduce(x, 4, minimum)
    ret2 = tl.reduce(ret1, 3, minimum)
    ret3 = tl.reduce(ret2, 2, minimum)
    ret4 = tl.reduce(ret3, 1, minimum)
    ret5 = tl.reduce(ret4, 0, minimum)
    tl.store(out_ptr0, ret5)


testlist = [
    (triton_min_5d_dim024, (1, 1, 1, 1, 1), "dim024"),
    (triton_min_5d_dim024, (2, 2, 2, 2, 2), "dim024"),
    (triton_min_5d_dim024, (3, 11, 1, 3, 42), "dim024"),
    (triton_min_5d_dim13, (1, 1, 1, 1, 1024), "dim13"),
    (triton_min_5d_dim0, (2, 2, 2, 2, 2), "dim0"),
    (triton_min_5d_dim1, (2, 2, 2, 2, 2), "dim1"),
    (triton_min_5d_dim2, (2, 2, 2, 2, 2), "dim2"),
    (triton_min_5d_dim3, (2, 2, 2, 2, 2), "dim3"),
    (triton_min_5d_dim4, (2, 2, 2, 2, 2), "dim4"),
    (triton_min_5d_all, (3, 11, 1, 3, 42), "all"),
]

typelist = ['int8', 'int16', 'int32', 'int64', 'float16', 'float32', 'bfloat16', 'bool']

ids = [
    "{}-{}-{}".format(testfunc.__name__, "-".join(map(str, shape)), dim_name) for testfunc, shape, dim_name in testlist
]


@pytest.mark.parametrize('testfunc, shape, dim_name', testlist, ids=ids)
@pytest.mark.parametrize('dtype', typelist)
def test_min(testfunc, dtype, shape, dim_name):
    x0 = test_common.generate_tensor(shape=shape, dtype=dtype).npu()

    if dim_name == "dim024":
        if 'int' in dtype:
            ans, _ = torch.min(x0.to(torch.int64), 4)
            ans, _ = torch.min(ans.to(torch.int64), 2)
            ans, _ = torch.min(ans.to(torch.int64), 0)
            ans = ans.to(dtype=eval('torch.' + dtype))
        else:
            ans, _ = torch.min(x0, 4)
            ans, _ = torch.min(ans, 2)
            ans, _ = torch.min(ans, 0)
        output = torch.zeros((shape[1], ) + (shape[3], ), dtype=eval('torch.' + dtype)).npu()

    elif dim_name == "dim13":
        if 'int' in dtype:
            ans, _ = torch.min(x0.to(torch.int64), 3)
            ans, _ = torch.min(ans.to(torch.int64), 1)
            ans = ans.to(dtype=eval('torch.' + dtype))
        else:
            ans, _ = torch.min(x0, 3)
            ans, _ = torch.min(ans, 1)
        output = torch.zeros((shape[0], ) + (shape[2], ) + (shape[4], ), dtype=eval('torch.' + dtype)).npu()

    elif dim_name == "dim0":
        if 'int' in dtype:
            ans, _ = torch.min(x0.to(torch.int64), 0)
            ans = ans.to(dtype=eval('torch.' + dtype))
        else:
            ans, _ = torch.min(x0, 0)
        output = torch.zeros((shape[1], ) + (shape[2], ) + (shape[3], ) + (shape[4], ),
                             dtype=eval('torch.' + dtype)).npu()

    elif dim_name == "dim1":
        if 'int' in dtype:
            ans, _ = torch.min(x0.to(torch.int64), 1)
            ans = ans.to(dtype=eval('torch.' + dtype))
        else:
            ans, _ = torch.min(x0, 1)
        output = torch.zeros((shape[0], ) + (shape[2], ) + (shape[3], ) + (shape[4], ),
                             dtype=eval('torch.' + dtype)).npu()

    elif dim_name == "dim2":
        if 'int' in dtype:
            ans, _ = torch.min(x0.to(torch.int64), 2)
            ans = ans.to(dtype=eval('torch.' + dtype))
        else:
            ans, _ = torch.min(x0, 2)
        output = torch.zeros((shape[0], ) + (shape[1], ) + (shape[3], ) + (shape[4], ),
                             dtype=eval('torch.' + dtype)).npu()

    elif dim_name == "dim3":
        if 'int' in dtype:
            ans, _ = torch.min(x0.to(torch.int64), 3)
            ans = ans.to(dtype=eval('torch.' + dtype))
        else:
            ans, _ = torch.min(x0, 3)
        output = torch.zeros((shape[0], ) + (shape[1], ) + (shape[2], ) + (shape[4], ),
                             dtype=eval('torch.' + dtype)).npu()

    elif dim_name == "dim4":
        if 'int' in dtype:
            ans, _ = torch.min(x0.to(torch.int64), 4)
            ans = ans.to(dtype=eval('torch.' + dtype))
        else:
            ans, _ = torch.min(x0, 4)
        output = torch.zeros((shape[0], ) + (shape[1], ) + (shape[2], ) + (shape[3], ),
                             dtype=eval('torch.' + dtype)).npu()

    elif dim_name == "all":
        if 'int' in dtype:
            ans = torch.min(x0.to(torch.int64))
            ans = torch.tensor([ans], dtype=eval('torch.' + dtype))
        else:
            ans = torch.tensor([torch.min(x0)], dtype=eval('torch.' + dtype))
        output = torch.zeros((1, ), dtype=eval('torch.' + dtype)).npu()

    testfunc[(1, )](x0, output, *shape)

    test_common.validate_cmp(dtype, output, ans)
