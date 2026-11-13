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
"""
Coverage tests for tl.map_elementwise decomposition.
One test per op category / control-flow scenario.
"""
import torch
import triton
import triton.language as tl

# ---- integer arith ----


@triton.jit
def _add_i(a, b):
    return tl.add(a, b, sanitize_overflow=False)


def test_map_arith_add_i():

    @triton.jit
    def kernel(X, Y, Z, N: tl.constexpr):
        offs = tl.arange(0, N)
        x = tl.load(X + offs)
        y = tl.load(Y + offs)
        z = tl.map_elementwise(_add_i, x, y)
        tl.store(Z + offs, z)

    N = 128
    x = torch.randint(-100, 100, (N, ), dtype=torch.int32, device='npu')
    y = torch.randint(-100, 100, (N, ), dtype=torch.int32, device='npu')
    z = torch.zeros(N, dtype=torch.int32, device='npu')
    kernel[(1, )](x, y, z, N=N)
    assert torch.equal(z, x + y)


# ---- float arith ----


@triton.jit
def _mul_f(a, b):
    return a * b


def test_map_arith_mul_f():

    @triton.jit
    def kernel(X, Y, Z, N: tl.constexpr):
        offs = tl.arange(0, N)
        x = tl.load(X + offs)
        y = tl.load(Y + offs)
        z = tl.map_elementwise(_mul_f, x, y)
        tl.store(Z + offs, z)

    N = 128
    x = torch.randn(N, dtype=torch.float32, device='npu')
    y = torch.randn(N, dtype=torch.float32, device='npu')
    z = torch.zeros(N, dtype=torch.float32, device='npu')
    kernel[(1, )](x, y, z, N=N)
    assert torch.allclose(z, x * y)


# ---- bitwise ----


@triton.jit
def _bitwise(a, b):
    return a & b, a | b, a ^ b


def test_map_arith_bitwise():

    @triton.jit
    def kernel(A, B, C, D, E, N: tl.constexpr):
        offs = tl.arange(0, N)
        a = tl.load(A + offs)
        b = tl.load(B + offs)
        c, d, e = tl.map_elementwise(_bitwise, a, b)
        tl.store(C + offs, c)
        tl.store(D + offs, d)
        tl.store(E + offs, e)

    N = 128
    a = torch.randint(0, 255, (N, ), dtype=torch.int32, device='npu')
    b = torch.randint(0, 255, (N, ), dtype=torch.int32, device='npu')
    c, d, e = [torch.zeros(N, dtype=torch.int32, device='npu') for _ in range(3)]
    kernel[(1, )](a, b, c, d, e, N=N)
    assert torch.equal(c, a & b) and torch.equal(d, a | b) and torch.equal(e, a ^ b)


# ---- neg ----


@triton.jit
def _neg(a):
    return -a


def test_map_arith_neg_f():

    @triton.jit
    def kernel(X, Z, N: tl.constexpr):
        offs = tl.arange(0, N)
        x = tl.load(X + offs)
        z = tl.map_elementwise(_neg, x)
        tl.store(Z + offs, z)

    N = 128
    x = torch.randn(N, dtype=torch.float32, device='npu')
    z = torch.zeros(N, dtype=torch.float32, device='npu')
    kernel[(1, )](x, z, N=N)
    assert torch.allclose(z, -x)


# ---- comparison + external constant ----


@triton.jit
def _gt_zero(a):
    if a > 0:
        return 1
    else:
        return 0


def test_map_cmp_i():

    @triton.jit
    def kernel(X, Z, N: tl.constexpr):
        offs = tl.arange(0, N)
        x = tl.load(X + offs)
        z = tl.map_elementwise(_gt_zero, x)
        tl.store(Z + offs, z)

    N = 128
    x = torch.randint(-10, 10, (N, ), dtype=torch.int32, device='npu')
    z = torch.zeros(N, dtype=torch.int32, device='npu')
    kernel[(1, )](x, z, N=N)
    expected = (x > 0).to(torch.int32)
    assert torch.equal(z, expected)


# ---- type cast ----


@triton.jit
def _cast(a):
    return a.to(tl.float32)


def test_map_cast_sitofp():

    @triton.jit
    def kernel(X, Z, N: tl.constexpr):
        offs = tl.arange(0, N)
        x = tl.load(X + offs)
        z = tl.map_elementwise(_cast, x)
        tl.store(Z + offs, z)

    N = 128
    x = torch.randint(-100, 100, (N, ), dtype=torch.int32, device='npu')
    z = torch.zeros(N, dtype=torch.float32, device='npu')
    kernel[(1, )](x, z, N=N)
    assert torch.allclose(z, x.float())


# ---- math (float) ----


@triton.jit
def _exp(a):
    return tl.exp(a)


def test_map_math_exp():

    @triton.jit
    def kernel(X, Z, N: tl.constexpr):
        offs = tl.arange(0, N)
        x = tl.load(X + offs)
        z = tl.map_elementwise(_exp, x)
        tl.store(Z + offs, z)

    N = 128
    x = torch.randn(N, dtype=torch.float32, device='npu') * 0.5
    z = torch.zeros(N, dtype=torch.float32, device='npu')
    kernel[(1, )](x, z, N=N)
    assert torch.allclose(z, torch.exp(x), atol=1e-4)


# ---- math (integer) ----


@triton.jit
def _abs_i(a):
    return tl.abs(a)


def test_map_math_abs_i():

    @triton.jit
    def kernel(X, Z, N: tl.constexpr):
        offs = tl.arange(0, N)
        x = tl.load(X + offs)
        z = tl.map_elementwise(_abs_i, x)
        tl.store(Z + offs, z)

    N = 128
    x = torch.randint(-100, 100, (N, ), dtype=torch.int32, device='npu')
    z = torch.zeros(N, dtype=torch.int32, device='npu')
    kernel[(1, )](x, z, N=N)
    assert torch.equal(z, torch.abs(x))


# ---- select ----


@triton.jit
def _where_max(a, b):
    return tl.where(a > b, a, b)


def test_map_select_direct():

    @triton.jit
    def kernel(X, Y, Z, N: tl.constexpr):
        offs = tl.arange(0, N)
        x = tl.load(X + offs)
        y = tl.load(Y + offs)
        z = tl.map_elementwise(_where_max, x, y)
        tl.store(Z + offs, z)

    N = 128
    x = torch.randint(0, 100, (N, ), dtype=torch.int32, device='npu')
    y = torch.randint(0, 100, (N, ), dtype=torch.int32, device='npu')
    z = torch.zeros(N, dtype=torch.int32, device='npu')
    kernel[(1, )](x, y, z, N=N)
    assert torch.equal(z, torch.maximum(x, y))


# ---- 2D tensor ----


@triton.jit
def _add(a, b):
    return a + b


def test_map_2d():

    @triton.jit
    def kernel(X, Y, Z, ROWS: tl.constexpr, COLS: tl.constexpr):
        offs = tl.arange(0, ROWS)[:, None] * COLS + tl.arange(0, COLS)[None, :]
        x = tl.load(X + offs)
        y = tl.load(Y + offs)
        z = tl.map_elementwise(_add, x, y)
        tl.store(Z + offs, z)

    rows, cols = 4, 8
    x = torch.randint(-100, 100, (rows, cols), dtype=torch.int32, device='npu')
    y = torch.randint(-100, 100, (rows, cols), dtype=torch.int32, device='npu')
    z = torch.zeros((rows, cols), dtype=torch.int32, device='npu')
    kernel[(1, )](x, y, z, ROWS=rows, COLS=cols)
    assert torch.equal(z, x + y)


# ---- for loop ----


@triton.jit
def _accumulate(a):
    result = 0
    for i in range(5):
        result = result + a
    return result


def test_map_for_loop():

    @triton.jit
    def kernel(X, Z, N: tl.constexpr):
        offs = tl.arange(0, N)
        x = tl.load(X + offs)
        z = tl.map_elementwise(_accumulate, x)
        tl.store(Z + offs, z)

    N = 128
    x = torch.randint(-10, 10, (N, ), dtype=torch.int32, device='npu')
    z = torch.zeros(N, dtype=torch.int32, device='npu')
    kernel[(1, )](x, z, N=N)
    assert torch.equal(z, x * 5)


# ---- community tests (from test_core.py) ----


@triton.jit
def _compare(x, y):
    if x < y:
        return -1
    elif x == y:
        return 0
    else:
        return 1


def test_map_elementwise():

    @triton.jit
    def kernel(X, Y, Z, BLOCK: tl.constexpr):
        x = tl.load(X + tl.arange(0, BLOCK))
        y = tl.load(Y + tl.arange(0, BLOCK))
        z = tl.map_elementwise(_compare, x, y)
        tl.store(Z + tl.arange(0, BLOCK), z)

    shape = (128, )
    x = torch.randint(-100, 100, shape, dtype=torch.int32, device='npu')
    y = torch.randint(-100, 100, shape, dtype=torch.int32, device='npu')
    z = torch.zeros(shape, dtype=torch.int32, device='npu')
    kernel[(1, )](x, y, z, BLOCK=shape[0])
    expected = (x > y).int() - (y > x).int()
    assert torch.equal(z, expected)


@triton.jit
def _divmod(a, b):
    return a // b, a % b


def test_map_elementwise_multiple_outputs():

    @triton.jit
    def kernel(A, B, C, D, BLOCK: tl.constexpr):
        a = tl.load(A + tl.arange(0, BLOCK))
        b = tl.load(B + tl.arange(0, BLOCK))
        c, d = tl.map_elementwise(_divmod, a, b)
        tl.store(C + tl.arange(0, BLOCK), c)
        tl.store(D + tl.arange(0, BLOCK), d)

    shape = (512, )
    A = torch.randint(1, 100, shape, dtype=torch.int32, device='npu')
    B = torch.randint(1, 10, shape, dtype=torch.int32, device='npu')
    C = torch.zeros(shape, dtype=torch.int32, device='npu')
    D = torch.zeros(shape, dtype=torch.int32, device='npu')
    kernel[(1, )](A, B, C, D, BLOCK=shape[0])
    assert torch.equal(C, A // B) and torch.equal(D, A % B)


@triton.jit
def _divmod_pack2(a0, a1, b0, b1):
    return a0 // b0, a1 // b1, a0 % b0, a1 % b1


def test_map_elementwise_pack():

    @triton.jit
    def kernel(A, B, C, D, BLOCK: tl.constexpr):
        a = tl.load(A + tl.arange(0, BLOCK))
        b = tl.load(B + tl.arange(0, BLOCK))
        c, d = tl.map_elementwise(_divmod_pack2, a, b, pack=2)
        tl.store(C + tl.arange(0, BLOCK), c)
        tl.store(D + tl.arange(0, BLOCK), d)

    shape = (512, )
    A = torch.randint(1, 100, shape, dtype=torch.int32, device='npu')
    B = torch.randint(1, 10, shape, dtype=torch.int32, device='npu')
    C = torch.zeros(shape, dtype=torch.int32, device='npu')
    D = torch.zeros(shape, dtype=torch.int32, device='npu')
    kernel[(1, )](A, B, C, D, BLOCK=shape[0])
    assert torch.equal(C, A // B) and torch.equal(D, A % B)
