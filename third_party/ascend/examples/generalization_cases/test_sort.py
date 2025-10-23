import triton
import pytest
import torch
import triton.language as tl
import test_common
from test_common import TestUtils


# ----------------------
# 1D sort kernel
# ----------------------
@triton.jit
def sort_kernel_1d(X, Z, M: tl.constexpr, descending: tl.constexpr):
    off = tl.arange(0, M)
    x = tl.load(X + off)
    x = tl.sort(x, descending=descending, dim=0)
    tl.store(Z + off, x)


@pytest.mark.interpreter
@pytest.mark.parametrize("shape", TestUtils.test_shape1d)
@pytest.mark.parametrize("descending", [False, True])
@pytest.mark.parametrize("dtype", ["int8", "int16", "float16", "float32", "bfloat16"])
def test_sort_1d(shape, descending, dtype):
    x = test_common.generate_tensor(shape, dtype).npu()
    torch_res = torch.sort(x, descending=descending, dim=0)[0]

    triton_res = torch.zeros_like(x)
    M = x.shape[0]
    sort_kernel_1d[(1, )](x, triton_res, M, descending)
    assert torch.equal(torch_res, triton_res)


# ----------------------
# 2D sort kernel (split by rows, not cutting M axis)
# ----------------------
@triton.jit
def sort_kernel_2d(X, Z, N: tl.constexpr, M: tl.constexpr, descending: tl.constexpr):
    pid = tl.program_id(0)
    offx = tl.arange(0, M)
    offy = pid * M
    off2d = offx + offy
    x = tl.load(X + off2d)
    x = tl.sort(x, descending=descending, dim=0)
    tl.store(Z + off2d, x)


@pytest.mark.interpreter
@pytest.mark.parametrize("shape", TestUtils.test_shape2d)
@pytest.mark.parametrize("descending", [False, True])
@pytest.mark.parametrize("dtype", ["int8", "int16", "float16", "float32", "bfloat16"])
def test_sort_2d(shape, descending, dtype):
    x = test_common.generate_tensor(shape, dtype).npu()
    torch_res = torch.sort(x, descending=descending, dim=1)[0]

    triton_res = torch.zeros_like(x)
    N, M = x.shape
    # 每行一个 block
    sort_kernel_2d[(N, )](x, triton_res, N, M, descending)
    assert torch.equal(torch_res, triton_res), (torch_res, triton_res)


# ----------------------
# 3D sort kernel (split by D0, D1, not cutting D2)
# ----------------------
@triton.jit
def sort_kernel_3d(X, Z, D0: tl.constexpr, D1: tl.constexpr, D2: tl.constexpr, descending: tl.constexpr):
    pid = tl.program_id(0)
    row_id = pid % D1
    batch_id = pid // D1

    off2 = tl.arange(0, D2)
    off1 = row_id * D2
    off0 = batch_id * D1 * D2
    off = off2 + off1 + off0

    x = tl.load(X + off)
    x = tl.sort(x, descending=descending, dim=0)  # 一整行排序
    tl.store(Z + off, x)


@pytest.mark.interpreter
@pytest.mark.parametrize("shape", TestUtils.test_shape3d)
@pytest.mark.parametrize("descending", [False, True])
@pytest.mark.parametrize("dtype", ["int8", "int16", "float16", "float32", "bfloat16"])
def test_sort_3d(shape, descending, dtype):
    x = test_common.generate_tensor(shape, dtype).npu()
    torch_res = torch.sort(x, descending=descending, dim=2)[0]

    triton_res = torch.zeros_like(x)
    D0, D1, D2 = x.shape
    # 每个 (D0,D1) 对应一个 block
    sort_kernel_3d[(D0 * D1, )](x, triton_res, D0, D1, D2, descending)
    assert torch.equal(torch_res, triton_res), (torch_res, triton_res)


# ----------------------
# 4D sort kernel
# ----------------------
@triton.jit
def sort_kernel_4d(X, Z,
                   D0: tl.constexpr, D1: tl.constexpr,
                   D2: tl.constexpr, D3: tl.constexpr,
                   descending: tl.constexpr):
    pid = tl.program_id(0)
    row_id = pid % D2
    col_id = (pid // D2) % D1
    batch_id = pid // (D1 * D2)

    off3 = tl.arange(0, D3)
    off2 = row_id * D3
    off1 = col_id * D2 * D3
    off0 = batch_id * D1 * D2 * D3
    off = off3 + off2 + off1 + off0

    x = tl.load(X + off)
    x = tl.sort(x, descending=descending, dim=0)
    tl.store(Z + off, x)


@pytest.mark.interpreter
@pytest.mark.parametrize("shape", TestUtils.test_shape4d)
@pytest.mark.parametrize("descending", [False, True])
@pytest.mark.parametrize("dtype", ["int8", "int16", "float16", "float32", "bfloat16"])
def test_sort_4d(shape, descending, dtype):
    x = test_common.generate_tensor(shape, dtype).npu()
    torch_res = torch.sort(x, descending=descending, dim=3)[0]

    triton_res = torch.zeros_like(x)
    D0, D1, D2, D3 = x.shape
    sort_kernel_4d[(D0 * D1 * D2, )](x, triton_res, D0, D1, D2, D3, descending)
    assert torch.equal(torch_res, triton_res)


# ----------------------
# 5D sort kernel
# ----------------------
@triton.jit
def sort_kernel_5d(X, Z,
                   D0: tl.constexpr, D1: tl.constexpr,
                   D2: tl.constexpr, D3: tl.constexpr,
                   D4: tl.constexpr,
                   descending: tl.constexpr):
    pid = tl.program_id(0)
    row_id = pid % D3
    col_id = (pid // D3) % D2
    depth_id = (pid // (D2 * D3)) % D1
    batch_id = pid // (D1 * D2 * D3)

    off4 = tl.arange(0, D4)
    off3 = row_id * D4
    off2 = col_id * D3 * D4
    off1 = depth_id * D2 * D3 * D4
    off0 = batch_id * D1 * D2 * D3 * D4
    off = off4 + off3 + off2 + off1 + off0

    x = tl.load(X + off)
    x = tl.sort(x, descending=descending, dim=0)
    tl.store(Z + off, x)


@pytest.mark.interpreter
@pytest.mark.parametrize("shape", TestUtils.test_shape5d)
@pytest.mark.parametrize("descending", [False, True])
@pytest.mark.parametrize("dtype", ["int8", "int16", "float16", "float32", "bfloat16"])
def test_sort_5d(shape, descending, dtype):
    x = test_common.generate_tensor(shape, dtype).npu()
    torch_res = torch.sort(x, descending=descending, dim=4)[0]

    triton_res = torch.zeros_like(x)
    D0, D1, D2, D3, D4 = x.shape
    sort_kernel_5d[(D0 * D1 * D2 * D3, )](x, triton_res, D0, D1, D2, D3, D4, descending)
    assert torch.equal(torch_res, triton_res)