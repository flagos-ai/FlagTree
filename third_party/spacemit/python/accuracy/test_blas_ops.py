import pytest
import torch

import triton
from triton.backends.spine_triton.driver import CPUDriver

triton.runtime.driver.set_active(CPUDriver())
import flag_gems

MN_SHAPES = [(1, 32), (5333, 497)]
MNK_SHAPES = ([(1, 1, 32), (15, 160, 1024), (495, 5333, 71)])
FLOAT_DTYPES = [torch.float32, torch.float16]
SCALARS = [0.001]


def to_cpu(res, ref):
    res = res.to("cpu")
    assert ref.device == torch.device("cpu")
    return res


def gems_assert_close(res, ref, dtype, equal_nan=False, reduce_dim=1):
    res = to_cpu(res, ref)
    flag_gems.testing.assert_close(res, ref, dtype, equal_nan=equal_nan, reduce_dim=reduce_dim)


@pytest.mark.addmm
@pytest.mark.linear
@pytest.mark.matmul
@pytest.mark.parametrize("M, N, K", MNK_SHAPES)
@pytest.mark.parametrize("scalar", SCALARS)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_accuracy_addmm(M, N, K, scalar, dtype):
    mat1 = torch.randn((M, K), dtype=dtype, device=flag_gems.device, requires_grad=False)
    mat2 = torch.randn((K, N), dtype=dtype, device=flag_gems.device, requires_grad=False)
    bias1 = torch.randn((N, ), dtype=dtype, device=flag_gems.device, requires_grad=False)

    alpha = beta = scalar

    ref_out1 = torch.addmm(bias1, mat1, mat2, alpha=alpha, beta=beta)
    with flag_gems.use_gems():
        with torch.no_grad():
            res_out1 = torch.addmm(bias1, mat1, mat2, alpha=alpha, beta=beta)

    gems_assert_close(res_out1, ref_out1, dtype)


@pytest.mark.mm
@pytest.mark.parametrize("M, N, K", MNK_SHAPES)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_accuracy_mm(M, N, K, dtype):
    mat1 = torch.randn((M, K), dtype=dtype, device=flag_gems.device)
    mat2 = torch.randn((K, N), dtype=dtype, device=flag_gems.device)

    ref_out = torch.mm(mat1, mat2)
    with flag_gems.use_gems():
        res_out = torch.mm(mat1, mat2)

    gems_assert_close(res_out, ref_out, dtype, reduce_dim=K)


@pytest.mark.bmm
@pytest.mark.parametrize("M, N, K", MNK_SHAPES)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_accuracy_bmm(M, N, K, dtype):
    mat1 = torch.randn((4, M, K), dtype=dtype, device=flag_gems.device)
    mat2 = torch.randn((4, K, N), dtype=dtype, device=flag_gems.device)

    ref_out = torch.bmm(mat1, mat2)
    with flag_gems.use_gems():
        with torch.no_grad():
            res_out = torch.bmm(mat1, mat2)

    gems_assert_close(res_out, ref_out, dtype, reduce_dim=K)


@pytest.mark.mv
@pytest.mark.parametrize("M, N", MN_SHAPES)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_accuracy_mv(M, N, dtype):
    matrix = torch.randn((N, M), dtype=dtype, device=flag_gems.device)
    vector = torch.randn((M, ), dtype=dtype, device=flag_gems.device)

    ref_out = torch.mv(matrix, vector)
    with flag_gems.use_gems():
        res_out = torch.mv(matrix, vector)

    gems_assert_close(res_out, ref_out, dtype, reduce_dim=M)


@pytest.mark.outer
@pytest.mark.parametrize("M, N", MN_SHAPES)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_accuracy_outer(M, N, dtype):
    inp1 = torch.randn(M, dtype=dtype, device=flag_gems.device, requires_grad=True)
    inp2 = torch.randn(N, dtype=dtype, device=flag_gems.device, requires_grad=True)

    ref_out = torch.outer(inp1, inp2)
    res_out = flag_gems.outer(inp1, inp2)
    gems_assert_close(res_out, ref_out, dtype)
