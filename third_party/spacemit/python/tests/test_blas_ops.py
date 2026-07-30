import pytest
import torch

import triton
from triton.backends.spine_triton.driver import CPUDriver

triton.runtime.driver.set_active(CPUDriver())
import flag_gems

from .accuracy_utils import (
    FLOAT_DTYPES,
    SCALARS,
    UT_SHAPES_1D,
    gems_assert_close,
    to_reference,
)

MN_SHAPES = [(1, 32), (160, 1024), (5333, 497)]
MNK_SHAPES = ([(1, 1, 32), (15, 160, 1024), (495, 5333, 71)])


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

    gems_assert_close(res_out1, ref_out1, dtype, reduce_dim=K)


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

    torch.testing.assert_close(ref_out, res_out, atol=1e-2, rtol=0)


@pytest.mark.outer
@pytest.mark.parametrize("M, N", MN_SHAPES)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_accuracy_outer(M, N, dtype):
    inp1 = torch.randn(M, dtype=dtype, device=flag_gems.device, requires_grad=True)
    inp2 = torch.randn(N, dtype=dtype, device=flag_gems.device, requires_grad=True)

    ref_out = torch.outer(inp1, inp2)
    res_out = flag_gems.outer(inp1, inp2)
    gems_assert_close(res_out, ref_out, dtype)


@pytest.mark.skipif(flag_gems.vendor_name == "spacemit", reason="TODO")
@pytest.mark.vdot
@pytest.mark.parametrize("M", UT_SHAPES_1D)
@pytest.mark.parametrize("is_conj", [(False, False), (False, True), (True, False), (True, True)])
@pytest.mark.parametrize("dtype", FLOAT_DTYPES + [torch.cfloat])
@pytest.mark.parametrize("stride", [1, 2])
def test_accuracy_vdot(M, is_conj, dtype, stride):
    inp1_is_conj, inp2_is_conj = is_conj

    if flag_gems.device == "musa":
        inp1 = torch.randn(M, dtype=dtype, device="cpu")
        inp2 = torch.randn(M, dtype=dtype, device="cpu")
    else:
        inp1 = torch.randn(M, dtype=dtype, device=flag_gems.device)
        inp2 = torch.randn(M, dtype=dtype, device=flag_gems.device)

    inp1 = inp1[::stride]
    inp2 = inp2[::stride]

    if inp1_is_conj:
        inp1 = inp1.conj()
    if inp2_is_conj:
        inp2 = inp2.conj()

    ref_inp1 = to_reference(inp1, True)
    ref_inp2 = to_reference(inp2, True)

    with flag_gems.use_gems():
        if flag_gems.device == "musa":
            res_out = torch.vdot(inp1.to(device=flag_gems.device), inp2.to(device=flag_gems.device))
        else:
            res_out = torch.vdot(inp1, inp2)
    ref_out = torch.vdot(ref_inp1, ref_inp2)
    gems_assert_close(res_out, ref_out, dtype)
