import pytest
import torch

import triton
import triton.language as tl
from triton._flagtree_backend import FLAGTREE_BACKEND
from triton.backends.compiler import GPUTarget
from triton.compiler import ASTSource

pytestmark = pytest.mark.skipif(FLAGTREE_BACKEND != "hcu", reason="HCU MMAC regression tests")

CONFIG = (32, 32, 16, 4)


@pytest.fixture
def fp64_mmac_device(device):
    target = triton.runtime.driver.active.get_current_target()
    if target.backend != "hip" or target.arch != "gfx936":
        pytest.skip("Requires gfx936")
    return device


@triton.jit
def _dot_kernel(A, B, C, D, M: tl.constexpr, N: tl.constexpr, K: tl.constexpr, STRIDE_AM: tl.constexpr,
                STRIDE_AK: tl.constexpr, STRIDE_BK: tl.constexpr, STRIDE_BN: tl.constexpr):
    m = tl.arange(0, M)
    n = tl.arange(0, N)
    k = tl.arange(0, K)
    a = tl.load(A + m[:, None] * STRIDE_AM + k[None, :] * STRIDE_AK)
    b = tl.load(B + k[:, None] * STRIDE_BK + n[None, :] * STRIDE_BN)
    offsets = m[:, None] * N + n[None, :]
    c = tl.load(C + offsets)
    d = tl.dot(a, b, c, input_precision="ieee", out_dtype=D.dtype.element_ty)
    tl.store(D + offsets, d)


def _compile_dot(kpack, arch="gfx936"):
    m, n, k, num_warps = CONFIG
    source = ASTSource(
        fn=_dot_kernel,
        signature={name: "*fp64"
                   for name in ("A", "B", "C", "D")},
        constexprs=dict(M=m, N=n, K=k, STRIDE_AM=k, STRIDE_AK=1, STRIDE_BK=n, STRIDE_BN=1),
    )
    return triton.compile(source, target=GPUTarget("hip", arch, 64), options=dict(kpack=kpack, num_warps=num_warps,
                                                                                  num_stages=1))


def _assert_fp64_mmac(kernel):
    assert "v_mmac_16x16x4_f64" in kernel.asm["amdgcn"]
    assert "v_fma_f64" not in kernel.asm["amdgcn"]
    assert "@llvm.hcu.mmac.16x16x4.f64(double " in kernel.asm["llir"]


@pytest.mark.parametrize("kpack", [1, 2])
def test_fp64_mmac_codegen(kpack):
    kernel = _compile_dot(kpack)
    _assert_fp64_mmac(kernel)


@pytest.mark.parametrize("arch", ["gfx928"])
def test_fp64_fma_codegen(arch):
    kernel = _compile_dot(kpack=1, arch=arch)
    assert "llvm.hcu.mmac.16x16x4.f64" not in kernel.asm["llir"]
    assert "v_mmac_16x16x4_f64" not in kernel.asm["amdgcn"]
    assert "@llvm.fmuladd.f64(double " in kernel.asm["llir"]


@pytest.mark.parametrize("kpack", [1, 2])
@pytest.mark.parametrize("transpose_a,transpose_b", [(False, False), (True, False), (False, True), (True, True)])
def test_fp64_mmac(kpack, transpose_a, transpose_b, fp64_mmac_device):
    device = fp64_mmac_device
    m, n, k, num_warps = CONFIG
    torch.manual_seed(1231)
    a = torch.randn((k, m) if transpose_a else (m, k), dtype=torch.float64, device=device)
    b = torch.randn((n, k) if transpose_b else (k, n), dtype=torch.float64, device=device)
    if transpose_a:
        a = a.T
    if transpose_b:
        b = b.T
    # Check the full A @ B + C operation with a nonzero initial accumulator.
    c = torch.randn((m, n), dtype=torch.float64, device=device)
    d = torch.empty_like(c)
    kernel = _dot_kernel[(1, )](
        a,
        b,
        c,
        d,
        m,
        n,
        k,
        *a.stride(),
        *b.stride(),
        num_warps=num_warps,
        num_stages=1,
        kpack=kpack,
    )
    _assert_fp64_mmac(kernel)
    torch.testing.assert_close(d, a @ b + c, atol=1e-12, rtol=1e-12)


@pytest.mark.parametrize("kpack", [1, 2])
def test_fp32_mmac(kpack, fp64_mmac_device):
    device = fp64_mmac_device
    m, n, k, num_warps = CONFIG
    torch.manual_seed(1231)
    a = torch.randn((m, k), dtype=torch.float32, device=device)
    b = torch.randn((k, n), dtype=torch.float32, device=device)
    c = torch.randn((m, n), dtype=torch.float32, device=device)
    d = torch.empty_like(c)
    kernel = _dot_kernel[(1, )](a, b, c, d, m, n, k, k, 1, n, 1, num_warps=num_warps, num_stages=1, kpack=kpack)
    assert "v_mmac_16x16x8_f32" in kernel.asm["amdgcn"]
    assert "v_mmac_16x16x4_f64" not in kernel.asm["amdgcn"]
    reference = (a.double() @ b.double() + c.double()).float()
    torch.testing.assert_close(d, reference, atol=1e-4, rtol=1e-4)
