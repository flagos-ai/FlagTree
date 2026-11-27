"""Exercise matrix LDS selection and its register-staged fallback on gfx938."""

import pytest
import torch

import triton
import triton.language as tl


@triton.jit
def _gemm(
    a, b, c,
    n: tl.constexpr, k: tl.constexpr, bsk: tl.constexpr, bsn: tl.constexpr,
    tm: tl.constexpr, tn: tl.constexpr, tk: tl.constexpr,
):
    m_ids = tl.program_id(0) * tm + tl.arange(0, tm)
    n_ids = tl.program_id(1) * tn + tl.arange(0, tn)
    r = tl.arange(0, tk)
    acc = tl.full((tm, tn), 0, tl.float32)
    for step in range(tl.cdiv(k, tk)):
        k_ids = step * tk + r
        lhs = tl.load(a + m_ids[:, None] * k + k_ids[None, :])
        rhs = tl.load(b + k_ids[:, None] * bsk + n_ids[None, :] * bsn)
        acc = tl.dot(lhs, rhs, acc)
    tl.store(c + m_ids[:, None] * n + n_ids[None, :], acc)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float8_e4m3fn])
def test_matrix_lds_gemm(dtype):
    if not torch.cuda.is_available():
        pytest.skip("requires gfx938")
    target = triton.runtime.driver.active.get_current_target()
    if target.arch != "gfx938":
        pytest.skip("requires gfx938")

    from triton.backends.hcu.compiler_hcu import HIPBackend

    clang = HIPBackend.path_to_rocm_clang().resolve()
    if not HIPBackend._supports_gemm_backoff(str(clang)):
        pytest.skip("requires the qualified HCU code generator")

    m, n, k = 256, 256, 512
    tm, tn = 128, 128
    fp8 = dtype == torch.float8_e4m3fn
    tk = 128 if fp8 else 64
    opcode = "ds_read_m32x32_b8" if fp8 else "ds_read_m32x16_b16"
    generator = torch.Generator().manual_seed(2207)
    a_ref = torch.randint(-2, 3, (m, k), generator=generator)
    b_ref = torch.randint(-2, 3, (k, n), generator=generator)
    expected = (a_ref @ b_ref).to(device="cuda", dtype=torch.float32)
    a = a_ref.to(device="cuda", dtype=dtype)
    b = b_ref.to(device="cuda", dtype=dtype)
    c = torch.empty((m, n), device="cuda", dtype=torch.float32)
    k_contiguous_b = b.T.contiguous().T

    for rhs, enable, selected in ((b, True, True), (b, False, False), (k_contiguous_b, True, False)):
        compiled = _gemm[(m // tm, n // tn)](
            a, rhs, c, n, k, rhs.stride(0), rhs.stride(1), tm, tn, tk,
            num_warps=4, num_stages=1, matrix_instr_nonkdim=16,
            hcu_gemm_matrix_lds=enable,
        )
        assert (opcode in compiled.asm["amdgcn"]) == selected
        torch.testing.assert_close(c, expected, rtol=0, atol=0)
