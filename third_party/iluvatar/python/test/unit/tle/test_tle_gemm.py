# flagtree tle
"""
TLE GEMM on Iluvatar with TCU shared layout + SME G2S.

The trunk ``python/test/tle/integration/test_tle_gemm.py`` keeps
``nv_mma_shared_layout=False``. On Iluvatar that flag remaps to
``#ttg.swizzled_shared{useTcu=true}``, which is the destination encoding SME
writes. Combined with launch-time ``use_sme``, the local-pointer async-store
pass recovers the row stride and emits SME ``async_copy_global_to_local``.
"""

import re

import pytest
import torch
import triton
import triton.language as tl
import triton.experimental.tle.language as tle
from triton.experimental.tle.language.gpu.iluvatar import layout as iluvatar_layout

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

requires_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires CUDA GPU")


def _make_tle_gemm_kernel(tl_dtype, acc_tl_dtype, out_tl_dtype, use_ieee_flag):
    use_ieee = tl.constexpr(use_ieee_flag)

    @triton.jit
    def _kernel(
        a_ptr,
        b_ptr,
        c_ptr,
        M,
        N,
        K,
        stride_am,
        stride_ak,
        stride_bk,
        stride_bn,
        stride_cm,
        stride_cn,
        A_LAYOUT: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_K: tl.constexpr,
    ):
        pid_m = tl.program_id(0)
        pid_n = tl.program_id(1)
        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

        accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=acc_tl_dtype)
        # A_LAYOUT=None keeps the row-major nv_mma_shared_layout default; a
        # col-major A needs an explicitly matching TCU buffer instead.
        a_smem = tle.gpu.alloc([BLOCK_M, BLOCK_K], dtype=tl_dtype, layout=A_LAYOUT, scope=tle.gpu.smem,
                               nv_mma_shared_layout=True)
        b_smem = tle.gpu.alloc([BLOCK_K, BLOCK_N], dtype=tl_dtype, layout=None, scope=tle.gpu.smem,
                               nv_mma_shared_layout=True)
        a_row_ids = tl.broadcast_to(tl.arange(0, BLOCK_M)[:, None], (BLOCK_M, BLOCK_K))
        a_col_ids = tl.broadcast_to(tl.arange(0, BLOCK_K)[None, :], (BLOCK_M, BLOCK_K))
        b_row_ids = tl.broadcast_to(tl.arange(0, BLOCK_K)[:, None], (BLOCK_K, BLOCK_N))
        b_col_ids = tl.broadcast_to(tl.arange(0, BLOCK_N)[None, :], (BLOCK_K, BLOCK_N))
        a_smem_ptrs = tle.gpu.local_ptr(a_smem, (a_row_ids, a_col_ids))
        b_smem_ptrs = tle.gpu.local_ptr(b_smem, (b_row_ids, b_col_ids))

        for k_start in range(0, K, BLOCK_K):
            k_offs = k_start + tl.arange(0, BLOCK_K)
            a_ptrs = a_ptr + offs_m[:, None] * stride_am + k_offs[None, :] * stride_ak
            b_ptrs = b_ptr + k_offs[:, None] * stride_bk + offs_n[None, :] * stride_bn
            tle.gpu.copy(a_ptrs, a_smem, [BLOCK_M, BLOCK_K])
            tle.gpu.copy(b_ptrs, b_smem, [BLOCK_K, BLOCK_N])
            a_tile = tl.load(a_smem_ptrs)
            b_tile = tl.load(b_smem_ptrs)
            if use_ieee:
                accumulator = tl.dot(a_tile, b_tile, accumulator, input_precision="ieee")
            else:
                accumulator = tl.dot(a_tile, b_tile, accumulator, out_dtype=acc_tl_dtype)

        c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
        tl.store(c_ptrs, accumulator.to(out_tl_dtype), mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))

    return _kernel


# name, torch_in, torch_out, tl_in, tl_acc, tl_out, use_ieee
_DTYPE_SPECS = [
    ("float16", torch.float16, torch.float16, tl.float16, tl.float32, tl.float16, False),
    ("float16_out32", torch.float16, torch.float32, tl.float16, tl.float32, tl.float32, False),
    ("bfloat16", torch.bfloat16, torch.bfloat16, tl.bfloat16, tl.float32, tl.bfloat16, False),
    ("float32", torch.float32, torch.float32, tl.float32, tl.float32, tl.float32, True),
    ("int8", torch.int8, torch.int8, tl.int8, tl.int32, tl.int8, False),
]

_GEMM_KERNELS = {
    name: _make_tle_gemm_kernel(tl_in, tl_acc, tl_out, use_ieee)
    for name, _, _, tl_in, tl_acc, tl_out, use_ieee in _DTYPE_SPECS
}


def _launch_gemm(a, b, c, BLOCK_M, BLOCK_N, BLOCK_K, dtype_name, a_layout=None, **kwargs):
    M, K = a.shape
    Kb, N = b.shape
    assert K == Kb and c.shape == (M, N)
    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))
    return _GEMM_KERNELS[dtype_name][grid](a, b, c, M, N, K, a.stride(0), a.stride(1), b.stride(0), b.stride(1),
                                           c.stride(0), c.stride(1), a_layout, BLOCK_M, BLOCK_N, BLOCK_K, **kwargs)


def _run_fp32_gemm(size=256, block=64, **kwargs):
    torch.manual_seed(42)
    a = torch.randn(size, size, device="cuda", dtype=torch.float32).contiguous()
    b = torch.randn(size, size, device="cuda", dtype=torch.float32).contiguous()
    c = torch.empty(size, size, device="cuda", dtype=torch.float32).contiguous()
    compiled = _launch_gemm(a, b, c, block, block, block, "float32", **kwargs)
    torch.testing.assert_close(c, torch.matmul(a, b), atol=1e-4, rtol=1e-4)
    return compiled


def _run_int8_gemm(size=256, block=64):
    torch.manual_seed(42)
    a = torch.randint(-8, 8, (size, size), device="cuda", dtype=torch.int8).contiguous()
    b = torch.randint(-8, 8, (size, size), device="cuda", dtype=torch.int8).contiguous()
    c = torch.empty(size, size, device="cuda", dtype=torch.int8).contiguous()
    compiled = _launch_gemm(a, b, c, block, block, block, "int8")
    # Same default as triton.ops.matmul: acc int32, store as input dtype.
    expected = torch.matmul(a.to(torch.float32), b.to(torch.float32)).to(torch.int8)
    torch.testing.assert_close(c, expected)
    return compiled


@requires_cuda
def test_tle_gemm_tcu_shared_layout():
    compiled = _run_fp32_gemm(num_stages=1)
    ttgir = compiled.asm["ttgir"]
    assert "useTcu = true" in ttgir, ttgir
    assert "#ttg.nvmma_shared" not in ttgir, ttgir
    assert "ttg.async_copy_global_to_local" in ttgir, ttgir
    assert "iluvatar_tle.local_ptr_async_store" in ttgir, ttgir


@requires_cuda
def test_tle_gemm_uses_sme_g2s():
    compiled = _run_fp32_gemm()
    ttgir = compiled.asm["ttgir"]
    llir = compiled.asm["llir"]
    assert "isSme = true" in ttgir, ttgir
    assert "inputStride" in ttgir, ttgir
    assert ttgir.count("ttg.async_copy_global_to_local") >= 2, ttgir
    assert "llvm.bi.sme.load.16x1b64" in llir, llir
    assert "llvm.bi.load.kop" not in llir, llir


@requires_cuda
def test_tle_gemm_int8_row_major_uses_sme_g2s():
    # int8 row-major SME writes the rowxfb8 layout, whose read-back needs the
    # bit-7 offset correction. The correction only fires when the dot operand
    # carries a non-zero useSme, which is what
    # triton-iluvatar-tle-mark-sme-dot-operands adds for a fused tle.gpu.copy.
    compiled = _run_int8_gemm()
    ttgir = compiled.asm["ttgir"]
    llir = compiled.asm["llir"]
    assert "isSme = true" in ttgir, ttgir
    assert "inputStride" in ttgir, ttgir
    assert "llvm.bi.sme.load.16x1b64.rowxfb8" in llir, llir
    for operand in re.findall(r"#ttg\.dot_op<\{[^}]*\}>", ttgir):
        assert "useSme = 0" not in operand, ttgir


@requires_cuda
@pytest.mark.parametrize("num_stages", [2, 3])
def test_tle_gemm_promoted_staging_is_pipelined(num_stages):
    # The async-store fusion steps aside for a whole-buffer staging that only
    # feeds a dot, so promote-local-store-staging turns it into the canonical
    # load -> local_alloc shape and the software pipeliner multi-buffers it.
    compiled = _run_fp32_gemm(num_stages=num_stages)
    ttgir = compiled.asm["ttgir"]
    llir = compiled.asm["llir"]

    # One shared slot per stage, and no leftover fusion marker on this path.
    assert re.search(rf"memdesc<{num_stages}x\d+x\d+xf32", ttgir), ttgir
    assert "iluvatar_tle.local_ptr_async_store" not in ttgir, ttgir
    # SME has to survive: mark-sme-dot-operands restores the flag that
    # AccelerateMatmul cannot derive across the staged buffer, which is what
    # lets matmul-smeload issue the transfer.
    assert "isSme = true" in ttgir, ttgir
    assert "inputStride" in ttgir, ttgir
    assert "llvm.bi.sme.load.16x1b64" in llir, llir


# ---------------------------------------------------------------------------
# Numerical coverage
#
# operators/test_matmul.py is the reference: same-dtype fp16 / bf16 / fp32,
# plus the unaligned (M, N) and n-stage shapes Iluvatar actually runs.
# Mixed-precision, fp8 and float64 are skipped there (TCU / CoreX) and
# are not repeated here. Col-major follows test_dot's col_a/col_b: same
# logical C=A@B, operands allocated as (K,M).T / (N,K).T.
#
# tle.gpu.copy has no K-mask, so K is kept a multiple of BLOCK_K.
# ---------------------------------------------------------------------------

# (M, N, K, BLOCK_M, BLOCK_N, BLOCK_K) drawn from operators/test_matmul.py.
# K is a multiple of BLOCK_K so the unmasked tle.gpu.copy does not pollute acc.
_SHAPE_CASES = [
    (64, 64, 64, 64, 64, 64),
    (107, 233, 128, 64, 64, 64),
    (256, 128, 64, 128, 64, 32),
]


def _init_operand(m, n, torch_dtype):
    if torch_dtype == torch.int8:
        return torch.randint(-8, 8, (m, n), device="cuda", dtype=torch.int8).contiguous()
    # Small powers of two, same idea as operators/test_matmul.py init_input.
    exponents = torch.randint(-10, 0, size=(m, n))
    return (2.**exponents).to(torch_dtype).to("cuda").contiguous()


def _atol_rtol(torch_out):
    if torch_out == torch.float32:
        return 1e-4, 1e-4
    if torch_out in (torch.float16, torch.bfloat16):
        return 1e-2, 1e-2
    return 0, 0


def _run_tle_gemm_accuracy(dtype_name, torch_in, torch_out, M, N, K, BLOCK_M, BLOCK_N, BLOCK_K, num_stages, col_a,
                           col_b, a_layout=None):
    torch.manual_seed(0)
    # test_dot: col_a → (K, M).T so A is (M, K) with stride (1, M).
    a = _init_operand(K, M, torch_in).T if col_a else _init_operand(M, K, torch_in)
    b = _init_operand(N, K, torch_in).T if col_b else _init_operand(K, N, torch_in)
    c = torch.empty((M, N), device="cuda", dtype=torch_out).contiguous()
    try:
        compiled = _launch_gemm(a, b, c, BLOCK_M, BLOCK_N, BLOCK_K, dtype_name, a_layout=a_layout,
                                num_stages=num_stages)
    except triton.OutOfResources as e:
        pytest.skip(str(e))

    ref = torch.matmul(a.to(torch.float32), b.to(torch.float32)).to(torch_out)
    atol, rtol = _atol_rtol(torch_out)
    torch.testing.assert_close(c, ref, atol=atol, rtol=rtol)
    return compiled


@requires_cuda
@pytest.mark.parametrize("dtype_name,torch_in,torch_out", [(name, tin, tout) for name, tin, tout, *_ in _DTYPE_SPECS],
                         ids=[spec[0] for spec in _DTYPE_SPECS])
@pytest.mark.parametrize("M,N,K,BLOCK_M,BLOCK_N,BLOCK_K", _SHAPE_CASES)
@pytest.mark.parametrize("num_stages", [1, 2])
def test_tle_gemm_accuracy(dtype_name, torch_in, torch_out, M, N, K, BLOCK_M, BLOCK_N, BLOCK_K, num_stages):
    _run_tle_gemm_accuracy(dtype_name, torch_in, torch_out, M, N, K, BLOCK_M, BLOCK_N, BLOCK_K, num_stages, False,
                           False)


# Storage layout only, same logical C=A@B as test_dot col_a/col_b.
# Shapes stay on the 64-tile cases so stride (1, M) still aligns with BLOCK_*.
_COLMAJOR_SHAPES = [
    (64, 64, 64, 64, 64, 64),
    (107, 233, 128, 64, 64, 64),
]


@requires_cuda
@pytest.mark.parametrize("dtype_name,torch_in,torch_out", [(name, tin, tout) for name, tin, tout, *_ in _DTYPE_SPECS],
                         ids=[spec[0] for spec in _DTYPE_SPECS])
@pytest.mark.parametrize("M,N,K,BLOCK_M,BLOCK_N,BLOCK_K", _COLMAJOR_SHAPES)
@pytest.mark.parametrize("num_stages", [1, 2])
@pytest.mark.parametrize("col_a,col_b", [(True, False), (False, True), (True, True)], ids=["col_a", "col_b", "col_ab"])
def test_tle_gemm_accuracy_col_major(dtype_name, torch_in, torch_out, M, N, K, BLOCK_M, BLOCK_N, BLOCK_K, num_stages,
                                     col_a, col_b):
    compiled = _run_tle_gemm_accuracy(dtype_name, torch_in, torch_out, M, N, K, BLOCK_M, BLOCK_N, BLOCK_K, num_stages,
                                      col_a, col_b)
    # These allocate the row-major default, so the col-major operand may not go
    # through SME: the colxfb pattern reads back only from a matching buffer.
    assert ".colxfb" not in compiled.asm["llir"], compiled.asm["llir"]


def _colxfb_intrinsic(tl_in):
    return f"llvm.bi.sme.load.16x1b64.colxfb{int(tl_in.primitive_bitwidth)}"


def _launch_col_major_sme(M, tl_in):
    # get_corex_sme: col-major needs dim_m % (64 / element_size) == 0.
    return M % (512 // int(tl_in.primitive_bitwidth)) == 0


# Matching buffer + same dtype/shape/stage sweep as test_tle_gemm_accuracy.
# A col-major operand cannot use the row-major alloc default.
@requires_cuda
@pytest.mark.parametrize("dtype_name,torch_in,torch_out", [(name, tin, tout) for name, tin, tout, *_ in _DTYPE_SPECS],
                         ids=[spec[0] for spec in _DTYPE_SPECS])
@pytest.mark.parametrize("M,N,K,BLOCK_M,BLOCK_N,BLOCK_K", _SHAPE_CASES)
@pytest.mark.parametrize("num_stages", [1, 2])
def test_tle_gemm_col_major_uses_sme(dtype_name, torch_in, torch_out, M, N, K, BLOCK_M, BLOCK_N, BLOCK_K, num_stages):
    spec = next(s for s in _DTYPE_SPECS if s[0] == dtype_name)
    tl_in = spec[3]
    a_shape = [BLOCK_M, BLOCK_K]
    assert iluvatar_layout.is_tcu_eligible(a_shape, tl_in, col_major=True)
    a_layout = iluvatar_layout.make_tcu_swizzled_layout(a_shape, tl_in, col_major=True)

    compiled = _run_tle_gemm_accuracy(dtype_name, torch_in, torch_out, M, N, K, BLOCK_M, BLOCK_N, BLOCK_K, num_stages,
                                      col_a=True, col_b=False, a_layout=a_layout)
    ttgir = compiled.asm["ttgir"]
    llir = compiled.asm["llir"]
    # A gets the col-major TCU buffer, B keeps the row-major default.
    assert "order = [0, 1], useTcu = true" in ttgir, ttgir
    assert "order = [1, 0], useTcu = true" in ttgir, ttgir
    if not _launch_col_major_sme(M, tl_in):
        assert ".colxfb" not in llir, llir
        assert "iluvatar_tle.local_ptr_async_store" in ttgir, ttgir
        return

    assert "isSme = true" in ttgir, ttgir
    assert _colxfb_intrinsic(tl_in) in llir, llir
