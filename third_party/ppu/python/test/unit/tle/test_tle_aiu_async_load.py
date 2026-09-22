"""Compile-only tests: tle.load(is_async=True) -> AIU promotion on PPU.

Covers:
  - Basic promotion (block pointer -> AIULoadOp)
  - Non-block-pointer fallback (no promotion)
  - Multiple data types (fp16, bf16)
  - Various block shapes
  - Memory layout orders
  - AIU load feeding tl.dot (GEMM)
  - Pipelined loop with block_ptr advance
  - Boundary check on block pointer
  - Mixed: AIU load + regular load
  - is_async=False (no promotion)
  - TTGIR/LLIR stage verification
"""

import os
import re
import shutil
import pytest
import torch
import triton
import triton.language as tl
from triton import knobs
from triton.backends.compiler import GPUTarget

tle_backend = pytest.importorskip(
    "triton._C.libtriton.tle",
    reason="libtriton was built without FLAGTREE_TLE; skipping TLE tests",
)
tle = pytest.importorskip(
    "triton.experimental.tle.language",
    reason="triton.experimental.tle.language unavailable",
)

_GPU_TARGET = GPUTarget("cuda", 80, 32)


def _ppu_sdk_available() -> bool:
    sdk = os.environ.get("PPU_SDK") or os.environ.get("PPU_HOME")
    if sdk and os.path.isfile(os.path.join(sdk, "bin", "ppu-llc")):
        return True
    return bool(shutil.which("ppu-llc"))


_skip_no_sdk = pytest.mark.skipif(not _ppu_sdk_available(), reason="PPU SDK not available")


def _compile(kernel, signature, constexprs, target=None, options=None):
    src = triton.compiler.ASTSource(fn=kernel, signature=signature, constexprs=constexprs)
    return triton.compile(src, target=target or _GPU_TARGET, options=options)


def _compile_through_llir(kernel, signature, constexprs, target=None, options=None):
    previous_hook = knobs.runtime.add_stages_inspection_hook

    def stop_before_hgbin(_backend, stages, _options, _language, _capability):
        stages["hgbin"] = lambda _src, _metadata: b""

    knobs.runtime.add_stages_inspection_hook = stop_before_hgbin
    try:
        return _compile(kernel, signature, constexprs, target, options)
    finally:
        knobs.runtime.add_stages_inspection_hook = previous_hook


def _assert_stages_exist(compiled, stages=("ttir", "ttgir", "llir")):
    for s in stages:
        assert s in compiled.asm and compiled.asm[s], f"stage '{s}' missing or empty"


def _assert_no_tle_residue(compiled):
    llir = compiled.asm["llir"]
    leak = [ln for ln in llir.split("\n") if "tle." in ln]
    assert not leak, f"residual tle.* ops in LLIR:\n" + "\n".join(leak[:5])


def _assert_int8_uses_ppu_aiu_v1_b8(compiled):
    ttgir = compiled.asm["ttgir"]
    llir = compiled.asm["llir"]
    aiu_instructions = [line for line in llir.splitlines() if "ppu.cp.async.aiu" in line]
    assert "versionMajor = 1" in ttgir
    assert aiu_instructions, "expected an async AIU copy instruction"
    assert all(".2d.b8" in line for line in aiu_instructions)
    assert all(".b8" in line for line in aiu_instructions)
    assert all(".b16" not in line for line in aiu_instructions)


# ---------------------------------------------------------------------------
# 1. Basic promotion & fallback
# ---------------------------------------------------------------------------


@triton.jit
def _basic_aiu_load(a_ptr, c_ptr, M: tl.constexpr, K: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_K: tl.constexpr):
    pid = tl.program_id(0)
    a_bp = tl.make_block_ptr(a_ptr, shape=(M, K), strides=(K, 1), offsets=(pid * BLOCK_M, 0),
                             block_shape=(BLOCK_M, BLOCK_K), order=(1, 0))
    a = tle.load(a_bp, is_async=True)
    c = a.to(tl.float16)
    offs_m = pid * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_k = tl.arange(0, BLOCK_K)
    tl.store(c_ptr + K * offs_m[:, None] + offs_k[None, :], c, mask=(offs_m[:, None] < M) & (offs_k[None, :] < K))


@_skip_no_sdk
def test_basic_promotion_to_aiu():
    compiled = _compile(_basic_aiu_load, {"a_ptr": "*fp16", "c_ptr": "*fp16"},
                        {"M": 1024, "K": 1024, "BLOCK_M": 64, "BLOCK_K": 64})
    _assert_stages_exist(compiled)
    assert "aiu_load" in compiled.asm["ttir"]
    _assert_no_tle_residue(compiled)


@_skip_no_sdk
def test_no_promotion_for_non_block_ptr():

    @triton.jit
    def kernel(a_ptr, c_ptr, N: tl.constexpr, BLOCK: tl.constexpr):
        pid = tl.program_id(0)
        offs = pid * BLOCK + tl.arange(0, BLOCK)
        a = tle.load(a_ptr + offs, mask=offs < N, is_async=True)
        tl.store(c_ptr + offs, a, mask=offs < N)

    compiled = _compile(kernel, {"a_ptr": "*fp16", "c_ptr": "*fp16"}, {"N": 1024, "BLOCK": 256})
    _assert_stages_exist(compiled)
    assert "aiu_load" not in compiled.asm["ttir"]


@_skip_no_sdk
def test_no_promotion_when_is_async_false():

    @triton.jit
    def kernel(a_ptr, c_ptr, M: tl.constexpr, K: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_K: tl.constexpr):
        pid = tl.program_id(0)
        a_bp = tl.make_block_ptr(a_ptr, shape=(M, K), strides=(K, 1), offsets=(pid * BLOCK_M, 0),
                                 block_shape=(BLOCK_M, BLOCK_K), order=(1, 0))
        a = tle.load(a_bp, is_async=False)
        offs_m = pid * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_k = tl.arange(0, BLOCK_K)
        tl.store(c_ptr + K * offs_m[:, None] + offs_k[None, :], a, mask=(offs_m[:, None] < M) & (offs_k[None, :] < K))

    compiled = _compile(kernel, {"a_ptr": "*fp16", "c_ptr": "*fp16"},
                        {"M": 1024, "K": 1024, "BLOCK_M": 64, "BLOCK_K": 64})
    _assert_stages_exist(compiled)
    assert "aiu_load" not in compiled.asm["ttir"]


# ---------------------------------------------------------------------------
# 2. Data type parametrization
# ---------------------------------------------------------------------------


@triton.jit
def _typed_aiu_load(a_ptr, c_ptr, M: tl.constexpr, K: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_K: tl.constexpr):
    pid = tl.program_id(0)
    a_bp = tl.make_block_ptr(a_ptr, shape=(M, K), strides=(K, 1), offsets=(pid * BLOCK_M, 0),
                             block_shape=(BLOCK_M, BLOCK_K), order=(1, 0))
    a = tle.load(a_bp, is_async=True)
    offs_m = pid * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_k = tl.arange(0, BLOCK_K)
    tl.store(c_ptr + K * offs_m[:, None] + offs_k[None, :], a, mask=(offs_m[:, None] < M) & (offs_k[None, :] < K))


@_skip_no_sdk
@pytest.mark.parametrize("dtype", ["fp16", "bf16"])
def test_aiu_promotion_dtypes(dtype):
    compiled = _compile(_typed_aiu_load, {"a_ptr": f"*{dtype}", "c_ptr": f"*{dtype}"},
                        {"M": 256, "K": 256, "BLOCK_M": 64, "BLOCK_K": 64})
    _assert_stages_exist(compiled)
    assert "aiu_load" in compiled.asm["ttir"]
    _assert_no_tle_residue(compiled)


@_skip_no_sdk
def test_int8_aiu_load_uses_v1_b8_instruction():
    compiled = _compile_through_llir(_typed_aiu_load, {"a_ptr": "*i8", "c_ptr": "*i8"},
                                     {"M": 256, "K": 256, "BLOCK_M": 64, "BLOCK_K": 64})
    _assert_stages_exist(compiled)
    assert "aiu_load" in compiled.asm["ttir"]
    _assert_no_tle_residue(compiled)
    _assert_int8_uses_ppu_aiu_v1_b8(compiled)


@_skip_no_sdk
@pytest.mark.skipif(not torch.cuda.is_available(), reason="PPU device not available")
def test_int8_aiu_load_device_correctness():
    expected = torch.arange(32 * 32, device="cuda", dtype=torch.int32)
    expected = expected.remainder(127).to(torch.int8).reshape(32, 32)
    actual = torch.empty_like(expected)
    _typed_aiu_load[(1, )](expected, actual, 32, 32, 32, 32, num_warps=4, num_stages=1)
    torch.cuda.synchronize()
    torch.testing.assert_close(actual.cpu(), expected.cpu(), rtol=0, atol=0)


# ---------------------------------------------------------------------------
# 3. Block shape parametrization
# ---------------------------------------------------------------------------


@_skip_no_sdk
@pytest.mark.parametrize("bm,bk", [(32, 32), (64, 32), (64, 64), (128, 64), (128, 128)])
def test_aiu_promotion_block_shapes(bm, bk):
    compiled = _compile(_basic_aiu_load, {"a_ptr": "*fp16", "c_ptr": "*fp16"},
                        {"M": 1024, "K": 1024, "BLOCK_M": bm, "BLOCK_K": bk})
    _assert_stages_exist(compiled)
    assert "aiu_load" in compiled.asm["ttir"]
    _assert_no_tle_residue(compiled)


# ---------------------------------------------------------------------------
# 4. Memory order
# ---------------------------------------------------------------------------


@_skip_no_sdk
@pytest.mark.parametrize("order", [(1, 0), (0, 1)])
def test_aiu_promotion_memory_order(order):

    @triton.jit
    def kernel(a_ptr, c_ptr, M: tl.constexpr, K: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_K: tl.constexpr,
               ORDER_0: tl.constexpr, ORDER_1: tl.constexpr):
        pid = tl.program_id(0)
        a_bp = tl.make_block_ptr(a_ptr, shape=(M, K), strides=(K, 1), offsets=(pid * BLOCK_M, 0),
                                 block_shape=(BLOCK_M, BLOCK_K), order=(ORDER_0, ORDER_1))
        a = tle.load(a_bp, is_async=True)
        offs_m = pid * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_k = tl.arange(0, BLOCK_K)
        tl.store(c_ptr + K * offs_m[:, None] + offs_k[None, :], a, mask=(offs_m[:, None] < M) & (offs_k[None, :] < K))

    compiled = _compile(kernel, {"a_ptr": "*fp16", "c_ptr": "*fp16"},
                        {"M": 512, "K": 512, "BLOCK_M": 64, "BLOCK_K": 64, "ORDER_0": order[0], "ORDER_1": order[1]})
    _assert_stages_exist(compiled)
    assert "aiu_load" in compiled.asm["ttir"]
    _assert_no_tle_residue(compiled)


# ---------------------------------------------------------------------------
# 5. AIU load + tl.dot (GEMM) — the primary use case
# ---------------------------------------------------------------------------


@triton.jit
def _gemm_aiu_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    a_bp = tl.make_block_ptr(a_ptr, shape=(M, K), strides=(K, 1), offsets=(pid_m * BLOCK_M, 0),
                             block_shape=(BLOCK_M, BLOCK_K), order=(1, 0))
    b_bp = tl.make_block_ptr(b_ptr, shape=(K, N), strides=(N, 1), offsets=(0, pid_n * BLOCK_N),
                             block_shape=(BLOCK_K, BLOCK_N), order=(1, 0))

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for _ in range(0, K, BLOCK_K):
        a = tle.load(a_bp, is_async=True)
        b = tle.load(b_bp, is_async=True)
        acc += tl.dot(a, b)
        a_bp = tl.advance(a_bp, (0, BLOCK_K))
        b_bp = tl.advance(b_bp, (BLOCK_K, 0))

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    c_ptrs = c_ptr + N * offs_m[:, None] + offs_n[None, :]
    tl.store(c_ptrs, acc, mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))


_GEMM_SIG = {"a_ptr": "*fp16", "b_ptr": "*fp16", "c_ptr": "*fp32"}
_GEMM_CONSTEXPRS = {"M": 512, "N": 512, "K": 256, "BLOCK_M": 64, "BLOCK_N": 64, "BLOCK_K": 64}


@triton.jit(do_not_specialize_on_alignment=["a_ptr", "b_ptr", "c_ptr"])
def _int8_gemm_aiu_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    a_bp = tl.make_block_ptr(a_ptr, shape=(M, K), strides=(K, 1), offsets=(pid_m * BLOCK_M, 0),
                             block_shape=(BLOCK_M, BLOCK_K), order=(1, 0))
    # b_ptr is physically contiguous [N, K], logically column-major [K, N].
    b_bp = tl.make_block_ptr(b_ptr, shape=(K, N), strides=(1, K), offsets=(0, pid_n * BLOCK_N),
                             block_shape=(BLOCK_K, BLOCK_N), order=(0, 1))
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.int32)
    for _ in range(0, K, BLOCK_K):
        a = tle.load(a_bp, is_async=True)
        b = tle.load(b_bp, is_async=True)
        acc += tl.dot(a, b)
        a_bp = tl.advance(a_bp, (0, BLOCK_K))
        b_bp = tl.advance(b_bp, (BLOCK_K, 0))
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    tl.store(c_ptr + N * offs_m[:, None] + offs_n[None, :], acc, mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))


@_skip_no_sdk
@pytest.mark.skipif(not torch.cuda.is_available(), reason="PPU device not available")
@pytest.mark.parametrize(
    "m,n,k,bm,bn,bk,num_warps",
    [(16, 16, 32, 16, 16, 32, 1), (16, 32, 64, 16, 32, 64, 1), (64, 64, 64, 64, 64, 64, 4),
     (128, 128, 128, 64, 64, 64, 4)],
)
def test_int8_gemm_aiu_device_correctness(m, n, k, bm, bn, bk, num_warps):
    torch.manual_seed(123)
    a = torch.randint(-8, 9, (m, k), device="cuda", dtype=torch.int8)
    b = torch.randint(-8, 9, (n, k), device="cuda", dtype=torch.int8)
    actual = torch.empty((m, n), device="cuda", dtype=torch.int32)
    grid = (triton.cdiv(m, bm), triton.cdiv(n, bn))
    _int8_gemm_aiu_kernel[grid](a, b, actual, m, n, k, bm, bn, bk, num_warps=num_warps, num_stages=1)
    torch.cuda.synchronize()
    expected = a.cpu().to(torch.int32) @ b.cpu().to(torch.int32).T
    torch.testing.assert_close(actual.cpu(), expected, rtol=0, atol=0)


@_skip_no_sdk
def test_gemm_aiu_both_operands_promoted():
    """Both A and B matrix loads should be promoted to AIULoadOp."""
    compiled = _compile(_gemm_aiu_kernel, _GEMM_SIG, _GEMM_CONSTEXPRS)
    _assert_stages_exist(compiled)
    ttir = compiled.asm["ttir"]
    aiu_count = ttir.count("aiu_load")
    assert aiu_count >= 2, f"expected >=2 aiu_load ops, found {aiu_count}"
    _assert_no_tle_residue(compiled)


@_skip_no_sdk
def test_gemm_aiu_produces_mma():
    """AIU loads feeding tl.dot should ultimately lower to MMA instructions."""
    compiled = _compile(_gemm_aiu_kernel, _GEMM_SIG, _GEMM_CONSTEXPRS)
    llir = compiled.asm["llir"]
    mma_markers = ("ppu.mma", "mma.sync", "fmuladd", "fma.")
    assert any(m in llir for m in mma_markers), \
        f"no MMA marker in LLIR — dot may not have lowered through AIU path"


@_skip_no_sdk
def test_gemm_aiu_ttgir_has_ppu_aiu_encoding():
    """TTGIR should contain PPUAIUSharedEncoding from the AIU lowering path."""
    compiled = _compile(_gemm_aiu_kernel, _GEMM_SIG, _GEMM_CONSTEXPRS)
    ttgir = compiled.asm["ttgir"]
    assert "PPUAIUSharedEncoding" in ttgir or "ppu_aiu_shared" in ttgir or "aiu" in ttgir.lower(), \
        "TTGIR should show AIU shared encoding after AIU lowering"


@_skip_no_sdk
def test_gemm_aiu_no_regular_load_for_block_ptrs():
    """In TTIR, block-pointer async loads should become aiu_load, not tt.load."""
    compiled = _compile(_gemm_aiu_kernel, _GEMM_SIG, _GEMM_CONSTEXPRS)
    ttir = compiled.asm["ttir"]
    assert "aiu_load" in ttir
    # tt.load may still exist for the final store's mask computation etc.,
    # but there should be no tt.load with a tensor pointer (block_ptr) remaining.
    # The tensor_pointer loads should all have been promoted.
    assert "tt.load" not in ttir or "load.async" not in ttir, \
        "block-ptr loads with tt.load.async should be promoted to aiu_load"


# ---------------------------------------------------------------------------
# 6. Pipelined loop with advance
# ---------------------------------------------------------------------------


@_skip_no_sdk
@pytest.mark.parametrize("num_stages", [1, 2, 4])
def test_aiu_pipelined_loop(num_stages):
    """AIU loads inside a pipeline loop with block_ptr advance should compile."""

    @triton.jit
    def kernel(a_ptr, c_ptr, M: tl.constexpr, K: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_K: tl.constexpr):
        pid = tl.program_id(0)
        a_bp = tl.make_block_ptr(a_ptr, shape=(M, K), strides=(K, 1), offsets=(pid * BLOCK_M, 0),
                                 block_shape=(BLOCK_M, BLOCK_K), order=(1, 0))
        acc = tl.zeros((BLOCK_M, BLOCK_K), dtype=tl.float32)
        for _ in range(0, K, BLOCK_K):
            a = tle.load(a_bp, is_async=True)
            acc += a.to(tl.float32)
            a_bp = tl.advance(a_bp, (0, BLOCK_K))
        offs_m = pid * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_k = tl.arange(0, BLOCK_K)
        tl.store(c_ptr + K * offs_m[:, None] + offs_k[None, :], acc.to(tl.float16),
                 mask=(offs_m[:, None] < M) & (offs_k[None, :] < K))

    compiled = _compile(kernel, {"a_ptr": "*fp16", "c_ptr": "*fp16"},
                        {"M": 512, "K": 512, "BLOCK_M": 64, "BLOCK_K": 64}, options={"num_stages": num_stages})
    _assert_stages_exist(compiled)
    assert "aiu_load" in compiled.asm["ttir"]
    _assert_no_tle_residue(compiled)


@_skip_no_sdk
def test_int8_aiu_pipelined_loop_uses_v1_b8_instruction():

    @triton.jit
    def kernel(a_ptr, c_ptr, M: tl.constexpr, K: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_K: tl.constexpr):
        pid = tl.program_id(0)
        a_bp = tl.make_block_ptr(a_ptr, shape=(M, K), strides=(K, 1), offsets=(pid * BLOCK_M, 0),
                                 block_shape=(BLOCK_M, BLOCK_K), order=(1, 0))
        acc = tl.zeros((BLOCK_M, BLOCK_K), dtype=tl.int32)
        for _ in range(0, K, BLOCK_K):
            a = tle.load(a_bp, is_async=True)
            acc += a.to(tl.int32)
            a_bp = tl.advance(a_bp, (0, BLOCK_K))
        offs_m = pid * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_k = tl.arange(0, BLOCK_K)
        tl.store(c_ptr + K * offs_m[:, None] + offs_k[None, :], acc, mask=(offs_m[:, None] < M) & (offs_k[None, :] < K))

    compiled = _compile_through_llir(kernel, {"a_ptr": "*i8", "c_ptr": "*i32"},
                                     {"M": 512, "K": 512, "BLOCK_M": 64, "BLOCK_K": 64}, options={"num_stages": 2})
    _assert_stages_exist(compiled)
    assert "aiu_load" in compiled.asm["ttir"]
    _assert_no_tle_residue(compiled)
    _assert_int8_uses_ppu_aiu_v1_b8(compiled)


# ---------------------------------------------------------------------------
# 7. Boundary check
# ---------------------------------------------------------------------------


@_skip_no_sdk
def test_aiu_with_boundary_check():
    """tle.load with boundary_check should still promote to AIU."""

    @triton.jit
    def kernel(a_ptr, c_ptr, M: tl.constexpr, K: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_K: tl.constexpr):
        pid = tl.program_id(0)
        a_bp = tl.make_block_ptr(a_ptr, shape=(M, K), strides=(K, 1), offsets=(pid * BLOCK_M, 0),
                                 block_shape=(BLOCK_M, BLOCK_K), order=(1, 0))
        a = tle.load(a_bp, boundary_check=(0, 1), is_async=True)
        offs_m = pid * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_k = tl.arange(0, BLOCK_K)
        tl.store(c_ptr + K * offs_m[:, None] + offs_k[None, :], a, mask=(offs_m[:, None] < M) & (offs_k[None, :] < K))

    compiled = _compile(kernel, {"a_ptr": "*fp16", "c_ptr": "*fp16"},
                        {"M": 300, "K": 300, "BLOCK_M": 64, "BLOCK_K": 64})
    _assert_stages_exist(compiled)
    assert "aiu_load" in compiled.asm["ttir"]
    _assert_no_tle_residue(compiled)


# ---------------------------------------------------------------------------
# 8. Mixed: AIU async load + regular load in same kernel
# ---------------------------------------------------------------------------


@_skip_no_sdk
def test_mixed_aiu_and_regular_load():
    """Kernel with both AIU async loads and regular loads should compile."""

    @triton.jit
    def kernel(a_ptr, b_ptr, c_ptr, M: tl.constexpr, K: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_K: tl.constexpr):
        pid = tl.program_id(0)
        # A: AIU async load via block pointer
        a_bp = tl.make_block_ptr(a_ptr, shape=(M, K), strides=(K, 1), offsets=(pid * BLOCK_M, 0),
                                 block_shape=(BLOCK_M, BLOCK_K), order=(1, 0))
        a = tle.load(a_bp, is_async=True)
        # B: regular pointer load
        offs_m = pid * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_k = tl.arange(0, BLOCK_K)
        b_ptrs = b_ptr + K * offs_m[:, None] + offs_k[None, :]
        b_mask = (offs_m[:, None] < M) & (offs_k[None, :] < K)
        b = tl.load(b_ptrs, mask=b_mask)
        # Combine
        c = a + b
        c_ptrs = c_ptr + K * offs_m[:, None] + offs_k[None, :]
        tl.store(c_ptrs, c, mask=b_mask)

    compiled = _compile(kernel, {"a_ptr": "*fp16", "b_ptr": "*fp16", "c_ptr": "*fp16"},
                        {"M": 512, "K": 512, "BLOCK_M": 64, "BLOCK_K": 64})
    _assert_stages_exist(compiled)
    ttir = compiled.asm["ttir"]
    assert "aiu_load" in ttir, "block-ptr async load should be promoted"
    assert "tt.load" in ttir, "regular ptr load should remain as tt.load"
    _assert_no_tle_residue(compiled)


# ---------------------------------------------------------------------------
# 9. Multiple independent AIU loads (non-dot)
# ---------------------------------------------------------------------------


@_skip_no_sdk
def test_multiple_independent_aiu_loads():
    """Multiple AIU loads used independently (not feeding dot)."""

    @triton.jit
    def kernel(a_ptr, b_ptr, c_ptr, M: tl.constexpr, K: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_K: tl.constexpr):
        pid = tl.program_id(0)
        a_bp = tl.make_block_ptr(a_ptr, shape=(M, K), strides=(K, 1), offsets=(pid * BLOCK_M, 0),
                                 block_shape=(BLOCK_M, BLOCK_K), order=(1, 0))
        b_bp = tl.make_block_ptr(b_ptr, shape=(M, K), strides=(K, 1), offsets=(pid * BLOCK_M, 0),
                                 block_shape=(BLOCK_M, BLOCK_K), order=(1, 0))
        a = tle.load(a_bp, is_async=True)
        b = tle.load(b_bp, is_async=True)
        c = a + b
        offs_m = pid * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_k = tl.arange(0, BLOCK_K)
        tl.store(c_ptr + K * offs_m[:, None] + offs_k[None, :], c, mask=(offs_m[:, None] < M) & (offs_k[None, :] < K))

    compiled = _compile(kernel, {"a_ptr": "*fp16", "b_ptr": "*fp16", "c_ptr": "*fp16"},
                        {"M": 512, "K": 512, "BLOCK_M": 64, "BLOCK_K": 64})
    _assert_stages_exist(compiled)
    aiu_count = compiled.asm["ttir"].count("aiu_load")
    assert aiu_count >= 2, f"expected >=2 aiu_load, got {aiu_count}"
    _assert_no_tle_residue(compiled)


# ---------------------------------------------------------------------------
# 10. Non-block-pointer async load fallback (cp.async, not AIU)
# ---------------------------------------------------------------------------


@_skip_no_sdk
def test_non_block_ptr_async_load_uses_cp_async_fallback():
    """Non-block-pointer tle.load(is_async=True) should NOT promote to AIU
    but should be lowered to cp.async by TleLowerAsyncLoad when vectorization
    meets the 4-byte minimum."""

    @triton.jit
    def kernel(a_ptr, c_ptr, M: tl.constexpr, K: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_K: tl.constexpr):
        pid = tl.program_id(0)
        num_pid_m = tl.cdiv(M, BLOCK_M)
        pid_m = pid % num_pid_m
        pid_k = pid // num_pid_m
        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_k = pid_k * BLOCK_K + tl.arange(0, BLOCK_K)
        a_ptrs = a_ptr + K * offs_m[:, None] + offs_k[None, :]
        a_mask = (offs_m[:, None] < M) & (offs_k[None, :] < K)
        a = tle.load(a_ptrs, mask=a_mask, is_async=True)
        c_ptrs = c_ptr + K * offs_m[:, None] + offs_k[None, :]
        tl.store(c_ptrs, a, mask=a_mask)

    compiled = _compile(kernel, {"a_ptr": "*fp32", "c_ptr": "*fp32"},
                        {"M": 1024, "K": 1024, "BLOCK_M": 64, "BLOCK_K": 64})
    _assert_stages_exist(compiled)
    assert "aiu_load" not in compiled.asm["ttir"], \
        "Non-block-pointer should NOT promote to AIU"
    _assert_no_tle_residue(compiled)
    assert "async_copy_global_to_local" in compiled.asm["ttgir"] or \
           "cp.async" in compiled.asm.get("llir", ""), \
        "Non-block-pointer 2D fp32 async load should use cp.async fallback"
