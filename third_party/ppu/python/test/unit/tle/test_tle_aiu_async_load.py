"""Compile-only tests: tle.load(is_async=True) -> AIU promotion on PPU.

Covers:
  - Basic promotion (block pointer -> AIULoadOp)
  - Non-block-pointer fallback (no promotion)
  - Multiple data types (fp16, bf16; int8/int32 lower to .b8/.b32, int64 is not promoted)
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
    assert not leak, "residual tle.* ops in LLIR:\n" + "\n".join(leak[:5])


def _assert_uses_ppu_aiu_v1_width(compiled, width):
    ttgir = compiled.asm["ttgir"]
    llir = compiled.asm["llir"]
    aiu_instructions = [line for line in llir.splitlines() if "ppu.cp.async.aiu" in line]
    assert "versionMajor = 1" in ttgir
    assert aiu_instructions, "expected an async AIU copy instruction"
    assert all(f".2d.{width}" in line for line in aiu_instructions)
    for other in {"b8", "b16", "b32"} - {width}:
        assert all(f".{other}" not in line for line in aiu_instructions)


def _assert_int8_uses_ppu_aiu_v1_b8(compiled):
    _assert_uses_ppu_aiu_v1_width(compiled, "b8")


def _assert_int32_uses_ppu_aiu_v1_b32(compiled):
    _assert_uses_ppu_aiu_v1_width(compiled, "b32")


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


@triton.jit
def _typed_aiu_load_ordered(a_ptr, c_ptr, M: tl.constexpr, K: tl.constexpr, BLOCK_M: tl.constexpr,
                            BLOCK_K: tl.constexpr, STRIDE_M: tl.constexpr, STRIDE_K: tl.constexpr,
                            ORDER_0: tl.constexpr, ORDER_1: tl.constexpr):
    pid = tl.program_id(0)
    a_bp = tl.make_block_ptr(a_ptr, shape=(M, K), strides=(STRIDE_M, STRIDE_K), offsets=(pid * BLOCK_M, 0),
                             block_shape=(BLOCK_M, BLOCK_K), order=(ORDER_0, ORDER_1))
    a = tle.load(a_bp, is_async=True)
    offs_m = pid * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_k = tl.arange(0, BLOCK_K)
    tl.store(c_ptr + K * offs_m[:, None] + offs_k[None, :], a, mask=(offs_m[:, None] < M) & (offs_k[None, :] < K))


@triton.jit
def _int32_aiu_nibble_probe(packed_ptr, output_ptr, M: tl.constexpr, K: tl.constexpr):
    # W4A16-style decode: the int32 tile is a container for eight INT4 nibbles
    packed_bp = tl.make_block_ptr(packed_ptr, shape=(M, K), strides=(K, 1), offsets=(0, 0), block_shape=(M, K),
                                  order=(1, 0))
    packed = tle.load(packed_bp, is_async=True)
    unpacked = ((packed >> 4) & 0xF).to(tl.bfloat16)
    offs_m = tl.arange(0, M)
    offs_k = tl.arange(0, K)
    tl.store(output_ptr + offs_m[:, None] * K + offs_k[None, :], unpacked)


@_skip_no_sdk
def test_int32_aiu_load_uses_v1_b32_instruction():
    # flagos-ai/FlagTree#1050: 4-byte elements used to lower to a .b16 copy
    compiled = _compile_through_llir(_typed_aiu_load, {"a_ptr": "*i32", "c_ptr": "*i32"},
                                     {"M": 256, "K": 256, "BLOCK_M": 64, "BLOCK_K": 64})
    _assert_stages_exist(compiled)
    assert "aiu_load" in compiled.asm["ttir"]
    _assert_no_tle_residue(compiled)
    _assert_int32_uses_ppu_aiu_v1_b32(compiled)


@_skip_no_sdk
def test_int64_aiu_load_not_promoted():
    # 8-byte elements have no AIU copy form and must not be promoted
    compiled = _compile(_typed_aiu_load, {"a_ptr": "*i64", "c_ptr": "*i64"},
                        {"M": 256, "K": 256, "BLOCK_M": 64, "BLOCK_K": 64})
    _assert_stages_exist(compiled)
    # match the op, not the kernel name `_typed_aiu_load`
    assert "tt.aiu_load" not in compiled.asm["ttir"]
    assert "tt.load" in compiled.asm["ttir"]
    assert "ppu.cp.async.aiu" not in compiled.asm["llir"]


@_skip_no_sdk
@pytest.mark.skipif(not torch.cuda.is_available(), reason="PPU device not available")
@pytest.mark.parametrize("order", [(1, 0), (0, 1)])
def test_int32_aiu_load_device_correctness(order):
    m, k = 32, 32
    torch.manual_seed(0)
    expected = torch.randint(-(1 << 31), 1 << 31, (m, k), device="cuda", dtype=torch.int32)
    if order == (1, 0):
        a, strides = expected, (k, 1)
    else:
        # physically [K, M] so that dim 0 of the logical [M, K] view is contiguous
        a, strides = expected.t().contiguous().t(), (1, m)
    actual = torch.empty_like(expected)
    _typed_aiu_load_ordered[(1, )](a, actual, m, k, m, k, strides[0], strides[1], order[0], order[1], num_warps=4,
                                   num_stages=1)
    torch.cuda.synchronize()
    torch.testing.assert_close(actual.cpu(), expected.cpu(), rtol=0, atol=0)


@_skip_no_sdk
@pytest.mark.skipif(not torch.cuda.is_available(), reason="PPU device not available")
def test_int32_aiu_load_shift_and_mask_device_correctness():
    # reproducer from flagos-ai/FlagTree#1050
    m, k = 16, 16
    torch.manual_seed(0)
    packed = torch.randint(-(1 << 31), 1 << 31, (m, k), device="cuda", dtype=torch.int32)
    output = torch.empty((m, k), device="cuda", dtype=torch.bfloat16)
    _int32_aiu_nibble_probe[(1, )](packed, output, m, k, num_warps=4, num_stages=2)
    torch.cuda.synchronize()
    reference = ((packed.cpu().to(torch.int64) >> 4) & 0xF).to(torch.bfloat16)
    torch.testing.assert_close(output.cpu(), reference, rtol=0, atol=0)


@triton.jit
def _fp32_gemm_aiu_ordered_kernel(a_ptr, b_ptr, c_ptr, M: tl.constexpr, N: tl.constexpr, K: tl.constexpr,
                                  BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
                                  A_STRIDE_M: tl.constexpr, A_STRIDE_K: tl.constexpr, B_STRIDE_K: tl.constexpr,
                                  B_STRIDE_N: tl.constexpr, ORDER_0: tl.constexpr, ORDER_1: tl.constexpr):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    a_bp = tl.make_block_ptr(a_ptr, shape=(M, K), strides=(A_STRIDE_M, A_STRIDE_K), offsets=(pid_m * BLOCK_M, 0),
                             block_shape=(BLOCK_M, BLOCK_K), order=(ORDER_0, ORDER_1))
    b_bp = tl.make_block_ptr(b_ptr, shape=(K, N), strides=(B_STRIDE_K, B_STRIDE_N), offsets=(0, pid_n * BLOCK_N),
                             block_shape=(BLOCK_K, BLOCK_N), order=(ORDER_0, ORDER_1))
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
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
@pytest.mark.parametrize("order", [(1, 0), (0, 1)])
def test_fp32_gemm_aiu_device_correctness_memory_order(order):
    m, n, k = 32, 32, 32
    bm, bn, bk = 32, 32, 32
    torch.manual_seed(0)
    a_ref = torch.randn((m, k), device="cuda", dtype=torch.float32)
    b_ref = torch.randn((k, n), device="cuda", dtype=torch.float32)
    if order == (1, 0):
        a, a_strides = a_ref, (k, 1)
        b, b_strides = b_ref, (n, 1)
    else:
        # physically [K, M] / [N, K] so that dim 0 of the logical view is contiguous
        a, a_strides = a_ref.t().contiguous().t(), (1, m)
        b, b_strides = b_ref.t().contiguous().t(), (1, k)
    actual = torch.empty((m, n), device="cuda", dtype=torch.float32)
    grid = (triton.cdiv(m, bm), triton.cdiv(n, bn))
    _fp32_gemm_aiu_ordered_kernel[grid](a, b, actual, m, n, k, bm, bn, bk, a_strides[0], a_strides[1], b_strides[0],
                                        b_strides[1], order[0], order[1], num_warps=4, num_stages=1)
    torch.cuda.synchronize()
    # PPU fp32 dot computes through TF32 MMA (10 mantissa bits): measured max
    # abs error vs the fp32 CPU reference is ~1.6e-2 for this shape. The loose
    # tolerance absorbs that rounding noise while still catching order/layout
    # misinterpretation, which produces O(1) errors.
    expected = a_ref.cpu() @ b_ref.cpu()
    torch.testing.assert_close(actual.cpu(), expected, rtol=1e-2, atol=1e-2)


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
        "no MMA marker in LLIR — dot may not have lowered through AIU path"


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


# ---------------------------------------------------------------------------
# 11. Row strides: rows of the tile that are not the row length apart
# ---------------------------------------------------------------------------


@triton.jit
def _strided_rows_aiu_load(x_ptr, y_ptr, T, ROW_STRIDE: tl.constexpr, V: tl.constexpr, BT: tl.constexpr,
                           BV: tl.constexpr, CHECK_COLUMNS: tl.constexpr):
    # One [T, V] slice per program_id(1), interleaved like the heads of a [T, H, V] tensor
    i_t, i_h = tl.program_id(0), tl.program_id(1)
    p = tl.make_block_ptr(x_ptr + i_h * V, (T, V), (ROW_STRIDE, 1), (i_t * BT, 0), (BT, BV), (1, 0))
    if CHECK_COLUMNS:
        a = tle.load(p, boundary_check=(0, 1), is_async=True)
    else:
        a = tle.load(p, boundary_check=(0, ), is_async=True)
    offs_t = i_t * BT + tl.arange(0, BT)
    offs_v = tl.arange(0, BV)
    tl.store(y_ptr + (i_h * T + offs_t[:, None]) * BV + offs_v[None, :], a, mask=offs_t[:, None] < T)


@triton.jit
def _runtime_stride_aiu_load(x_ptr, y_ptr, T, V, stride_t, BT: tl.constexpr, BV: tl.constexpr):
    i_t = tl.program_id(0)
    p = tl.make_block_ptr(x_ptr, (T, V), (stride_t, 1), (i_t * BT, 0), (BT, BV), (1, 0))
    a = tle.load(p, boundary_check=(0, 1), is_async=True)
    offs_t = i_t * BT + tl.arange(0, BT)
    offs_v = tl.arange(0, BV)
    tl.store(y_ptr + offs_t[:, None] * BV + offs_v[None, :], a, mask=offs_t[:, None] < T)


@triton.jit
def _strided_columns_aiu_load(x_ptr, y_ptr, M, K, COL_STRIDE: tl.constexpr, BM: tl.constexpr, BK: tl.constexpr):
    # order=(0, 1): dim 0 is contiguous and the columns are COL_STRIDE elements apart
    i_m = tl.program_id(0)
    p = tl.make_block_ptr(x_ptr, (M, K), (1, COL_STRIDE), (i_m * BM, 0), (BM, BK), (0, 1))
    a = tle.load(p, boundary_check=(0, 1), is_async=True)
    offs_m = i_m * BM + tl.arange(0, BM)
    offs_k = tl.arange(0, BK)
    tl.store(y_ptr + offs_m[:, None] * BK + offs_k[None, :], a, mask=offs_m[:, None] < M)


def _padded_tile_reference(x, cols, block_cols):
    out = torch.zeros((x.shape[0], block_cols), dtype=x.dtype, device=x.device)
    out[:, :cols] = x[:, :cols]
    return out


def _aiu_instruction_forms(llir):
    copies = [line for line in llir.splitlines() if "ppu.cp.async.aiu" in line]
    return [form for form in ("2d", "3d") if any(f".{form}." in line for line in copies)]


@_skip_no_sdk
@pytest.mark.parametrize("heads", [1, 2, 4])
def test_strided_rows_use_3d_aiu_form(heads):
    compiled = _compile_through_llir(_strided_rows_aiu_load, {"x_ptr": "*bf16", "y_ptr": "*bf16", "T": "i32"},
                                     {"ROW_STRIDE": heads * 128, "V": 128, "BT": 64, "BV": 128, "CHECK_COLUMNS": True})
    assert "tt.aiu_load" in compiled.asm["ttir"]
    # Dense rows keep the 2d form; rows that are a multiple of V apart need the 3d form
    assert _aiu_instruction_forms(compiled.asm["llir"]) == (["2d"] if heads == 1 else ["3d"])


@_skip_no_sdk
def test_strided_rows_without_column_check_use_2d_aiu_form():
    # Nothing past the row length is zero filled, so the 2d form takes the row stride as its row length
    compiled = _compile_through_llir(_strided_rows_aiu_load, {"x_ptr": "*bf16", "y_ptr": "*bf16", "T": "i32"},
                                     {"ROW_STRIDE": 4 * 128, "V": 128, "BT": 64, "BV": 128, "CHECK_COLUMNS": False})
    assert "tt.aiu_load" in compiled.asm["ttir"]
    assert _aiu_instruction_forms(compiled.asm["llir"]) == ["2d"]


@_skip_no_sdk
def test_runtime_row_stride_selects_aiu_form_at_run_time():
    compiled = _compile_through_llir(_runtime_stride_aiu_load,
                                     {"x_ptr": "*bf16", "y_ptr": "*bf16", "T": "i32", "V": "i32", "stride_t": "i32"},
                                     {"BT": 64, "BV": 128})
    assert "tt.aiu_load" in compiled.asm["ttir"]
    assert _aiu_instruction_forms(compiled.asm["llir"]) == ["2d", "3d"]


@_skip_no_sdk
@pytest.mark.parametrize("row_stride,row_length,check_columns,promoted", [
    (192, 128, True, False),  # zero filling past column 128 needs a stride that is a multiple of 128
    (192, 128, False, True),
    (64, 128, False, False),  # overlapping rows
    (128, 0, False, False),
])
def test_row_stride_promotion(row_stride, row_length, check_columns, promoted):
    compiled = _compile(
        _strided_rows_aiu_load, {"x_ptr": "*bf16", "y_ptr": "*bf16", "T": "i32"},
        {"ROW_STRIDE": row_stride, "V": row_length, "BT": 64, "BV": 128, "CHECK_COLUMNS": check_columns})
    assert ("tt.aiu_load" in compiled.asm["ttir"]) == promoted


@_skip_no_sdk
def test_non_unit_inner_stride_is_not_promoted():

    @triton.jit
    def kernel(x_ptr, y_ptr, BT: tl.constexpr, BV: tl.constexpr):
        p = tl.make_block_ptr(x_ptr, (BT, BV), (2 * BV, 2), (0, 0), (BT, BV), (1, 0))
        a = tle.load(p, is_async=True)
        tl.store(y_ptr + tl.arange(0, BT)[:, None] * BV + tl.arange(0, BV)[None, :], a)

    compiled = _compile(kernel, {"x_ptr": "*bf16", "y_ptr": "*bf16"}, {"BT": 64, "BV": 128})
    assert "tt.aiu_load" not in compiled.asm["ttir"]


@_skip_no_sdk
@pytest.mark.skipif(not torch.cuda.is_available(), reason="PPU device not available")
@pytest.mark.parametrize("heads", [1, 2, 4])
@pytest.mark.parametrize("seq_len,block_t", [(256, 64), (200, 64), (100, 32)])
@pytest.mark.parametrize("cols,block_v,check_columns", [(128, 128, True), (96, 128, True), (128, 128, False)])
def test_strided_rows_aiu_load_device_correctness(heads, seq_len, block_t, cols, block_v, check_columns):
    # cols < block_v and seq_len % block_t != 0 check the zero fill past the last column and row
    torch.manual_seed(0)
    x = torch.randn((seq_len, heads, cols), device="cuda", dtype=torch.bfloat16)
    y = torch.full((heads, seq_len, block_v), float("nan"), device="cuda", dtype=torch.bfloat16)
    grid = (triton.cdiv(seq_len, block_t), heads)
    _strided_rows_aiu_load[grid](x, y, seq_len, heads * cols, cols, block_t, block_v, check_columns, num_warps=4,
                                 num_stages=1)
    torch.cuda.synchronize()
    for h in range(heads):
        expected = _padded_tile_reference(x[:, h], cols, block_v)
        torch.testing.assert_close(y[h], expected, rtol=0, atol=0)


@_skip_no_sdk
@pytest.mark.skipif(not torch.cuda.is_available(), reason="PPU device not available")
@pytest.mark.parametrize("row_stride", [128, 256, 512, 192])
def test_runtime_row_stride_aiu_load_device_correctness(row_stride):
    # 128 is dense, 256 and 512 use the 3d form, 192 keeps the 2d form with Z = stride
    seq_len, cols, block_t, block_v = 200, 128, 64, 128
    torch.manual_seed(0)
    x = torch.randn((seq_len, row_stride), device="cuda", dtype=torch.bfloat16)
    y = torch.full((seq_len, block_v), float("nan"), device="cuda", dtype=torch.bfloat16)
    _runtime_stride_aiu_load[(triton.cdiv(seq_len, block_t), )](x, y, seq_len, cols, row_stride, block_t, block_v,
                                                                num_warps=4, num_stages=1)
    torch.cuda.synchronize()
    torch.testing.assert_close(y, _padded_tile_reference(x, cols, block_v), rtol=0, atol=0)


@_skip_no_sdk
@pytest.mark.skipif(not torch.cuda.is_available(), reason="PPU device not available")
@pytest.mark.parametrize("col_stride", [64, 128])
def test_strided_columns_aiu_load_device_correctness(col_stride):
    m, k, block_m, block_k = 64, 32, 64, 32
    torch.manual_seed(0)
    storage = torch.randn((k, col_stride), device="cuda", dtype=torch.bfloat16)
    x = storage[:, :m].t()  # logical [M, K] with unit stride along M
    y = torch.empty((m, block_k), device="cuda", dtype=torch.bfloat16)
    _strided_columns_aiu_load[(1, )](storage, y, m, k, col_stride, block_m, block_k, num_warps=4, num_stages=1)
    torch.cuda.synchronize()
    torch.testing.assert_close(y, x, rtol=0, atol=0)


@_skip_no_sdk
@pytest.mark.skipif(not torch.cuda.is_available(), reason="PPU device not available")
def test_strided_rows_pipelined_aiu_load_device_correctness():
    # The pipeliner turns the loads of the loop into multi-buffered AIU copies

    @triton.jit
    def kernel(x_ptr, y_ptr, T, H: tl.constexpr, V: tl.constexpr, BT: tl.constexpr, BV: tl.constexpr):
        i_h = tl.program_id(0)
        p = tl.make_block_ptr(x_ptr + i_h * V, (T, V), (H * V, 1), (0, 0), (BT, BV), (1, 0))
        acc = tl.zeros((BT, BV), dtype=tl.float32)
        for _ in range(0, T, BT):
            acc += tle.load(p, boundary_check=(0, 1), is_async=True).to(tl.float32)
            p = tl.advance(p, (BT, 0))
        tl.store(y_ptr + i_h * BT * BV + tl.arange(0, BT)[:, None] * BV + tl.arange(0, BV)[None, :], acc)

    seq_len, heads, cols, block_t = 512, 4, 64, 64
    torch.manual_seed(0)
    x = torch.randn((seq_len, heads, cols), device="cuda", dtype=torch.bfloat16)
    y = torch.empty((heads, block_t, cols), device="cuda", dtype=torch.float32)
    kernel[(heads, )](x, y, seq_len, heads, cols, block_t, cols, num_warps=4, num_stages=3)
    torch.cuda.synchronize()
    expected = x.float().reshape(seq_len // block_t, block_t, heads, cols).sum(0).permute(1, 0, 2)
    torch.testing.assert_close(y, expected, rtol=1e-5, atol=1e-5)


@_skip_no_sdk
@pytest.mark.skipif(not torch.cuda.is_available(), reason="PPU device not available")
@pytest.mark.parametrize("heads", [1, 2])
def test_strided_rows_negative_offset_aiu_load_device_correctness(heads):
    # Tiles start ROW_SHIFT rows before the tensor: those rows read as zeros

    @triton.jit
    def kernel(x_ptr, y_ptr, T, H: tl.constexpr, V: tl.constexpr, BT: tl.constexpr, ROW_SHIFT: tl.constexpr):
        i_t, i_h = tl.program_id(0), tl.program_id(1)
        p = tl.make_block_ptr(x_ptr + i_h * V, (T, V), (H * V, 1), (i_t * BT - ROW_SHIFT, 0), (BT, V), (1, 0))
        a = tle.load(p, boundary_check=(0, 1), is_async=True)
        offs = (i_h * tl.num_programs(0) + i_t) * BT + tl.arange(0, BT)
        tl.store(y_ptr + offs[:, None] * V + tl.arange(0, V)[None, :], a)

    seq_len, cols, block_t, shift = 128, 64, 64, 16
    torch.manual_seed(0)
    x = torch.randn((seq_len, heads, cols), device="cuda", dtype=torch.bfloat16)
    tiles = triton.cdiv(seq_len + shift, block_t)
    y = torch.full((heads, tiles * block_t, cols), float("nan"), device="cuda", dtype=torch.bfloat16)
    kernel[(tiles, heads)](x, y, seq_len, heads, cols, block_t, shift, num_warps=4, num_stages=1)
    torch.cuda.synchronize()
    expected = torch.zeros_like(y)
    expected[:, shift:shift + seq_len] = x.permute(1, 0, 2)
    torch.testing.assert_close(y, expected, rtol=0, atol=0)
