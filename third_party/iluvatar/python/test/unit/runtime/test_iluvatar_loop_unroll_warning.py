"""
Verify that ixcc "loop with constant trip count not unrolled" warnings
are suppressed during Triton kernel compilation on Iluvatar GPU by default,
but can be re-enabled with TRITON_ENABLE_LOOP_UNROLL_WARNING.

Background: ixcc (commit 4b80df81) emits DiagnosticInfoOptimizationFailure
from LoopUnrollPass whenever a constant-trip-count loop is not fully
unrolled on a GPU target. This floods stderr when torch.compile JIT-compiles
hundreds of Triton kernels. The fix installs a callback-based diagnostic
handler (triton/Tools/LLVMWarningFilter.h) on each LLVMContext to suppress
only this class of warning while preserving all other diagnostics unless
TRITON_ENABLE_LOOP_UNROLL_WARNING is set.
"""

import pytest
import triton
import triton.language as tl
from triton._internal_testing import is_corex


@triton.jit
def _matmul(
    a_ptr,
    b_ptr,
    c_ptr,
    stride_am,
    stride_ak,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    NUM_K_BLOCKS: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """Matmul with a constant-trip-count K-reduction loop.

    NUM_K_BLOCKS=32 gives trip count 32 > MaxIterationsCountToAnalyze(10),
    so ixcc's LoopUnrollPass emits the warning on the unfixed build.
    """
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    a_ptrs = a_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
    b_ptrs = b_ptr + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn
    acc = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
    for _ in range(NUM_K_BLOCKS):
        a = tl.load(a_ptrs)
        b = tl.load(b_ptrs)
        acc = tl.dot(a, b, acc)
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk
    c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    tl.store(c_ptrs, acc)


@pytest.mark.skipif(not is_corex(), reason="Iluvatar backend only")
def test_loop_unroll_warning_suppressed_by_default(capfd, fresh_triton_cache):
    """Compiling a matmul kernel must not produce loop-unroll warnings."""
    constexprs = {
        "NUM_K_BLOCKS": 32,
        "BLOCK_M": 32,
        "BLOCK_N": 32,
        "BLOCK_K": 32,
    }
    signature = {
        "a_ptr": "*fp16",
        "b_ptr": "*fp16",
        "c_ptr": "*fp32",
        "stride_am": "i32",
        "stride_ak": "i32",
        "stride_bk": "i32",
        "stride_bn": "i32",
        "stride_cm": "i32",
        "stride_cn": "i32",
        "NUM_K_BLOCKS": "constexpr",
        "BLOCK_M": "constexpr",
        "BLOCK_N": "constexpr",
        "BLOCK_K": "constexpr",
    }
    triton.compile(triton.compiler.ASTSource(
        fn=_matmul,
        signature=signature,
        constexprs=constexprs,
    ), )

    captured = capfd.readouterr()
    assert "loop with constant trip count not unrolled" not in captured.err, (
        "ixcc loop-unroll warning still leaking to stderr. "
        "Check that LLVMWarningFilter is installed in "
        "translate_llvmir_to_cubin (triton_iluvatar.cc) and "
        "translate_to_asm / optimize_module (python/src/llvm.cc).\n"
        f"Captured stderr:\n{captured.err[:2000]}")


@pytest.mark.skipif(not is_corex(), reason="Iluvatar backend only")
def test_loop_unroll_warning_can_be_reenabled(capfd, fresh_triton_cache, monkeypatch):
    """Setting TRITON_ENABLE_LOOP_UNROLL_WARNING restores the warning."""
    monkeypatch.setenv("TRITON_ENABLE_LOOP_UNROLL_WARNING", "1")
    constexprs = {
        "NUM_K_BLOCKS": 32,
        "BLOCK_M": 32,
        "BLOCK_N": 32,
        "BLOCK_K": 32,
    }
    signature = {
        "a_ptr": "*fp16",
        "b_ptr": "*fp16",
        "c_ptr": "*fp32",
        "stride_am": "i32",
        "stride_ak": "i32",
        "stride_bk": "i32",
        "stride_bn": "i32",
        "stride_cm": "i32",
        "stride_cn": "i32",
        "NUM_K_BLOCKS": "constexpr",
        "BLOCK_M": "constexpr",
        "BLOCK_N": "constexpr",
        "BLOCK_K": "constexpr",
    }
    triton.compile(triton.compiler.ASTSource(
        fn=_matmul,
        signature=signature,
        constexprs=constexprs,
    ), )

    captured = capfd.readouterr()
    assert "loop with constant trip count not unrolled" in captured.err, (
        "TRITON_ENABLE_LOOP_UNROLL_WARNING did not re-enable the ixcc "
        "loop-unroll warning.\n"
        f"Captured stderr:\n{captured.err[:2000]}")
