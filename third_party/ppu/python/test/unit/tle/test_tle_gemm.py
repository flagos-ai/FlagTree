"""TLE GEMM compile-only test for PPU.

Adapted from python/test/tle/integration/test_tle_gemm.py. This is the most
ambitious workload we can compile against the PPU TLE wiring today and the
primary vehicle for the AxisInfoExt-on-PPU-encodings risk identified in the
adaptation plan: the kernel exercises

  * 2D ``tle.gpu.local_ptr`` (smoke only covered 1D)
  * ``tle.gpu.copy`` global -> smem (smoke did the load/store dance manually)
  * ``tl.dot`` whose operands flow through ``tle.local_pointers`` -> ``tl.load``
    (forces DotOperandEncoding selection through the TLE-aware axis-info
    analysis)
  * A K-axis loop with shared-memory reuse across iterations

We compile to llir/hgbin (no GPU execution) and assert that:
  * the kernel reaches the final stage,
  * no ``tle.*`` op survives into llir,
  * the dot is materialized in llir (so we did not get short-circuited out).
"""

import os
import shutil

import pytest

import triton
import triton.language as tl
from triton.backends.compiler import GPUTarget

tle_backend = pytest.importorskip(
    "triton._C.libtriton.tle",
    reason="libtriton built without FLAGTREE_TLE",
)
tle = pytest.importorskip(
    "triton.experimental.tle.language.gpu",
    reason="triton.experimental.tle.language.gpu unavailable",
)


def _ppu_sdk_available() -> bool:
    sdk = os.environ.get("PPU_SDK") or os.environ.get("PPU_HOME")
    if sdk and os.path.isfile(os.path.join(sdk, "bin", "ppu-llc")):
        return True
    return bool(shutil.which("ppu-llc"))


_GPU_TARGET = GPUTarget("cuda", 80, 32)


@triton.jit
def _gemm_kernel(
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
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    # nv_mma_shared_layout=False to use the generic swizzled_shared_layout path
    # that does not require Hopper-specific encodings.
    a_smem = tle.alloc(
        [BLOCK_M, BLOCK_N],
        dtype=tl.float32,
        layout=None,
        scope=tle.smem,
        nv_mma_shared_layout=False,
    )
    b_smem = tle.alloc(
        [BLOCK_M, BLOCK_N],
        dtype=tl.float32,
        layout=None,
        scope=tle.smem,
        nv_mma_shared_layout=False,
    )
    row_ids = tl.broadcast_to(tl.arange(0, BLOCK_M)[:, None], (BLOCK_M, BLOCK_N))
    col_ids = tl.broadcast_to(tl.arange(0, BLOCK_N)[None, :], (BLOCK_M, BLOCK_N))
    a_smem_ptrs = tle.local_ptr(a_smem, (row_ids, col_ids))
    b_smem_ptrs = tle.local_ptr(b_smem, (row_ids, col_ids))

    for k_start in range(0, K, BLOCK_K):
        k_offs = k_start + tl.arange(0, BLOCK_K)
        a_ptrs = a_ptr + offs_m[:, None] * stride_am + k_offs[None, :] * stride_ak
        b_ptrs = b_ptr + k_offs[:, None] * stride_bk + offs_n[None, :] * stride_bn

        tle.copy(a_ptrs, a_smem, [BLOCK_M, BLOCK_N])
        tle.copy(b_ptrs, b_smem, [BLOCK_M, BLOCK_N])
        a_tile = tl.load(a_smem_ptrs)
        b_tile = tl.load(b_smem_ptrs)
        accumulator += tl.dot(a_tile, b_tile, input_precision="ieee")

    c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    tl.store(c_ptrs, accumulator, mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))


_SIGNATURE = {
    "a_ptr": "*fp32",
    "b_ptr": "*fp32",
    "c_ptr": "*fp32",
    "M": "i32",
    "N": "i32",
    "K": "i32",
    "stride_am": "i32",
    "stride_ak": "i32",
    "stride_bk": "i32",
    "stride_bn": "i32",
    "stride_cm": "i32",
    "stride_cn": "i32",
    "BLOCK_M": "constexpr",
    "BLOCK_N": "constexpr",
    "BLOCK_K": "constexpr",
}


def _compile(block_m=64, block_n=64, block_k=64):
    src = triton.compiler.ASTSource(
        fn=_gemm_kernel,
        signature=_SIGNATURE,
        constexprs={"BLOCK_M": block_m, "BLOCK_N": block_n, "BLOCK_K": block_k},
    )
    return triton.compile(src, target=_GPU_TARGET)


def test_tle_gemm_64_compiles_to_hgbin():
    """BLOCK 64 — exact shape from upstream test, the canonical TLE GEMM."""
    if not _ppu_sdk_available():
        pytest.skip("PPU SDK not available; cannot run hgbin stage")
    compiled = _compile(64, 64, 64)
    for stage in ("ttir", "ttgir", "llir", "hgbin"):
        assert stage in compiled.asm, f"missing stage: {stage}"
        assert len(compiled.asm[stage]) > 0


def test_tle_gemm_no_tle_residue_in_llir():
    """The TLE C++ conversion patterns we registered for PPU must consume
    every tle.* op by the time llir is emitted. A residue here would mean
    AxisInfoExt or pattern matching disagreed with the PPU encoding choices
    and left an op behind."""
    if not _ppu_sdk_available():
        pytest.skip("PPU SDK not available; cannot run hgbin stage")
    compiled = _compile(64, 64, 64)
    llir = compiled.asm["llir"]
    residual = "\n".join(ln for ln in llir.split("\n") if "tle." in ln)
    assert not residual, f"unexpected tle.* op in llir:\n{residual}"


def test_tle_gemm_emits_dot_in_llir():
    """Guard that the kernel actually reaches LLVM-level matmul codegen rather
    than being silently DCE'd."""
    if not _ppu_sdk_available():
        pytest.skip("PPU SDK not available; cannot run hgbin stage")
    compiled = _compile(64, 64, 64)
    llir = compiled.asm["llir"]
    # ppu lowering goes through ldmatrix-style intrinsics or mma-equivalent
    # instructions; any one of these markers means we got to the dot.
    markers = ("ldmatrix", "mma.", "ppu.mma", "fmadd", "fmuladd", "fma.")
    assert any(m in llir for m in markers), (f"llir contains none of {markers}; tl.dot may have been lost.\n"
                                             f"--- llir tail ---\n{llir[-2000:]}")


def test_tle_gemm_ttgir_carries_dot_operand_encoding():
    """Sanity-check that the TLE TTGIR passes selected a DotOperand encoding
    for the local-pointer loads feeding tl.dot. Without it, AxisInfoExt would
    have failed to recognize the PPU dot path."""
    if not _ppu_sdk_available():
        pytest.skip("PPU SDK not available; cannot run hgbin stage")
    compiled = _compile(64, 64, 64)
    ttgir = compiled.asm["ttgir"]
    assert "DotOperandEncoding" in ttgir or "dot_op" in ttgir, (
        f"no DotOperand encoding in ttgir — TLE pipeline may not have picked "
        f"a dot-friendly layout for the local_ptr loads.\n--- ttgir tail ---\n"
        f"{ttgir[-2000:]}")
