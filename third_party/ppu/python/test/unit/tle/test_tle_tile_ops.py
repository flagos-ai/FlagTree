"""TLE extract_tile / insert_tile compile-only tests for PPU.

Adapted from the four upstream tests in python/test/tle/unit/:
  * test_extract_tile_static_index.py
  * test_extract_tile_dynamic_index.py
  * test_insert_tile_static_index.py
  * test_insert_tile_dynamic_index.py

These exist primarily to exercise the C++ conversion patterns that
TritonPPUGPUToLLVM/TritonGPUToLLVM.cpp registers but the existing PPU TLE
tests do not trigger:

  * mlir::triton::tle::populateExtractTileOpToLLVMPatterns
  * mlir::triton::tle::populateInsertTileOpToLLVMPatterns

Upstream's "dynamic_index" tests do not in fact emit a dynamic MLIR index
operand — each branch of an ``if`` uses a static index, so the same op is
exercised twice through control flow. We keep that structure so we cover
exactly what upstream covers.
"""

import os
import shutil

import pytest

import triton
import triton.language as tl
from triton.backends.compiler import GPUTarget

tle_backend = pytest.importorskip("triton._C.libtriton.tle", reason="libtriton built without FLAGTREE_TLE")
tle = pytest.importorskip("triton.experimental.tle.language", reason="tle language unavailable")


def _ppu_sdk_available() -> bool:
    sdk = os.environ.get("PPU_SDK") or os.environ.get("PPU_HOME")
    if sdk and os.path.isfile(os.path.join(sdk, "bin", "ppu-llc")):
        return True
    return bool(shutil.which("ppu-llc"))


_GPU_TARGET = GPUTarget("cuda", 80, 32)

# ---------------------------------------------------------------------------
# Kernels (mirrors of the upstream tests)
# ---------------------------------------------------------------------------


@triton.jit
def _extract_tile_static(x_ptr, out_ptr, M: tl.constexpr, N: tl.constexpr):
    offs_m = tl.arange(0, M)
    offs_n = tl.arange(0, N)
    x = tl.load(x_ptr + offs_m[:, None] * N + offs_n[None, :])
    tile = tle.extract_tile(x, index=[1, 1], tile_shape=[128, 128])
    out_offs_m = tl.arange(0, 128)
    out_offs_n = tl.arange(0, 128)
    tl.store(out_ptr + out_offs_m[:, None] * 128 + out_offs_n[None, :], tile)


@triton.jit
def _extract_tile_dynamic(x_ptr, out_ptr, stride_xb, stride_xm, stride_xn, stride_ob, stride_om, stride_on,
                          BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, TILE_M: tl.constexpr, TILE_N: tl.constexpr):
    pid_z = tl.program_id(0)
    offs_m = tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    x_ptrs = (x_ptr + pid_z * stride_xb + offs_m[:, None] * stride_xm + offs_n[None, :] * stride_xn)
    bg_tile = tl.load(x_ptrs)
    if pid_z % 2 == 0:
        extracted_tile = tle.extract_tile(bg_tile, index=[0, 0], tile_shape=[TILE_M, TILE_N])
    else:
        extracted_tile = tle.extract_tile(bg_tile, index=[1, 1], tile_shape=[TILE_M, TILE_N])
    offs_tm = tl.arange(0, TILE_M)
    offs_tn = tl.arange(0, TILE_N)
    out_ptrs = (out_ptr + pid_z * stride_ob + offs_tm[:, None] * stride_om + offs_tn[None, :] * stride_on)
    tl.store(out_ptrs, extracted_tile)


@triton.jit
def _insert_tile_static(x_ptr, y_ptr, out_ptr, M: tl.constexpr, N: tl.constexpr, TM: tl.constexpr, TN: tl.constexpr):
    offs_m = tl.arange(0, M)
    offs_n = tl.arange(0, N)
    x = tl.load(x_ptr + offs_m[:, None] * N + offs_n[None, :])
    tile_m = tl.arange(0, TM)
    tile_n = tl.arange(0, TN)
    y = tl.load(y_ptr + tile_m[:, None] * TN + tile_n[None, :])
    z = tle.insert_tile(x, y, index=[1, 1])
    tl.store(out_ptr + offs_m[:, None] * N + offs_n[None, :], z)


@triton.jit
def _insert_tile_dynamic(x_ptr, y_ptr, stride_xb, stride_xm, stride_xn, stride_ym, stride_yn, BLOCK_M: tl.constexpr,
                         BLOCK_N: tl.constexpr, TILE_M: tl.constexpr, TILE_N: tl.constexpr):
    pid_z = tl.program_id(0)
    pid_m = tl.program_id(1)
    pid_n = tl.program_id(2)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    x_ptrs = (x_ptr + pid_z * stride_xb + offs_m[:, None] * stride_xm + offs_n[None, :] * stride_xn)
    bg_tile = tl.load(x_ptrs)
    offs_tm = tl.arange(0, TILE_M)
    offs_tn = tl.arange(0, TILE_N)
    y_ptrs = (y_ptr + offs_tm[:, None] * stride_ym + offs_tn[None, :] * stride_yn)
    small_tile = tl.load(y_ptrs)
    if pid_z % 2 == 0:
        res_tile = tle.insert_tile(bg_tile, small_tile, index=[0, 0])
    else:
        res_tile = tle.insert_tile(bg_tile, small_tile, index=[1, 1])
    tl.store(x_ptrs, res_tile)


# ---------------------------------------------------------------------------
# Compile helpers
# ---------------------------------------------------------------------------


def _compile(fn, signature, constexprs):
    src = triton.compiler.ASTSource(fn=fn, signature=signature, constexprs=constexprs)
    return triton.compile(src, target=_GPU_TARGET)


def _assert_no_tile_residue(compiled, op_name):
    """Check the C++ tile lowering pattern actually consumed the op. We match
    the MLIR op syntax (``tle.extract_tile`` / ``tle.insert_tile`` followed by
    whitespace or ``(``) rather than the substring, because the kernel
    function symbol names (e.g. ``@_extract_tile_static``) contain the
    substring and would produce false positives in LLVM IR."""
    import re
    llir = compiled.asm["llir"]
    pat = re.compile(r"\btle\.(extract_tile|insert_tile)[\s(]")
    leak = [ln for ln in llir.split("\n") if pat.search(ln)]
    assert not leak, (f"{op_name} pattern did not consume the op — found residue in llir:\n" + "\n".join(leak[:8]))


def _assert_full_pipeline(compiled):
    for stage in ("ttir", "ttgir", "llir", "hgbin"):
        assert stage in compiled.asm and len(compiled.asm[stage]) > 0, \
            f"stage {stage} missing or empty"


# ---------------------------------------------------------------------------
# The four ported tests
# ---------------------------------------------------------------------------


def test_extract_tile_static_index_compiles():
    """Mirror of test_extract_tile_static_index.py — extract_tile with a
    compile-time multi-dim index list."""
    if not _ppu_sdk_available():
        pytest.skip("PPU SDK not available")
    compiled = _compile(
        _extract_tile_static,
        signature={"x_ptr": "*fp32", "out_ptr": "*fp32", "M": "constexpr", "N": "constexpr"},
        constexprs={"M": 512, "N": 512},
    )
    _assert_full_pipeline(compiled)
    _assert_no_tile_residue(compiled, "extract_tile")


def test_extract_tile_dynamic_index_compiles():
    """Mirror of test_extract_tile_dynamic_index.py — same op used inside both
    branches of an ``if pid_z % 2`` so the codegen sees the op twice through
    control flow."""
    if not _ppu_sdk_available():
        pytest.skip("PPU SDK not available")
    compiled = _compile(
        _extract_tile_dynamic,
        signature={
            "x_ptr": "*fp32", "out_ptr": "*fp32", "stride_xb": "i32", "stride_xm": "i32", "stride_xn": "i32",
            "stride_ob": "i32", "stride_om": "i32", "stride_on": "i32", "BLOCK_M": "constexpr", "BLOCK_N": "constexpr",
            "TILE_M": "constexpr", "TILE_N": "constexpr"
        },
        constexprs={"BLOCK_M": 32, "BLOCK_N": 32, "TILE_M": 16, "TILE_N": 16},
    )
    _assert_full_pipeline(compiled)
    _assert_no_tile_residue(compiled, "extract_tile")


def test_insert_tile_static_index_compiles():
    """Mirror of test_insert_tile_static_index.py."""
    if not _ppu_sdk_available():
        pytest.skip("PPU SDK not available")
    compiled = _compile(
        _insert_tile_static,
        signature={
            "x_ptr": "*fp32", "y_ptr": "*fp32", "out_ptr": "*fp32", "M": "constexpr", "N": "constexpr", "TM":
            "constexpr", "TN": "constexpr"
        },
        constexprs={"M": 512, "N": 512, "TM": 128, "TN": 128},
    )
    _assert_full_pipeline(compiled)
    _assert_no_tile_residue(compiled, "insert_tile")


def test_insert_tile_dynamic_index_compiles():
    """Mirror of test_insert_tile_dynamic_index.py."""
    if not _ppu_sdk_available():
        pytest.skip("PPU SDK not available")
    compiled = _compile(
        _insert_tile_dynamic,
        signature={
            "x_ptr": "*fp32", "y_ptr": "*fp32", "stride_xb": "i32", "stride_xm": "i32", "stride_xn": "i32", "stride_ym":
            "i32", "stride_yn": "i32", "BLOCK_M": "constexpr", "BLOCK_N": "constexpr", "TILE_M": "constexpr", "TILE_N":
            "constexpr"
        },
        constexprs={"BLOCK_M": 32, "BLOCK_N": 32, "TILE_M": 16, "TILE_N": 16},
    )
    _assert_full_pipeline(compiled)
    _assert_no_tile_residue(compiled, "insert_tile")
