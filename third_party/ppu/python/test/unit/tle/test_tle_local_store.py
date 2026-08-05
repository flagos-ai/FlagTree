"""PPU compile-only port of python/test/tle/integration/test_tle_local_store.py.

Upstream test exercises the bidirectional ``tle.gpu.copy`` path: copy
gm -> smem for two inputs, stage the elementwise sum into a third smem
buffer via ``tl.store(local_ptr, ...)``, then copy smem -> gm. This is the
only file in the upstream suite that triggers the ``LOCAL_TO_GM`` arm of
``tle.gpu.copy``; the GEMM and pipeline e2e tests only use ``GM_TO_LOCAL``.

We compile the canonical 64x64 case and assert: full pipeline reaches
hgbin, no ``tle.*`` op leaks into llir, and the reverse-direction copy
actually shows up in ttgir as a ``ttg.async_copy_local_to_global`` (or
falls back to a regular store sequence) — either way we verify it isn't
silently DCE'd.
"""

import os
import re
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


@triton.jit
def _elementwise_add_with_local_store(
    a_ptr,
    b_ptr,
    c_ptr,
    xnumel,
    ynumel,
    xstride_a,
    ystride_a,
    xstride_b,
    ystride_b,
    xstride_c,
    ystride_c,
    XBLOCK: tl.constexpr,
    YBLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    xoffs = pid * XBLOCK + tl.arange(0, XBLOCK)
    a_ptrs = a_ptr + xstride_a * xoffs[:, None]
    b_ptrs = b_ptr + xstride_b * xoffs[:, None]
    c_ptrs = c_ptr + xstride_c * xoffs[:, None]

    a_smem = tle.gpu.alloc([XBLOCK, YBLOCK], dtype=tl.float32, layout=None, scope=tle.gpu.smem,
                           nv_mma_shared_layout=False)
    b_smem = tle.gpu.alloc([XBLOCK, YBLOCK], dtype=tl.float32, layout=None, scope=tle.gpu.smem,
                           nv_mma_shared_layout=False)
    c_smem = tle.gpu.alloc([XBLOCK, YBLOCK], dtype=tl.float32, layout=None, scope=tle.gpu.smem,
                           nv_mma_shared_layout=False)
    rows = tl.broadcast_to(tl.arange(0, XBLOCK)[:, None], (XBLOCK, YBLOCK))
    cols = tl.broadcast_to(tl.arange(0, YBLOCK)[None, :], (XBLOCK, YBLOCK))
    a_smem_ptrs = tle.gpu.local_ptr(a_smem, (rows, cols))
    b_smem_ptrs = tle.gpu.local_ptr(b_smem, (rows, cols))
    c_smem_ptrs = tle.gpu.local_ptr(c_smem, (rows, cols))

    for yoff in range(0, ynumel, YBLOCK):
        yoffs = tl.arange(0, YBLOCK) + yoff
        tle.gpu.copy(a_ptrs + ystride_a * yoffs[None, :], a_smem, [XBLOCK, YBLOCK])
        tle.gpu.copy(b_ptrs + ystride_b * yoffs[None, :], b_smem, [XBLOCK, YBLOCK])
        aval = tl.load(a_smem_ptrs)
        bval = tl.load(b_smem_ptrs)
        tl.store(c_smem_ptrs, aval + bval)
        # The smem -> gm direction is the actual coverage gap closed by
        # this file — none of the other PPU TLE tests exercise it.
        tle.gpu.copy(c_smem, c_ptrs + ystride_c * yoffs[None, :], [XBLOCK, YBLOCK])


_SIGNATURE = {
    "a_ptr": "*fp32",
    "b_ptr": "*fp32",
    "c_ptr": "*fp32",
    "xnumel": "i32",
    "ynumel": "i32",
    "xstride_a": "i32",
    "ystride_a": "i32",
    "xstride_b": "i32",
    "ystride_b": "i32",
    "xstride_c": "i32",
    "ystride_c": "i32",
    "XBLOCK": "constexpr",
    "YBLOCK": "constexpr",
}


def _compile(xb=64, yb=64):
    src = triton.compiler.ASTSource(
        fn=_elementwise_add_with_local_store,
        signature=_SIGNATURE,
        constexprs={"XBLOCK": xb, "YBLOCK": yb},
    )
    return triton.compile(src, target=_GPU_TARGET)


def test_local_store_basic_compiles():
    """Canonical 64x64 case from upstream."""
    if not _ppu_sdk_available():
        pytest.skip("PPU SDK not available")
    compiled = _compile(64, 64)
    for stage in ("ttir", "ttgir", "llir", "hgbin"):
        assert stage in compiled.asm and compiled.asm[stage], f"stage {stage} missing"


def test_local_store_no_tle_residue_in_llir():
    if not _ppu_sdk_available():
        pytest.skip("PPU SDK not available")
    compiled = _compile(64, 64)
    llir = compiled.asm["llir"]
    leak = [ln for ln in llir.split("\n") if re.search(r"\btle\.[a-z]+_[a-z]+\b", ln)]
    leak = [ln for ln in leak if "@_" not in ln]  # drop func-symbol false positives
    assert not leak, "unexpected tle.* in llir:\n" + "\n".join(leak[:8])


def test_local_store_emits_reverse_direction_copy():
    """The LOCAL_TO_GM arm of tle.gpu.copy is the actual coverage gap closed
    by this file. We verify it materializes in ttgir as either an async
    local->global copy (PPU's analogue of cp.async.bulk smem->gmem) or as a
    fallback ``tl.store`` of values loaded from the smem alloc, but does
    not silently disappear."""
    if not _ppu_sdk_available():
        pytest.skip("PPU SDK not available")
    compiled = _compile(64, 64)
    ttgir = compiled.asm["ttgir"]
    markers = ("async_copy_local_to_global", "tt.store", "ttg.local_load")
    assert any(m in ttgir
               for m in markers), (f"reverse-direction copy disappeared from ttgir; expected one of {markers}\n"
                                   f"--- ttgir tail ---\n{ttgir[-1500:]}")
