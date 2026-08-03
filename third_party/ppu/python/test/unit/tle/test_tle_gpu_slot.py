"""PPU compile-only port of python/test/tle/unit/test_tle_gpu_slot.py.

Upstream test pins down the ``buffered_tensor.slot(stage)`` lowering:
allocating a 2-stage ring buffer in smem and indexing one stage should
materialize as a ``ttg.memdesc_index`` over an outer
``!ttg.memdesc<2x64xi32, ...>``, projecting to an inner
``!ttg.memdesc<64xi32, ...>``. The lowering is shared with the
TritonGPU multi-stage pipelining machinery, so this exercises a piece
of TLE that the canonical local_ptr / GEMM tests don't reach.
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


_PPU_TARGET = GPUTarget("ppu", 80, 32)


@triton.jit
def _slot_local_ptr_store_kernel(out_ptr, BLOCK: tl.constexpr):
    idx = tl.arange(0, BLOCK)
    # Two-stage ring buffer in smem; .slot(0) projects to the first stage.
    smem = tle.gpu.alloc([2, BLOCK], dtype=tl.int32, layout=None, scope=tle.gpu.smem, nv_mma_shared_layout=False)
    slot = smem.slot(0)
    ptrs = tle.gpu.local_ptr(slot, (idx, ))
    tl.store(ptrs, idx + 7)
    vals = tl.load(ptrs)
    tl.store(out_ptr + idx, vals)


def _compile(block=64):
    src = triton.compiler.ASTSource(
        fn=_slot_local_ptr_store_kernel,
        signature={"out_ptr": "*i32", "BLOCK": "constexpr"},
        constexprs={"BLOCK": block},
    )
    return triton.compile(src, target=_PPU_TARGET)


def test_buffered_tensor_slot_lowers_to_memdesc_index():
    """Mirror of the upstream invariant: ``.slot(0)`` must materialize as a
    ``ttg.memdesc_index`` and the outer/inner memdesc types must both
    appear in ttgir."""
    if not _ppu_sdk_available():
        pytest.skip("PPU SDK not available")
    compiled = _compile(64)
    ttgir = compiled.asm["ttgir"]
    assert "ttg.memdesc_index" in ttgir, ("no ttg.memdesc_index in ttgir; .slot(0) failed to lower\n"
                                          f"--- ttgir tail ---\n{ttgir[-1500:]}")
    assert "!ttg.memdesc<2x64xi32" in ttgir, "outer 2x64 alloc type missing"
    assert "!ttg.memdesc<64xi32" in ttgir, "inner per-slot 64 type missing"


def test_buffered_tensor_slot_compiles_to_hgbin():
    if not _ppu_sdk_available():
        pytest.skip("PPU SDK not available")
    compiled = _compile(64)
    for stage in ("ttir", "ttgir", "llir", "hgbin"):
        assert stage in compiled.asm and compiled.asm[stage], \
            f"stage {stage} missing or empty"


def test_buffered_tensor_slot_no_tle_residue_in_llir():
    if not _ppu_sdk_available():
        pytest.skip("PPU SDK not available")
    compiled = _compile(64)
    llir = compiled.asm["llir"]
    leak = [ln for ln in llir.split("\n") if re.search(r"\btle\.[a-z]+_[a-z]+\b", ln) and "@_" not in ln]
    assert not leak, "unexpected tle.* in llir:\n" + "\n".join(leak[:8])
