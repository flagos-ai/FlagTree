"""Smoke tests for the TLE (Triton Language Extensions) wiring on the PPU
backend.

These tests are compile-only — they walk a kernel through ttir -> ttgir ->
llir (and through hgbin when PPU_SDK is available) and assert that:

  1. A baseline kernel that uses no TLE op still lowers cleanly. This guards
     against regressions in PPU's existing pipeline from the TLE plumbing.
  2. A kernel that uses ``tle.alloc`` + ``tle.local_ptr`` reaches llir, the
     ``tle.local_pointers`` op is preserved through ttgir as expected (the
     ttgir-level TLE passes only decorate / optimize it), and the
     TleLLVMConversionTarget plus populateLocalPointersOpToLLVMPatterns in
     third_party/ppu/lib/TritonPPUGPUToLLVM/TritonGPUToLLVM.cpp consume every
     ``tle.*`` op by the time llir is emitted.
"""

import os
import shutil

import pytest

import triton
import triton.language as tl
from triton.backends.compiler import GPUTarget

tle_backend = pytest.importorskip(
    "triton._C.libtriton.tle",
    reason="libtriton was built without FLAGTREE_TLE; PPU TLE smoke tests are not applicable.",
)
tle = pytest.importorskip(
    "triton.experimental.tle.language.gpu",
    reason="triton.experimental.tle.language.gpu unavailable",
)


def _ppu_sdk_available() -> bool:
    """The hgbin stage shells out to ppu-llc + llvm-irformatter. Skip it when
    no SDK is reachable so the test still exercises the MLIR-side plumbing on
    machines without a PPU toolchain."""
    sdk = os.environ.get("PPU_SDK") or os.environ.get("PPU_HOME")
    if sdk and os.path.isfile(os.path.join(sdk, "bin", "ppu-llc")):
        return True
    return bool(shutil.which("ppu-llc"))


_PPU_TARGET = GPUTarget("ppu", 80, 32)
_SIGNATURE = {
    "x_ptr": "*fp32",
    "y_ptr": "*fp32",
    "out_ptr": "*fp32",
    "n": "i32",
    "BLOCK": "constexpr",
}
_CONSTEXPRS = {"BLOCK": 64}


@triton.jit
def _vector_add_baseline(x_ptr, y_ptr, out_ptr, n, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    x = tl.load(x_ptr + offs, mask=mask, other=0.0)
    y = tl.load(y_ptr + offs, mask=mask, other=0.0)
    tl.store(out_ptr + offs, x + y, mask=mask)


@triton.jit
def _vector_add_tle(x_ptr, y_ptr, out_ptr, n, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    smem = tle.alloc([BLOCK], dtype=tl.float32, scope=tle.smem)
    ptrs = tle.local_ptr(smem, (tl.arange(0, BLOCK),))
    x = tl.load(x_ptr + offs, mask=mask, other=0.0)
    y = tl.load(y_ptr + offs, mask=mask, other=0.0)
    tl.store(ptrs, x + y, mask=mask)
    out = tl.load(ptrs, mask=mask, other=0.0)
    tl.store(out_ptr + offs, out, mask=mask)


def _compile(kernel):
    src = triton.compiler.ASTSource(
        fn=kernel, signature=_SIGNATURE, constexprs=_CONSTEXPRS)
    return triton.compile(src, target=_PPU_TARGET)


def test_baseline_kernel_still_lowers():
    """No TLE op anywhere — guards against regressions in the existing PPU
    pipeline from the TLE wiring."""
    if not _ppu_sdk_available():
        pytest.skip("PPU SDK not available; cannot run hgbin stage")
    compiled = _compile(_vector_add_baseline)
    assert "ttir" in compiled.asm
    assert "ttgir" in compiled.asm
    assert "llir" in compiled.asm
    assert "hgbin" in compiled.asm
    assert "tle." not in compiled.asm["ttgir"], \
        "baseline kernel must not emit any tle.* op"
    assert "tle." not in compiled.asm["llir"]


def test_tle_local_ptr_lowers_to_llir():
    """tle.alloc + tle.local_ptr round-trip — exercises the new TLE
    conversion patterns registered in TritonPPUGPUToLLVM."""
    if not _ppu_sdk_available():
        pytest.skip("PPU SDK not available; cannot run hgbin stage")
    compiled = _compile(_vector_add_tle)
    ttgir = compiled.asm["ttgir"]
    llir = compiled.asm["llir"]

    # The TTGIR-level TLE passes (early_assign_memory_space, select_encodings,
    # insert_local_pointer_barriers, optimize_local_pointer_loads/stores) only
    # decorate or optimize tle.local_pointers — the op itself survives to
    # llir-time lowering.
    assert "tle.local_pointers" in ttgir, \
        "TLE local_pointers op should survive the ttgir-level TLE passes"

    # By llir there should be no tle.* op left: the C++ TleLLVMConversionTarget
    # in TritonPPUGPUToLLVM.cpp populated patterns that consume every TLE op
    # we register today (LocalPointers / Extract / Pack / ExtractTile /
    # InsertTile / DSLRegion).
    assert "tle." not in llir, \
        f"unexpected residual tle.* op in llir:\n{_grep(llir, 'tle.')}"

    # Sanity: the shared-memory allocation made it to LLVM.
    assert "@global_smem" in llir
    assert "addrspace(3)" in llir


def test_smem_size_metadata_present():
    """BLOCK=64 fp32 should produce smemsize=256 metadata on the kernel."""
    if not _ppu_sdk_available():
        pytest.skip("PPU SDK not available; cannot run hgbin stage")
    compiled = _compile(_vector_add_tle)
    llir = compiled.asm["llir"]
    # ppu sets the smemsize via nvvm.annotations: !{ptr @k, !"smemsize", i32 N}
    assert '"smemsize"' in llir
    assert "i32 256" in llir, "expected 256-byte shared allocation (64 * fp32)"


def _grep(text: str, needle: str) -> str:
    return "\n".join(ln for ln in text.split("\n") if needle in ln)


# --- Rejection of unsupported TLE features on PPU --------------------------
# These verify the capability registry in
# triton.experimental.tle._capabilities (populated by
# third_party/ppu/backend/__init__.py) raises a clear Python error before MLIR
# legalization would otherwise fail with a cryptic "failed to legalize" message.


import triton.experimental.tle.language as _tle_lang


@triton.jit
def _kernel_cumsum(x_ptr, out_ptr, n, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    x = tl.load(x_ptr + offs, mask=mask, other=0.0)
    excl, _total = _tle_lang.cumsum(x)
    tl.store(out_ptr + offs, excl, mask=mask)


@triton.jit
def _kernel_cumsum_simple(x_ptr, out_ptr, BLOCK: tl.constexpr):
    offs = tl.arange(0, BLOCK)
    x = tl.load(x_ptr + offs)
    excl, _total = _tle_lang.cumsum(x)
    tl.store(out_ptr + offs, excl)


@triton.jit
def _kernel_distributed_barrier():
    _tle_lang.distributed_barrier()


@triton.jit
def _kernel_warp_specialize():
    # NV-style warp_specialize partition list; on PPU this must be rejected
    # at the Python frontend before any IR is emitted.
    _tle_lang.gpu.warp_specialize(((), ), (1, ), (32, ))


def _compile_2arg(kernel, sig, const):
    return triton.compile(
        triton.compiler.ASTSource(fn=kernel, signature=sig, constexprs=const),
        target=_PPU_TARGET,
    )


def test_ppu_supports_cumsum():
    """tle.cumsum compiles to hgbin on PPU — the ExclusiveCumsumOp conversion
    pattern and Allocation.cpp handler are both wired up."""
    if not _ppu_sdk_available():
        pytest.skip("PPU SDK not available")
    compiled = _compile_2arg(
        _kernel_cumsum_simple,
        {"x_ptr": "*fp32", "out_ptr": "*fp32", "BLOCK": "constexpr"},
        {"BLOCK": 64},
    )
    for stage in ("ttir", "ttgir", "llir", "hgbin"):
        assert stage in compiled.asm and compiled.asm[stage]


def test_ppu_rejects_distributed_barrier_with_clear_error():
    with pytest.raises(Exception) as excinfo:
        triton.compile(
            triton.compiler.ASTSource(
                fn=_kernel_distributed_barrier, signature={}, constexprs={}),
            target=_PPU_TARGET,
        )
    chain = _exception_chain(excinfo.value)
    full = str(excinfo.value) + "".join(str(c) for c in chain)
    assert "tle.distributed_barrier" in full
    assert "ppu" in full.lower()


def test_ppu_rejects_warp_specialize_with_clear_error():
    with pytest.raises(Exception) as excinfo:
        triton.compile(
            triton.compiler.ASTSource(
                fn=_kernel_warp_specialize, signature={}, constexprs={}),
            target=_PPU_TARGET,
        )
    chain = _exception_chain(excinfo.value)
    full = str(excinfo.value) + "".join(str(c) for c in chain)
    assert "tle.warp_specialize" in full
    assert "ppu" in full.lower()


def _exception_chain(exc):
    chain = []
    while exc is not None:
        chain.append(exc)
        exc = exc.__cause__ or exc.__context__
    return chain
