"""Iluvatar-specific coverage for tle_raw.call / tle_raw.call_smem.

The frontend bookkeeping of tle_raw (@dialect kwargs, source cache keys) is
builder-agnostic and already covered by python/test/tle/unit/
test_tle_raw_cache_key.py, which iluvatar CI runs directly. This file pins the
backend-local part: the corex clang driver, the iluvatar_tle op prefix, the
address spaces the descriptor protocol has to produce, and the two invariants
that broke while bringing tle_raw up on corex -- that no dsl_region reaches the
LLVM conversion, and that the raw device function does not steal the kernel's
calling convention. The deferred path is pinned here too, since its stub is
filled by the corex-local IluvatarMaterializeDeferredRaw pass.

TestCorexNative goes past the CUDA-portable sources and pins what tle_raw is
for on this backend: a region that reaches corex-only hardware. It drives the
SME global->shared engine through the corex clang builtins.
"""
import re
from pathlib import Path

import pytest
import torch
import triton
import triton.language as tl
from triton.experimental.tle.raw import dialect
import triton.experimental.tle.language.gpu as tle_gpu
import triton.experimental.tle.language.raw as tle_raw

from utils import compile_iluvatar

RAW_DIR = Path(__file__).parent / "raw"

BLOCK = 16

# One SME transfer is a 16 row x 64 byte hardware tile, so 16 fp32 columns.
SME_TILE_ROWS = 16
SME_TILE_COLS = 16


@dialect(name="cuda", file=RAW_DIR / "vector_add.cu")
def vector_add_edsl(*args, **kwargs):
    ...


@dialect(name="cuda", file=RAW_DIR / "smem_accumulate.cu")
def smem_accumulate_edsl(*args, **kwargs):
    ...


@dialect(name="cuda", file=RAW_DIR / "smem_accumulate.cu", deferred=True, extern_func_name="SmemAccumulate")
def smem_accumulate_deferred_edsl(*args, **kwargs):
    ...


# The corex-native spelling of the same JIT; "cuda" is only an alias for it.
@dialect(name="corex", file=RAW_DIR / "sme_load_tile.cu")
def sme_load_tiles_edsl(*args, **kwargs):
    ...


@triton.jit
def _vector_add_kernel(x_ptr, y_ptr, out_ptr, n_elements):
    tle_raw.call(vector_add_edsl, [out_ptr, x_ptr, y_ptr, n_elements])


@triton.jit
def _call_kernel(x_ptr, out_ptr, SIZE: tl.constexpr):
    rows = tl.broadcast_to(tl.arange(0, SIZE)[:, None], (SIZE, SIZE))
    cols = tl.broadcast_to(tl.arange(0, SIZE)[None, :], (SIZE, SIZE))
    offs = rows * SIZE + cols
    acc = tl.zeros((SIZE, SIZE), dtype=tl.float32)
    val = tl.load(x_ptr + offs)
    acc = tle_raw.call(smem_accumulate_edsl, [acc, val])
    tl.store(out_ptr + offs, acc)


@triton.jit
def _call_smem_kernel(x_ptr, out_ptr, SIZE: tl.constexpr):
    rows = tl.broadcast_to(tl.arange(0, SIZE)[:, None], (SIZE, SIZE))
    cols = tl.broadcast_to(tl.arange(0, SIZE)[None, :], (SIZE, SIZE))
    offs = rows * SIZE + cols
    acc_smem = tle_gpu.alloc(shape=[SIZE, SIZE], dtype=tl.float32, layout=None, scope=tle_gpu.smem,
                             nv_mma_shared_layout=False)
    val_smem = tle_gpu.alloc(shape=[SIZE, SIZE], dtype=tl.float32, layout=None, scope=tle_gpu.smem,
                             nv_mma_shared_layout=False)
    acc_ptrs = tle_gpu.local_ptr(acc_smem, (rows, cols))
    tl.store(acc_ptrs, tl.zeros((SIZE, SIZE), dtype=tl.float32))
    val_ptrs = tle_gpu.local_ptr(val_smem, (rows, cols))
    tl.store(val_ptrs, tl.load(x_ptr + offs))
    acc_smem = tle_raw.call_smem(smem_accumulate_edsl, [acc_smem, val_smem])
    tl.store(out_ptr + offs, tl.load(acc_ptrs))


@triton.jit
def _call_smem_deferred_kernel(x_ptr, out_ptr, SIZE: tl.constexpr):
    rows = tl.broadcast_to(tl.arange(0, SIZE)[:, None], (SIZE, SIZE))
    cols = tl.broadcast_to(tl.arange(0, SIZE)[None, :], (SIZE, SIZE))
    offs = rows * SIZE + cols
    acc_smem = tle_gpu.alloc(shape=[SIZE, SIZE], dtype=tl.float32, layout=None, scope=tle_gpu.smem,
                             nv_mma_shared_layout=False)
    val_smem = tle_gpu.alloc(shape=[SIZE, SIZE], dtype=tl.float32, layout=None, scope=tle_gpu.smem,
                             nv_mma_shared_layout=False)
    acc_ptrs = tle_gpu.local_ptr(acc_smem, (rows, cols))
    tl.store(acc_ptrs, tl.zeros((SIZE, SIZE), dtype=tl.float32))
    val_ptrs = tle_gpu.local_ptr(val_smem, (rows, cols))
    tl.store(val_ptrs, tl.load(x_ptr + offs))
    acc_smem = tle_raw.call_smem(smem_accumulate_deferred_edsl, [acc_smem, val_smem], output_indices=[0])
    tl.store(out_ptr + offs, tl.load(acc_ptrs))


@triton.jit
def _sme_load_kernel(src_ptr, out_ptr, stride_bytes, ROWS: tl.constexpr, COLS: tl.constexpr):
    rows = tl.broadcast_to(tl.arange(0, ROWS)[:, None], (ROWS, COLS))
    cols = tl.broadcast_to(tl.arange(0, COLS)[None, :], (ROWS, COLS))
    smem = tle_gpu.alloc(shape=[ROWS, COLS], dtype=tl.float32, layout=None, scope=tle_gpu.smem,
                         nv_mma_shared_layout=True)
    smem = tle_raw.call_smem(sme_load_tiles_edsl, [smem, src_ptr, stride_bytes])
    tl.store(out_ptr + rows * COLS + cols, tl.load(tle_gpu.local_ptr(smem, (rows, cols))))


def _compile_vector_add():
    return compile_iluvatar(
        _vector_add_kernel,
        signature={"x_ptr": "*fp32", "y_ptr": "*fp32", "out_ptr": "*fp32", "n_elements": "i32"},
    )


def _compile_call():
    return compile_iluvatar(
        _call_kernel,
        signature={"x_ptr": "*fp32", "out_ptr": "*fp32", "SIZE": "constexpr"},
        constexprs={"SIZE": BLOCK},
    )


def _compile_call_smem():
    return compile_iluvatar(
        _call_smem_kernel,
        signature={"x_ptr": "*fp32", "out_ptr": "*fp32", "SIZE": "constexpr"},
        constexprs={"SIZE": BLOCK},
    )


def _compile_call_smem_deferred():
    return compile_iluvatar(
        _call_smem_deferred_kernel,
        signature={"x_ptr": "*fp32", "out_ptr": "*fp32", "SIZE": "constexpr"},
        constexprs={"SIZE": BLOCK},
    )


def _compile_sme_load(rows=SME_TILE_ROWS):
    return compile_iluvatar(
        _sme_load_kernel,
        signature={
            "src_ptr": "*fp32", "out_ptr": "*fp32", "stride_bytes": "i32", "ROWS": "constexpr", "COLS": "constexpr"
        },
        constexprs={"ROWS": rows, "COLS": SME_TILE_COLS},
    )


def _run_sme_load(rows=SME_TILE_ROWS):
    device = triton.runtime.driver.active.get_active_torch_device()
    src = torch.randn((rows, SME_TILE_COLS), device=device)
    out = torch.empty_like(src)
    _sme_load_kernel[(1, )](src, out, src.stride(0) * src.element_size(), rows, SME_TILE_COLS)
    return src, out


class TestDialectPrefix:

    def test_call_uses_iluvatar_tle_ops(self):
        ttir = _compile_vector_add().asm["ttir"]
        assert "iluvatar_tle.dsl_region" in ttir, ttir
        assert "iluvatar_tle.yield" in ttir, ttir
        # The trunk `tle` dialect is not built for corex; nothing may reference it.
        assert re.search(r"(?<!iluvatar_)\btle\.", ttir) is None, ttir


class TestSignatureProtocol:

    def test_pointer_operands_lower_to_addrspace_1(self):
        ttir = _compile_vector_add().asm["ttir"]
        assert "iluvatar_tle.extract_ptr" in ttir, ttir
        call = re.search(r"llvm\.call @\w*VectorAdd\w*\([^\n]*", ttir)
        assert call, ttir
        assert "!llvm.ptr<1>" in call.group(0), call.group(0)
        assert "!llvm.ptr<3>" not in call.group(0), call.group(0)

    # A shared-memory buffer expands to the flattened memref descriptor:
    # allocated ptr, aligned ptr, offset, sizes, strides.
    def test_smem_operands_expand_to_memref_descriptor(self):
        ttir = _compile_call_smem().asm["ttir"]
        for op in (
                "iluvatar_tle.extract_allocated_ptr",
                "iluvatar_tle.extract_aligned_ptr",
                "iluvatar_tle.extract_offset",
                "iluvatar_tle.extract_sizes",
                "iluvatar_tle.extract_strides",
        ):
            assert op in ttir, f"{op} missing from\n{ttir}"
        call = re.search(r"llvm\.call @\w*SmemAccumulate\w*\([^\n]*", ttir)
        assert call, ttir
        assert "!llvm.ptr<3>" in call.group(0), call.group(0)

    # A non-void raw function returns the descriptor struct, which comes back
    # through tle.pack.
    def test_struct_return_goes_through_pack(self):
        ttir = _compile_call_smem().asm["ttir"]
        assert "iluvatar_tle.pack" in ttir, ttir


class TestResultWrapping:

    # call_smem keeps the result as a shared-memory buffer, so the value stays
    # a memdesc and no register round trip is inserted for it.
    def test_call_smem_yields_memdesc(self):
        ttir = _compile_call_smem().asm["ttir"]
        lines = ttir.splitlines()
        start = next(i for i, line in enumerate(lines) if "iluvatar_tle.dsl_region" in line)
        # The op's type signature closes the region, several lines below.
        signature = next(line for line in lines[start:] if re.match(r"\s*\}\) : \(", line))
        result_type = signature.split("->", 1)[1]
        assert "ttg.memdesc" in result_type, signature

    # call hands back an ordinary tensor, so convert-arg-to-memdesc has to
    # stage the register operands into shared memory and read the alias back.
    def test_call_spills_register_tensors(self):
        ttgir = _compile_call().asm["ttgir"]
        assert "ttg.local_alloc" in ttgir, ttgir
        assert "ttg.local_store" in ttgir, ttgir
        assert "ttg.local_load" in ttgir, ttgir


class TestLowering:

    # dsl_region has no LLVM lowering on corex; the inline pass must consume
    # every one of them before TritonGPU->LLVM runs.
    def test_dsl_region_does_not_survive_to_llir(self):
        llir = _compile_call_smem().asm["llir"]
        assert "iluvatar_tle" not in llir, llir
        assert "dsl_region" not in llir, llir

    def test_kernel_keeps_iluvatar_calling_convention(self):
        llir = _compile_vector_add().asm["llir"]
        kernels = re.findall(r"define\s+(?:\S+\s+)*iluvatar_kernel\s+\S+\s+@(\w+)\(", llir)
        assert kernels == ["_vector_add_kernel"], llir

    # always_inline on the call plus a plain calling convention on the callee is
    # what lets the corex inliner fold the raw body into the kernel.
    def test_raw_body_is_inlined_into_kernel(self):
        llir = _compile_vector_add().asm["llir"]
        kernel_body = llir.split("define")[1].split("\n}")[0]
        # The call site is gone (only the inlined-return label keeps the name)
        # and the body now reads the block geometry the .cu asked for.
        assert re.search(r"call\b[^\n]*@\w*VectorAdd", kernel_body) is None, kernel_body
        assert "llvm.nvvm.read.ptx.sreg.ctaid.x" in kernel_body, kernel_body


class TestDeferred:

    # Tracing only stamps the stub: the source id is what make_llir looks the
    # pending corex source up by, so it has to reach TTGIR intact.
    def test_stub_carries_source_id(self):
        ttir = _compile_call_smem_deferred().asm["ttir"]
        assert "tle_raw.source_id" in ttir, ttir

    # IluvatarMaterializeDeferredRaw runs before dsl_region_inline, so by llir
    # both the stub marker and the region itself have to be gone.
    def test_materialize_runs_before_inline(self):
        llir = _compile_call_smem_deferred().asm["llir"]
        assert "tle_raw.source_id" not in llir, llir
        assert "dsl_region" not in llir, llir
        assert "iluvatar_tle" not in llir, llir

    # Without a compiled body there is no return type to analyze, so the alias
    # indices cannot be inferred the way the eager path infers them.
    def test_requires_output_indices(self):

        @triton.jit
        def kernel(x_ptr, out_ptr, SIZE: tl.constexpr):
            acc = tl.zeros((SIZE, SIZE), dtype=tl.float32)
            tle_raw.call(smem_accumulate_deferred_edsl, [acc, acc])

        with pytest.raises(Exception, match="output_indices"):
            compile_iluvatar(
                kernel,
                signature={"x_ptr": "*fp32", "out_ptr": "*fp32", "SIZE": "constexpr"},
                constexprs={"SIZE": BLOCK},
            )

    def test_matches_eager_result(self):
        device = triton.runtime.driver.active.get_active_torch_device()
        x = torch.randn((BLOCK, BLOCK), device=device)
        eager_out = torch.empty_like(x)
        deferred_out = torch.empty_like(x)
        _call_smem_kernel[(1, )](x, eager_out, SIZE=BLOCK)
        _call_smem_deferred_kernel[(1, )](x, deferred_out, SIZE=BLOCK)
        torch.testing.assert_close(deferred_out, eager_out)
        torch.testing.assert_close(deferred_out, x)


class TestCorexNative:
    """A raw region that only corex can compile: the SME G2S engine."""

    # The builtins lower to sl_sme_load_16x1b64 / sl_wait g2scnt, so seeing the
    # intrinsics in llir is what proves the region reached corex hardware
    # instead of a portable CUDA subset.
    def test_sme_builtins_reach_llir(self):
        llir = _compile_sme_load().asm["llir"]
        assert "llvm.bi.sme.load.16x1b64" in llir, llir
        assert "llvm.bi.sl.waitcnt" in llir, llir
        assert "iluvatar_tle" not in llir, llir

    # name="corex" is the native spelling, so the dsl_region it stamps has to
    # carry the corex dialect rather than the "cuda" frontend alias.
    def test_region_is_tagged_corex(self):
        ttir = _compile_sme_load().asm["ttir"]
        assert 'region_dialect = "corex"' in ttir, ttir

    @pytest.mark.parametrize("rows", [SME_TILE_ROWS, 2 * SME_TILE_ROWS])
    def test_sme_load_matches_source(self, rows):
        src, out = _run_sme_load(rows)
        torch.testing.assert_close(out, src, atol=0, rtol=0)


class TestUnsupported:

    # register_pending_source has no way to find the device symbol in the .cu
    # once tracing no longer compiles it, so the name has to be declared.
    def test_deferred_without_extern_func_name_is_rejected(self):

        @dialect(name="cuda", file=RAW_DIR / "smem_accumulate.cu", deferred=True)
        def deferred_edsl(*args, **kwargs):
            ...

        with pytest.raises(RuntimeError, match="extern_func_name"):
            deferred_edsl.register_pending_source()

    def test_library_is_rejected(self):
        with pytest.raises(RuntimeError, match="library"):

            @dialect(name="cuda", file=RAW_DIR / "vector_add.cu", library="nvshmem")
            def nvshmem_edsl(*args, **kwargs):
                ...


class TestRuntime:

    def test_call_vector_add(self):
        device = triton.runtime.driver.active.get_active_torch_device()
        n = 2048
        x = torch.randn(n, device=device)
        y = torch.randn(n, device=device)
        out = torch.empty_like(x)
        _vector_add_kernel[(triton.cdiv(n, 1024), )](x, y, out, n)
        torch.testing.assert_close(out, x + y)

    def test_call_accumulates_register_tensor(self):
        device = triton.runtime.driver.active.get_active_torch_device()
        x = torch.randn((BLOCK, BLOCK), device=device)
        out = torch.empty_like(x)
        _call_kernel[(1, )](x, out, SIZE=BLOCK)
        torch.testing.assert_close(out, x)

    def test_call_smem_accumulates_shared_buffer(self):
        device = triton.runtime.driver.active.get_active_torch_device()
        x = torch.randn((BLOCK, BLOCK), device=device)
        out = torch.empty_like(x)
        _call_smem_kernel[(1, )](x, out, SIZE=BLOCK)
        torch.testing.assert_close(out, x)
