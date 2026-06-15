import pytest
import torch
import triton
import triton.language as tl
import triton.experimental.tle.language as tle
from triton.compiler.errors import CompilationError

from test_tle_utils import compile_musa, require_mthreads_libtriton

require_mthreads_libtriton()


@triton.jit(noinline=True)
def _consume_alloc(buf, out_ptr):
    tl.store(out_ptr, 0.0)


@triton.jit
def _alloc_kernel(out_ptr):
    buf = tle.gpu.alloc((16, ), dtype=tl.float32, nv_mma_shared_layout=False)
    _consume_alloc(buf, out_ptr)


@triton.jit
def _alloc_nv_mma_kernel(out_ptr):
    buf = tle.gpu.alloc((16, ), dtype=tl.float32, nv_mma_shared_layout=True)
    _consume_alloc(buf, out_ptr)


@triton.jit
def _alloc_default_kernel(out_ptr):
    buf = tle.gpu.alloc((16, ), dtype=tl.float32)
    _consume_alloc(buf, out_ptr)


@triton.jit
def _alloc_init_kernel(out_ptr):
    init = tl.full((16, ), 1.0, tl.float32)
    buf = tle.gpu.alloc((16, ), dtype=tl.float32, init_value=init, nv_mma_shared_layout=False)
    _consume_alloc(buf, out_ptr)


@triton.jit
def _alloc_tmem_kernel(out_ptr):
    buf = tle.gpu.alloc((16, 16), dtype=tl.float32, scope=tle.gpu.tmem)
    _consume_alloc(buf, out_ptr)


@triton.jit
def _alloc_explicit_layout_kernel(out_ptr, LAYOUT: tl.constexpr):
    buf = tle.gpu.alloc((16, ), dtype=tl.float32, layout=LAYOUT)
    _consume_alloc(buf, out_ptr)


@triton.jit
def _alloc_explicit_swizzled_layout_roundtrip_kernel(src_ptr, out_ptr, LAYOUT: tl.constexpr):
    offs = tl.arange(0, 16)
    values = tl.load(src_ptr + offs)
    buf = tle.gpu.alloc((16, ), dtype=tl.float32, layout=LAYOUT, init_value=values)
    _consume_alloc(buf, out_ptr)
    tl.store(out_ptr + offs, values + 1.0)


def test_tle_alloc_ttgir_emits_smem_memdesc():
    compiled = compile_musa(_alloc_kernel, signature={"out_ptr": "*fp32"})
    ttgir = compiled.asm["ttgir"]

    assert "ttg.local_alloc" in ttgir, ttgir
    assert "!ttg.memdesc<16xf32" in ttgir, ttgir
    assert "#smem" in ttgir, ttgir
    assert "#ttg.swizzled_shared" in ttgir, ttgir
    assert "#ttg.nvmma_shared" not in ttgir, ttgir
    assert "tensor_memory" not in ttgir, ttgir


def test_tle_alloc_nv_mma_shared_layout_true_raises():
    with pytest.raises(CompilationError, match="mthreads TLE alloc does not support nv_mma_shared_layout=True"):
        compile_musa(_alloc_nv_mma_kernel, signature={"out_ptr": "*fp32"})


def test_tle_alloc_default_nv_mma_shared_layout_raises():
    with pytest.raises(CompilationError, match="mthreads TLE alloc does not support nv_mma_shared_layout=True"):
        compile_musa(_alloc_default_kernel, signature={"out_ptr": "*fp32"})


def test_tle_alloc_explicit_swizzled_shared_layout_ttgir_emits_smem_memdesc():
    layout = tle.gpu.swizzled_shared_layout.make_default(rank=1)
    compiled = compile_musa(_alloc_explicit_layout_kernel, signature={"out_ptr": "*fp32", "LAYOUT": "constexpr"},
                            constexprs={"LAYOUT": layout})
    ttgir = compiled.asm["ttgir"]

    assert "ttg.local_alloc" in ttgir, ttgir
    assert "!ttg.memdesc<16xf32" in ttgir, ttgir
    assert "#smem" in ttgir, ttgir
    assert "#ttg.swizzled_shared" in ttgir, ttgir
    assert "#ttg.nvmma_shared" not in ttgir, ttgir
    assert "tensor_memory" not in ttgir, ttgir


def test_tle_alloc_explicit_nv_mma_shared_layout_raises():
    layout = tle.gpu.nv_mma_shared_layout.make_default((16, ), tl.float32)
    with pytest.raises(CompilationError, match="mthreads TLE alloc does not support nv_mma_shared_layout=True"):
        compile_musa(_alloc_explicit_layout_kernel, signature={"out_ptr": "*fp32", "LAYOUT": "constexpr"},
                     constexprs={"LAYOUT": layout})


def test_tle_alloc_with_init_value_ttgir_emits_initialized_alloc():
    compiled = compile_musa(_alloc_init_kernel, signature={"out_ptr": "*fp32"})
    ttgir = compiled.asm["ttgir"]

    assert "ttg.local_alloc %" in ttgir, ttgir
    assert "(tensor<16xf32" in ttgir, ttgir
    assert "!ttg.memdesc<16xf32" in ttgir, ttgir
    assert "tensor_memory" not in ttgir, ttgir


def test_tle_alloc_tmem_scope_raises():
    with pytest.raises(CompilationError, match="mthreads TLE alloc does not support tmem storage"):
        compile_musa(_alloc_tmem_kernel, signature={"out_ptr": "*fp32"})


@pytest.mark.skipif(not torch.musa.is_available(), reason="MUSA device is not available")
def test_tle_alloc_runtime_launch_smoke():
    out = torch.empty((1, ), device="musa", dtype=torch.float32)

    _alloc_kernel[(1, )](out, num_warps=4)

    torch.testing.assert_close(out.cpu(), torch.zeros((1, ), dtype=torch.float32), rtol=0, atol=0)


@pytest.mark.skipif(not torch.musa.is_available(), reason="MUSA device is not available")
def test_tle_alloc_explicit_swizzled_shared_layout_runtime_precision():
    src = torch.arange(16, device="musa", dtype=torch.float32)
    out = torch.empty((16, ), device="musa", dtype=torch.float32)
    layout = tle.gpu.swizzled_shared_layout.make_default(rank=1)

    _alloc_explicit_swizzled_layout_roundtrip_kernel[(1, )](src, out, layout, num_warps=4)

    torch.testing.assert_close(out.cpu(), torch.arange(16, dtype=torch.float32) + 1.0, rtol=0, atol=0)
