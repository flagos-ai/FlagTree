import pytest
import torch
import triton
import triton.language as tl
import triton.experimental.tle.language as tle
from triton.compiler.errors import CompilationError

from test_tle_utils import compile_musa, require_mthreads_libtriton

require_mthreads_libtriton()


@triton.jit
def _local_ptr_subview_kernel(out_ptr, BLOCK: tl.constexpr):
    init = tl.arange(0, 64).to(tl.float32) + 1.0
    smem = tle.gpu.alloc((64, ), dtype=tl.float32, init_value=init, nv_mma_shared_layout=False)
    offsets = tl.arange(0, BLOCK) * 2
    ptrs = tle.gpu.local_ptr(smem, (offsets, ))
    values = tl.load(ptrs)
    tl.store(out_ptr + tl.arange(0, BLOCK), values)


@triton.jit
def _local_ptr_scalar_kernel(out_ptr):
    init = tl.full((16, ), 0.0, tl.float32)
    smem = tle.gpu.alloc((16, ), dtype=tl.float32, init_value=init, nv_mma_shared_layout=False)
    ptr = tle.gpu.local_ptr(smem, (5, ))
    tl.store(ptr, 42.0)
    value = tl.load(ptr)
    tl.store(out_ptr, value)


@triton.jit
def _local_ptr_full_view_kernel(out_ptr):
    smem = tle.gpu.alloc((16, ), dtype=tl.float32, nv_mma_shared_layout=False)
    values = tl.arange(0, 16).to(tl.float32) + 7.0
    ptrs = tle.gpu.local_ptr(smem)
    tl.store(ptrs, values)
    loaded = tl.load(ptrs)
    tl.store(out_ptr + tl.arange(0, 16), loaded)


@triton.jit
def _local_ptr_non_integer_index_kernel(out_ptr):
    smem = tle.gpu.alloc((16, ), dtype=tl.float32, nv_mma_shared_layout=False)
    idx = tl.arange(0, 16).to(tl.float32)
    ptrs = tle.gpu.local_ptr(smem, (idx, ))
    values = tl.load(ptrs)
    tl.store(out_ptr + tl.arange(0, 16), values)


@triton.jit
def _local_ptr_mixed_scalar_tensor_index_kernel(out_ptr):
    smem = tle.gpu.alloc((4, 4), dtype=tl.float32, nv_mma_shared_layout=False)
    cols = tl.arange(0, 4)
    ptrs = tle.gpu.local_ptr(smem, (0, cols))
    values = tl.load(ptrs)
    tl.store(out_ptr + cols, values)


@triton.jit
def _local_ptr_wrong_rank_index_kernel(out_ptr):
    smem = tle.gpu.alloc((4, 4), dtype=tl.float32, nv_mma_shared_layout=False)
    rows = tl.arange(0, 4)
    ptrs = tle.gpu.local_ptr(smem, (rows, ))
    values = tl.load(ptrs)
    tl.store(out_ptr + rows, values)


@triton.jit
def _local_ptr_tmem_kernel(out_ptr):
    smem = tle.gpu.alloc((16, 16), dtype=tl.float32, scope=tle.gpu.tmem)
    idx = tl.arange(0, 16)
    ptrs = tle.gpu.local_ptr(smem, (idx, idx))
    values = tl.load(ptrs)
    tl.store(out_ptr + idx, values)


def test_tle_local_ptr_subview_lowers_through_mthreads_llvm():
    compiled = compile_musa(
        _local_ptr_subview_kernel,
        signature={"out_ptr": "*fp32", "BLOCK": "constexpr"},
        constexprs={"BLOCK": 16},
    )

    ttgir = compiled.asm["ttgir"]
    llir = compiled.asm["llir"]
    assert "musa_tle.local_pointers" in ttgir, ttgir
    assert "tensor<16x!tt.ptr<f32, 3>" in ttgir, ttgir
    assert "musa_tle.local_pointers" not in llir, llir


def test_tle_local_ptr_scalar_lowers_through_mthreads_llvm():
    compiled = compile_musa(_local_ptr_scalar_kernel, signature={"out_ptr": "*fp32"})

    ttgir = compiled.asm["ttgir"]
    llir = compiled.asm["llir"]
    assert "musa_tle.local_pointers" in ttgir, ttgir
    assert "-> !tt.ptr<f32, 3>" in ttgir, ttgir
    assert "musa_tle.local_pointers" not in llir, llir


def test_tle_local_ptr_full_view_store_load_rewrites_to_memdesc_ops():
    compiled = compile_musa(_local_ptr_full_view_kernel, signature={"out_ptr": "*fp32"})

    ttgir = compiled.asm["ttgir"]
    llir = compiled.asm["llir"]
    assert "ttg.local_store" in ttgir, ttgir
    assert "ttg.local_load" in ttgir, ttgir
    assert "musa_tle.local_pointers" not in llir, llir


def test_tle_local_ptr_rejects_non_integer_indices():
    with pytest.raises(CompilationError, match="local_ptr indices must use integer dtypes"):
        compile_musa(_local_ptr_non_integer_index_kernel, signature={"out_ptr": "*fp32"})


def test_tle_local_ptr_rejects_mixed_scalar_tensor_indices():
    with pytest.raises(CompilationError, match="local_ptr indices must be either all scalar or all tensors"):
        compile_musa(_local_ptr_mixed_scalar_tensor_index_kernel, signature={"out_ptr": "*fp32"})


def test_tle_local_ptr_rejects_wrong_index_rank():
    with pytest.raises(CompilationError, match="local_ptr indices must provide 2 tensors, got 1"):
        compile_musa(_local_ptr_wrong_rank_index_kernel, signature={"out_ptr": "*fp32"})


def test_tle_local_ptr_unsupported_storage_keeps_mthreads_error():
    with pytest.raises(CompilationError, match="mthreads TLE alloc does not support tmem storage"):
        compile_musa(_local_ptr_tmem_kernel, signature={"out_ptr": "*fp32"})


@pytest.mark.skipif(not torch.musa.is_available(), reason="MUSA device is not available")
def test_tle_local_ptr_subview_runtime_loads_shared_values():
    block = 16
    out = torch.empty((block, ), device="musa", dtype=torch.float32)

    _local_ptr_subview_kernel[(1, )](out, BLOCK=block, num_warps=1)

    ref = torch.arange(0, block * 2, 2, dtype=torch.float32) + 1.0
    torch.testing.assert_close(out.cpu(), ref, rtol=0, atol=0)


@pytest.mark.skipif(not torch.musa.is_available(), reason="MUSA device is not available")
def test_tle_local_ptr_scalar_runtime_store_load():
    out = torch.empty((1, ), device="musa", dtype=torch.float32)

    _local_ptr_scalar_kernel[(1, )](out, num_warps=1)

    torch.testing.assert_close(out.cpu(), torch.tensor([42.0], dtype=torch.float32), rtol=0, atol=0)


@pytest.mark.skipif(not torch.musa.is_available(), reason="MUSA device is not available")
def test_tle_local_ptr_full_view_runtime_round_trip():
    out = torch.empty((16, ), device="musa", dtype=torch.float32)

    _local_ptr_full_view_kernel[(1, )](out, num_warps=1)

    ref = torch.arange(0, 16, dtype=torch.float32) + 7.0
    torch.testing.assert_close(out.cpu(), ref, rtol=0, atol=0)
