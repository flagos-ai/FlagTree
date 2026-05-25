import pytest
import torch
import triton
import triton.language as tl
import triton.experimental.tle.language as tle
from triton.tools.tensor_descriptor import TensorDescriptor

from test_tle_utils import compile_musa, require_mthreads_libtriton

require_mthreads_libtriton()


@triton.jit
def _normal_copy_roundtrip_kernel(src, dst, BLOCK: tl.constexpr):
    offsets = tl.arange(0, BLOCK)
    smem = tle.gpu.alloc((BLOCK, ), dtype=tl.float32, nv_mma_shared_layout=False)
    tle.gpu.copy(src + offsets, smem, (BLOCK, ))
    tle.gpu.copy(smem, dst + offsets, (BLOCK, ))


@triton.jit
def _normal_copy_shape_mismatch_kernel(src, dst, BLOCK: tl.constexpr):
    offsets = tl.arange(0, BLOCK)
    smem = tle.gpu.alloc((BLOCK, ), dtype=tl.float32, nv_mma_shared_layout=False)
    tle.gpu.copy(src + offsets, smem, (BLOCK // 2, ))
    tle.gpu.copy(smem, dst + offsets, (BLOCK, ))


@triton.jit
def _tma_copy_desc_to_smem_kernel(desc, dst, BLOCK: tl.constexpr):
    offsets = tl.arange(0, BLOCK)
    smem = tle.gpu.alloc((BLOCK, ), dtype=tl.float16, nv_mma_shared_layout=False)
    tle.gpu.copy(desc, smem, (BLOCK, ), (0, ))
    values = tl.load(tle.gpu.local_ptr(smem))
    tl.store(dst + offsets, values)


@triton.jit
def _tma_copy_smem_to_desc_kernel(desc, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):
    rows = tl.arange(0, BLOCK_M)[:, None]
    cols = tl.arange(0, BLOCK_N)[None, :]
    values = (rows * 10 + cols).to(tl.float16)
    smem = tle.gpu.alloc((BLOCK_M, BLOCK_N), dtype=tl.float16, nv_mma_shared_layout=False)
    tl.store(tle.gpu.local_ptr(smem), values)
    tle.gpu.copy(smem, desc, (BLOCK_M, BLOCK_N), (0, 0))


@triton.jit
def _tma_copy_missing_offsets_kernel(desc, BLOCK: tl.constexpr):
    smem = tle.gpu.alloc((BLOCK, ), dtype=tl.float16, nv_mma_shared_layout=False)
    tle.gpu.copy(desc, smem, (BLOCK, ))


@triton.jit
def _tma_copy_wrong_offset_rank_kernel(desc, BLOCK: tl.constexpr):
    smem = tle.gpu.alloc((BLOCK, ), dtype=tl.float16, nv_mma_shared_layout=False)
    tle.gpu.copy(desc, smem, (BLOCK, ), (0, 0))


def test_tle_copy_normal_gmem_to_smem_lowers_to_async_copy():
    compiled = compile_musa(
        _normal_copy_roundtrip_kernel,
        signature={"src": "*fp32", "dst": "*fp32", "BLOCK": "constexpr"},
        constexprs={"BLOCK": 64},
    )

    ttgir = compiled.asm["ttgir"]
    assert "ttg.async_copy_global_to_local" in ttgir, ttgir
    assert "musa_tle.local_ptr_async_store" in ttgir, ttgir
    assert "ttg.local_load" in ttgir, ttgir
    assert "musa_tle.local_pointers" not in compiled.asm["llir"], compiled.asm["llir"]


def test_tle_copy_normal_rejects_shape_mismatch():
    from triton.compiler.errors import CompilationError

    with pytest.raises(CompilationError, match="copy shape .* must match"):
        compile_musa(
            _normal_copy_shape_mismatch_kernel,
            signature={"src": "*fp32", "dst": "*fp32", "BLOCK": "constexpr"},
            constexprs={"BLOCK": 64},
        )


@pytest.mark.skipif(not torch.musa.is_available(), reason="MUSA device is not available")
def test_tle_copy_normal_roundtrip_runtime():
    block = 64
    src = torch.arange(0, block, device="musa", dtype=torch.float32)
    dst = torch.empty((block, ), device="musa", dtype=torch.float32)

    _normal_copy_roundtrip_kernel[(1, )](src, dst, BLOCK=block, num_warps=1)

    torch.testing.assert_close(dst.cpu(), src.cpu(), rtol=0, atol=0)


@pytest.mark.skipif(not torch.musa.is_available(), reason="MUSA device is not available")
def test_tle_copy_tma_desc_to_smem_lowers_to_musa_tme():
    block = 128
    src = torch.arange(0, block, device="musa", dtype=torch.float16)
    dst = torch.empty((block, ), device="musa", dtype=torch.float16)
    desc = TensorDescriptor.from_tensor(src, [block])

    compiled = _tma_copy_desc_to_smem_kernel.warmup(desc, dst, BLOCK=block, grid=(1, ), num_warps=4)
    ttgir = compiled.asm["ttgir"]
    llir = compiled.asm["llir"]

    assert "ttg.tma_copy" not in ttgir, ttgir
    assert "ttmg.async_tme_copy_global_to_local" in ttgir, ttgir
    assert "ttmg.wait_barrier" in ttgir, ttgir
    assert "llvm.musa.tme.ld.tile.1d" in llir, llir


@pytest.mark.skipif(not torch.musa.is_available(), reason="MUSA device is not available")
def test_tle_copy_tma_rejects_missing_offsets():
    from triton.compiler.errors import CompilationError

    block = 128
    src = torch.empty((block, ), device="musa", dtype=torch.float16)
    desc = TensorDescriptor.from_tensor(src, [block])

    with pytest.raises(CompilationError, match="descriptor-based tle.gpu.copy requires offsets"):
        _tma_copy_missing_offsets_kernel.warmup(desc, BLOCK=block, grid=(1, ), num_warps=4)


@pytest.mark.skipif(not torch.musa.is_available(), reason="MUSA device is not available")
def test_tle_copy_tma_rejects_wrong_offset_rank():
    from triton.compiler.errors import CompilationError

    block = 128
    src = torch.empty((block, ), device="musa", dtype=torch.float16)
    desc = TensorDescriptor.from_tensor(src, [block])

    with pytest.raises(CompilationError, match="offsets must provide 1 values, got 2"):
        _tma_copy_wrong_offset_rank_kernel.warmup(desc, BLOCK=block, grid=(1, ), num_warps=4)


@pytest.mark.skipif(not torch.musa.is_available(), reason="MUSA device is not available")
def test_tle_copy_tma_desc_to_smem_runtime():
    block = 128
    src = torch.arange(0, block, device="musa", dtype=torch.float16)
    dst = torch.empty((block, ), device="musa", dtype=torch.float16)
    desc = TensorDescriptor.from_tensor(src, [block])

    _tma_copy_desc_to_smem_kernel[(1, )](desc, dst, BLOCK=block, num_warps=4)

    torch.testing.assert_close(dst.cpu(), src.cpu(), rtol=0, atol=0)


@pytest.mark.skipif(not torch.musa.is_available(), reason="MUSA device is not available")
def test_tle_copy_tma_smem_to_desc_lowers_to_musa_tme():
    block_m = 16
    block_n = 32
    dst = torch.empty((block_m, block_n), device="musa", dtype=torch.float16)
    desc = TensorDescriptor.from_tensor(dst, [block_m, block_n])

    compiled = _tma_copy_smem_to_desc_kernel.warmup(
        desc,
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        grid=(1, ),
        num_warps=4,
    )
    ttgir = compiled.asm["ttgir"]
    llir = compiled.asm["llir"]

    assert "ttg.tma_copy" not in ttgir, ttgir
    assert "ttmg.async_tme_copy_local_to_global" in ttgir, ttgir
    assert "ttmg.tme_store_commit" in ttgir, ttgir
    assert "llvm.musa.tme.st.2d" in llir, llir


@pytest.mark.skipif(not torch.musa.is_available(), reason="MUSA device is not available")
def test_tle_copy_tma_smem_to_desc_runtime():
    block_m = 16
    block_n = 32
    dst = torch.empty((block_m, block_n), device="musa", dtype=torch.float16)
    desc = TensorDescriptor.from_tensor(dst, [block_m, block_n])

    _tma_copy_smem_to_desc_kernel[(1, )](desc, BLOCK_M=block_m, BLOCK_N=block_n, num_warps=4)

    rows = torch.arange(0, block_m, dtype=torch.float16)[:, None]
    cols = torch.arange(0, block_n, dtype=torch.float16)[None, :]
    ref = rows * 10 + cols
    torch.testing.assert_close(dst.cpu(), ref, rtol=0, atol=0)
