import pytest
import torch
import triton
import triton.language as tl
import triton.experimental.tle.language as tle

from test_tle_utils import compile_musa, require_mthreads_libtriton

require_mthreads_libtriton()


@triton.jit
def _memory_space_load_kernel(x_ptr, out_ptr, BLOCK: tl.constexpr):
    offs = tl.arange(0, BLOCK)
    vals = tle.load(x_ptr + offs)
    vals = tle.gpu.memory_space(vals, "shared_memory")
    tl.store(out_ptr + offs, vals)


@triton.jit
def _memory_space_non_load_kernel(out_ptr, BLOCK: tl.constexpr):
    offs = tl.arange(0, BLOCK)
    vals = offs.to(tl.float32) + 3.0
    vals = tle.gpu.memory_space(vals, "shared_memory")
    tl.store(out_ptr + offs, vals)


@triton.jit
def _memory_space_unsupported_kernel(out_ptr, BLOCK: tl.constexpr):
    offs = tl.arange(0, BLOCK)
    vals = offs.to(tl.float32)
    vals = tle.gpu.memory_space(vals, "tensor_memory")
    tl.store(out_ptr + offs, vals)


def test_tle_memory_space_load_uses_shared_async_copy():
    compiled = compile_musa(
        _memory_space_load_kernel,
        signature={"x_ptr": "*fp32", "out_ptr": "*fp32", "BLOCK": "constexpr"},
        constexprs={"BLOCK": 64},
    )

    ttgir = compiled.asm["ttgir"]
    assert "tt.memory_space" not in ttgir
    assert "ttg.async_copy_global_to_local" in ttgir, ttgir
    assert "ttg.local_alloc" in ttgir, ttgir
    assert "ttg.local_load" in ttgir, ttgir
    assert "llvm.musa.memcpy.g2s" in compiled.asm["llir"]


def test_tle_memory_space_non_load_uses_initialized_shared_alloc():
    compiled = compile_musa(
        _memory_space_non_load_kernel,
        signature={"out_ptr": "*fp32", "BLOCK": "constexpr"},
        constexprs={"BLOCK": 64},
    )

    ttgir = compiled.asm["ttgir"]
    assert "tt.memory_space" not in ttgir
    assert "ttg.async_copy_global_to_local" not in ttgir
    assert "ttg.local_alloc" in ttgir, ttgir
    assert "ttg.local_load" in ttgir, ttgir


def test_tle_memory_space_rejects_unsupported_space(capfd):
    with pytest.raises(RuntimeError, match="PassManager::run failed"):
        compile_musa(
            _memory_space_unsupported_kernel,
            signature={"out_ptr": "*fp32", "BLOCK": "constexpr"},
            constexprs={"BLOCK": 64},
        )
    assert "unsupported MUSA TLE memory space: tensor_memory" in capfd.readouterr().err


@pytest.mark.skipif(not torch.musa.is_available(), reason="MUSA device is not available")
def test_tle_memory_space_runtime_matches_input():
    block = 64
    x = torch.arange(block, device="musa", dtype=torch.float32)
    out = torch.empty_like(x)

    _memory_space_load_kernel[(1, )](x, out, BLOCK=block, num_warps=1)

    torch.testing.assert_close(out.cpu(), x.cpu(), rtol=0, atol=0)


@pytest.mark.skipif(not torch.musa.is_available(), reason="MUSA device is not available")
def test_tle_memory_space_non_load_runtime():
    block = 64
    out = torch.empty((block, ), device="musa", dtype=torch.float32)

    _memory_space_non_load_kernel[(1, )](out, BLOCK=block, num_warps=1)

    ref = torch.arange(block, dtype=torch.float32) + 3.0
    torch.testing.assert_close(out.cpu(), ref, rtol=0, atol=0)
