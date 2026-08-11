import pytest
import torch
import triton
import triton.language as tl
import triton.experimental.tle.language as tle

from utils import compile_iluvatar

DTYPE_CASES = [
    pytest.param("fp32", tl.float32, torch.float32, id="fp32"),
    pytest.param("fp16", tl.float16, torch.float16, id="fp16"),
    pytest.param("bf16", tl.bfloat16, torch.bfloat16, id="bf16"),
]

BLOCK = 64


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
def _direct_load_kernel(x_ptr, out_ptr, BLOCK: tl.constexpr):
    offs = tl.arange(0, BLOCK)
    vals = tl.load(x_ptr + offs)
    tl.store(out_ptr + offs, vals)


@pytest.mark.parametrize("dtype_str,tl_dtype,torch_dtype", DTYPE_CASES)
def test_tle_memory_space_load_uses_shared_async_copy(dtype_str, tl_dtype, torch_dtype):
    compiled = compile_iluvatar(
        _memory_space_load_kernel,
        signature={"x_ptr": f"*{dtype_str}", "out_ptr": f"*{dtype_str}", "BLOCK": "constexpr"},
        constexprs={"BLOCK": BLOCK},
    )

    ttgir = compiled.asm["ttgir"]
    assert "tt.memory_space" not in ttgir
    assert "ttg.async_copy_global_to_local" in ttgir, ttgir
    assert "ttg.local_alloc" in ttgir, ttgir
    assert "ttg.local_load" in ttgir, ttgir


def test_tle_memory_space_non_load_uses_initialized_shared_alloc():
    compiled = compile_iluvatar(
        _memory_space_non_load_kernel,
        signature={"out_ptr": "*fp32", "BLOCK": "constexpr"},
        constexprs={"BLOCK": BLOCK},
    )

    ttgir = compiled.asm["ttgir"]
    assert "tt.memory_space" not in ttgir
    # A non-load producer is staged through an initialized shared alloc, never
    # the async copy path.
    assert "ttg.async_copy_global_to_local" not in ttgir
    assert "ttg.local_alloc" in ttgir, ttgir
    assert "ttg.local_load" in ttgir, ttgir


@pytest.mark.parametrize("dtype_str,tl_dtype,torch_dtype", DTYPE_CASES)
def test_tle_memory_space_load_runtime(device, dtype_str, tl_dtype, torch_dtype):
    x = torch.arange(BLOCK, device=device, dtype=torch_dtype)
    out = torch.empty_like(x)
    ref = torch.empty_like(x)

    grid = (1, )
    _memory_space_load_kernel[grid](x, out, BLOCK=BLOCK)
    _direct_load_kernel[grid](x, ref, BLOCK=BLOCK)

    torch.testing.assert_close(out.cpu(), ref.cpu(), rtol=0, atol=0)
    torch.testing.assert_close(out.cpu(), x.cpu(), rtol=0, atol=0)


def test_tle_memory_space_non_load_runtime(device):
    out = torch.empty((BLOCK, ), device=device, dtype=torch.float32)

    _memory_space_non_load_kernel[(1, )](out, BLOCK=BLOCK)

    ref = torch.arange(BLOCK, dtype=torch.float32) + 3.0
    torch.testing.assert_close(out.cpu(), ref, rtol=0, atol=0)
