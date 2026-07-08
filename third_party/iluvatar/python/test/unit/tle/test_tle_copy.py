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


def _make_tle_copy_roundtrip_kernel(tl_dtype):
    """Instantiate a dtype-specific copy kernel (tle.gpu.alloc requires tl.dtype, not tl.constexpr)."""

    @triton.jit
    def _tle_copy_roundtrip_kernel(src, dst, BLOCK: tl.constexpr):
        offsets = tl.arange(0, BLOCK)
        smem = tle.gpu.alloc((BLOCK, ), dtype=tl_dtype, nv_mma_shared_layout=False)
        tle.gpu.copy(src + offsets, smem, (BLOCK, ))
        tle.gpu.copy(smem, dst + offsets, (BLOCK, ))

    return _tle_copy_roundtrip_kernel


TLE_COPY_KERNELS = {
    dtype_str: _make_tle_copy_roundtrip_kernel(tl_dtype)
    for dtype_str, tl_dtype, _ in (case.values for case in DTYPE_CASES)
}


@triton.jit
def _direct_copy_kernel(src, dst, BLOCK: tl.constexpr):
    offsets = tl.arange(0, BLOCK)
    vals = tl.load(src + offsets)
    tl.store(dst + offsets, vals)


@pytest.mark.parametrize("dtype_str,tl_dtype,torch_dtype", DTYPE_CASES)
def test_tle_copy_async_codegen(dtype_str, tl_dtype, torch_dtype):
    kernel = TLE_COPY_KERNELS[dtype_str]
    compiled = compile_iluvatar(
        kernel,
        signature={"src": f"*{dtype_str}", "dst": f"*{dtype_str}", "BLOCK": "constexpr"},
        constexprs={"BLOCK": 64},
    )

    ttgir = compiled.asm["ttgir"]
    # The GM->local staging (global tt.load + local_pointers tt.store) must be
    # fused into an async copy chain, and the marker attribute must be present.
    assert "ttg.async_copy_global_to_local" in ttgir, ttgir
    assert "iluvatar_tle.local_ptr_async_store" in ttgir, ttgir
    # The local->GM leg still reads shared memory back via local_load.
    assert "ttg.local_load" in ttgir, ttgir
    # The pointer op must not survive to LLVM.
    assert "iluvatar_tle.local_pointers" not in compiled.asm["llir"], compiled.asm["llir"]


@pytest.mark.parametrize("dtype_str,tl_dtype,torch_dtype", DTYPE_CASES)
def test_tle_copy_roundtrip(device, dtype_str, tl_dtype, torch_dtype):
    block = 64
    torch.manual_seed(0)
    src = torch.randn(block, device=device, dtype=torch_dtype)
    dst_tle = torch.empty_like(src)
    dst_ref = torch.empty_like(src)

    grid = (1, )
    TLE_COPY_KERNELS[dtype_str][grid](src, dst_tle, BLOCK=block)
    _direct_copy_kernel[grid](src, dst_ref, BLOCK=block)

    torch.testing.assert_close(dst_tle.cpu(), dst_ref.cpu(), atol=0, rtol=0)
    torch.testing.assert_close(dst_tle.cpu(), src.cpu(), atol=0, rtol=0)
