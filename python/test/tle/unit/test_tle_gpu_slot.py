# flagtree tle
"""
Unit tests for TLE buffered_tensor.slot(stage).
"""

import pytest
import torch
import triton
import triton.language as tl
import triton.experimental.tle.language as tle


def _is_enflame_backend():
    target = triton.runtime.driver.active.get_current_target()
    return target.backend == "gcu"


def _require_cuda():
    try:
        if _is_enflame_backend():
            pass
        else:
            torch.cuda.init()
    except Exception as exc:
        pytest.skip(f"CUDA init failed: {exc}")


@pytest.fixture(scope="module", autouse=True)
def _cuda_guard():
    _require_cuda()


@triton.jit
def _slot_local_ptr_store_kernel(out_ptr, BLOCK: tl.constexpr):
    idx = tl.arange(0, BLOCK)
    smem = tle.gpu.alloc([2, BLOCK], dtype=tl.int32, layout=None, scope=tle.gpu.smem, nv_mma_shared_layout=False)
    slot = smem.slot(0)
    ptrs = tle.gpu.local_ptr(slot, (idx, ))
    tl.store(ptrs, idx + 7)
    vals = tl.load(ptrs)
    tl.store(out_ptr + idx, vals)


def test_buffered_tensor_slot_lowers_to_memdesc_index_and_executes():
    block = 64
    out = torch.empty((block, ), device="cuda", dtype=torch.int32)

    compiled = _slot_local_ptr_store_kernel.warmup(out, BLOCK=block, grid=(1, ), num_warps=4)
    ttgir = compiled.asm["ttgir"]
    assert "ttg.memdesc_index" in ttgir
    assert "!ttg.memdesc<2x64xi32" in ttgir
    assert "!ttg.memdesc<64xi32" in ttgir

    _slot_local_ptr_store_kernel[(1, )](out, BLOCK=block, num_warps=4)
    expected = torch.arange(0, block, device="cuda", dtype=torch.int32) + 7
    torch.testing.assert_close(out, expected, atol=0, rtol=0)
