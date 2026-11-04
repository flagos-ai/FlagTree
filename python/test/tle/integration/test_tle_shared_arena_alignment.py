# flagtree tle
"""Propagate a byte arena's explicit alignment into the normal allocator."""

import pytest
import torch
import triton
import triton.language as tl
import triton.experimental.tle.language as tle


@triton.jit
def _aligned_arena(output, ALIGNMENT: tl.constexpr):
    arena = tle.gpu.alloc([4096], tl.uint8, layout=None, scope=tle.gpu.smem,
                          nv_mma_shared_layout=False, alignment_bytes=ALIGNMENT)
    view = tle.gpu.alloc([128], tl.int32, layout=None, scope=tle.gpu.smem,
                         alias=arena, alias_offset_bytes=1024, nv_mma_shared_layout=False)
    i = tl.arange(0, 128)
    pointers = tle.gpu.local_ptr(view, (i,), (128,))
    tl.store(pointers, i * 3)
    tl.debug_barrier()
    tl.store(output + i, tl.load(pointers))


@pytest.mark.parametrize("alignment", (1024, 2048, 4096))
def test_shared_arena_allocation_has_explicit_alignment(alignment):
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required")
    out = torch.empty((128,), device="cuda", dtype=torch.int32)
    compiled = _aligned_arena[(1,)](out, alignment, num_warps=4)
    assert f"alignment = {alignment}" in compiled.asm["ttir"]
    torch.testing.assert_close(out, torch.arange(128, device="cuda", dtype=torch.int32) * 3, rtol=0, atol=0)
