# flagtree tle
"""
TLE tle.gpu.copy (normcopy / no-TMA) end-to-end test (sunrise)
=============================================================

tle.gpu.copy has two paths, chosen by operand type:

  * normcopy   -- src/dst are tl.tensor <-> tle.buffered_tensor (plain pointers).
                  Lowers to tl.load + local_ptr + tl.store. NO TMA. Portable.
  * tmacopy    -- src/dst involve tl.tensor_descriptor. Lowers to create_tma_copy
                  (NVIDIA Hopper TMA). NOT supported on sunrise.

This test exercises ONLY the normcopy path (plain pointer tensors), in both
directions:

  * GM_TO_LOCAL : copy(global_ptrs, smem_buffer, shape)
  * LOCAL_TO_GM : copy(smem_buffer, global_ptrs, shape)

It relies on the local_ptr support wired into the sunrise backend. Because
normcopy is just load/store through local pointers, no TMA is involved and it
should run on sunrise.
"""

import torch
import triton
import triton.language as tl
import triton.experimental.tle.language as tle
import pytest

DEVICE = triton.runtime.driver.active.get_active_torch_device()


@triton.jit
def copy_roundtrip_kernel(
    a_ptr,
    c_ptr,
    xnumel,
    ynumel,
    xstride,
    ystride,
    XBLOCK: tl.constexpr,
    YBLOCK: tl.constexpr,
):
    """For each X-block, for each Y-block:

        gmem --copy(GM_TO_LOCAL)--> smem --(scale via local_ptr load/store)-->
        smem --copy(LOCAL_TO_GM)--> gmem

    Exercises both copy directions plus local_ptr load/store on the staged tile.
    """
    pid = tl.program_id(0)
    xoffs = pid * XBLOCK + tl.arange(0, XBLOCK)
    a_rows = a_ptr + xstride * xoffs[:, None]
    c_rows = c_ptr + xstride * xoffs[:, None]

    smem = tle.gpu.alloc([XBLOCK, YBLOCK], dtype=tl.float32, layout=None, scope=tle.gpu.smem,
                         nv_mma_shared_layout=False)

    row_ids = tl.arange(0, XBLOCK)[:, None]
    col_ids = tl.arange(0, YBLOCK)[None, :]
    row_ids = tl.broadcast_to(row_ids, (XBLOCK, YBLOCK))
    col_ids = tl.broadcast_to(col_ids, (XBLOCK, YBLOCK))
    smem_ptrs = tle.gpu.local_ptr(smem, (row_ids, col_ids))

    for yoff in range(0, ynumel, YBLOCK):
        yoffs = tl.arange(0, YBLOCK) + yoff

        # GM_TO_LOCAL: plain pointer tensor src -> smem buffer dst (normcopy)
        tle.gpu.copy(a_rows + ystride * yoffs[None, :], smem, [XBLOCK, YBLOCK])

        # scale in shared memory through local pointers
        val = tl.load(smem_ptrs)
        tl.store(smem_ptrs, val * 2.0)

        # LOCAL_TO_GM: smem buffer src -> plain pointer tensor dst (normcopy)
        tle.gpu.copy(smem, c_rows + ystride * yoffs[None, :], [XBLOCK, YBLOCK])


def copy_roundtrip(a, c, XBLOCK, YBLOCK):
    assert a.shape == c.shape
    xnumel, ynumel = a.shape
    assert ynumel % YBLOCK == 0
    grid = (triton.cdiv(xnumel, XBLOCK), )
    return copy_roundtrip_kernel[grid](a, c, xnumel, ynumel, a.stride(0), a.stride(1), XBLOCK, YBLOCK)


@pytest.mark.parametrize("XBLOCK,YBLOCK", [(32, 64), (64, 64), (32, 128)])
def test_copy_roundtrip(XBLOCK, YBLOCK):
    torch.manual_seed(0)
    xnumel, ynumel = 256, YBLOCK * 4

    a = torch.randn(xnumel, ynumel, device=DEVICE, dtype=torch.float32)
    c = torch.empty_like(a)

    copy_roundtrip(a, c, XBLOCK, YBLOCK)

    torch.testing.assert_close(c.cpu(), (a * 2.0).cpu(), atol=1e-5, rtol=1e-5)


@triton.jit
def copy_gm_to_local_kernel(
    a_ptr,
    c_ptr,
    ynumel,
    xstride,
    ystride,
    XBLOCK: tl.constexpr,
    YBLOCK: tl.constexpr,
):
    """Pure GM_TO_LOCAL copy then read back via local_ptr and store to gmem."""
    pid = tl.program_id(0)
    xoffs = pid * XBLOCK + tl.arange(0, XBLOCK)
    a_rows = a_ptr + xstride * xoffs[:, None]
    c_rows = c_ptr + xstride * xoffs[:, None]

    smem = tle.gpu.alloc([XBLOCK, YBLOCK], dtype=tl.float32, layout=None, scope=tle.gpu.smem,
                         nv_mma_shared_layout=False)
    row_ids = tl.broadcast_to(tl.arange(0, XBLOCK)[:, None], (XBLOCK, YBLOCK))
    col_ids = tl.broadcast_to(tl.arange(0, YBLOCK)[None, :], (XBLOCK, YBLOCK))
    smem_ptrs = tle.gpu.local_ptr(smem, (row_ids, col_ids))

    for yoff in range(0, ynumel, YBLOCK):
        yoffs = tl.arange(0, YBLOCK) + yoff
        tle.gpu.copy(a_rows + ystride * yoffs[None, :], smem, [XBLOCK, YBLOCK])
        val = tl.load(smem_ptrs)
        tl.store(c_rows + ystride * yoffs[None, :], val)


def test_copy_gm_to_local_identity():
    torch.manual_seed(1)
    XBLOCK, YBLOCK = 32, 64
    xnumel, ynumel = 256, YBLOCK * 4

    a = torch.randn(xnumel, ynumel, device=DEVICE, dtype=torch.float32)
    c = torch.empty_like(a)

    grid = (triton.cdiv(xnumel, XBLOCK), )
    copy_gm_to_local_kernel[grid](a, c, ynumel, a.stride(0), a.stride(1), XBLOCK, YBLOCK)

    torch.testing.assert_close(c.cpu(), a.cpu(), atol=0, rtol=0)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
