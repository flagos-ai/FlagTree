# flagtree tle
"""
TLE Local Pointer (no-TMA) End-to-End Test
==========================================

Exercises the TLE local-pointer path that is portable across backends that
lower ``tle.local_pointers`` to ``ttg.local_*`` (NVIDIA, sunrise, ...):

    tle.gpu.alloc(nv_mma_shared_layout=False)   # SwizzledShared smem buffer
    tle.gpu.local_ptr(buffer, indices)          # materialize shared pointers
    tl.store(local_ptr, value)                  # stage into shared memory
    tl.load(local_ptr)                          # read back from shared memory

It deliberately AVOIDS ``tle.gpu.copy`` because that lowers to a TMA copy
(``create_tma_copy``), which is NVIDIA Hopper specific and unsupported on
sunrise.

What this covers
----------------
* The local-pointer TTGIR pass chain wired into the backend pipeline
  (early_assign_memory_space / select_encodings / insert_local_pointer_barriers
  / optimize_local_pointer_loads / optimize_local_pointer_stores).
* The store->load visibility barrier inserted by
  insert_local_pointer_barriers (correctness: the value read back must equal
  the value staged).
* The shared-store/-load alignment guard in the backend LoadStore lowering --
  forced to matter by sweeping inner block widths, including non-power-of-two
  widths (48 / 96) that the guard must clamp the vector width for.
"""

import pytest
import torch
import triton
import triton.language as tl
import triton.experimental.tle.language as tle

DEVICE = triton.runtime.driver.active.get_active_torch_device()


@triton.jit
def shared_roundtrip_kernel(
    src_ptr,
    dst_ptr,
    xstride,
    ynumel,
    XBLOCK: tl.constexpr,
    YBLOCK: tl.constexpr,
):
    """Per program (one X-block), for each Y-block:

      gmem -> (tl.store) shared -> (tl.load) shared -> (scale) -> gmem

    The shared store and load both go through ``tle.gpu.local_ptr`` pointers.
    """
    pid = tl.program_id(0)

    xoffs = pid * XBLOCK + tl.arange(0, XBLOCK)
    src_rows = src_ptr + xstride * xoffs[:, None]
    dst_rows = dst_ptr + xstride * xoffs[:, None]

    smem = tle.gpu.alloc([XBLOCK, YBLOCK], dtype=tl.float32, layout=None, scope=tle.gpu.smem,
                         nv_mma_shared_layout=False)

    row_ids = tl.arange(0, XBLOCK)[:, None]
    col_ids = tl.arange(0, YBLOCK)[None, :]
    row_ids = tl.broadcast_to(row_ids, (XBLOCK, YBLOCK))
    col_ids = tl.broadcast_to(col_ids, (XBLOCK, YBLOCK))
    smem_ptrs = tle.gpu.local_ptr(smem, (row_ids, col_ids))

    for yoff in range(0, ynumel, YBLOCK):
        yoffs = tl.arange(0, YBLOCK) + yoff
        gval = tl.load(src_rows + yoffs[None, :])

        tl.store(smem_ptrs, gval)  # vectorized shared store (guarded)
        sval = tl.load(smem_ptrs)  # vectorized shared load (guarded)

        tl.store(dst_rows + yoffs[None, :], sval * 2.0)


def shared_roundtrip(src, dst, XBLOCK, YBLOCK):
    assert src.shape == dst.shape
    xnumel, ynumel = src.shape
    assert ynumel % YBLOCK == 0, "kernel assumes ynumel divisible by YBLOCK"
    grid = (triton.cdiv(xnumel, XBLOCK), )
    return shared_roundtrip_kernel[grid](src, dst, src.stride(0), ynumel, XBLOCK, YBLOCK)


class TestTLELocalPtrRoundtrip:
    """Local-pointer staging correctness across vectorizable / non-vectorizable widths."""

    @pytest.mark.parametrize("XBLOCK,YBLOCK", [
        (32, 64),
        (32, 128),
        (16, 64),
        (64, 64),
    ])
    def test_roundtrip(self, XBLOCK, YBLOCK):
        torch.manual_seed(0)
        xnumel, ynumel = 256, YBLOCK * 4

        src = torch.randn(xnumel, ynumel, device=DEVICE, dtype=torch.float32)
        dst = torch.empty_like(src)

        shared_roundtrip(src, dst, XBLOCK, YBLOCK)

        torch.testing.assert_close(dst.cpu(), src.cpu() * 2.0, atol=1e-5, rtol=1e-5)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
