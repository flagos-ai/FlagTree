# Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.

# =============================================================================
# MTE (Memory Tag Extension) OOB regression test for DiscreteMaskAccessConversionPass
#
# This file verifies that DiscreteMaskAccessConversionPass correctly bounds
# global-memory accesses when the load/store mask is a combined discrete mask
#
# Test strategy
# -------------
# The test engineers this condition in four steps:
#   Step 1 — probe:    Trigger a fresh 2 MB NPU segment; measure its size.
#   Step 2 — pre_fill: Fill the segment with small tensors until the remaining
#                      free space is in [IN_BYTES, TARGET_FREE].
#   Step 3 — in_tensor: Allocate the test tensor; it lands at the segment tail
#                       with only ~7680 bytes gap to the boundary.
#   Step 4 — kernel:   Run the kernel + synchronize.  Before the fix the
#                      OOB read (24576 bytes) crosses the boundary → MTE.
#                      After the fix the copy is bounded to IN_BYTES → no MTE.
#
# Memory layout at the time of the kernel call (before fix):
#
#   ┌──────────────────────────── 2 MB segment ────────────────────────────────┐
#   │ probe(512 B) │←────── pre_fill (~2025 KB) ──────→│ in_tensor(8192 B) │gap│
#   └──────────────────────────────────────────────────────────────────────────┘
#                                                                              ↑ segment end
#                                              ├──── OOB_BYTES (24576 B) ─────→
#                                                              crosses boundary → MTE ✓
#
# =============================================================================

import math
import torch
import triton
import triton.language as tl
import torch_npu
import pytest


@triton.jit
def cont_disc_oob_inplace_2d_kernel(
    ptr,
    M,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid_m = tl.program_id(0)
    row_offs = tl.arange(0, BLOCK_M)
    col_offs = tl.arange(0, BLOCK_N)

    # Continuous bound (contMask): marks the M valid rows in this tile.
    row_boundary = row_offs < (M - pid_m * BLOCK_M)
    row_disc = (row_offs * 2) < BLOCK_M
    combined = row_boundary[:, None] & row_disc[:, None] & (col_offs < BLOCK_N)[None, :]

    row_start = pid_m * BLOCK_M
    ptr_2d = ptr + (row_start + row_offs[:, None]) * BLOCK_N + col_offs[None, :]

    # load triggers DiscreteMaskAccessConversionPass.
    # Before fix: copy size = BLOCK_M × BLOCK_N × 2 bytes = 32768 bytes (OOB).
    # After fix:  copy size = M      × BLOCK_N × 2 bytes =  8192 bytes (safe).
    data = tl.load(ptr_2d, mask=combined, other=0.0)
    tl.store(ptr_2d, data, mask=row_boundary[:, None])


# =============================================================================
# Memory setup helper
# =============================================================================
def _fill_segment_to_boundary(dtype, device, in_bytes, target_free, chunk_max_bytes):
    """Allocate a fresh NPU segment and fill it so that only ~target_free bytes remain.

    Returns
    -------
    pre_fillers : list of torch.Tensor
        All tensors allocated (probe + fill chunks).  The caller is responsible
        for deleting them in `finally`.
    pool_free_after_fill : int
        Segment free space after filling, in bytes.
    seg_size : int
        Total size of the triggered segment, in bytes.
    """
    elem_size = torch.finfo(dtype).bits // 8

    # --- Step 1: probe — trigger a fresh 2 MB small-alloc segment ----------
    pool0 = torch.npu.memory_reserved(0)
    alloc0 = torch.npu.memory_allocated(0)

    probe = torch.empty(1, dtype=dtype, device=device)

    pool1 = torch.npu.memory_reserved(0)
    alloc1 = torch.npu.memory_allocated(0)

    seg_size = pool1 - pool0  # should be 2 MB = 2097152 bytes
    probe_actual = alloc1 - alloc0  # NPU 512-byte aligned → 512 bytes

    print(f"\n[mte] Step 1: probe")
    print(f"[mte]   segment_size = {seg_size} bytes ({seg_size // 1024} KB)")
    print(f"[mte]   probe_actual = {probe_actual} bytes")
    print(f"[mte]   pool_free    = {seg_size - probe_actual} bytes")

    # --- Step 2: pre_fill — leave only [in_bytes, target_free] bytes free ---
    # Chunks are kept ≤ chunk_max_bytes to stay in the small-alloc pool and
    # avoid opening a new segment via the large-alloc path.
    pre_fillers = [probe]

    for chunk in [
            chunk_max_bytes, chunk_max_bytes // 2, chunk_max_bytes // 4, chunk_max_bytes // 8, 32 * 1024, 16 * 1024,
            8 * 1024, 4 * 1024, 2 * 1024, 1024, 512
    ]:
        while True:
            free = torch.npu.memory_reserved(0) - torch.npu.memory_allocated(0)
            if free <= in_bytes:
                break  # not enough room even for in_tensor; stop
            if free <= target_free:
                break  # already in target range; try smaller chunk
            if free <= target_free + chunk:
                break  # this chunk would overshoot; try smaller chunk
            try:
                t = torch.empty(chunk // elem_size, dtype=dtype, device=device)
                pre_fillers.append(t)
            except RuntimeError:
                break  # segment exhausted; try smaller chunk

    pool_free_after_fill = torch.npu.memory_reserved(0) - torch.npu.memory_allocated(0)
    pre_bytes = sum(t.numel() * elem_size for t in pre_fillers)
    print(f"\n[mte] Step 2: pre_fill")
    print(f"[mte]   tensors = {len(pre_fillers)},  total = {pre_bytes} bytes ({pre_bytes // 1024} KB)")
    print(f"[mte]   pool_free = {pool_free_after_fill} bytes  (target [{in_bytes}, {target_free}] bytes)")

    return pre_fillers, pool_free_after_fill, seg_size


# =============================================================================
# Test: MTE OOB via segment-boundary placement
# =============================================================================
@pytest.mark.parametrize("BLOCK_M,BLOCK_N,M", [
    (4, 4096, 1),
])
def test_mte_segment_boundary_oob(BLOCK_M, BLOCK_N, M):
    """Regression: combined discrete mask load causes OOB on tail blocks.

    Verifies that DiscreteMaskAccessConversionPass correctly bounds
    the memory copy to M rows (the contiguous range), not BLOCK_M rows (the full tile).

    Test outcome:
    - Before fix: RuntimeError (MTE OOB) — the test would fail.
    - After fix:  no exception — the test passes.
    """
    dtype = torch.float16
    device = 'npu'
    elem_size = 2  # float16

    in_bytes = M * BLOCK_N * elem_size  # 8192 bytes
    oob_bytes = (BLOCK_M - M) * BLOCK_N * elem_size  # 24576 bytes
    # TARGET_FREE: midpoint between in_bytes and oob_bytes.
    # Ensures in_tensor fits AND gap < oob_bytes so OOB crosses segment boundary.
    target_free = (in_bytes + oob_bytes) // 2  # 16384 bytes
    chunk_max_bytes = 512 * 1024  # 512 KB

    print(f"\n[mte] BLOCK_M={BLOCK_M}  BLOCK_N={BLOCK_N}  M={M}")
    print(f"[mte] in_bytes    = {in_bytes} bytes  (in_tensor: {M}×{BLOCK_N}×{elem_size})")
    print(f"[mte] oob_bytes   = {oob_bytes} bytes  (unfixed copy: {BLOCK_M}×{BLOCK_N}×{elem_size} - in_bytes)")
    print(f"[mte] target_free = {target_free} bytes  (must satisfy in_bytes < target_free < oob_bytes)")

    torch.npu.empty_cache()

    pre_fillers = []
    in_tensor = None

    try:
        pre_fillers, pool_free_after_fill, _ = _fill_segment_to_boundary(dtype, device, in_bytes, target_free,
                                                                         chunk_max_bytes)
    except Exception as exc:
        torch.npu.empty_cache()
        pytest.skip(f"Memory layout setup failed (allocator behaviour may differ): {exc}")

    # Verify pre_fill achieved the required free-space window.
    if not (in_bytes <= pool_free_after_fill <= target_free):
        for t in reversed(pre_fillers):
            del t
        torch.npu.empty_cache()
        pytest.skip(f"pre_fill did not reach target range [{in_bytes}, {target_free}] bytes; "
                    f"got {pool_free_after_fill} bytes.  "
                    f"Skipping MTE check (NPU allocator behaviour may differ).")

    try:
        # Step 3: allocate in_tensor — lands at the very end of the segment.
        # NPU 512-byte alignment means the allocator consumes
        # in_bytes + 512 = 8704 bytes, leaving gap ≈ target_free - 8704 = 7680 bytes.
        in_tensor = torch.ones(M * BLOCK_N, dtype=dtype, device=device).view(M, BLOCK_N)

        gap = torch.npu.memory_reserved(0) - torch.npu.memory_allocated(0)
        print(f"\n[mte] Step 3: in_tensor")
        print(f"[mte]   address = [{in_tensor.data_ptr():#x}, {in_tensor.data_ptr() + in_bytes:#x})")
        print(f"[mte]   gap     = {gap} bytes  (in_tensor end → segment end)")

        if oob_bytes <= gap:
            pytest.skip(f"gap ({gap} bytes) >= oob_bytes ({oob_bytes} bytes): "
                        f"OOB would not cross the segment boundary.  "
                        f"Skipping MTE check.")
        print(f"[mte]   oob_bytes({oob_bytes} B) > gap({gap} B) → MTE expected if unfixed ✓")

        # Step 4: run kernel
        num_pids_m = math.ceil(M / BLOCK_M)
        print(f"\n[mte] Step 4: kernel  (grid=({num_pids_m},))")
        cont_disc_oob_inplace_2d_kernel[(num_pids_m, )](in_tensor, M=M, BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N)
        torch.npu.synchronize()
        print("[mte] PASSED: fix is effective, no OOB.")

    except RuntimeError as exc:
        pytest.fail(f"MTE OOB triggered — DiscreteMaskAccessConversionPass fix "
                    f"may not be applied or is incomplete.\nError: {exc}")

    finally:
        if in_tensor is not None:
            del in_tensor
        for t in reversed(pre_fillers):
            del t
        torch.npu.empty_cache()
        print("[mte] Memory released.")
