# flagtree tle
"""E2E tests: verify alloc alias shares the same physical smem via modify-and-observe.
Also some corner-case run test: offset != 0, and transtype aliasing.
"""

import pytest
import torch
import triton
import triton.language as tl
import triton.experimental.tle.language as tle

try:
    from triton._flagtree_backend import FLAGTREE_BACKEND
except ModuleNotFoundError:  # pragma: no cover
    FLAGTREE_BACKEND = ""

BLOCK = 128
DEVICE = ({
    "nvidia": "cuda",
    "mthreads": "musa",
}).get(FLAGTREE_BACKEND, "cuda")


def _synchronize():
    torch.get_device_module().synchronize()


@triton.jit
def alias_e2e_kernel(in_ptr, out_ptr, N, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)

    # Phase 1: allocate v_smem, write the first batch of data (original input)
    v_smem = tle.gpu.alloc([BLOCK], dtype=tl.float32, scope=tle.gpu.smem, nv_mma_shared_layout=False)
    tle.gpu.copy(in_ptr + offs, v_smem, [BLOCK])

    # Phase 2: alias — o_smem reuses v_smem's physical memory
    o_smem = tle.gpu.alloc(
        [BLOCK],
        dtype=tl.float32,
        scope=tle.gpu.smem,
        alias=v_smem,
        alias_offset_bytes=0,
        nv_mma_shared_layout=False,
    )

    # Phase 3: critical — write second batch of data via v_smem (input + BLOCK offset),
    #          overwriting the same physical memory. If alias is correct, o_smem
    #          should see the overwritten values.
    offs2 = offs + BLOCK
    tle.gpu.copy(in_ptr + offs2, v_smem, [BLOCK])

    # Phase 4: read from o_smem
    tle.gpu.copy(o_smem, out_ptr + offs, [BLOCK])


@triton.jit
def alias_offset_kernel(in_ptr, out_ptr, BLOCK: tl.constexpr, OFFSET_BYTES: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * 2 * BLOCK + tl.arange(0, 2 * BLOCK)

    # Phase 1: source holds two BLOCK-sized chunks; alias the second one
    v_smem = tle.gpu.alloc([2 * BLOCK], dtype=tl.float32, scope=tle.gpu.smem, nv_mma_shared_layout=False)
    o_smem = tle.gpu.alloc(
        [BLOCK],
        dtype=tl.float32,
        scope=tle.gpu.smem,
        alias=v_smem,
        alias_offset_bytes=OFFSET_BYTES,
        nv_mma_shared_layout=False,
    )
    tle.gpu.copy(in_ptr + offs, v_smem, [2 * BLOCK])

    # Phase 2: critical — overwrite the whole source with the next batch. The
    #          write covers the aliased region too, so a correct alias at
    #          OFFSET_BYTES must observe the overwritten second chunk.
    offs2 = offs + 2 * BLOCK
    tle.gpu.copy(in_ptr + offs2, v_smem, [2 * BLOCK])

    # Phase 3: read from o_smem
    o_offs = pid * BLOCK + tl.arange(0, BLOCK)
    tle.gpu.copy(o_smem, out_ptr + o_offs, [BLOCK])


@triton.jit
def alias_trans_kernel(in_ptr, out_ptr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * 2 * BLOCK + tl.arange(0, 2 * BLOCK)

    # Phase 1: u8 source buffer (2 * BLOCK bytes), filled with the first batch
    # (LE byte layout of u16 0xABCD: 0xCD, 0xAB alternating)
    v_smem = tle.gpu.alloc([2 * BLOCK], dtype=tl.uint8, scope=tle.gpu.smem, nv_mma_shared_layout=False)
    tle.gpu.copy(in_ptr + offs, v_smem, [2 * BLOCK])

    # Phase 2: transtype alias — the same bytes viewed as u16: identical byte
    # range, but half the elements, so the alias is never larger than the source
    o_smem = tle.gpu.alloc(
        [BLOCK],
        dtype=tl.uint16,
        scope=tle.gpu.smem,
        alias=v_smem,
        alias_offset_bytes=0,
        nv_mma_shared_layout=False,
    )

    # Phase 3: overwrite the source with the second batch (position-unique bytes)
    offs2 = offs + 2 * BLOCK
    tle.gpu.copy(in_ptr + offs2, v_smem, [2 * BLOCK])

    # Phase 4: read back through the u16 view
    o_offs = pid * BLOCK + tl.arange(0, BLOCK)
    tle.gpu.copy(o_smem, out_ptr + o_offs, [BLOCK])


@pytest.mark.require_tle("gpu.alloc", "gpu.copy")
def test_alias_shares_smem_offset_zero():
    """o_smem observes v_smem's overwrite when alias_offset_bytes == 0."""
    N = 2048
    x = torch.randn(N, device=DEVICE, dtype=torch.float32)
    y = torch.zeros(1024, device=DEVICE, dtype=torch.float32)  # only first 1024

    grid = (triton.cdiv(1024, BLOCK), )
    alias_e2e_kernel[grid](x, y, 1024, BLOCK=BLOCK)
    _synchronize()

    # Verify: y should equal x[128:1152] (second batch), not x[0:1024] (first batch).
    # This confirms that v_smem's overwrite was observed by o_smem.
    expected = x[128:1152]
    max_diff_alias = (expected - y).abs().max().item()

    # Control: y should NOT equal the first batch
    first_batch = x[:1024]
    max_diff_original = (first_batch - y).abs().max().item()

    assert max_diff_alias < 1e-5 and max_diff_original > 1e-5


@pytest.mark.require_tle("gpu.alloc", "gpu.copy")
def test_alias_nonzero_offset_bytes():
    """alias_offset_bytes != 0: o_smem observes the overwrite at the byte offset."""
    offset_bytes = BLOCK * 4  # second float32 chunk of the 2 * BLOCK source
    x = torch.randn(4 * BLOCK, device=DEVICE, dtype=torch.float32)
    y = torch.empty(BLOCK, device=DEVICE, dtype=torch.float32)

    alias_offset_kernel[(1, )](x, y, BLOCK=BLOCK, OFFSET_BYTES=offset_bytes)
    _synchronize()

    # y should equal the overwritten second chunk x[3 * BLOCK:4 * BLOCK].
    expected = x[3 * BLOCK:4 * BLOCK]
    max_diff_alias = (expected - y).abs().max().item()

    # Control: y should NOT equal the first chunk of batch B (offset ignored),
    # nor a stale pre-overwrite chunk (alias not sharing the source).
    first_chunk = x[2 * BLOCK:3 * BLOCK]
    max_diff_original = (first_chunk - y).abs().max().item()

    assert max_diff_alias < 1e-5 and max_diff_original > 1e-5


@pytest.mark.require_tle("gpu.alloc", "gpu.copy")
def test_alias_cross_dtype_u8_to_u16():
    """Transtype alias: byte pattern in a u8 smem buffer, read back as u16.

    Sizes: source [2 * BLOCK] u8 = 2 * BLOCK bytes = alias [BLOCK] u16 —
    equal byte range, and the alias has half the elements of the source
    (never larger than the original buffer).

    The overwritten batch carries position-unique bytes (byte j == j), so the
    assertion is per byte: every one of the 2 * BLOCK positions must hold its
    own value — any shift, swap or element mix-up fails. The first batch is
    the LE byte layout of u16 0xABCD ([0xCD, 0xAB] alternating) and serves as
    the stale control.
    """
    first_batch = torch.tensor([0xCD, 0xAB] * BLOCK, device=DEVICE, dtype=torch.uint8)
    second_batch = torch.arange(2 * BLOCK, device=DEVICE).to(torch.uint8)  # byte j == j
    x = torch.cat([first_batch, second_batch])  # two batches for fill/overwrite
    y = torch.empty(BLOCK, device=DEVICE, dtype=torch.uint16)

    alias_trans_kernel[(1, )](x, y, BLOCK=BLOCK)
    _synchronize()

    y_bytes = y.view(torch.uint8)
    assert torch.equal(y_bytes, second_batch) and not torch.equal(y_bytes, first_batch)


if __name__ == "__main__":
    test_alias_shares_smem_offset_zero()
    test_alias_nonzero_offset_bytes()
    test_alias_cross_dtype_u8_to_u16()
    print("PASSED: alias e2e tests (offset 0, non-zero offset, transtype u8->u16)")
