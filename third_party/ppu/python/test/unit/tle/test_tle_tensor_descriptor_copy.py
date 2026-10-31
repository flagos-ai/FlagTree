"""PPU runtime tests of tle.gpu.copy between a tensor descriptor and shared memory.

PPU has no TMA: the backend rewrites a descriptor copy into a bounds-checked
pointer copy, and a global-to-shared copy becomes a cp.async copy when the
descriptor's alignment allows 16-byte vectors. The tests check the values,
the out-of-bounds fill and which form of the copy the kernel got.
"""

import subprocess
import sys
import textwrap

import pytest
import torch

import triton
import triton.language as tl
from triton._internal_testing import is_ppu
from triton.tools.tensor_descriptor import TensorDescriptor

tle = pytest.importorskip("triton.experimental.tle.language", reason="tle language unavailable")

pytestmark = pytest.mark.skipif(not is_ppu(), reason="requires the PPU backend")


@triton.jit
def _ring_sum_kernel(desc, y, n_tiles, R: tl.constexpr, C: tl.constexpr):
    ring = tle.gpu.alloc([2, R, C], dtype=tl.bfloat16, layout=None, scope=tle.gpu.smem)
    rows = tl.arange(0, R)[:, None] + tl.zeros([R, C], dtype=tl.int32)
    cols = tl.arange(0, C)[None, :] + tl.zeros([R, C], dtype=tl.int32)
    acc = tl.zeros([R, C], dtype=tl.float32)
    for i in range(n_tiles):
        slot = ring.slot(i % 2)
        tle.gpu.copy(desc, slot, [R, C], [i * R, 0])
        acc += tl.load(tle.gpu.local_ptr(slot, (rows, cols))).to(tl.float32)
    tl.store(y + rows * C + cols, acc)


def _reference(x, n_tiles, block):
    expected = torch.zeros(block, device="cuda")
    for i in range(n_tiles):
        tile = x[i * block[0]:(i + 1) * block[0]].float()
        expected[:tile.shape[0], :tile.shape[1]] += tile
    return expected


# rows, columns, and whether the copy is expected to be asynchronous.
@pytest.mark.parametrize("rows, cols, is_async", [
    (256, 64, True),
    (200, 64, True),
    (64, 64, True),
    (256, 56, False),
])
def test_descriptor_copy_to_shared(rows, cols, is_async):
    block = [64, 64]
    n_tiles = (rows + block[0] - 1) // block[0]
    # A column count that is not a multiple of 16 elements needs a padded allocation:
    # the base and the row pitch stay 16-byte aligned, as the descriptor requires.
    padded = (cols + 7) // 8 * 8
    storage = torch.randn(rows, padded, dtype=torch.bfloat16, device="cuda")
    x = storage[:, :cols]
    desc = TensorDescriptor(x, x.shape, x.stride(), block)
    y = torch.zeros(block, device="cuda")
    ck = _ring_sum_kernel[(1, )](desc, y, n_tiles, R=block[0], C=block[1], num_warps=4)
    torch.cuda.synchronize()
    torch.testing.assert_close(y, _reference(x, n_tiles, block), rtol=0, atol=0)
    ttgir = ck.asm["ttgir"]
    assert "ttg.tma_copy" not in ttgir
    assert ("ttg.async_copy_global_to_local" in ttgir) == is_async


@triton.jit
def _store_kernel(desc, x, R: tl.constexpr, C: tl.constexpr):
    buf = tle.gpu.alloc([R, C], dtype=tl.float32, layout=None, scope=tle.gpu.smem)
    rows = tl.arange(0, R)[:, None] + tl.zeros([R, C], dtype=tl.int32)
    cols = tl.arange(0, C)[None, :] + tl.zeros([R, C], dtype=tl.int32)
    tl.store(tle.gpu.local_ptr(buf, (rows, cols)), tl.load(x + rows * C + cols) * 2.0)
    tle.gpu.copy(buf, desc, [R, C], [R, 0])


def test_descriptor_copy_to_global():
    r, c = 32, 64
    x = torch.randn(r, c, device="cuda")
    out = torch.zeros(3 * r, c, device="cuda")
    desc = TensorDescriptor(out, out.shape, out.stride(), [r, c])
    _store_kernel[(1, )](desc, x, R=r, C=c, num_warps=4)
    torch.cuda.synchronize()
    expected = torch.zeros_like(out)
    expected[r:2 * r] = x * 2.0
    torch.testing.assert_close(out, expected, rtol=0, atol=0)


def test_descriptor_copy_with_barrier_is_rejected():

    @triton.jit
    def kernel(desc, y, R: tl.constexpr, C: tl.constexpr):
        buf = tle.gpu.alloc([R, C], dtype=tl.bfloat16, layout=None, scope=tle.gpu.smem)
        bar = tle.gpu.alloc_barrier(expect_bytes=R * C * 2)
        tle.gpu.copy(desc, buf, [R, C], [0, 0], barrier=bar)
        tle.gpu.barrier_wait(bar, phaseIdx=0)

    x = torch.randn(64, 64, dtype=torch.bfloat16, device="cuda")
    desc = TensorDescriptor.from_tensor(x, [64, 64])
    with pytest.raises(Exception, match="completion barrier|barrier="):
        kernel[(1, )](desc, x, R=64, C=64, num_warps=4)


# A device assertion poisons the CUDA context, so the NaN-padding check runs in its own process.
_NAN_PADDING_SCRIPT = textwrap.dedent("""
    import torch
    import triton
    import triton.language as tl
    import triton.experimental.tle.language as tle
    from triton.tools.tensor_descriptor import TensorDescriptor

    @triton.jit
    def kernel(desc, y, R: tl.constexpr, C: tl.constexpr):
        buf = tle.gpu.alloc([R, C], dtype=tl.bfloat16, layout=None, scope=tle.gpu.smem)
        tle.gpu.copy(desc, buf, [R, C], [0, 0])
        rows = tl.arange(0, R)[:, None] + tl.zeros([R, C], dtype=tl.int32)
        cols = tl.arange(0, C)[None, :] + tl.zeros([R, C], dtype=tl.int32)
        tl.store(y + rows * C + cols, tl.load(tle.gpu.local_ptr(buf, (rows, cols))))

    x = torch.randn(64, 64, dtype=torch.bfloat16, device="cuda")
    y = torch.zeros(64, 64, dtype=torch.bfloat16, device="cuda")
    kernel[(1, )](TensorDescriptor.from_tensor(x, [64, 64], padding="nan"), y, R=64, C=64, num_warps=4)
    torch.cuda.synchronize()
""")


def test_descriptor_copy_rejects_nan_padding(tmp_path):
    script = tmp_path / "nan_padding.py"
    script.write_text(_NAN_PADDING_SCRIPT)
    result = subprocess.run([sys.executable, str(script)], capture_output=True, text=True, timeout=300)
    assert result.returncode != 0
    assert "the descriptor asks for NaN padding" in result.stdout + result.stderr
