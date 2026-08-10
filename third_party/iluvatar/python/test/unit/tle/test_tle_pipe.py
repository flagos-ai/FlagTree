"""Iluvatar tle.pipe + warp_specialize e2e.

Frontend contract tests live in python/test/tle/unit/test_tle.py (main tree).
This file covers Iluvatar-specific lowering: pipe -> software mbarrier ->
ivcore11 warp-specialize execution.
"""

import pytest
import torch
import triton
import triton.language as tl
import triton.experimental.tle.language as tle


def _is_corex():
    try:
        target = triton.runtime.driver.active.get_current_target()
    except Exception:
        return False
    return target is not None and target.backend == "corex"


requires_corex = pytest.mark.skipif(not _is_corex(), reason="Requires an Iluvatar (corex) device")


@triton.jit
def _pipe_producer(writer, x_ptr, n_tiles: tl.constexpr, BLOCK: tl.constexpr):
    offs = tl.arange(0, BLOCK)
    for i in tl.static_range(n_tiles):
        slot = writer.acquire(i)
        vals = tl.load(x_ptr + i * BLOCK + offs)
        tl.store(tle.gpu.local_ptr(slot.tile), vals)
        writer.commit(i)


@triton.jit
def _pipe_consumer(reader, out_ptr, n_tiles: tl.constexpr, BLOCK: tl.constexpr):
    offs = tl.arange(0, BLOCK)
    acc = tl.zeros([BLOCK], dtype=tl.float32)
    for i in tl.static_range(n_tiles):
        ready = reader.wait(i)
        tile = tl.load(tle.gpu.local_ptr(ready.slot.tile))
        acc += tile
        reader.release(i)
    tl.store(out_ptr + offs, acc)


@triton.jit
def _pipe_ws_sum_kernel(x_ptr, out_ptr, n_tiles: tl.constexpr, BLOCK: tl.constexpr):
    smem = tle.gpu.alloc([2, BLOCK], dtype=tl.float32, layout=None, scope=tle.gpu.smem, nv_mma_shared_layout=False)
    pipe = tle.pipe(capacity=2, scope="cta", name="x_pipe", tile=smem)
    tle.gpu.warp_specialize(
        [
            (_pipe_producer, (pipe.writer(), x_ptr, n_tiles, BLOCK)),
            (_pipe_consumer, (pipe.reader(), out_ptr, n_tiles, BLOCK)),
        ],
        worker_num_warps=[4],
        worker_num_regs=[80],
    )


@requires_corex
def test_tle_pipe_warp_specialize_sum_e2e():
    """Producer partition fills a capacity-2 SMEM pipe; consumer sums tiles."""
    BLOCK = 64
    n_tiles = 4
    torch.manual_seed(0)
    x = torch.randn(n_tiles * BLOCK, dtype=torch.float32, device="cuda")
    out = torch.zeros(BLOCK, dtype=torch.float32, device="cuda")

    compiled = _pipe_ws_sum_kernel[(1, )](x, out, n_tiles=n_tiles, BLOCK=BLOCK, num_warps=4)

    ttgir = compiled.asm["ttgir"]
    assert "iluvatar_tle.pipe" not in ttgir, ttgir
    assert ("iluvatar_tle.wait_barrier" in ttgir or "iluvatar_tle.arrive_barrier" in ttgir), ttgir

    llir = compiled.asm["llir"]
    code_lines = [ln for ln in llir.splitlines() if not ln.lstrip().startswith("!") and "DIFile" not in ln]
    assert "warp_specialize" not in "\n".join(code_lines), llir

    expected = x.view(n_tiles, BLOCK).sum(dim=0)
    torch.testing.assert_close(out, expected, atol=1e-4, rtol=1e-4)
