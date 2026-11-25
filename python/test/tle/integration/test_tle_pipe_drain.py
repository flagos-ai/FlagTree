"""Closing and draining cyclic pipes before whole-stage storage reuse."""
import pytest
import torch
import triton
import triton.language as tl
import triton.experimental.tle.language as tle

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a CUDA GPU")

_LAYOUT = tl.constexpr(tle.gpu.swizzled_shared_layout.make_default(rank=2))


@triton.jit(noinline=True)
def _drain_writer(writer, storage, n):
    for step in range(n):
        slot = writer.acquire(step)
        tl.store(tle.gpu.local_ptr(slot.data), step + tl.arange(0, 16))
        writer.commit(step)
    writer.close(n)
    writer.pipe.wait_drained()
    # Only the writer overwrites storage after the final reader has left.
    slot = storage.slot(0)
    tl.store(tle.gpu.local_ptr(slot), 99 + tl.arange(0, 16))


@triton.jit(noinline=True)
def _drain_reader(reader, output, closed, n):
    for step in range(n + 1):
        ready = reader.wait(step)
        if not ready.is_closed:
            value = tl.load(tle.gpu.local_ptr(ready.slot.data))
            tl.store(output + step * 16 + tl.arange(0, 16), value)
        else:
            tl.store(closed, 1)
        reader.release(step)
    reader.pipe.wait_drained()


@triton.jit
def _drain_kernel(output, closed, n, CAPACITY: tl.constexpr):
    storage = tle.gpu.alloc([CAPACITY, 16], dtype=tl.int32, scope=tle.gpu.smem, layout=_LAYOUT)
    edge = tle.pipe(capacity=CAPACITY, data=storage)
    tle.gpu.warp_specialize([(_drain_reader, (edge.reader(), output, closed, n)),
                             (_drain_writer, (edge.writer(), storage, n))], [1], [24])


@pytest.mark.skip_if_tle_unsupported("pipe", "gpu.alloc", "gpu.local_ptr", "gpu.warp_specialize")
@pytest.mark.parametrize("capacity", [1, 3])
@pytest.mark.parametrize("n", [0, 17])
def test_pipe_drains_after_close_and_last_release(capacity, n, with_allocator):
    if not torch.cuda.is_available() or torch.version.cuda is None or torch.cuda.get_device_capability()[0] < 9:
        pytest.skip("requires Hopper or newer CUDA GPU")
    output = torch.full((n, 16), -1, device="cuda", dtype=torch.int32)
    closed = torch.zeros((), device="cuda", dtype=torch.int32)
    kernel = _drain_kernel[(1, )](output, closed, n, capacity, num_warps=4)
    torch.cuda.synchronize()
    expected = torch.arange(n, device="cuda", dtype=torch.int32)[:, None] + torch.arange(
        16, device="cuda", dtype=torch.int32)[None, :]
    torch.testing.assert_close(output, expected, rtol=0, atol=0)
    assert closed.item() == 1
    assert "tle.pipe.drain" in kernel.asm["ttir"]
    assert "tle.pipe." not in kernel.asm["ttgir"]
