# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Logical ring capacities survive physical control padding and noinline ABI."""

import pytest
import torch
import triton
import triton.language as tl
import triton.experimental.tle.language as tle


@triton.jit(noinline=True)
def _produce(writer):
    for step in range(17):
        slot = writer.acquire(step)
        tl.store(tle.gpu.local_ptr(slot.data), step * 16 + tl.arange(0, 16))
        writer.commit(step)
    writer.close(17)


@triton.jit(noinline=True)
def _consume(reader, output, closed):
    for step in range(18):
        ready = reader.wait(step)
        tl.store(closed + tl.program_id(0) * 18 + step, ready.is_closed.to(tl.int32))
        if not ready.is_closed:
            value = tl.load(tle.gpu.local_ptr(ready.slot.data))
            tl.store(output + tl.program_id(0) * 272 + step * 16 + tl.arange(0, 16), value)
        reader.release(step)


@triton.jit(noinline=True)
def _produce_empty(writer):
    writer.close(0)


@triton.jit(noinline=True)
def _consume_empty(reader, closed):
    ready = reader.wait(0)
    tl.store(closed + tl.program_id(0), ready.is_closed.to(tl.int32))
    reader.release(0)


@triton.jit
def _empty_ring(closed, CAPACITY: tl.constexpr):
    storage = tle.gpu.alloc([CAPACITY, 16], dtype=tl.int32, scope=tle.gpu.smem)
    pipe = tle.pipe(capacity=CAPACITY, data=storage)
    tle.gpu.warp_specialize([(_consume_empty, (pipe.reader(), closed)), (_produce_empty, (pipe.writer(),))], [1], [24])


@triton.jit
def _ring(output, closed, CAPACITY: tl.constexpr):
    storage = tle.gpu.alloc([CAPACITY, 16], dtype=tl.int32, scope=tle.gpu.smem)
    pipe = tle.pipe(capacity=CAPACITY, data=storage)
    tle.gpu.warp_specialize([(_consume, (pipe.reader(), output, closed)), (_produce, (pipe.writer(),))], [1], [24])


@pytest.mark.require_tle("pipe", "gpu.warp_specialize")
@pytest.mark.parametrize("capacity", (1, 2, 3, 5))
def test_cyclic_pipe_wrap_and_close_across_noinline(capacity):
    if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 9:
        pytest.skip("requires NVIDIA Hopper")
    output = torch.full((4, 272), -1, dtype=torch.int32, device="cuda")
    closed = torch.full((4, 18), -1, dtype=torch.int32, device="cuda")
    kernel = _ring[(4,)](output, closed, capacity, num_warps=4)
    torch.cuda.synchronize()
    torch.testing.assert_close(output, torch.arange(272, dtype=torch.int32, device="cuda").expand(4, -1), atol=0, rtol=0)
    expected_closed = torch.zeros_like(closed)
    expected_closed[:, -1] = 1
    torch.testing.assert_close(closed, expected_closed, atol=0, rtol=0)
    assert f"pipe_cta_{capacity}_cyclic" in kernel.asm["llir"]
    assert kernel.metadata.ptxas_spill_load_bytes == kernel.metadata.ptxas_spill_store_bytes == 0


@pytest.mark.require_tle("pipe", "gpu.warp_specialize")
@pytest.mark.parametrize("capacity", (1, 3, 5))
def test_empty_pipe_closes_without_a_data_participant_contract(capacity):
    if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 9:
        pytest.skip("requires NVIDIA Hopper")
    closed = torch.zeros(4, dtype=torch.int32, device="cuda")
    _empty_ring[(4,)](closed, capacity, num_warps=4)
    torch.cuda.synchronize()
    torch.testing.assert_close(closed, torch.ones_like(closed), atol=0, rtol=0)
