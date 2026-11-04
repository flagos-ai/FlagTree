# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Explicit TLE pipes must lower on Blackwell and later CUDA targets."""
import re

import pytest
import torch
import triton
import triton.language as tl
import triton.experimental.tle.language as tle


@triton.jit(noinline=True)
def _producer(writer):
    for stage in range(13):
        slot = writer.acquire(stage)
        tl.store(tle.gpu.local_ptr(slot.data), stage * 32 + tl.arange(0, 32))
        writer.commit(stage)
    writer.close(13)


@triton.jit(noinline=True)
def _consumer(reader, output):
    for stage in range(14):
        ready = reader.wait(stage)
        if not ready.is_closed:
            value = tl.load(tle.gpu.local_ptr(ready.slot.data))
            tl.store(output + tl.program_id(0) * 416 + stage * 32 + tl.arange(0, 32), value)
        reader.release(stage)


@triton.jit
def _kernel(output):
    storage = tle.gpu.alloc((2, 32), dtype=tl.int32, scope=tle.gpu.smem)
    pipe = tle.pipe(capacity=2, data=storage)
    tle.gpu.warp_specialize([(_consumer, (pipe.reader(), output)), (_producer, (pipe.writer(), ))], [1], [32])


@pytest.mark.require_tle('pipe', 'gpu.warp_specialize')
@pytest.mark.parametrize('warps', [4, 8])
def test_explicit_pipe_wrap_noinline_and_close(warps):
    if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] < 10:
        pytest.skip('requires Blackwell or newer CUDA GPU')
    output = torch.full((3, 416), -1, dtype=torch.int32, device='cuda')
    kernel = _kernel[(3, )](output, num_warps=warps)
    torch.cuda.synchronize()
    torch.testing.assert_close(output,
                               torch.arange(416, dtype=torch.int32, device='cuda').expand(3, -1), atol=0, rtol=0)
    assert 'tle.pipe.' not in kernel.asm['ttgir']
    assert kernel.metadata.ptxas_spill_load_bytes == kernel.metadata.ptxas_spill_store_bytes == 0


@pytest.mark.require_tle('pipe', 'gpu.warp_specialize')
def test_explicit_pipe_from_ttir(tmp_path):
    """An IR entry must not depend on frontend module markers for lowering."""
    if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] < 10:
        pytest.skip('requires Blackwell or newer CUDA GPU')
    output = torch.full((3, 416), -1, dtype=torch.int32, device='cuda')
    source_kernel = _kernel.warmup(output, grid=(3, ), num_warps=4)
    source = tmp_path / 'pipe.ttir'
    # Drop all frontend module metadata to exercise the standalone IR entry.
    standalone_ir = re.sub(r'(?m)^module(?: attributes \{.*\})? \{', 'module {', source_kernel.asm['ttir'], count=1)
    source.write_text(standalone_ir)
    kernel = triton.compile(str(source), options={'num_warps': 4})
    kernel[(3, 1, 1)](output)
    torch.cuda.synchronize()
    torch.testing.assert_close(output,
                               torch.arange(416, dtype=torch.int32, device='cuda').expand(3, -1), atol=0, rtol=0)
    assert 'tle.pipe.' not in kernel.asm['ttgir']
    assert kernel.metadata.ptxas_spill_load_bytes == kernel.metadata.ptxas_spill_store_bytes == 0
