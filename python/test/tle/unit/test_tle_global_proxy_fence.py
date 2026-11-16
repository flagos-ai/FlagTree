# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Explicit global generic-to-async publication retains its native ordering."""

import pytest
import re
import triton
import triton.language as tl
import triton.experimental.tle.language as tle
from triton.backends.compiler import GPUTarget
from triton.compiler import ASTSource


@triton.jit
def _fence_only():
    tle.gpu.fence_proxy_async_global()


@triton.jit
def _publish_to_tensor_map(data, descriptor, output, ENABLE: tl.constexpr):
    owner = tl.program_id(0)
    lane = tl.arange(0, 16)
    tl.store(data + owner * 16 + lane, owner * 16 + lane + 1000)
    tl.debug_barrier()
    if ENABLE:
        tle.gpu.fence_proxy_async_global()
    shared = tle.gpu.alloc([1, 16], dtype=tl.int32, scope=tle.gpu.smem)
    tle.gpu.tensor_map_fenceproxy_acquire(descriptor)
    tensor_map = tle.gpu.reinterpret_tensor_map(descriptor, shared)
    tle.gpu.copy(tensor_map, shared, [1, 16], [owner, 0])
    values = tl.reshape(tl.load(tle.gpu.local_ptr(shared)), (16,))
    tl.store(output + owner * 16 + lane, values)


def compile_publication(enable=True):
    return triton.compile(
        ASTSource(_publish_to_tensor_map,
                  {'data': '*i32', 'descriptor': '*u8', 'output': '*i32', 'ENABLE': 'constexpr'},
                  constexprs={'ENABLE': enable},
                  attrs={(i,): [('tt.divisibility', 16)] for i in range(3)}),
        target=GPUTarget('cuda', 90, 32), options={'num_warps': 4, 'ptx_version': 83})


def test_global_proxy_fence_orders_data_before_tma():
    compiled = compile_publication()
    ptx = compiled.asm['ptx']
    data_fence = ptx.index('fence.proxy.async.global;')
    assert ptx.count('fence.proxy.async.global;') == 1
    assert ptx.index('st.global') < data_fence < ptx.index('cp.async.bulk.tensor.2d')
    # The descriptor and data publications are distinct instructions.
    assert 'fence.proxy.tensormap::generic.acquire.gpu' in ptx


def test_global_proxy_fence_minimum_ptx():
    # The data fence itself needs PTX 8.0; descriptor acquire needs PTX 8.3.
    compiled = triton.compile(ASTSource(_fence_only, {}), target=GPUTarget('cuda', 90, 32),
                              options={'num_warps': 4, 'ptx_version': 80})
    assert 'fence.proxy.async.global;' in compiled.asm['ptx']


@pytest.mark.parametrize('sm,ptx', [(80, 80), (90, 78)])
def test_global_proxy_fence_rejects_unsupported_target(sm, ptx, capfd):
    with pytest.raises(RuntimeError, match='PassManager::run failed'):
        triton.compile(ASTSource(_fence_only, {}), target=GPUTarget('cuda', sm, 32),
                       options={'num_warps': 4, 'ptx_version': ptx})
    assert 'requires SM90 or newer and PTX ISA 8.0 or newer' in capfd.readouterr().err


def test_unused_proxy_fence_is_not_inserted():
    assert 'fence.proxy.async.global' not in compile_publication(False).asm['ptx']


_GRID = tl.constexpr(tle.device_mesh({'block': [('block_y', 8), ('block_x', 16)]}))


@triton.jit(noinline=True)
def _cross_cta_consumer(handoff_writer, data_reader, data, output):
    owner = tl.program_id(0)
    lane = tl.arange(0, 16)
    for epoch in tl.range(0, 4, loop_unroll_factor=1):
        handoff_writer.acquire(epoch)
        offset = (epoch * 128 + owner) * 16 + lane
        tl.store(data + offset, offset + 1000)
        tle.distributed_barrier(_GRID)
        handoff_writer.commit(epoch)
        ready = data_reader.wait(epoch)
        payload = ready.slot.raw.subslice([0, 0], [1, 16])
        value = tl.reshape(tl.load(tle.gpu.local_ptr(payload)), (16,))
        tl.store(output + offset, value)
        data_reader.release(epoch)
    handoff_writer.close(4)


@triton.jit(noinline=True)
def _cross_cta_producer(handoff_reader, data_writer, descriptor):
    owner = tl.program_id(0)
    tle.gpu.tensor_map_fenceproxy_acquire(descriptor)
    for epoch in tl.range(0, 4, loop_unroll_factor=1):
        handoff_reader.wait(epoch)
        tle.gpu.fence_proxy_async_global()
        slot = data_writer.acquire(epoch)
        payload = slot.raw.subslice([0, 0], [1, 16])
        tensor_map = tle.gpu.reinterpret_tensor_map(descriptor, payload)
        tle.gpu.copy(tensor_map, payload, [1, 16], [epoch * 128 + (owner + 1) % 128, 0])
        data_writer.commit(epoch)
        handoff_reader.release(epoch)
    data_writer.close(4)


@triton.jit
def _cross_cta_handoff(data, descriptor, output):
    # Tensor TMA destinations require 128-byte alignment, including every slot.
    # Each epoch still transfers and checks exactly 16 INT32 payload words.
    shared = tle.gpu.alloc([2, 1, 32], dtype=tl.int32, scope=tle.gpu.smem)
    data_pipe = tle.pipe(capacity=2, scope='cta', name='published_data', raw=shared)
    handoff = tle.pipe(capacity=1, scope='cta', name='data_ready')
    tle.gpu.warp_specialize([
        (_cross_cta_consumer, (handoff.writer(), data_pipe.reader(), data, output)),
        (_cross_cta_producer, (handoff.reader(), data_pipe.writer(), descriptor)),
    ], [1], [32])


def compile_cross_cta_handoff():
    return triton.compile(
        ASTSource(_cross_cta_handoff, {'data': '*i32', 'descriptor': '*u8', 'output': '*i32'},
                  attrs={(i,): [('tt.divisibility', 16)] for i in range(3)}),
        target=GPUTarget('cuda', 90, 32), options={'num_warps': 8, 'ptx_version': 83})


def test_proxy_fence_follows_cross_cta_handoff():
    compiled = compile_cross_cta_handoff()
    ptx = compiled.asm['ptx']
    # Pipe lowering materializes the four epochs. Each distinct TMA issuer
    # reaches its fence after a successful handoff wait in that epoch.
    assert ptx.count('fence.proxy.async.global;') == 4
    assert ptx.count('cp.async.bulk.tensor.2d') == 4
    start = ptx.rfind('\n.func ', 0, ptx.index('fence.proxy.async.global;'))
    assert '_cross_cta_producer__' in ptx[start:ptx.index('\n', start + 1)]
    for match in re.finditer('fence[.]proxy[.]async[.]global;', ptx[start:]):
        fence = start + match.start()
        assert ptx.rfind('mbarrier.try_wait', start, fence) >= start
        assert ptx.find('cp.async.bulk.tensor.2d', fence) > fence
    assert compiled.metadata.num_warps == 12
    assert compiled.metadata.launch_cooperative_grid


def test_cross_cta_global_proxy_publication(with_allocator):
    import torch
    from triton.flagmega.runtime.nvidia_tensor_map import encode_tma_descriptor

    if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] < 9:
        pytest.skip('requires SM90 or newer')
    if torch.cuda.get_device_properties(0).multi_processor_count < 128:
        pytest.skip('requires a resident 128-CTA cooperative grid')
    data = torch.zeros((512, 16), dtype=torch.int32, device='cuda')
    payload = encode_tma_descriptor(data.data_ptr(), 0, 4, 3, (1, 16), (512, 16), (16, 1), 0)
    descriptor = torch.frombuffer(bytearray(payload), dtype=torch.uint8).to('cuda')
    output = torch.empty_like(data)
    _cross_cta_handoff[(128,)](data, descriptor, output, num_warps=8, ptx_version=83)
    expected = (torch.arange(8192, device='cuda', dtype=torch.int32) + 1000).reshape(4, 128, 16)
    torch.testing.assert_close(data.reshape(4, 128, 16), expected, rtol=0, atol=0)
    torch.testing.assert_close(output.reshape(4, 128, 16), torch.roll(expected, -1, 1), rtol=0, atol=0)
