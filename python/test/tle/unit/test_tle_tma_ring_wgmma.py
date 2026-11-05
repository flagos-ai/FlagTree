# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: MIT
"""A caller-owned TMA ring cannot be refilled before all WGMMA readers retire."""

import os
from pathlib import Path

import pytest
import torch
import triton
import triton.language as tl
import triton.experimental.tle.language as tle
from triton.flagmega.runtime.nvidia_tensor_map import encode_tma_descriptor


@triton.jit(noinline=True)
def _tma_wgmma_ring_body(arena, weight_descriptor, source_descriptor, output, OWN_ARENA: tl.constexpr,
                         GROUP_DELAY_NS: tl.constexpr):
    if OWN_ARENA:
        arena = tle.gpu.alloc([147456], tl.uint8, nv_mma_shared_layout=False, alignment_bytes=1024)
    weight = tle.gpu.alloc([2, 64, 256], tl.bfloat16, alias=arena, alias_offset_bytes=0, nv_mma_shared_layout=True)
    activation = tle.gpu.alloc([2, 8, 256], tl.bfloat16, alias=arena, alias_offset_bytes=65536,
                               nv_mma_shared_layout=True)
    weight_full = tle.gpu.alloc_barriers(2, expect_bytes=64 * 256 * 2)
    activation_full = tle.gpu.alloc_barriers(2, expect_bytes=8 * 256 * 2)
    tle.gpu.tensor_map_fenceproxy_acquire(source_descriptor)
    tle.gpu.tensor_map_fenceproxy_acquire(weight_descriptor)
    sequence_base = tl.full((), 0, tl.int32)
    for task in tl.range(tl.program_id(0), 101, tl.num_programs(0)):
        for initial in tl.static_range(0, 2):
            slot = (sequence_base + initial) % 2
            w = weight.slot(slot)
            x = activation.slot(slot)
            wd = tle.gpu.reinterpret_tensor_map(weight_descriptor, w)
            xd = tle.gpu.reinterpret_tensor_map(source_descriptor, x)
            tle.gpu.copy(wd, w, [64, 256], [task * 64, initial * 256], barrier=weight_full[slot])
            tle.gpu.copy(xd, x, [8, 256], [0, initial * 256], barrier=activation_full[slot])
        total = tl.zeros((64, 8), tl.float32)
        for step in tl.range(0, 16):
            sequence = sequence_base + step
            slot = sequence % 2
            phase = (sequence // 2) % 2
            tle.gpu.barrier_wait(weight_full[slot], phaseIdx=phase)
            tle.gpu.barrier_wait(activation_full[slot], phaseIdx=phase)
            if GROUP_DELAY_NS:
                # Uniform within each 128-thread WGMMA group; stress legal
                # independent scheduling between the two consumer groups.
                tl.inline_asm_elementwise(
                    "{ .reg .u32 tid; .reg .pred slow; mov.u32 tid, %tid.x; "
                    "and.b32 tid, tid, 255; setp.ge.u32 slow, tid, 128; "
                    "@slow nanosleep.u32 $1; mov.u32 $0, 0; }", constraints="=r,r", args=[GROUP_DELAY_NS],
                    dtype=tl.int32, is_pure=False, pack=1)
            total = tle.gpu.wgmma(weight.slot(slot), activation.slot(slot), total, trans_b=True)
            total = tle.gpu.wgmma_wait(0, total)
            if step + 2 < 16:
                w = weight.slot(slot)
                x = activation.slot(slot)
                wd = tle.gpu.reinterpret_tensor_map(weight_descriptor, w)
                xd = tle.gpu.reinterpret_tensor_map(source_descriptor, x)
                tle.gpu.copy(wd, w, [64, 256], [task * 64, (step + 2) * 256], barrier=weight_full[slot])
                tle.gpu.copy(xd, x, [8, 256], [0, (step + 2) * 256], barrier=activation_full[slot])
        sequence_base += 16
        result = tl.reshape(tl.gather(total, tl.full((64, 1), 0, tl.int32), 1), (64, ))
        tl.store(output + task * 64 + tl.arange(0, 64), result.to(tl.bfloat16))


@triton.jit
def _tma_ring_consumer(arena, weight_descriptor, source_descriptor, output, OWN_ARENA: tl.constexpr,
                       GROUP_DELAY_NS: tl.constexpr):
    _tma_wgmma_ring_body(arena, weight_descriptor, source_descriptor, output, OWN_ARENA, GROUP_DELAY_NS)


@triton.jit
def _tma_ring_readonly_producer(weight):
    for index in tl.range(0, 64):
        tl.inline_asm_elementwise("prefetch.global.L2 [$1]; mov.u32 $0, 0;", constraints="=r,l",
                                  args=[weight + index * 128], dtype=tl.int32, is_pure=False, pack=1)


@triton.jit
def _tma_wgmma_ring(weight, weight_descriptor, source_descriptor, output, OWN_ARENA: tl.constexpr, OUTER: tl.constexpr,
                    GROUP_DELAY_NS: tl.constexpr):
    if OWN_ARENA:
        arena = None
    else:
        arena = tle.gpu.alloc([147456], tl.uint8, nv_mma_shared_layout=False, alignment_bytes=1024)
    if OUTER:
        tle.gpu.warp_specialize([
            (_tma_ring_consumer, (arena, weight_descriptor, source_descriptor, output, OWN_ARENA, GROUP_DELAY_NS)),
            (_tma_ring_readonly_producer, (weight, )),
        ], [4], [24])
    else:
        _tma_wgmma_ring_body(arena, weight_descriptor, source_descriptor, output, OWN_ARENA, GROUP_DELAY_NS)


@pytest.mark.parametrize("ownership,outer", [
    ("borrowed", False),
    ("owned", False),
    ("borrowed", True),
], ids=["caller_arena", "callee_arena", "global_caller_arena"])
@pytest.mark.parametrize("num_warps,delay_ns", [
    (4, 0),
    (8, 0),
    (8, 20000),
], ids=["one_group", "two_groups", "two_groups_skew"])
def test_tma_ring_wait_retires_every_shared_reader(ownership, outer, num_warps, delay_ns):
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required")
    if torch.cuda.get_device_capability()[0] != 9:
        pytest.skip("This reproducer exercises SM90 WGMMA")
    generator = torch.Generator().manual_seed(2761)
    weights = torch.randint(-2, 3, (101 * 64, 4096), generator=generator).float() / 2
    activation = torch.randint(-2, 3, (1, 4096), generator=generator).float() / 2
    expected = (weights @ activation.T).flatten().to(torch.bfloat16)
    weight = weights.to(torch.bfloat16).cuda()
    source = activation.to(torch.bfloat16).cuda()
    payload = encode_tma_descriptor(weight.data_ptr(), 3, 2, 9, (64, 64), (101 * 64, 4096), (4096, 1), 0)
    payload += encode_tma_descriptor(source.data_ptr(), 3, 2, 9, (8, 64), (1, 4096), (4096, 1), 0)
    descriptors = torch.frombuffer(bytearray(payload), dtype=torch.uint8).cuda()
    assert descriptors.data_ptr() % 128 == 0
    source_descriptor = descriptors[128:]
    assert source_descriptor.data_ptr() == descriptors.data_ptr() + 128
    output = torch.empty((101 * 64, ), dtype=torch.bfloat16, device="cuda")
    args = (weight, descriptors, source_descriptor, output, ownership == "owned", outer, delay_ns)
    compiled = _tma_wgmma_ring.warmup(*args, grid=(128, ), num_warps=num_warps)
    diagnostic = os.environ.get("FLAGMEGA_TMA_RING_DIAGNOSTIC_DIR")
    if diagnostic:
        root = Path(diagnostic) / f"{ownership}-outer-{outer}-warps-{num_warps}-delay-{delay_ns}"
        root.mkdir(parents=True, exist_ok=True)
        for suffix in ("ttir", "ttgir", "llir", "ptx"):
            (root / f"kernel.{suffix}").write_text(compiled.asm[suffix])
    for replay in range(32):
        _tma_wgmma_ring[(128, )](*args, num_warps=num_warps)
        actual = output.cpu()
        if diagnostic and not torch.equal(actual, expected):
            torch.save(
                dict(actual=actual, expected=expected, replay=replay, ownership=ownership, outer=outer,
                     num_warps=num_warps, delay_ns=delay_ns), root / "failure.pt")
        torch.testing.assert_close(
            actual, expected, rtol=0, atol=0, msg=f"{ownership} arena, outer={outer}, warps={num_warps}, "
            f"delay={delay_ns}, replay={replay}")
