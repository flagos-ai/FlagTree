"""Numerical coverage for an explicit copy ring inside a pipelined WGMMA loop."""

import os
import re
from pathlib import Path

import pytest
import torch
import triton
import triton.language as tl
from triton.language.extra.cuda import libdevice
import triton.experimental.tle.language as tle


@triton.jit(noinline=True)
def _manual_ring_wgmma_body(arena, weight, activation, scales, block_scales, output):
    weights = tle.gpu.alloc([8, 64, 128], tl.float8e4nv, nv_mma_shared_layout=True, alias=arena, alias_offset_bytes=0)
    activations = tle.gpu.alloc([8, 16, 128], tl.float8e4nv, nv_mma_shared_layout=True, alias=arena,
                                alias_offset_bytes=65536)
    rows = tl.arange(0, 64)
    columns = tl.arange(0, 16)
    kk = tl.arange(0, 128)
    for task in tl.range(0, 2):
        for initial in tl.static_range(0, 8):
            k = initial * 128 + kk
            w = weight + task * 64 * 4096 + rows[:, None] * 4096 + k[None, :]
            a = activation + task * 4096 + k[None, :] + columns[:, None] * 4096
            tle.gpu.copy(w, weights.slot(initial), [64, 128], is_async=True)
            tle.gpu.copy(a, activations.slot(initial), [16, 128], is_async=True, mask=columns[:, None] == 0)
            tle.gpu.async_commit_group()
        total = tl.zeros((64, 16), tl.float32)
        for block in tl.range(0, 32, num_stages=4, loop_unroll_factor=2):
            av = tl.load(block_scales + task * 32 + block)
            if block + 7 < 32:
                tle.gpu.async_wait_group(7)
            elif block + 6 < 32:
                tle.gpu.async_wait_group(6)
            elif block + 5 < 32:
                tle.gpu.async_wait_group(5)
            elif block + 4 < 32:
                tle.gpu.async_wait_group(4)
            elif block + 3 < 32:
                tle.gpu.async_wait_group(3)
            elif block + 2 < 32:
                tle.gpu.async_wait_group(2)
            elif block + 1 < 32:
                tle.gpu.async_wait_group(1)
            else:
                tle.gpu.async_wait_group(0)
            bs = tl.load(scales + task * 64 + rows)
            product = tle.gpu.wgmma(weights.slot(block % 8), activations.slot(block % 8), out_dtype=tl.float32,
                                    trans_b=True)
            product = tle.gpu.wgmma_wait(0, product)
            total = tl.fma(libdevice.mul_rn(product, bs[:, None]), av, total)
            if block + 8 < 32:
                k = (block + 8) * 128 + kk
                w = weight + task * 64 * 4096 + rows[:, None] * 4096 + k[None, :]
                a = activation + task * 4096 + k[None, :] + columns[:, None] * 4096
                tle.gpu.copy(w, weights.slot(block % 8), [64, 128], is_async=True)
                tle.gpu.copy(a, activations.slot(block % 8), [16, 128], is_async=True, mask=columns[:, None] == 0)
                tle.gpu.async_commit_group()
        result = tl.reshape(tl.gather(total, tl.full((64, 1), 0, tl.int32), 1), (64, ))
        tl.store(output + task * 64 + rows, result)


@triton.jit
def _manual_ring_consumer(arena, weight, activation, scales, block_scales, output):
    _manual_ring_wgmma_body(arena, weight, activation, scales, block_scales, output)


@triton.jit
def _readonly_ring_producer(weight):
    for index in tl.range(0, 64):
        tl.inline_asm_elementwise("prefetch.global.L2 [$1]; mov.u32 $0, 0;", constraints="=r,l",
                                  args=[weight + index * 128], dtype=tl.int32, is_pure=False, pack=1)


@triton.jit
def _manual_ring_wgmma(weight, activation, scales, block_scales, output, OUTER_SPECIALIZATION: tl.constexpr):
    arena = tle.gpu.alloc([147456], tl.uint8, nv_mma_shared_layout=False, alignment_bytes=128)
    if OUTER_SPECIALIZATION:
        tle.gpu.warp_specialize([
            (_manual_ring_consumer, (arena, weight, activation, scales, block_scales, output)),
            (_readonly_ring_producer, (weight, )),
        ], [4], [24])
    else:
        _manual_ring_wgmma_body(arena, weight, activation, scales, block_scales, output)


@pytest.mark.parametrize("num_warps", [4, 8])
@pytest.mark.parametrize("outer_specialization", [False, True])
def test_manual_copy_ring_retains_order_in_pipelined_wgmma(num_warps, outer_specialization):
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required")
    generator = torch.Generator().manual_seed(173)
    weights = torch.randint(-2, 3, (2, 64, 4096), generator=generator).float() / 2
    activations = torch.randint(-2, 3, (2, 4096), generator=generator).float() / 2
    scales = torch.ones((2, 64))
    block_scales = (torch.arange(64).reshape(2, 32) % 3 + 1).float()
    expected = torch.zeros((2, 64))
    for block in range(32):
        product = (weights[:, :, block * 128:(block + 1) * 128] *
                   activations[:, None, block * 128:(block + 1) * 128]).sum(-1)
        expected += product * block_scales[:, block, None]
    args = (weights.to(torch.float8_e4m3fn).cuda(), activations.to(torch.float8_e4m3fn).cuda(), scales.cuda(),
            block_scales.cuda(), torch.empty((2, 64), device="cuda"))
    compiled = _manual_ring_wgmma.warmup(*args, outer_specialization, grid=(1, ), num_warps=num_warps)
    diagnostic = os.environ.get("FLAGMEGA_RING_DIAGNOSTIC_DIR")
    if diagnostic:
        root = Path(diagnostic) / f"warps-{num_warps}-outer-{outer_specialization}"
        root.mkdir(parents=True, exist_ok=True)
        for suffix in ("ttir", "ttgir", "llir", "ptx"):
            (root / f"kernel.{suffix}").write_text(compiled.asm[suffix])
    assert re.search(r"cp\.async\.wait_group\s+7\b",
                     compiled.asm["ptx"]), ("The explicitly submitted eight-group ring must retain its partial wait")
    for replay in range(32):
        _manual_ring_wgmma[(1, )](*args, outer_specialization, num_warps=num_warps)
        actual = args[-1].cpu()
        torch.testing.assert_close(actual, expected, rtol=0, atol=0,
                                   msg=f"manual ring mismatch with {num_warps} warps at replay {replay}")
