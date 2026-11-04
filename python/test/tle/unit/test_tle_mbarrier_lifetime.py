# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Owned TMA barriers close before another helper reuses shared storage."""

import re

import pytest
import torch
import triton
import triton.language as tl
import triton.experimental.tle.language as tle
from triton.flagmega.runtime.nvidia_tensor_map import encode_tma_descriptor


@triton.jit(noinline=True)
def _owned_copy(descriptor_pointer, output, width: tl.constexpr):
    destination = tle.gpu.alloc([8, width], tl.bfloat16, scope=tle.gpu.smem)
    complete = tle.gpu.alloc_barrier(expect_bytes=16 * width)
    descriptor = tle.gpu.reinterpret_tensor_map(descriptor_pointer, destination)
    tle.gpu.copy(descriptor, destination, [8, width], [0, 0], barrier=complete)
    tle.gpu.barrier_wait(complete, phaseIdx=0)
    row = tl.arange(0, 8)[:, None]
    column = tl.arange(0, width)[None, :]
    value = tl.load(tle.gpu.local_ptr(destination, (row, column), [8, width]))
    tl.store(output + row * width + column, value)


@triton.jit
def _repeat_owned_copies(table, small_output, large_output,
                         repeats: tl.constexpr):
    for iteration in tl.range(0, repeats):
        _owned_copy(table, small_output + tl.program_id(0) * 512, 64)
        _owned_copy(table + 128, large_output + tl.program_id(0) * 4096, 512)


@triton.jit(noinline=True)
def _borrowed_copy(descriptor_pointer, destination, complete, phase):
    descriptor = tle.gpu.reinterpret_tensor_map(descriptor_pointer, destination)
    tle.gpu.copy(descriptor, destination, [8, 64], [0, 0], barrier=complete)
    tle.gpu.barrier_wait(complete, phaseIdx=phase)


@triton.jit
def _loop_barrier_copy(table, output, repeats: tl.constexpr,
                       inner_owned: tl.constexpr):
    destination = tle.gpu.alloc([8, 64], tl.bfloat16, scope=tle.gpu.smem)
    if not inner_owned:
        complete = tle.gpu.alloc_barrier(expect_bytes=1024)
    for iteration in tl.range(0, repeats):
        if inner_owned:
            complete = tle.gpu.alloc_barrier(expect_bytes=1024)
            phase = 0
        else:
            phase = iteration % 2
        _borrowed_copy(table, destination, complete, phase)
        row = tl.arange(0, 8)[:, None]
        column = tl.arange(0, 64)[None, :]
        value = tl.load(tle.gpu.local_ptr(destination, (row, column), [8, 64]))
        tl.store(output + tl.program_id(0) * 512 + row * 64 + column, value)


def _descriptor_table(source):
    payload = bytearray()
    # Both copy widths use hardware boxes of 64 columns under B128 swizzling.
    # The 512-column destination is filled by eight such TMA transfers.
    for _ in range(2):
        payload.extend(encode_tma_descriptor(
            source.data_ptr(), 3, source.element_size(), 9,
            (8, 64), (8, 512), (512, 1), 0,
        ))
    table = torch.frombuffer(payload, dtype=torch.uint8).to(device="cuda")
    assert table.data_ptr() % 128 == 0
    return table


def _assert_barriers_closed(compiled):
    ptx = compiled.asm["ptx"]
    initialized = len(re.findall(r"mbarrier\.init\.shared::cta\.b64", ptx))
    invalidated = len(re.findall(r"mbarrier\.inval\.shared::cta\.b64", ptx))
    assert initialized > 0
    assert invalidated == initialized, (initialized, invalidated)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.require_tle("gpu.alloc_barrier", "gpu.reinterpret_tensor_map")
@pytest.mark.parametrize("warps", [4, 8])
@pytest.mark.parametrize("mode", ["different_helpers", "outer_owned", "inner_owned"])
def test_owned_mbarrier_storage_reuse(warps, mode):
    if torch.cuda.get_device_capability()[0] < 9:
        pytest.skip("TMA requires NVIDIA Hopper or later")
    source = ((torch.arange(4096, dtype=torch.int32, device="cuda") % 31) - 15)
    source = source.to(torch.bfloat16).reshape(8, 512)
    table = _descriptor_table(source)
    output = torch.empty((64, 8, 64), dtype=torch.bfloat16, device="cuda")
    expected = source[:, :64].expand(64, -1, -1)
    if mode == "different_helpers":
        large = torch.empty((64, 8, 512), dtype=torch.bfloat16, device="cuda")
        compiled = _repeat_owned_copies[(64,)](table, output, large, 32, num_warps=warps)
        torch.testing.assert_close(large, source.expand(64, -1, -1), rtol=0, atol=0)
    else:
        compiled = _loop_barrier_copy[(64,)](
            table, output, 32, mode == "inner_owned", num_warps=warps)
    torch.testing.assert_close(output, expected, rtol=0, atol=0)
    _assert_barriers_closed(compiled)
