# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Raw TMA destinations retain their shape across branches and function calls."""

import pytest
import torch
import triton
import triton.language as tl
import triton.experimental.tle.language as tle
from triton._filecheck import run_parser
from triton.backends.compiler import GPUTarget
from triton.flagmega.runtime.nvidia_tensor_map import encode_tma_descriptor


@triton.jit(noinline=True)
def _copy_raw_descriptor(pointer, destination):
    descriptor = tle.gpu.reinterpret_tensor_map(pointer, destination)
    complete = tle.gpu.alloc_barrier(expect_bytes=128)
    tle.gpu.copy(descriptor, destination, [8, 8], [0, 0], barrier=complete)
    tle.gpu.barrier_wait(complete, phaseIdx=0)


@triton.jit
def _parse_destination_shape(MODE: tl.constexpr):
    pointer = tl.full((), 0, tl.uint64).to(tl.pointer_type(tl.uint8))
    destination = tle.gpu.alloc([8, 8], tl.float16, scope=tle.gpu.smem)
    if MODE == 1:
        if tl.program_id(0) != 0:
            pointer += 128
    if MODE == 2:
        _copy_raw_descriptor(pointer, destination)
    else:
        descriptor = tle.gpu.reinterpret_tensor_map(pointer, destination)
        tle.gpu.copy(descriptor, destination, [8, 8], [0, 0])


@pytest.mark.require_tle("gpu.reinterpret_tensor_map")
@pytest.mark.parametrize("mode", [0, 1, 2], ids=["direct", "runtime_branch", "callee_argument"])
def test_raw_tma_destination_shape_survives_scope_cloning(mode):
    module = run_parser(
        _parse_destination_shape,
        kwargs={"MODE": mode, "num_warps": 4},
        target=GPUTarget("cuda", 90, 32),
    )
    source = module.str_nodebug()
    assert "!tt.tensordesc<tensor<8x8xf16>>" in source
    assert source.count("ttg.tma_copy") == 1


@triton.jit
def _copy_descriptor_table(table, output, BRANCH: tl.constexpr, CALLEE: tl.constexpr):
    owner = tl.program_id(0)
    destination = tle.gpu.alloc([8, 8], tl.float16, scope=tle.gpu.smem)
    pointer = table + owner * 128
    if BRANCH:
        pointer = table
        if owner != 0:
            pointer = table + 128
    tle.gpu.tensor_map_fenceproxy_acquire(pointer)
    if CALLEE:
        _copy_raw_descriptor(pointer, destination)
    else:
        descriptor = tle.gpu.reinterpret_tensor_map(pointer, destination)
        tle.gpu.copy(descriptor, destination, [8, 8], [0, 0])
    rows = tl.arange(0, 8)[:, None]
    columns = tl.arange(0, 8)[None, :]
    values = tl.load(tle.gpu.local_ptr(destination, (rows, columns), [8, 8]))
    tl.store(output + owner * 64 + rows * 8 + columns, values)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.require_tle("gpu.reinterpret_tensor_map")
@pytest.mark.parametrize("branch", [False, True])
@pytest.mark.parametrize("callee", [False, True])
def test_raw_tma_scope_cloning_copies_selected_descriptor(branch, callee):
    if torch.cuda.get_device_capability()[0] < 9:
        pytest.skip("TMA requires NVIDIA Hopper or later")
    source = torch.arange(128, dtype=torch.float16, device="cuda").reshape(2, 8, 8)
    payload = bytearray()
    for owner in range(2):
        payload.extend(encode_tma_descriptor(
            source[owner].data_ptr(), 0, source.element_size(), 6,
            (8, 8), (8, 8), (8, 1), 0,
        ))
    table = torch.frombuffer(payload, dtype=torch.uint8).to(device="cuda")
    assert table.data_ptr() % 128 == 0
    output = torch.empty_like(source)
    _copy_descriptor_table[(2,)](table, output, branch, callee, num_warps=4)
    torch.testing.assert_close(output, source, rtol=0, atol=0)
