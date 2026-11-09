# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.

import re

import pytest
import torch
import torch_npu  # noqa: F401
import triton
import triton.language as tl
import triton.language.extra.cann.extension as al

BLOCK = 32
N = 128


@triton.jit
def block_ptr_if_descriptor_kernel(a, b, out, choose_ptr, descriptor_ptr, BLOCK: tl.constexpr):
    choose = tl.load(choose_ptr)
    size_a = tl.load(descriptor_ptr)
    size_b = tl.load(descriptor_ptr + 1)
    stride_a = tl.load(descriptor_ptr + 2)
    stride_b = tl.load(descriptor_ptr + 3)
    # Block-pointer offsets are required to use Triton's 32-bit index type.
    # Keep the descriptor storage wide, but narrow the values at the frontend
    # boundary instead of changing the product's make_block_ptr contract.
    offset_a = tl.load(descriptor_ptr + 4).to(tl.int32)
    offset_b = tl.load(descriptor_ptr + 5).to(tl.int32)
    if choose != 0:
        pointer = tl.make_block_ptr(
            base=a + 3,
            shape=(size_a, ),
            strides=(stride_a, ),
            offsets=(offset_a, ),
            block_shape=(BLOCK, ),
            order=(0, ),
        )
    else:
        pointer = tl.make_block_ptr(
            base=b + 11,
            shape=(size_b, ),
            strides=(stride_b, ),
            offsets=(offset_b, ),
            block_shape=(BLOCK, ),
            order=(0, ),
        )
    value = tl.load(pointer, boundary_check=(0, ), padding_option="zero")
    tl.store(out + tl.arange(0, BLOCK), value)


@triton.jit
def block_ptr_for_descriptor_kernel(a, b, out, scalar_out, steps_ptr, descriptor_ptr, BLOCK: tl.constexpr):
    steps = tl.load(steps_ptr)
    size_a = tl.load(descriptor_ptr)
    size_b = tl.load(descriptor_ptr + 1)
    stride_a = tl.load(descriptor_ptr + 2)
    stride_b = tl.load(descriptor_ptr + 3)
    pointer = tl.make_block_ptr(
        base=a,
        shape=(size_a, ),
        strides=(stride_a, ),
        offsets=(1, ),
        block_shape=(BLOCK, ),
        order=(0, ),
    )
    ordinary_result = 17
    for i in tl.range(0, steps):
        if (i & 1) == 0:
            pointer = tl.advance(pointer, (2, ))
        else:
            pointer = tl.make_block_ptr(
                base=b,
                shape=(size_b, ),
                strides=(stride_b, ),
                offsets=(i + 3, ),
                block_shape=(BLOCK, ),
                order=(0, ),
            )
        ordinary_result = ordinary_result + i + 1
    value = tl.load(pointer, boundary_check=(0, ), padding_option="zero")
    tl.store(out + tl.arange(0, BLOCK), value)
    tl.store(scalar_out, ordinary_result)


@triton.jit
def block_ptr_while_descriptor_kernel(a, b, out, steps_ptr, switch_at_ptr, n: tl.constexpr, BLOCK: tl.constexpr):
    steps = tl.load(steps_ptr)
    switch_at = tl.load(switch_at_ptr)
    pointer = tl.make_block_ptr(
        base=a,
        shape=(n, ),
        strides=(1, ),
        offsets=(0, ),
        block_shape=(BLOCK, ),
        order=(0, ),
    )
    i = 0
    while i < steps:
        if i == switch_at:
            pointer = tl.make_block_ptr(
                base=b,
                shape=(n - 3, ),
                strides=(1, ),
                offsets=(1, ),
                block_shape=(BLOCK, ),
                order=(0, ),
            )
        else:
            pointer = tl.advance(pointer, (2, ))
        i += 1
    value = tl.load(pointer, boundary_check=(0, ), padding_option="zero")
    tl.store(out + tl.arange(0, BLOCK), value)


@triton.jit
def scalar_base_tensor_ptr_loop_kernel(x, out, steps_ptr, n: tl.constexpr, BLOCK: tl.constexpr):
    steps = tl.load(steps_ptr)
    lane = tl.arange(0, BLOCK)
    pointers = x + 3 + lane
    for _ in tl.range(0, steps):
        pointers = pointers + 2
    index = 3 + lane + 2 * steps
    value = tl.load(pointers, mask=index < n, other=0.0)
    tl.store(out + lane, value)


@triton.jit
def dense_splat_tensor_ptr_loop_kernel(x, out, steps_ptr, n: tl.constexpr, BLOCK: tl.constexpr):
    steps = tl.load(steps_ptr)
    lane = tl.arange(0, BLOCK)
    # Keep the loop delta tensor-valued so TTIR materializes a dense splat.
    delta = tl.full((BLOCK, ), BLOCK, tl.int32)
    pointers = x + lane
    for _ in tl.range(0, steps):
        pointers = pointers + delta
    index = lane + BLOCK * steps
    value = tl.load(pointers, mask=index < n, other=0.0)
    tl.store(out + lane, value)


@triton.jit
def dense_splat_affine_rank1_loop_kernel(x, out, steps_ptr, n: tl.constexpr, BLOCK: tl.constexpr, STRIDE: tl.constexpr):
    steps = tl.load(steps_ptr)
    lane = tl.arange(0, BLOCK)
    scale = tl.full((BLOCK, ), STRIDE, tl.int32)
    pointers = x + lane * scale
    for _ in tl.range(0, steps):
        pointers += 1
    index = lane * STRIDE + steps
    value = tl.load(pointers, mask=index < n, other=0.0)
    tl.store(out + lane, value)


@triton.jit
def dense_splat_affine_rank2_loop_kernel(x, out, steps_ptr, n: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr,
                                         ROW_STRIDE: tl.constexpr):
    steps = tl.load(steps_ptr)
    row = tl.arange(0, BM)[:, None]
    col = tl.arange(0, BN)[None, :]
    scale = tl.full((BM, BN), ROW_STRIDE, tl.int32)
    offsets = row * scale + col
    pointers = x + offsets
    for _ in tl.range(0, steps):
        pointers += 1
    value = tl.load(pointers, mask=offsets + steps < n, other=0.0)
    tl.store(out + row * BN + col, value)


@triton.jit
def broadcasted_pointer_rank2_loop_kernel(x, out, steps_ptr, BM: tl.constexpr, BN: tl.constexpr,
                                          ROW_STRIDE: tl.constexpr):
    steps = tl.load(steps_ptr)
    row = tl.arange(0, BM)[:, None]
    col = tl.arange(0, BN)[None, :]
    # Match grouped matmul's construction order: first create a lower-rank
    # pointer tensor, then broadcast it while adding the second axis.
    row_pointers = x + row * ROW_STRIDE
    pointers = row_pointers + col
    for _ in tl.range(0, steps):
        pointers += 1
    value = tl.load(pointers)
    tl.store(out + row * BN + col, value)


@triton.jit
def opaque_tensor_ptr_loop_kernel(a, b, out, steps_ptr, n: tl.constexpr, BLOCK: tl.constexpr):
    steps = tl.load(steps_ptr)
    lane = tl.arange(0, BLOCK)
    pointers = tl.where((lane & 1) == 0, a + lane, b + lane)
    for _ in tl.range(0, steps):
        pointers = pointers + 1
    index = lane + steps
    value = tl.load(pointers, mask=index < n, other=0.0)
    tl.store(out + lane, value)


@triton.jit
def scope_block_ptr_kernel(x, out, delta_ptr, n: tl.constexpr, BLOCK: tl.constexpr):
    delta = tl.load(delta_ptr).to(tl.int32)
    pointer = tl.make_block_ptr(
        base=x,
        shape=(n, ),
        strides=(1, ),
        offsets=(1, ),
        block_shape=(BLOCK, ),
        order=(0, ),
    )
    with al.scope(core_mode="vector", disable_auto_sync=True):
        pointer = tl.advance(pointer, (delta, ))
    value = tl.load(pointer, boundary_check=(0, ), padding_option="zero")
    tl.store(out + tl.arange(0, BLOCK), value)


@triton.jit
def scope_tensor_ptr_kernel(x, out, delta_ptr, n: tl.constexpr, BLOCK: tl.constexpr):
    lane = tl.arange(0, BLOCK)
    delta = tl.load(delta_ptr)
    pointers = x + lane
    with al.scope(core_mode="vector"):
        pointers = pointers + delta
    value = tl.load(pointers, mask=lane + delta < n, other=0.0)
    tl.store(out + lane, value)


def _inputs():
    a_cpu = torch.arange(2048, dtype=torch.float32)
    b_cpu = 100000.0 + torch.arange(2048, dtype=torch.float32) * 3.0
    return a_cpu, b_cpu, a_cpu.npu(), b_cpu.npu()


def _device_i32(value):
    return torch.tensor([value], dtype=torch.int32, device="npu")


def _slice(source, base_offset, logical_size, stride, offset):
    expected = torch.zeros(BLOCK, dtype=source.dtype)
    for lane in range(BLOCK):
        logical_index = offset + lane
        if 0 <= logical_index < logical_size:
            expected[lane] = source[base_offset + logical_index * stride]
    return expected


def _assert_output(actual, expected):
    torch.npu.synchronize()
    torch.testing.assert_close(actual.cpu(), expected, rtol=0, atol=0)


@pytest.mark.parametrize("choose", [0, 1])
def test_block_ptr_if_carries_complete_dynamic_descriptor(choose):
    a_cpu, b_cpu, a, b = _inputs()
    descriptor = torch.tensor([40, 29, 2, 3, 3, 5], dtype=torch.int64, device="npu")
    choose_ptr = _device_i32(choose)
    out = torch.empty(BLOCK, dtype=torch.float32, device="npu")

    block_ptr_if_descriptor_kernel[(1, )](a, b, out, choose_ptr, descriptor, BLOCK=BLOCK)

    if choose:
        expected = _slice(a_cpu, 3, 40, 2, 3)
    else:
        expected = _slice(b_cpu, 11, 29, 3, 5)
    _assert_output(out, expected)


@pytest.mark.parametrize("steps", [0, 2, 5])
def test_block_ptr_for_carries_changing_descriptor_and_ordinary_result(steps):
    a_cpu, b_cpu, a, b = _inputs()
    descriptor = torch.tensor([60, 55, 2, 3], dtype=torch.int64, device="npu")
    out = torch.empty(BLOCK, dtype=torch.float32, device="npu")
    scalar_out = torch.empty(1, dtype=torch.int32, device="npu")

    block_ptr_for_descriptor_kernel[(1, )](a, b, out, scalar_out, _device_i32(steps), descriptor, BLOCK=BLOCK)

    source, logical_size, stride, offset = a_cpu, 60, 2, 1
    for i in range(steps):
        if (i & 1) == 0:
            offset += 2
        else:
            source, logical_size, stride, offset = b_cpu, 55, 3, i + 3
    _assert_output(out, _slice(source, 0, logical_size, stride, offset))
    assert scalar_out.cpu().item() == 17 + steps * (steps + 1) // 2


@pytest.mark.parametrize("steps,switch_at", [(0, -1), (4, -1), (4, 2)])
def test_block_ptr_while_carries_descriptor(steps, switch_at):
    a_cpu, b_cpu, a, b = _inputs()
    out = torch.empty(BLOCK, dtype=torch.float32, device="npu")

    block_ptr_while_descriptor_kernel[(1, )](a, b, out, _device_i32(steps), _device_i32(switch_at), n=N, BLOCK=BLOCK)

    source, logical_size, offset = a_cpu, N, 0
    for i in range(steps):
        if i == switch_at:
            source, logical_size, offset = b_cpu, N - 3, 1
        else:
            offset += 2
    _assert_output(out, _slice(source, 0, logical_size, 1, offset))


@pytest.mark.parametrize("steps", [0, 2, 4])
def test_scalar_base_tensor_pointer_loop(steps):
    a_cpu, _, a, _ = _inputs()
    out = torch.empty(BLOCK, dtype=torch.float32, device="npu")

    scalar_base_tensor_ptr_loop_kernel[(1, )](a, out, _device_i32(steps), n=N, BLOCK=BLOCK)

    _assert_output(out, _slice(a_cpu, 0, N, 1, 3 + 2 * steps))


@pytest.mark.parametrize("steps", [0, 1, 3])
def test_dense_splat_tensor_pointer_loop(steps):
    a_cpu, _, a, _ = _inputs()
    out = torch.empty(BLOCK, dtype=torch.float32, device="npu")

    dense_splat_tensor_ptr_loop_kernel[(1, )](a, out, _device_i32(steps), n=N, BLOCK=BLOCK)

    _assert_output(out, _slice(a_cpu, 0, N, 1, BLOCK * steps))


@pytest.mark.parametrize("stride,steps", [(16, 0), (16, 3), (256, 0), (256, 3)])
def test_dense_splat_affine_rank1_loop(stride, steps):
    n = 8192
    source_cpu = torch.arange(n, dtype=torch.float32)
    source = source_cpu.npu()
    out = torch.empty(16, dtype=torch.float32, device="npu")

    dense_splat_affine_rank1_loop_kernel[(1, )](source, out, _device_i32(steps), n=n, BLOCK=16, STRIDE=stride)

    expected = source_cpu[torch.arange(16) * stride + steps]
    _assert_output(out, expected)


@pytest.mark.parametrize("steps", [0, 2])
def test_dense_splat_affine_rank2_loop(steps):
    bm, bn, row_stride = 8, 16, 64
    n = 1024
    source_cpu = torch.arange(n, dtype=torch.float32)
    source = source_cpu.npu()
    out = torch.empty((bm, bn), dtype=torch.float32, device="npu")

    dense_splat_affine_rank2_loop_kernel[(1, )](source, out, _device_i32(steps), n=n, BM=bm, BN=bn,
                                                ROW_STRIDE=row_stride)

    rows = torch.arange(bm)[:, None]
    cols = torch.arange(bn)[None, :]
    expected = source_cpu[rows * row_stride + cols + steps]
    _assert_output(out, expected)


@pytest.mark.parametrize("steps", [0, 2])
def test_broadcasted_pointer_rank2_loop(steps):
    bm, bn, row_stride = 8, 16, 64
    n = 1024
    source_cpu = torch.arange(n, dtype=torch.float32)
    source = source_cpu.npu()
    out = torch.empty((bm, bn), dtype=torch.float32, device="npu")

    compiled = broadcasted_pointer_rank2_loop_kernel[(1, )](
        source,
        out,
        _device_i32(steps),
        BM=bm,
        BN=bn,
        ROW_STRIDE=row_stride,
    )

    rows = torch.arange(bm)[:, None]
    cols = torch.arange(bn)[None, :]
    expected = source_cpu[rows * row_stride + cols + steps]
    _assert_output(out, expected)

    # Before descriptor propagation, the pointer broadcast becomes an opaque
    # tensor<8x16xi32> loop carrier even though both axes are affine.
    adapter = compiled.asm["ttadapter"]
    assert not re.search(r"scf\.for[^\n]*-> \(tensor<8x16xi32>\)", adapter)
    assert re.search(r"scf\.for[^\n]*-> \(i32\)", adapter)


@pytest.mark.parametrize("steps", [0, 2, 4])
def test_opaque_tensor_pointer_loop(steps):
    a_cpu, b_cpu, a, b = _inputs()
    out = torch.empty(BLOCK, dtype=torch.float32, device="npu")

    opaque_tensor_ptr_loop_kernel[(1, )](a, b, out, _device_i32(steps), n=N, BLOCK=BLOCK)

    expected = torch.zeros(BLOCK, dtype=torch.float32)
    for lane in range(BLOCK):
        index = lane + steps
        if index < N:
            expected[lane] = a_cpu[index] if (lane & 1) == 0 else b_cpu[index]
    _assert_output(out, expected)


@pytest.mark.parametrize("delta", [0, 3, 37])
def test_scope_block_pointer_result(delta):
    a_cpu, _, a, _ = _inputs()
    out = torch.empty(BLOCK, dtype=torch.float32, device="npu")

    scope_block_ptr_kernel[(1, )](a, out, _device_i32(delta), n=N, BLOCK=BLOCK)

    _assert_output(out, _slice(a_cpu, 0, N, 1, 1 + delta))


@pytest.mark.parametrize("delta", [0, 2, 5])
def test_scope_tensor_pointer_result(delta):
    a_cpu, _, a, _ = _inputs()
    out = torch.empty(BLOCK, dtype=torch.float32, device="npu")

    scope_tensor_ptr_kernel[(1, )](a, out, _device_i32(delta), n=N, BLOCK=BLOCK)

    _assert_output(out, _slice(a_cpu, 0, N, 1, delta))
