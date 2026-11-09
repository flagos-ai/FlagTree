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

import struct

import torch
import torch_npu
import triton
import triton.language as tl
import pytest


def _write_u32_le(buffer, byte_offsets, values):
    for byte_offset, value in zip(byte_offsets, values):
        for byte_index, byte_value in enumerate(struct.pack("<I", value)):
            buffer[byte_offset + byte_index] = byte_value


def _write_bf16_le(buffer, byte_offset, values):
    for index, value in enumerate(values):
        fp32_bits = struct.unpack("<I", struct.pack("<f", value))[0]
        bf16_bits = fp32_bits >> 16
        buffer[byte_offset + index * 2] = bf16_bits & 0xFF
        buffer[byte_offset + index * 2 + 1] = bf16_bits >> 8


@triton.jit
def _paged_scale_load(
    kv_ptr,
    block_tables_ptr,
    out_ptr,
    n_positions,
    stride_kvblk,
    stride_kvpos,
    stride_kvbyte,
    DIM: tl.constexpr,
    CACHE_BLOCK_SIZE: tl.constexpr,
    BLOCK: tl.constexpr,
):
    positions = tl.arange(0, BLOCK)
    mask = positions < n_positions
    logical_block = positions // CACHE_BLOCK_SIZE
    intra_block_pos = positions % CACHE_BLOCK_SIZE
    physical_block = tl.load(block_tables_ptr + logical_block, mask=mask, other=0)
    kv_base = (physical_block * stride_kvblk + intra_block_pos * stride_kvpos)
    scale_addr = kv_base + DIM * stride_kvbyte
    scale_ptr = (kv_ptr + scale_addr).to(tl.pointer_type(tl.uint32, 1), bitcast=True)
    scale_u32 = tl.load(scale_ptr, mask=mask, other=0)
    tl.store(out_ptr + positions, scale_u32.to(tl.int32), mask=mask)


@triton.jit
def _multi_addptr_bf16_store(
    values_ptr,
    cache_ptr,
    block_idx,
    token_pos,
    block_stride,
    N: tl.constexpr,
    TOKEN_BYTES: tl.constexpr,
    BF16_OFFSET: tl.constexpr,
    BLOCK: tl.constexpr,
):
    lanes = tl.arange(0, BLOCK)
    block_base = cache_ptr + block_idx * tl.multiple_of(block_stride, 16)
    token_base = block_base + token_pos * TOKEN_BYTES
    bf16_ptr = (token_base + BF16_OFFSET).to(tl.pointer_type(tl.bfloat16))
    mask = lanes < N
    values = tl.load(values_ptr + lanes, mask=mask, other=0.0)
    tl.store(bf16_ptr + lanes, values, mask=mask)


@triton.jit
def _loop_bf16_load(
    src,
    out,
    count,
    NOPE_DIM: tl.constexpr,
    TOKEN_BYTES: tl.constexpr,
    ROPE_DIM: tl.constexpr,
    BLOCK: tl.constexpr,
    NUM_WORKERS: tl.constexpr,
):
    worker = tl.program_id(0)
    lanes = tl.arange(0, BLOCK)
    mask = lanes < ROPE_DIM
    for token_idx in range(worker, count, NUM_WORKERS):
        token_data = src + token_idx * TOKEN_BYTES
        bf16_ptr = (token_data + NOPE_DIM).to(tl.pointer_type(tl.bfloat16))
        values = tl.load(bf16_ptr + lanes, mask=mask, other=0.0)
        tl.store(out + token_idx * ROPE_DIM + lanes, values, mask=mask)


@triton.jit
def _tensor_multi_addptr_bf16_load(
    cache_ptr,
    block_ids_ptr,
    positions_ptr,
    out_ptr,
    block_stride,
    token_bytes,
    BF16_OFFSET: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    cols = tl.arange(0, BLOCK_N)
    block_ids = tl.load(block_ids_ptr + cols)
    positions = tl.load(positions_ptr + cols)

    block_offsets = block_ids.to(tl.int64) * block_stride
    block_ptrs = cache_ptr + block_offsets
    token_ptrs = block_ptrs + positions * token_bytes
    rope_bytes = token_ptrs + BF16_OFFSET
    rope_ptrs = rope_bytes.to(tl.pointer_type(tl.bfloat16), bitcast=True)

    rows = tl.arange(0, BLOCK_M)
    values = tl.load(rope_ptrs[None, :] + rows[:, None])
    output_offsets = rows[:, None] * BLOCK_N + cols[None, :]
    tl.store(out_ptr + output_offsets, values)


@pytest.mark.skip(reason="The case is not supported on Ascend 950, skipping for now. Will be fixed in future.")
def test_pointer_bitcast_paged_scale_tensor_offset():
    cache_block_size = 4
    dim = 8
    record_bytes = dim + 4
    stride_kvpos = record_bytes
    stride_kvblk = cache_block_size * record_bytes
    n_positions = 8
    block_table = [2, 0]
    values = [100 + index for index in range(n_positions)]

    host = torch.zeros(3 * stride_kvblk, dtype=torch.uint8)
    byte_offsets = []
    for position in range(n_positions):
        logical_block = position // cache_block_size
        intra_block_pos = position % cache_block_size
        physical_block = block_table[logical_block]
        byte_offsets.append(physical_block * stride_kvblk + intra_block_pos * stride_kvpos + dim)
    _write_u32_le(host, byte_offsets, values)

    src = host.npu()
    block_tables = torch.tensor(block_table, dtype=torch.int32).npu()
    out = torch.empty((n_positions, ), dtype=torch.int32, device="npu")
    _paged_scale_load[(1, )](
        src,
        block_tables,
        out,
        n_positions,
        stride_kvblk,
        stride_kvpos,
        1,
        DIM=dim,
        CACHE_BLOCK_SIZE=cache_block_size,
        BLOCK=n_positions,
    )

    torch.testing.assert_close(out.cpu(), torch.tensor(values, dtype=torch.int32), rtol=0, atol=0)


def test_pointer_bitcast_scalar_multi_addptr_store():
    block_idx = 1
    token_pos = 2
    block_stride = 2048
    token_bytes = 576
    bf16_offset = 448
    values = [float(index + 1) for index in range(16)]
    byte_offset = (block_idx * block_stride + token_pos * token_bytes + bf16_offset)

    expected = torch.zeros(byte_offset + len(values) * 2 + 64, dtype=torch.uint8)
    _write_bf16_le(expected, byte_offset, values)
    cache = torch.zeros_like(expected).npu()
    source = torch.tensor(values, dtype=torch.float32).to(torch.bfloat16).npu()
    _multi_addptr_bf16_store[(1, )](
        source,
        cache,
        block_idx,
        token_pos,
        block_stride,
        N=len(values),
        TOKEN_BYTES=token_bytes,
        BF16_OFFSET=bf16_offset,
        BLOCK=32,
    )

    torch.testing.assert_close(cache.cpu(), expected, rtol=0, atol=0)


def test_pointer_bitcast_inside_dynamic_loop():
    token_count = 5
    nope_dim = 448
    rope_dim = 8
    token_bytes = 576
    num_workers = 2
    host = torch.zeros(token_count * token_bytes, dtype=torch.uint8)
    expected_values = []
    for token_idx in range(token_count):
        values = [float(token_idx * rope_dim + index + 1) for index in range(rope_dim)]
        expected_values.extend(values)
        _write_bf16_le(host, token_idx * token_bytes + nope_dim, values)

    src = host.npu()
    out = torch.empty((token_count * rope_dim, ), dtype=torch.bfloat16, device="npu")
    _loop_bf16_load[(num_workers, )](
        src,
        out,
        token_count,
        NOPE_DIM=nope_dim,
        TOKEN_BYTES=token_bytes,
        ROPE_DIM=rope_dim,
        BLOCK=16,
        NUM_WORKERS=num_workers,
    )

    expected = torch.tensor(expected_values, dtype=torch.float32).to(torch.bfloat16)
    torch.testing.assert_close(out.cpu(), expected, rtol=0, atol=0)


@pytest.mark.skip(reason="The case is not supported on Ascend 950, skipping for now. Will be fixed in future.")
def test_pointer_bitcast_tensor_multi_addptr_broadcast_load():
    block_m = 4
    block_n = 32
    block_stride = 2048
    token_bytes = 576
    bf16_offset = 448

    block_ids = torch.arange(block_n, dtype=torch.int32)
    positions = torch.arange(block_n, dtype=torch.int32) % 3
    host = torch.zeros(block_n * block_stride, dtype=torch.uint8)
    expected = torch.empty((block_m, block_n), dtype=torch.bfloat16)
    for col in range(block_n):
        values = [float(col * block_m + row + 1) for row in range(block_m)]
        byte_offset = (block_ids[col].item() * block_stride + positions[col].item() * token_bytes + bf16_offset)
        _write_bf16_le(host, byte_offset, values)
        expected[:, col] = torch.tensor(values, dtype=torch.bfloat16)

    cache = host.npu()
    out = torch.empty((block_m, block_n), dtype=torch.bfloat16, device="npu")
    _tensor_multi_addptr_bf16_load[(1, )](
        cache,
        block_ids.npu(),
        positions.npu(),
        out,
        block_stride,
        token_bytes,
        BF16_OFFSET=bf16_offset,
        BLOCK_M=block_m,
        BLOCK_N=block_n,
    )

    torch.testing.assert_close(out.cpu(), expected, rtol=0, atol=0)
