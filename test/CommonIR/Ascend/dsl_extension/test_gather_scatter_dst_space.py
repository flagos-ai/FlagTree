# Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
#
# Validation for the tile.load/tile.store GatherScatterView path via
# CommonIRToHIVM. A sparse dimension is addressed by a tensor of per-lane
# coordinates; every other dimension is addressed by a scalar and read
# contiguously. Out-of-bounds gather lanes take the view's padding value;
# out-of-bounds scatter lanes are dropped.

import torch
import torch_npu  # noqa: F401
import triton
import triton.language as tl
import triton.experimental.tle as tle


@triton.jit
def gather_load_kernel(src, indices, dst, n_elements, BLOCK_SIZE: tl.constexpr):
    block = tl.program_id(0)
    offsets = block * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    data_indices = tl.load(indices + offsets)
    src_gather = tle.make_gather_scatter_view(src, shape=[n_elements], strides=[1], tile=[BLOCK_SIZE], sparse_dim=[0])
    dst_view = tle.make_partition_view(dst, shape=[n_elements], strides=[1], tile=[BLOCK_SIZE])

    value = tle.tile_load(src_gather, index=(data_indices, ), dst_space=tle.language.dsa.ascend.UB)
    tle.tile_store(dst_view, value=value, index=(block, ), src_space=tle.language.dsa.ascend.UB)


@triton.jit
def scatter_store_kernel(src, indices, dst, n_elements, BLOCK_SIZE: tl.constexpr):
    block = tl.program_id(0)
    offsets = block * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    data_indices = tl.load(indices + offsets)
    src_view = tle.make_partition_view(src, shape=[n_elements], strides=[1], tile=[BLOCK_SIZE])
    dst_scatter = tle.make_gather_scatter_view(dst, shape=[n_elements], strides=[1], tile=[BLOCK_SIZE], sparse_dim=[0])

    value = tle.tile_load(src_view, index=(block, ), dst_space=tle.language.dsa.ascend.UB)
    tle.tile_store(dst_scatter, value=value, index=(data_indices, ), src_space=tle.language.dsa.ascend.UB)


@triton.jit
def gather_padding_kernel(src, indices, dst, src_elements, BLOCK_SIZE: tl.constexpr):
    offsets = tl.arange(0, BLOCK_SIZE)
    data_indices = tl.load(indices + offsets)
    src_gather = tle.make_gather_scatter_view(src, shape=[src_elements], strides=[1], tile=[BLOCK_SIZE], sparse_dim=[0],
                                              padding_value="nan", bounds_check=True)
    dst_view = tle.make_partition_view(dst, shape=[BLOCK_SIZE], strides=[1], tile=[BLOCK_SIZE])

    value = tle.tile_load(src_gather, index=(data_indices, ), dst_space=tle.language.dsa.ascend.UB)
    tle.tile_store(dst_view, value=value, index=(0, ), src_space=tle.language.dsa.ascend.UB)


@triton.jit
def scatter_bounds_kernel(src, indices, dst, dst_elements, BLOCK_SIZE: tl.constexpr):
    offsets = tl.arange(0, BLOCK_SIZE)
    data_indices = tl.load(indices + offsets)
    src_view = tle.make_partition_view(src, shape=[BLOCK_SIZE], strides=[1], tile=[BLOCK_SIZE])
    dst_scatter = tle.make_gather_scatter_view(dst, shape=[dst_elements], strides=[1], tile=[BLOCK_SIZE],
                                               sparse_dim=[0], bounds_check=True)

    value = tle.tile_load(src_view, index=(0, ), dst_space=tle.language.dsa.ascend.UB)
    tle.tile_store(dst_scatter, value=value, index=(data_indices, ), src_space=tle.language.dsa.ascend.UB)


@triton.jit
def gather_embedding_kernel(embed_table, token_ids, output, VOCAB, DIM: tl.constexpr, SEQ_LEN: tl.constexpr,
                            BLOCK_SEQ: tl.constexpr):
    # Each program gathers BLOCK_SEQ embedding rows: sparse_dim=0 (the row is
    # picked per lane by ids), dim 1 contiguous. index=(ids, 0) pairs the sparse
    # row dimension with a tensor index and the contiguous column dimension with
    # a scalar.
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SEQ + tl.arange(0, BLOCK_SEQ)
    ids = tl.load(token_ids + offsets)

    embed_gs = tle.make_gather_scatter_view(embed_table, shape=[VOCAB, DIM], strides=[DIM, 1], tile=[BLOCK_SEQ, DIM],
                                            sparse_dim=[0])
    gathered = tle.tile_load(embed_gs, index=(ids, 0), dst_space=tle.language.dsa.ascend.UB)

    out_view = tle.make_partition_view(output, shape=[SEQ_LEN, DIM], strides=[DIM, 1], tile=[BLOCK_SEQ, DIM])
    tle.tile_store(out_view, value=gathered, index=(pid, 0), src_space=tle.language.dsa.ascend.UB)


def test_gather_scatter_gather_load():
    n_elements = 1024
    block_size = 256
    src = torch.rand(n_elements, device="npu")
    indices = torch.randint(0, n_elements, (n_elements, ), dtype=torch.int32, device="npu")
    actual = torch.empty_like(src)

    gather_load_kernel[(triton.cdiv(n_elements, block_size), )](src, indices, actual, n_elements, BLOCK_SIZE=block_size)

    torch.testing.assert_close(actual, src[indices.long()])


def test_gather_scatter_scatter_store():
    n_elements = 1024
    block_size = 256
    src = torch.rand(n_elements, device="npu")
    indices = torch.randperm(n_elements, device="npu").to(torch.int32)
    actual = torch.zeros_like(src)
    expected = torch.zeros_like(src)
    expected[indices.long()] = src

    scatter_store_kernel[(triton.cdiv(n_elements, block_size), )](src, indices, actual, n_elements,
                                                                  BLOCK_SIZE=block_size)

    torch.testing.assert_close(actual, expected)


def test_gather_scatter_gather_padding():
    src = torch.arange(8, dtype=torch.float32, device="npu")
    indices = torch.tensor([0, 7, -1, 8], dtype=torch.int32, device="npu")
    actual = torch.empty(4, dtype=torch.float32, device="npu")

    gather_padding_kernel[(1, )](src, indices, actual, src.numel(), BLOCK_SIZE=4)

    expected = torch.tensor([0, 7, torch.nan, torch.nan], dtype=torch.float32, device="npu")
    torch.testing.assert_close(actual, expected, equal_nan=True)


def test_gather_scatter_scatter_bounds():
    src = torch.tensor([11, 23, 45, 67], dtype=torch.float32, device="npu")
    indices = torch.tensor([-1, 0, 7, 8], dtype=torch.int32, device="npu")
    actual = torch.zeros(8, dtype=torch.float32, device="npu")

    scatter_bounds_kernel[(1, )](src, indices, actual, actual.numel(), BLOCK_SIZE=4)

    expected = torch.zeros_like(actual)
    expected[0] = src[1]
    expected[7] = src[2]
    torch.testing.assert_close(actual, expected)


def test_gather_scatter_embedding_2d():
    vocab, dim, seq_len, block_seq = 128, 16, 64, 64
    embed_table = torch.rand((vocab, dim), device="npu")
    token_ids = torch.randint(0, vocab, (seq_len, ), dtype=torch.int32, device="npu")
    output = torch.zeros((seq_len, dim), device="npu")

    grid = (triton.cdiv(seq_len, block_seq), )
    gather_embedding_kernel[grid](embed_table, token_ids, output, vocab, DIM=dim, SEQ_LEN=seq_len, BLOCK_SEQ=block_seq)

    torch.testing.assert_close(output, embed_table[token_ids.long()])
