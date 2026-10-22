# Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
#
# Validation for the tile.load/tile.store PartitionView path with an explicit
# dst_space/src_space argument (tile.load/tile.store -> CommonIRToHIVM ->
# hivm.load/hivm.store).

import torch
import torch_npu  # noqa: F401
import triton
import triton.language as tl
import triton.experimental.tle as tle


@triton.jit
def partition_dst_space_ub_kernel(src, dst, rows, cols, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):
    # Reference the space inline (like native_matmul) so the JIT does not treat
    # it as a mutable module global.
    row_block = tl.program_id(0)
    col_block = tl.program_id(1)
    src_view = tle.make_partition_view(src, shape=[rows, cols], strides=[cols, 1], tile=[BLOCK_M, BLOCK_N])
    dst_view = tle.make_partition_view(dst, shape=[rows, cols], strides=[cols, 1], tile=[BLOCK_M, BLOCK_N])

    value = tle.tile_load(src_view, index=(row_block, col_block), dst_space=tle.language.dsa.ascend.UB)
    tle.tile_store(dst_view, value=value, index=(row_block, col_block), src_space=tle.language.dsa.ascend.UB)


@triton.jit
def partition_default_space_kernel(src, dst, rows, cols, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):
    # dst_space/src_space omitted -> defaults to UB.
    row_block = tl.program_id(0)
    col_block = tl.program_id(1)
    src_view = tle.make_partition_view(src, shape=[rows, cols], strides=[cols, 1], tile=[BLOCK_M, BLOCK_N])
    dst_view = tle.make_partition_view(dst, shape=[rows, cols], strides=[cols, 1], tile=[BLOCK_M, BLOCK_N])

    value = tle.tile_load(src_view, index=(row_block, col_block))
    tle.tile_store(dst_view, value=value, index=(row_block, col_block))


def _run(kernel):
    rows, cols = 128, 128
    block_m, block_n = 64, 64
    src = torch.rand((rows, cols), device="npu")
    actual = torch.zeros_like(src)
    grid = (triton.cdiv(rows, block_m), triton.cdiv(cols, block_n))
    kernel[grid](src, actual, rows, cols, BLOCK_M=block_m, BLOCK_N=block_n)
    torch.testing.assert_close(actual, src)


def test_partition_dst_space_ub_explicit():
    _run(partition_dst_space_ub_kernel)


def test_partition_dst_space_default_is_ub():
    _run(partition_default_space_kernel)


@triton.jit
def partition_padding_ub_kernel(src, dst, src_elements, dst_elements, BLOCK_SIZE: tl.constexpr):
    # Padding on the tile.load/tile.store path: the trailing partition slot runs
    # past the source signal, so the out-of-bounds lanes must be pre-filled with
    # the view's padding_value ("-inf") instead of leaking stale UB data.
    block = tl.program_id(0)
    src_view = tle.make_partition_view(src, shape=[src_elements], strides=[1], tile=[BLOCK_SIZE], padding_value="-inf")
    dst_view = tle.make_partition_view(dst, shape=[dst_elements], strides=[1], tile=[BLOCK_SIZE])

    value = tle.tile_load(src_view, index=(block, ), dst_space=tle.language.dsa.ascend.UB)
    tle.tile_store(dst_view, value=value, index=(block, ), src_space=tle.language.dsa.ascend.UB)


@triton.jit
def partition_padding_2d_kernel(src, dst, sh, sw, dh, dw, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):
    # 2-D padding on the tile.load/tile.store gather path. The source's logical
    # valid region is [sh, sw]; a 2x2 grid of [BLOCK_M, BLOCK_N] windows covers
    # [0, 2*BLOCK_M) x [0, 2*BLOCK_N). Any lane past sh (rows) or sw (cols) is
    # out of bounds and must be padded with -inf. dst is [dh, dw] = the full
    # 2x2 tile grid, so every window is in-bounds on the store side and the
    # padded tile is written out verbatim (observable).
    row_block = tl.program_id(0)
    col_block = tl.program_id(1)
    src_view = tle.make_partition_view(src, shape=[sh, sw], strides=[sw, 1], tile=[BLOCK_M, BLOCK_N],
                                       padding_value="-inf")
    dst_view = tle.make_partition_view(dst, shape=[dh, dw], strides=[dw, 1], tile=[BLOCK_M, BLOCK_N])

    value = tle.tile_load(src_view, index=(row_block, col_block), dst_space=tle.language.dsa.ascend.UB)
    tle.tile_store(dst_view, value=value, index=(row_block, col_block), src_space=tle.language.dsa.ascend.UB)


@triton.jit
def partition_padding_3d_kernel(src, dst, sd, sh, sw, dd, dh, dw, BLOCK_D: tl.constexpr, BLOCK_M: tl.constexpr,
                                BLOCK_N: tl.constexpr):
    # 3-D padding on the tile.load/tile.store gather path. The source's logical
    # valid region is [sd, sh, sw]; a 2x2x2 grid of [BLOCK_D, BLOCK_M, BLOCK_N]
    # windows covers [0,2*BLOCK_D) x [0,2*BLOCK_M) x [0,2*BLOCK_N). Any lane past
    # sd (depth), sh (rows) or sw (cols) is out of bounds and must be padded with
    # -inf. dst is [dd, dh, dw] = the full 2x2x2 tile grid, so every window is
    # in-bounds on the store side and the padded tile is written out verbatim
    # (observable). This exercises the rank-agnostic element-wise gather
    # (emitPaddedGatherLoad) at rank 3: it nests one scf.for per tile dim, so the
    # only thing that changes vs the 1-D/2-D cases is loop-nest depth.
    d_block = tl.program_id(0)
    row_block = tl.program_id(1)
    col_block = tl.program_id(2)
    src_view = tle.make_partition_view(src, shape=[sd, sh, sw], strides=[sh * sw, sw, 1],
                                       tile=[BLOCK_D, BLOCK_M, BLOCK_N], padding_value="-inf")
    dst_view = tle.make_partition_view(dst, shape=[dd, dh, dw], strides=[dh * dw, dw, 1],
                                       tile=[BLOCK_D, BLOCK_M, BLOCK_N])

    value = tle.tile_load(src_view, index=(d_block, row_block, col_block), dst_space=tle.language.dsa.ascend.UB)
    tle.tile_store(dst_view, value=value, index=(d_block, row_block, col_block), src_space=tle.language.dsa.ascend.UB)


def test_partition_dst_space_padding_3d():
    # Valid src region 3x3x3; tile 2x2x2; 2x2x2 window grid covers [0,4)^3, so the
    # out-of-bounds region (depth/row/col >= 3) is padding. dst 4x4x4 is fully
    # in-bounds so the padded region is observable. Confirms the element-wise
    # gather pads correctly at rank 3, not just rank 1/2.
    sd, sh, sw = 3, 3, 3
    block_d, block_m, block_n = 2, 2, 2
    dd, dh, dw = 4, 4, 4
    n = sd * sh * sw
    src = torch.arange(n, dtype=torch.float32, device="npu").reshape(sd, sh, sw)
    actual = torch.full((dd, dh, dw), 123.0, dtype=torch.float32, device="npu")

    grid = (dd // block_d, dh // block_m, dw // block_n)
    partition_padding_3d_kernel[grid](src, actual, sd, sh, sw, dd, dh, dw, BLOCK_D=block_d, BLOCK_M=block_m,
                                      BLOCK_N=block_n)

    expected = torch.full((dd, dh, dw), -torch.inf, device="npu")
    expected[:sd, :sh, :sw] = src
    torch.testing.assert_close(actual, expected)


def test_partition_dst_space_padding_2d():
    # Valid src region 6x6; tile 4x4; 2x2 window grid covers [0,8)x[0,8), so the
    # L-shaped tail (rows >= 6 or cols >= 6) is padding. dst 8x8 is fully
    # in-bounds so the padded tail is observable. This exercises the multi-dim
    # element-wise gather (emitPaddedGatherLoad), where the pad value is the
    # scf.if else-yield and cannot be dead-code-eliminated by the backend.
    sh, sw = 6, 6
    block_m, block_n = 4, 4
    dh, dw = 8, 8
    src = torch.arange(sh * sw, dtype=torch.float32, device="npu").reshape(sh, sw)
    actual = torch.full((dh, dw), 123.0, dtype=torch.float32, device="npu")

    grid = (dh // block_m, dw // block_n)
    partition_padding_2d_kernel[grid](src, actual, sh, sw, dh, dw, BLOCK_M=block_m, BLOCK_N=block_n)

    expected = torch.full((dh, dw), -torch.inf, device="npu")
    expected[:sh, :sw] = src
    torch.testing.assert_close(actual, expected)


def test_partition_dst_space_padding():
    # src has 10 valid elements; block_size 8 with 2 windows covers [0,16), so the
    # second window's lanes [10,16) are padding. dst is 16 wide (fully in-bounds on
    # the store side) so the padded tail is written out verbatim and observable.
    src_elements = 10
    block_size = 8
    dst_elements = 16
    src = torch.arange(src_elements, dtype=torch.float32, device="npu")
    actual = torch.empty(dst_elements, dtype=torch.float32, device="npu")

    partition_padding_ub_kernel[(2, )](src, actual, src_elements, dst_elements, BLOCK_SIZE=block_size)

    expected = torch.full_like(actual, -torch.inf)
    expected[:src_elements] = src
    torch.testing.assert_close(actual, expected)


def test_partition_dst_space_mixed_grid():
    # 10x10 valid region, tile 4x4, 3x3 grid covers [0,12)x[0,12): the top-left
    # 2x2 block of tiles is fully in bounds, the last row/column of tiles runs
    # past the valid region. Both the whole-tile DMA and the element-wise
    # boundary path run within the same kernel launch, so this exercises the
    # runtime branch picked per tile rather than per kernel.
    sh, sw = 10, 10
    block_m, block_n = 4, 4
    dh, dw = 12, 12
    src = torch.arange(sh * sw, dtype=torch.float32, device="npu").reshape(sh, sw)
    actual = torch.full((dh, dw), 123.0, dtype=torch.float32, device="npu")

    grid = (dh // block_m, dw // block_n)
    partition_padding_2d_kernel[grid](src, actual, sh, sw, dh, dw, BLOCK_M=block_m, BLOCK_N=block_n)

    expected = torch.full((dh, dw), -torch.inf, device="npu")
    expected[:sh, :sw] = src
    torch.testing.assert_close(actual, expected)
