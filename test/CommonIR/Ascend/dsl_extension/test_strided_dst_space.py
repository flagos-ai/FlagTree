# Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
#
# Validation for the tile.load/tile.store StridedView path with an explicit
# dst_space/src_space argument (tile.load/tile.store -> CommonIRToHIVM ->
# hivm.load/hivm.store). StridedView differs from PartitionView only in the
# traversal stride, so these cases exercise traversal > tile (gapped windows,
# with a tail window running past the signal end) and traversal < tile
# (overlapping windows packed into partition slots).

import torch
import torch_npu  # noqa: F401
import triton
import triton.language as tl
import triton.experimental.tle as tle


@triton.jit
def strided_gap_ub_kernel(src, dst, signal_len, WINDOW: tl.constexpr, STRIDE: tl.constexpr):
    # Reference the space inline (like native_matmul) so the JIT does not treat
    # it as a mutable module global.
    pid = tl.program_id(0)
    src_view = tle.make_strided_view(src, shape=[signal_len], strides=[1], tile=[WINDOW], traversal_strides=[STRIDE])
    dst_view = tle.make_strided_view(dst, shape=[signal_len], strides=[1], tile=[WINDOW], traversal_strides=[STRIDE])

    value = tle.tile_load(src_view, index=(pid, ), dst_space=tle.language.dsa.ascend.UB)
    tle.tile_store(dst_view, value=value, index=(pid, ), src_space=tle.language.dsa.ascend.UB)


@triton.jit
def strided_gap_default_kernel(src, dst, signal_len, WINDOW: tl.constexpr, STRIDE: tl.constexpr):
    # dst_space/src_space omitted -> defaults to UB.
    pid = tl.program_id(0)
    src_view = tle.make_strided_view(src, shape=[signal_len], strides=[1], tile=[WINDOW], traversal_strides=[STRIDE])
    dst_view = tle.make_strided_view(dst, shape=[signal_len], strides=[1], tile=[WINDOW], traversal_strides=[STRIDE])

    value = tle.tile_load(src_view, index=(pid, ))
    tle.tile_store(dst_view, value=value, index=(pid, ))


@triton.jit
def strided_overlap_pack_kernel(src, out, signal_len, out_len, WINDOW: tl.constexpr, STRIDE: tl.constexpr):
    # Overlapping strided read windows (STRIDE < WINDOW) packed into disjoint
    # partition slots on the store side, so overlap is exercised on LOAD only.
    pid = tl.program_id(0)
    src_view = tle.make_strided_view(src, shape=[signal_len], strides=[1], tile=[WINDOW], traversal_strides=[STRIDE])
    out_view = tle.make_partition_view(out, shape=[out_len], strides=[1], tile=[WINDOW])

    value = tle.tile_load(src_view, index=(pid, ))
    tle.tile_store(out_view, value=value, index=(pid, ))


def _run_gap(kernel):
    signal_len = 4000
    WINDOW, STRIDE = 1024, 1536
    num_windows = 3
    src = torch.rand((signal_len, ), device="npu")
    dst = torch.zeros_like(src)
    expected = torch.zeros_like(src)
    for w in range(num_windows):
        start = w * STRIDE
        valid = max(0, min(WINDOW, signal_len - start))
        expected[start:start + valid] = src[start:start + valid]
    kernel[(num_windows, )](src, dst, signal_len, WINDOW=WINDOW, STRIDE=STRIDE)
    torch.testing.assert_close(dst, expected)


def _run_overlap():
    signal_len = 4096
    WINDOW, STRIDE = 1024, 256
    num_windows = (signal_len - WINDOW) // STRIDE + 1
    out_len = num_windows * WINDOW
    src = torch.rand((signal_len, ), device="npu")
    out = torch.zeros((out_len, ), device="npu")
    expected = torch.zeros((out_len, ), device="npu")
    for w in range(num_windows):
        start = w * STRIDE
        expected[w * WINDOW:(w + 1) * WINDOW] = src[start:start + WINDOW]
    strided_overlap_pack_kernel[(num_windows, )](src, out, signal_len, out_len, WINDOW=WINDOW, STRIDE=STRIDE)
    torch.testing.assert_close(out, expected)


def test_strided_gap_ub_explicit():
    _run_gap(strided_gap_ub_kernel)


def test_strided_gap_default_is_ub():
    _run_gap(strided_gap_default_kernel)


def test_strided_overlap_load():
    _run_overlap()


@triton.jit
def strided_padding_tail_kernel(src, dst, signal_len, dst_len, WINDOW: tl.constexpr, STRIDE: tl.constexpr):
    # StridedView padding: overlapping windows (STRIDE < WINDOW) whose last window
    # runs past the signal end. The out-of-bounds tail of that window must be
    # filled with the view's padding_value ("-inf"). Each window is packed into a
    # disjoint partition slot on the store side so the padded tail is observable.
    pid = tl.program_id(0)
    src_view = tle.make_strided_view(src, shape=[signal_len], strides=[1], tile=[WINDOW], traversal_strides=[STRIDE],
                                     padding_value="-inf")
    dst_view = tle.make_partition_view(dst, shape=[dst_len], strides=[1], tile=[WINDOW])

    value = tle.tile_load(src_view, index=(pid, ), dst_space=tle.language.dsa.ascend.UB)
    tle.tile_store(dst_view, value=value, index=(pid, ), src_space=tle.language.dsa.ascend.UB)


def test_strided_padding_tail():
    signal_len = 20
    WINDOW, STRIDE = 8, 6
    num_windows = 4  # window origins 0, 6, 12, 18; the last runs past signal_len=20
    dst_len = num_windows * WINDOW
    src = torch.arange(signal_len, dtype=torch.float32, device="npu")
    dst = torch.empty((dst_len, ), dtype=torch.float32, device="npu")

    expected = torch.full((dst_len, ), -torch.inf, dtype=torch.float32, device="npu")
    for w in range(num_windows):
        start = w * STRIDE
        valid = max(0, min(WINDOW, signal_len - start))
        if valid > 0:
            expected[w * WINDOW:w * WINDOW + valid] = src[start:start + valid]

    strided_padding_tail_kernel[(num_windows, )](src, dst, signal_len, dst_len, WINDOW=WINDOW, STRIDE=STRIDE)
    torch.testing.assert_close(dst, expected)
