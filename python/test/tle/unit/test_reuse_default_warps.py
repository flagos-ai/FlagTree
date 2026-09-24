import re

import pytest
import torch
import triton
import triton.language as tl
import triton.experimental.tle.language as tle


@pytest.fixture(scope="module", autouse=True)
def _require_nvidia_gpu():
    if not torch.cuda.is_available():
        pytest.skip("requires a CUDA GPU")
    target = triton.runtime.driver.active.get_current_target()
    if target.backend != "cuda" or int(target.arch) < 80:
        pytest.skip("requires NVIDIA Ampere or newer")


@triton.constexpr_function
def _layout(is_mma, row_warps, col_warps):
    if is_mma:
        return tle.gpu.MmaEncoding([2, 0], [row_warps, col_warps], [16, 8])
    return tle.gpu.BlockEncoding([4, 1], [4, 8], [row_warps, col_warps], [1, 0])


@triton.jit
def _transform_tile(tile, out, IS_MMA: tl.constexpr, PARTITION: tl.constexpr, WORKER_WARPS: tl.constexpr):
    COLS: tl.constexpr = 8 * WORKER_WARPS
    offsets = tle.gpu.set_layout(tl.reshape(tl.arange(0, 16 * COLS), (16, COLS)), _layout(IS_MMA, 1, WORKER_WARPS))
    if PARTITION == 0:
        result = tile * 2.0 + 17.0
    else:
        result = 1000.0 - tile * 3.0
    tl.store(out + PARTITION * 16 * COLS + offsets, result)


@triton.jit
def _extract_and_reuse_kernel(
    x,
    out,
    SOURCE_ROWS: tl.constexpr,
    TILE_BASE: tl.constexpr,
    PAIR_LAYOUT: tl.constexpr,
    HALF_LAYOUT: tl.constexpr,
    SWAP_OWNERS: tl.constexpr,
    IS_MMA: tl.constexpr,
    WORKER_WARPS: tl.constexpr,
):
    COLS: tl.constexpr = 8 * WORKER_WARPS
    offsets = tle.gpu.set_layout(tl.reshape(tl.arange(0, SOURCE_ROWS * COLS), (SOURCE_ROWS, COLS)), PAIR_LAYOUT)
    pair = tle.gpu.set_layout(tl.load(x + offsets), PAIR_LAYOUT)
    top = tle.gpu.set_layout(tle.extract_tile(pair, index=[TILE_BASE, 0], tile_shape=[16, COLS]), HALF_LAYOUT)
    bottom = tle.gpu.set_layout(tle.extract_tile(pair, index=[TILE_BASE + 1, 0], tile_shape=[16, COLS]), HALF_LAYOUT)
    if SWAP_OWNERS:
        first, second = bottom, top
    else:
        first, second = top, bottom
    tle.gpu.warp_specialize(
        (
            (_transform_tile, (first, out, IS_MMA, 0, WORKER_WARPS)),
            (_transform_tile, (second, out, IS_MMA, 1, WORKER_WARPS)),
        ),
        worker_num_warps=(WORKER_WARPS, WORKER_WARPS),
        reuse_default_warps=True,
    )


@pytest.mark.parametrize("layout_kind", ["mma", "blocked"])
@pytest.mark.parametrize("source_rows,tile_base", [(32, 0), (64, 2)])
@pytest.mark.parametrize("worker_warps", [1, 2, 4])
def test_extract_tile_reuses_owning_warps(layout_kind, source_rows, tile_base, worker_warps):
    # The 64-row case selects the later registers of each source thread,
    # exercising an actual register subset rather than a same-type bitcast.
    is_mma = layout_kind == "mma"
    pair_layout, half_layout = _layout(is_mma, 2, worker_warps), _layout(is_mma, 1, worker_warps)
    cols = 8 * worker_warps
    num_warps = 2 * worker_warps
    x = torch.arange(1, source_rows * cols + 1, device="cuda", dtype=torch.float32).reshape(source_rows, cols)
    out = torch.full((2, 16, cols), float("nan"), device="cuda", dtype=torch.float32)
    compiled = _extract_and_reuse_kernel[(1, )](x, out, source_rows, tile_base, pair_layout, half_layout, False, is_mma,
                                                worker_warps, num_warps=num_warps)
    first_row = tile_base * 16
    expected = torch.stack((x[first_row:first_row + 16] * 2.0 + 17.0, 1000.0 - x[first_row + 16:first_row + 32] * 3.0))
    torch.testing.assert_close(out, expected, rtol=0, atol=0)
    assert compiled.metadata.num_warps == num_warps
    assert compiled.metadata.shared == 0
    assert re.search(rf"\.reqntid\s+{num_warps * 32}\b", compiled.asm["ptx"])
    assert "bar.sync" not in compiled.asm["ptx"]


@pytest.mark.parametrize("layout_kind", ["mma", "blocked"])
def test_reuse_rejects_tile_owned_by_other_partition(layout_kind, capfd):
    pair_layout, half_layout = _layout(layout_kind == "mma", 2, 2), _layout(layout_kind == "mma", 1, 2)
    x = torch.arange(1, 513, device="cuda", dtype=torch.float32).reshape(32, 16)
    out = torch.empty((2, 16, 16), device="cuda", dtype=torch.float32)
    with pytest.raises(RuntimeError, match="PassManager::run failed"):
        _extract_and_reuse_kernel[(1, )](x, out, 32, 0, pair_layout, half_layout, True, layout_kind == "mma", 2,
                                         num_warps=4)
    diagnostic = capfd.readouterr().err
    assert "warp" in diagnostic and "partition" in diagnostic


@triton.jit
def _split_nested_parent(parent, MIDDLE_LAYOUT: tl.constexpr, LEAF_LAYOUT: tl.constexpr, VIEW_KIND: tl.constexpr):
    if VIEW_KIND == "identity":
        parent = tle.gpu.set_layout(tle.extract_tile(parent, index=[0, 0], tile_shape=[32, 32]), MIDDLE_LAYOUT)
    if VIEW_KIND == "narrow":
        # Keep the four owning warps while selecting the later column registers.
        parent = tle.gpu.set_layout(tle.extract_tile(parent, index=[0, 1], tile_shape=[32, 16]), MIDDLE_LAYOUT)
        first = tle.gpu.set_layout(tle.extract_tile(parent, index=[0, 0], tile_shape=[16, 16]), LEAF_LAYOUT)
        second = tle.gpu.set_layout(tle.extract_tile(parent, index=[1, 0], tile_shape=[16, 16]), LEAF_LAYOUT)
    else:
        # Both child extracts select nonzero registers from the packed parent.
        first = tle.gpu.set_layout(tle.extract_tile(parent, index=[0, 1], tile_shape=[16, 16]), LEAF_LAYOUT)
        second = tle.gpu.set_layout(tle.extract_tile(parent, index=[1, 1], tile_shape=[16, 16]), LEAF_LAYOUT)
    return first, second


@triton.jit
def _nested_extract_and_reuse_kernel(x, out, IS_MMA: tl.constexpr, VIEW_KIND: tl.constexpr):
    ROOT_LAYOUT: tl.constexpr = _layout(IS_MMA, 4, 2)
    MIDDLE_LAYOUT: tl.constexpr = _layout(IS_MMA, 2, 2)
    LEAF_LAYOUT: tl.constexpr = _layout(IS_MMA, 1, 2)
    offsets = tle.gpu.set_layout(tl.reshape(tl.arange(0, 128 * 64), (128, 64)), ROOT_LAYOUT)
    source = tle.gpu.set_layout(tl.load(x + offsets), ROOT_LAYOUT)
    # Rows 64..128 and columns 32..64 select later root registers. These two
    # parents belong to physical warps [0, 4) and [4, 8), respectively.
    top = tle.gpu.set_layout(tle.extract_tile(source, index=[2, 1], tile_shape=[32, 32]), MIDDLE_LAYOUT)
    bottom = tle.gpu.set_layout(tle.extract_tile(source, index=[3, 1], tile_shape=[32, 32]), MIDDLE_LAYOUT)
    first, second = _split_nested_parent(top, MIDDLE_LAYOUT, LEAF_LAYOUT, VIEW_KIND)
    third, fourth = _split_nested_parent(bottom, MIDDLE_LAYOUT, LEAF_LAYOUT, VIEW_KIND)
    tle.gpu.warp_specialize(
        (
            (_transform_tile, (first, out, IS_MMA, 0, 2)),
            (_transform_tile, (second, out, IS_MMA, 1, 2)),
            (_transform_tile, (third, out, IS_MMA, 2, 2)),
            (_transform_tile, (fourth, out, IS_MMA, 3, 2)),
        ),
        worker_num_warps=(2, 2, 2, 2),
        reuse_default_warps=True,
    )


@pytest.mark.parametrize("layout_kind", ["mma", "blocked"])
@pytest.mark.parametrize("view_kind", ["direct", "identity", "narrow"])
def test_nested_extract_tile_reuses_owning_warps(layout_kind, view_kind):
    # Exercise 8 -> 4 -> 2 warps with nonzero register selections in both the
    # parent and child, including a same-warp-count view between the splits.
    x = torch.arange(1, 128 * 64 + 1, device="cuda", dtype=torch.float32).reshape(128, 64)
    out = torch.full((4, 16, 16), float("nan"), device="cuda", dtype=torch.float32)
    compiled = _nested_extract_and_reuse_kernel[(1, )](x, out, layout_kind == "mma", view_kind, num_warps=8)
    tiles = x[64:128, 48:64].reshape(4, 16, 16)
    expected = 1000.0 - tiles * 3.0
    expected[0] = tiles[0] * 2.0 + 17.0
    torch.testing.assert_close(out, expected, rtol=0, atol=0)
    assert compiled.metadata.num_warps == 8
    assert compiled.metadata.shared == 0
    assert re.search(r"\.reqntid\s+256\b", compiled.asm["ptx"])
    assert "bar.sync" not in compiled.asm["ptx"]


@triton.jit(do_not_specialize=["index"])
def _materialized_extract_and_reuse_kernel(x, out, index):
    FULL_LAYOUT: tl.constexpr = _layout(False, 2, 2)
    HALF_LAYOUT: tl.constexpr = _layout(False, 1, 2)
    offsets = tle.gpu.set_layout(tl.reshape(tl.arange(0, 64 * 16), (64, 16)), FULL_LAYOUT)
    source = tle.gpu.set_layout(tl.load(x + offsets), FULL_LAYOUT)
    parent = tle.gpu.set_layout(tle.extract_tile(source, index=[index, 0], tile_shape=[32, 16]), FULL_LAYOUT)
    top = tle.gpu.set_layout(tle.extract_tile(parent, index=[0, 0], tile_shape=[16, 16]), HALF_LAYOUT)
    bottom = tle.gpu.set_layout(tle.extract_tile(parent, index=[1, 0], tile_shape=[16, 16]), HALF_LAYOUT)
    tle.gpu.warp_specialize(
        ((_transform_tile, (top, out, False, 0, 2)), (_transform_tile, (bottom, out, False, 1, 2))),
        worker_num_warps=(2, 2),
        reuse_default_warps=True,
    )


@pytest.mark.parametrize("index", [0, 1])
def test_materialized_extract_tile_reuses_owning_warps(index):
    x = torch.arange(1, 64 * 16 + 1, device="cuda", dtype=torch.float32).reshape(64, 16)
    out = torch.full((2, 16, 16), float("nan"), device="cuda", dtype=torch.float32)
    # Keep both values dynamic: specializing 1 would hide the failed-parent
    # case behind a register-local static parent proof.
    compiled = _materialized_extract_and_reuse_kernel[(1, )](x, out, index, num_warps=4)
    tiles = x[index * 32:(index + 1) * 32].reshape(2, 16, 16)
    expected = torch.stack((tiles[0] * 2.0 + 17.0, 1000.0 - tiles[1] * 3.0))
    torch.testing.assert_close(out, expected, rtol=0, atol=0)
    assert compiled.metadata.num_warps == 4
    assert compiled.metadata.shared == 32 * 16 * 4


@triton.jit
def _delayed_histogram(x, out):
    # Let the other partition reach the continuation while this partition still
    # needs its compiler-generated shared scratch.
    delay = tl.inline_asm_elementwise("nanosleep.u32 1000000; mov.u32 $0, 0;", "=r", [], dtype=tl.int32, is_pure=False,
                                      pack=1)
    offsets = tl.arange(0, 128)
    values = tl.load(x + offsets) + delay
    bins = tl.histogram(values, 128)
    tl.store(out + offsets, bins)


@triton.jit
def _empty_partition():
    pass


@triton.jit
def _ordinary_partition(out, INDEX: tl.constexpr):
    tl.store(out + 129 + INDEX, 37 + INDEX)


@triton.jit
def _reuse_scratch_kernel(x, out, WITH_ORDINARY_WORKERS: tl.constexpr):
    if WITH_ORDINARY_WORKERS:
        tle.gpu.warp_specialize(
            ((_ordinary_partition, (out, 0)), (_ordinary_partition, (out, 1))),
            worker_num_warps=(4, ),
            worker_num_regs=(64, ),
        )
    tle.gpu.warp_specialize(
        ((_delayed_histogram, (x, out)), (_empty_partition, ())),
        worker_num_warps=(2, 2),
        reuse_default_warps=True,
    )
    # This reduction uses different user data but can reuse the histogram's
    # shared scratch. It must wait until both reused partitions have finished.
    values = tl.load(x + tl.arange(0, 128))
    tl.store(out + 128, tl.sum(values, 0))


@pytest.mark.parametrize("with_ordinary_workers", [False, True])
def test_reuse_protects_shared_scratch_at_continuation(with_ordinary_workers):
    target = triton.runtime.driver.active.get_current_target()
    if with_ordinary_workers and int(target.arch) < 90:
        pytest.skip("ordinary worker register redistribution requires Hopper or newer")
    x = torch.arange(128, device="cuda", dtype=torch.int32)
    out = torch.full((131, ), -1, device="cuda", dtype=torch.int32)
    compiled = _reuse_scratch_kernel[(1, )](x, out, with_ordinary_workers, num_warps=4)
    expected = torch.ones_like(out)
    expected[128] = 8128
    expected[129:] = torch.tensor([37, 38] if with_ordinary_workers else [-1, -1], device="cuda", dtype=torch.int32)
    torch.testing.assert_close(out, expected, rtol=0, atol=0)
    assert compiled.metadata.shared > 0
    if with_ordinary_workers:
        # Four persistent workers wait elsewhere; the reused groups' barrier
        # must rendezvous only the four default warps in the eight-warp CTA.
        assert re.search(r"\.reqntid\s+256\b", compiled.asm["ptx"])
        assert re.search(r"bar\.sync\s+0,\s*128;", compiled.asm["ptx"])


@triton.jit(noinline=True)
def _noinline_store(out):
    tl.store(out, 1)


@triton.jit
def _nested_noinline_store(out):
    _noinline_store(out)


@triton.jit
def _reuse_noinline_kernel(out, NESTED: tl.constexpr):
    if NESTED:
        tle.gpu.warp_specialize(
            ((_empty_partition, ()), (_nested_noinline_store, (out, ))),
            worker_num_warps=(2, 2),
            reuse_default_warps=True,
        )
    else:
        tle.gpu.warp_specialize(
            ((_empty_partition, ()), (_noinline_store, (out, ))),
            worker_num_warps=(2, 2),
            reuse_default_warps=True,
        )


@pytest.mark.parametrize("nested", [False, True])
def test_reuse_rejects_noninlined_device_calls(nested, capfd):
    out = torch.full((1, ), -7, device="cuda", dtype=torch.int32)
    with pytest.raises(RuntimeError, match="PassManager::run failed"):
        _reuse_noinline_kernel[(1, )](out, nested, num_warps=4)
    assert "device calls in reuse_default_warps partitions must be inlined" in capfd.readouterr().err


@triton.jit
def _reuse_matmul(a_ptr, b_ptr, out):
    m = tl.arange(0, 64)
    n = tl.arange(0, 64)
    k = tl.arange(0, 32)
    a = tl.load(a_ptr + m[:, None] * 32 + k[None, :])
    b = tl.load(b_ptr + k[:, None] * 64 + n[None, :])
    result = tl.dot(a, b)
    tl.store(out + m[:, None] * 64 + n[None, :], result)


@triton.jit
def _reuse_aligned_matmul_kernel(a, b, out):
    tle.gpu.warp_specialize(
        ((_empty_partition, ()), (_reuse_matmul, (a, b, out)), (_empty_partition, ())),
        worker_num_warps=(2, 4, 2),
        reuse_default_warps=True,
    )


def test_reuse_aligns_wgmma_partition():
    target = triton.runtime.driver.active.get_current_target()
    if int(target.arch) != 90:
        pytest.skip("requires Hopper WGMMA lowering")
    # Compile and inspect assignment before launching: the old assignment
    # executes WGMMA from physical warps 2..5 and must never be launched.
    a = (torch.arange(64 * 32, device="cuda") % 7 - 3).to(torch.float16).reshape(64, 32)
    b = (torch.arange(32 * 64, device="cuda") % 5 - 2).to(torch.float16).reshape(32, 64)
    out = torch.empty((64, 64), device="cuda", dtype=torch.float32)
    compiled = _reuse_aligned_matmul_kernel.warmup(a, b, out, grid=(1, ), num_warps=8)
    assert "wgmma.mma_async" in compiled.asm["ptx"]
    assert re.search(r"shfl\.sync\.idx.*\n\s*setp\.gt\.u32\s+[^,]+,\s*%r\d+,\s*3;", compiled.asm["ptx"])
    assert compiled.metadata.num_warps == 8
    compiled[(1, 1, 1)](a, b, out)
    torch.testing.assert_close(out, a.float() @ b.float(), rtol=0, atol=0)
