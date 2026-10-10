"""Async copy completion and SSA pipe identity across unequal warp partitions."""
import re

import pytest
import torch
import triton
import triton.language as tl
import triton.experimental.tle.language as tle
from triton.compiler.compiler import ASTSource, GPUTarget

@triton.jit
def _produce(writer, src, B: tl.constexpr, STEPS: tl.constexpr, DRAIN: tl.constexpr):
    offsets = tl.arange(0, B)
    for step in range(STEPS):
        slot = writer.acquire(step)
        tle.gpu.copy(src + step * B + offsets, slot.tile, [B], is_async=True)
        tle.gpu.async_commit_group()
        tle.gpu.async_wait_group(0)
        writer.commit(step)
    if DRAIN:
        writer.close(STEPS)
        writer.pipe.wait_drained()

@triton.jit
def _consume(reader, dst, B: tl.constexpr, STEPS: tl.constexpr, DRAIN: tl.constexpr):
    offsets = tl.arange(0, B)
    for step in range(STEPS):
        ready = reader.wait(step)
        values = tl.load(tle.gpu.local_ptr(ready.slot.tile, (offsets,)))
        tl.store(dst + step * B + offsets, values)
        reader.release(step)
    if DRAIN:
        reader.pipe.wait_drained()

@triton.jit
def _copy_kernel(src, dst, B: tl.constexpr, STEPS: tl.constexpr, PW: tl.constexpr, ALIAS: tl.constexpr = False):
    if ALIAS:
        arena = tle.gpu.alloc([B * 8], dtype=tl.uint8, scope=tle.gpu.smem,
                              nv_mma_shared_layout=False)
        storage = tle.gpu.alloc([2, B], dtype=tl.bfloat16, scope=tle.gpu.smem,
                                nv_mma_shared_layout=False, alias=arena, alias_offset_bytes=128)
    else:
        storage = tle.gpu.alloc([2, B], dtype=tl.bfloat16, scope=tle.gpu.smem,
                                nv_mma_shared_layout=False)
    pipe = tle.pipe(capacity=2, scope="cta", name="transport", tile=storage)
    tle.gpu.warp_specialize([
        (_consume, (pipe.reader(), dst, B, STEPS, ALIAS)),
        (_produce, (pipe.writer(), src, B, STEPS, ALIAS)),
    ], [PW], [64])

@pytest.mark.parametrize("producer_warps", [4, 8])
@pytest.mark.parametrize("block", [256, 4096])
def test_ws_explicit_copy_wraps_slots_and_preserves_payload(producer_warps, block):
    torch.manual_seed(119)
    source = torch.randn(32 * block, device="cuda", dtype=torch.bfloat16)
    result = torch.empty_like(source)
    compiled = _copy_kernel[(1,)](source, result, block, 32, producer_warps, num_warps=8)
    assert "ttg.warp_specialize" in compiled.asm["ttgir"]
    assert "ttg.async_copy_global_to_local" in compiled.asm["ttgir"]
    for _ in range(10):
        source.normal_()
        result.fill_(float("nan"))
        _copy_kernel[(1,)](source, result, block, 32, producer_warps, num_warps=8)
        torch.testing.assert_close(result, source, rtol=0, atol=0)


def test_ws_copy_reinterprets_byte_arena_and_drains_both_partitions():
    source = torch.randn(32 * 4096, device="cuda", dtype=torch.bfloat16)
    destination = torch.empty_like(source)
    for _ in range(5):
        source.normal_()
        compiled = _copy_kernel[(1,)](source, destination, 4096, 32, 4, True, num_warps=8)
        torch.testing.assert_close(destination, source, rtol=0, atol=0)
    assert "iluvatar_tle.memdesc_alias" in compiled.asm["ttgir"]


def test_ws_copy_16_ctas_12_warps_single_pipe():
    """Match the artifact's launch geometry without its model-level work."""
    source = torch.randn(32 * 2048, device="cuda", dtype=torch.bfloat16)
    destination = torch.empty_like(source)
    for _ in range(2):
        source.normal_()
        _copy_kernel[(16,)](
            source,
            destination,
            2048,
            32,
            4,
            True,
            num_warps=16,
        )
        torch.testing.assert_close(destination, source, rtol=0, atol=0)


@triton.jit(noinline=True)
def _produce_2d(writer, src, STEPS: tl.constexpr):
    rows = tl.arange(0, 4)[:, None]
    cols = tl.arange(0, 512)[None, :]
    for step in range(STEPS):
        slot = writer.acquire(step)
        tle.gpu.copy(
            src + step * 2048 + rows * 512 + cols,
            slot.tile,
            [4, 512],
            is_async=True,
        )
        tle.gpu.async_commit_group()
        tle.gpu.async_wait_group(0)
        writer.commit(step)
    writer.close(STEPS)
    writer.pipe.wait_drained()


@triton.jit(noinline=True)
def _consume_2d(reader, dst, STEPS: tl.constexpr):
    rows = tl.arange(0, 4)[:, None]
    cols = tl.arange(0, 512)[None, :]
    for step in range(STEPS):
        ready = reader.wait(step)
        values = tl.load(tle.gpu.local_ptr(
            ready.slot.tile,
            (tl.broadcast_to(rows, [4, 512]), tl.broadcast_to(cols, [4, 512])),
            [4, 512],
        ))
        tl.store(dst + step * 2048 + rows * 512 + cols, values)
        reader.release(step)
    reader.pipe.wait_drained()


@triton.jit
def _copy_kernel_2d(src, dst, STEPS: tl.constexpr, PW: tl.constexpr):
    storage = tle.gpu.alloc(
        [2, 4, 512],
        dtype=tl.bfloat16,
        scope=tle.gpu.smem,
        nv_mma_shared_layout=False,
    )
    pipe = tle.pipe(capacity=2, scope="cta", name="transport_2d", tile=storage)
    tle.gpu.warp_specialize([
        (_consume_2d, (pipe.reader(), dst, STEPS)),
        (_produce_2d, (pipe.writer(), src, STEPS)),
    ], [PW], [64])


@pytest.mark.parametrize("producer_warps", [4, 8])
def test_ws_explicit_copy_2d_tile_is_deterministic(producer_warps):
    """The model's [4, 512] SME tile must survive the async pipe unchanged."""
    steps = 32
    source = torch.randn(steps * 2048, device="cuda", dtype=torch.bfloat16)
    destination = torch.empty_like(source)
    for _ in range(5):
        source.normal_()
        destination.fill_(float("nan"))
        _copy_kernel_2d[(1,)](
            source,
            destination,
            steps,
            producer_warps,
            num_warps=16,
        )
        torch.testing.assert_close(destination, source, rtol=0, atol=0)


@triton.jit(noinline=True)
def _produce_2d_partial_wait(writer, src, steps: tl.constexpr):
    rows = tl.arange(0, 4)[:, None]
    cols = tl.arange(0, 512)[None, :]
    for step in range(steps):
        if step >= 2:
            tle.gpu.async_wait_group(1)
            writer.commit(step - 2)
        slot = writer.acquire(step)
        tle.gpu.copy(
            src + step * 2048 + rows * 512 + cols,
            slot.tile,
            [4, 512],
            is_async=True,
        )
        tle.gpu.async_commit_group()
    tle.gpu.async_wait_group(0)
    for step in range(max(0, steps - 2), steps):
        writer.commit(step)
    writer.close(steps)
    writer.pipe.wait_drained()


@triton.jit(noinline=True)
def _copy_kernel_2d_partial_wait(src, dst, steps: tl.constexpr):
    rows = tl.arange(0, 4)[:, None]
    cols = tl.arange(0, 512)[None, :]
    storage = tle.gpu.alloc(
        [2, 4, 512],
        dtype=tl.bfloat16,
        scope=tle.gpu.smem,
        nv_mma_shared_layout=False,
    )
    pipe = tle.pipe(capacity=2, scope="cta", name="transport_2d_partial", tile=storage)
    tle.gpu.warp_specialize([
        (_consume_2d, (pipe.reader(), dst, steps)),
        (_produce_2d_partial_wait, (pipe.writer(), src, steps)),
    ], [4], [64])


def test_ws_explicit_copy_2d_partial_wait_preserves_payload():
    """A two-slot wait(1) producer must publish the completed oldest tile."""
    steps = 32
    source = torch.randn(steps * 2048, device="cuda", dtype=torch.bfloat16)
    destination = torch.empty_like(source)
    compiled = _copy_kernel_2d_partial_wait[(1,)](
        source, destination, steps, num_warps=16
    )
    assert "ttg.async_wait {num = 1" in compiled.asm["ttgir"]
    # The [4, 512] bf16 SME tile is four 1x64B G2S transactions.  wait(1)
    # uses that transaction count; wait(0) remains the full drain value.
    assert "llvm.bi.sl.waitcnt(i64 33554440)" in compiled.asm["llir"]
    assert "llvm.bi.sl.waitcnt(i64 8)" in compiled.asm["llir"]
    for _ in range(5):
        source.normal_()
        destination.fill_(float("nan"))
        _copy_kernel_2d_partial_wait[(1,)](
            source, destination, steps, num_warps=16
        )
        torch.testing.assert_close(destination, source, rtol=0, atol=0)


@triton.jit(noinline=True)
def _produce_2d_sme_4w(writer, src, steps: tl.constexpr):
    rows = tl.arange(0, 4)[:, None]
    cols = tl.arange(0, 512)[None, :]
    for step in range(steps):
        slot = writer.acquire(step)
        src_ptrs = tle.gpu.set_layout(
            src + step * 2048 + rows * 512 + cols,
            tle.gpu.IluvatarSmeBlockEncoding(
                [4, 512],
                tl.bfloat16,
                [2, 4],
                [2, 32],
                [1, 4],
                [1, 0],
            ),
        )
        tle.gpu.copy(
            src_ptrs,
            slot.tile,
            [4, 512],
            is_async=True,
            input_stride=512,
        )
        tle.gpu.async_commit_group()
        tle.gpu.async_wait_group(0)
        writer.commit(step)
    writer.close(steps)
    writer.pipe.wait_drained()


@triton.jit
def _copy_kernel_2d_sme_4w(src, dst, steps: tl.constexpr):
    storage = tle.gpu.alloc(
        [2, 4, 512],
        dtype=tl.bfloat16,
        layout=tle.gpu.iluvatar_sme_shared_layout(
            [2, 4, 512],
            tl.bfloat16,
            [2, 1, 0],
            [1, 1, 1],
            [1, 1, 1],
            [2, 1, 0],
        ),
        nv_mma_shared_layout=False,
    )
    pipe = tle.pipe(capacity=2, scope="cta", name="transport_2d_sme_4w", tile=storage)
    tle.gpu.warp_specialize([
        (_consume_2d, (pipe.reader(), dst, steps)),
        (_produce_2d_sme_4w, (pipe.writer(), src, steps)),
    ], [4], [64])


def test_ws_explicit_copy_2d_sme_codegen():
    """WS+SME must reach LLVM and emit the target SME tile load."""
    compiled = triton.compile(
        ASTSource(
            _copy_kernel_2d_sme_4w,
            signature={"src": "*bf16", "dst": "*bf16"},
            constexprs={"steps": 32},
        ),
        target=GPUTarget("corex", 71, 64),
        options={"num_warps": 16},
    )
    assert "isSme = true" in compiled.asm["ttgir"]
    assert "llvm.nvvm.barrier.cta.sync.aligned.all" in compiled.asm["llir"]
    assert "__ilu_sme_publication_barrier_" not in compiled.asm["llir"]
    assert 'i32 64 syncscope("workgroup") release' in compiled.asm["llir"]
    assert "static_arrive_count = 256 : i32" in compiled.asm["ttgir"]
    assert re.search(r"lshr i32 %\d+, 8", compiled.asm["llir"])
    assert "inputStride" in compiled.asm["ttgir"]
    assert "llvm.bi.sme.load.4x1b64.rowxfb16" in compiled.asm["llir"]


@triton.jit(noinline=True)
def _produce_2d_sme_2w(writer, src, steps: tl.constexpr):
    rows = tl.arange(0, 4)[:, None]
    cols = tl.arange(0, 512)[None, :]
    for step in range(steps):
        slot = writer.acquire(step)
        src_ptrs = tle.gpu.set_layout(
            src + step * 2048 + rows * 512 + cols,
            tle.gpu.IluvatarSmeBlockEncoding(
                [4, 512],
                tl.bfloat16,
                [2, 4],
                [2, 32],
                [1, 2],
                [1, 0],
            ),
        )
        tle.gpu.copy(
            src_ptrs,
            slot.tile,
            [4, 512],
            is_async=True,
            input_stride=512,
        )
        tle.gpu.async_commit_group()
        tle.gpu.async_wait_group(0)
        writer.commit(step)
    writer.close(steps)
    writer.pipe.wait_drained()


@triton.jit
def _copy_kernel_2d_sme_2w(src, dst, steps: tl.constexpr):
    storage = tle.gpu.alloc(
        [2, 4, 512],
        dtype=tl.bfloat16,
        scope=tle.gpu.smem,
        nv_mma_shared_layout=False,
    )
    pipe = tle.pipe(capacity=2, scope="cta", name="transport_2d_sme_2w",
                    tile=storage)
    tle.gpu.warp_specialize([
        (_consume_2d, (pipe.reader(), dst, steps)),
        (_produce_2d_sme_2w, (pipe.writer(), src, steps)),
    ], [2], [64])


def test_ws_explicit_copy_2d_sme_2w_codegen():
    """Padding a 2-warp producer must preserve the SME publication group."""
    compiled = triton.compile(
        ASTSource(
            _copy_kernel_2d_sme_2w,
            signature={"src": "*bf16", "dst": "*bf16"},
            constexprs={"steps": 32},
        ),
        target=GPUTarget("corex", 71, 64),
        options={"num_warps": 16},
    )
    assert "warpsPerCTA = [1, 2]" in compiled.asm["ttgir"]
    assert "llvm.bi.sme.load.4x1b64.rowxfb16" in compiled.asm["llir"]
    assert "llvm.bi.sl.waitcnt(i64 8)" in compiled.asm["llir"]


@triton.jit(noinline=True)
def _produce_2d_sme_4w_once(writer, src):
    rows = tl.arange(0, 4)[:, None]
    cols = tl.arange(0, 512)[None, :]
    slot = writer.acquire(0)
    src_ptrs = tle.gpu.set_layout(
        src + rows * 512 + cols,
        tle.gpu.IluvatarSmeBlockEncoding(
            [4, 512], tl.bfloat16, [2, 4], [2, 32], [1, 4], [1, 0]
        ),
    )
    tle.gpu.copy(
        src_ptrs,
        slot.tile,
        [4, 512],
        is_async=True,
        input_stride=512,
    )
    tle.gpu.async_commit_group()
    tle.gpu.async_wait_group(0)
    writer.commit(0)


@triton.jit(noinline=True)
def _consume_2d_sme_4w_once(reader, dst):
    rows = tl.arange(0, 4)[:, None]
    cols = tl.arange(0, 512)[None, :]
    ready = reader.wait(0)
    values = tl.load(tle.gpu.local_ptr(ready.slot.tile))
    tl.store(dst + rows * 512 + cols, values)
    reader.release(0)


@triton.jit(noinline=True)
def _consume_2d_sme_4w_raw(reader, dst, DRAIN: tl.constexpr):
    # Read the physical rowxfb16 offsets in linear order by applying the
    # inverse logical mapping to local_ptr indices.  This separates the SME
    # producer's raw shared payload from the normal logical local-load view.
    rows = tl.arange(0, 4)[:, None]
    cols = tl.arange(0, 512)[None, :]
    raw = rows * 512 + cols
    logical_row = (raw & 1) | (((raw >> 5) & 1) << 1)
    logical_col = ((raw >> 1) & 15) | (((raw >> 6) & 1) << 4) | ((raw >> 7) << 5)
    ready = reader.wait(0)
    values = tl.load(tle.gpu.local_ptr(
        ready.slot.tile,
        (logical_row, logical_col),
        [4, 512],
    ))
    tl.store(dst + rows * 512 + cols, values)
    reader.release(0)
    if DRAIN:
        reader.pipe.wait_drained()


@triton.jit(noinline=True)
def _consume_2d_sme_4w_physical_dump(reader, dst):
    rows = tl.arange(0, 4)[:, None]
    cols = tl.arange(0, 512)[None, :]
    ready = reader.wait(0)
    values = tl.load(tle.gpu.local_ptr(ready.slot.tile))
    tile_col = cols % 32
    warp_col = (cols // 32) % 4
    group = cols // 128
    physical = (
        group * 512
        + warp_col * 128
        + (tile_col // 16) * 64
        + (rows // 2) * 32
        + (tile_col % 16) * 2
        + (rows % 2)
    )
    tl.store(dst + physical, values)
    reader.release(0)


@triton.jit
def _copy_kernel_2d_sme_4w_physical_dump(src, dst):
    storage = tle.gpu.alloc(
        [2, 4, 512],
        dtype=tl.bfloat16,
        layout=tle.gpu.iluvatar_sme_shared_layout(
            [2, 4, 512],
            tl.bfloat16,
            [2, 1, 0],
            [1, 1, 1],
            [1, 1, 1],
            [2, 1, 0],
        ),
        nv_mma_shared_layout=False,
    )
    pipe = tle.pipe(capacity=2, scope="cta", name="transport_2d_sme_physical_dump", tile=storage)
    tle.gpu.warp_specialize(
        [
            (_consume_2d_sme_4w_physical_dump, (pipe.reader(), dst)),
            (_produce_2d_sme_4w_once, (pipe.writer(), src)),
        ],
        [4],
        [64],
    )


def test_ws_explicit_copy_2d_sme_physical_dump():
    source = torch.arange(4 * 512, device="cuda", dtype=torch.float32).to(
        torch.bfloat16
    ).reshape(4, 512)
    destination = torch.full((4 * 512,), -1, device="cuda", dtype=torch.bfloat16)
    _copy_kernel_2d_sme_4w_physical_dump[(1,)](source, destination, num_warps=8)
    torch.cuda.synchronize()
    expected = torch.empty_like(destination)
    rows = torch.arange(4, device="cuda")[:, None]
    cols = torch.arange(512, device="cuda")[None, :]
    tile_col = cols % 32
    physical = (
        (cols // 128) * 512
        + ((cols // 32) % 4) * 128
        + (tile_col // 16) * 64
        + (rows // 2) * 32
        + (tile_col % 16) * 2
        + (rows % 2)
    )
    expected[physical.reshape(-1)] = source.reshape(-1)
    torch.testing.assert_close(destination, expected, rtol=0, atol=0)


@triton.jit
def _copy_kernel_2d_sme_4w_raw(src, dst):
    storage = tle.gpu.alloc(
        [2, 4, 512],
        dtype=tl.bfloat16,
        layout=tle.gpu.iluvatar_sme_shared_layout(
            [2, 4, 512],
            tl.bfloat16,
            [2, 1, 0],
            [1, 1, 1],
            [1, 1, 1],
            [2, 1, 0],
        ),
        nv_mma_shared_layout=False,
    )
    pipe = tle.pipe(capacity=2, scope="cta", name="transport_2d_sme_4w_raw", tile=storage)
    tle.gpu.warp_specialize(
        [
            (_consume_2d_sme_4w_raw, (pipe.reader(), dst, False)),
            (_produce_2d_sme_4w_once, (pipe.writer(), src)),
        ],
        [4],
        [64],
    )


@triton.jit
def _copy_kernel_2d_nonsme_raw(src, dst):
    storage = tle.gpu.alloc(
        [2, 4, 512],
        dtype=tl.bfloat16,
        layout=tle.gpu.iluvatar_sme_shared_layout(
            [2, 4, 512],
            tl.bfloat16,
            [2, 1, 0],
            [1, 1, 1],
            [1, 1, 1],
            [2, 1, 0],
        ),
        nv_mma_shared_layout=False,
    )
    pipe = tle.pipe(capacity=2, scope="cta", name="transport_2d_nonsme_raw", tile=storage)
    tle.gpu.warp_specialize(
        [
            (_consume_2d_sme_4w_raw, (pipe.reader(), dst, True)),
            (_produce_2d, (pipe.writer(), src, 1)),
        ],
        [4],
        [64],
    )


def test_ws_explicit_copy_2d_sme_raw_shared_dump():
    source = torch.arange(4 * 512, device="cuda", dtype=torch.float32).to(
        torch.bfloat16
    ).reshape(4, 512)
    destination = torch.full_like(source, -1)
    compiled = _copy_kernel_2d_sme_4w_raw[(1,)](source, destination, num_warps=8)
    torch.cuda.synchronize()
    raw = torch.arange(4 * 512, device="cuda", dtype=torch.int32)
    expected_row = (raw & 1) | (((raw >> 5) & 1) << 1)
    expected_col = ((raw >> 1) & 15) | (((raw >> 6) & 1) << 4) | ((raw >> 7) << 5)
    expected = source[expected_row, expected_col]
    mismatch = destination.reshape(-1) != expected
    print("raw_shared_mismatch", int(mismatch.sum().item()))
    print("raw_shared_rows", mismatch.reshape(4, 512).sum(dim=1).tolist())
    bad = torch.nonzero(mismatch).reshape(-1)[:32]
    print("raw_shared_bad", bad.cpu().tolist())
    print("raw_shared_actual", destination.reshape(-1)[bad].cpu().tolist())
    print("raw_shared_expected", expected[bad].cpu().tolist())
    assert "llvm.bi.sme.load.4x1b64.rowxfb16" in compiled.asm["llir"]
    torch.testing.assert_close(destination.reshape(-1), expected, rtol=0, atol=0)


def test_ws_explicit_copy_2d_nonsme_raw_shared_dump():
    source = torch.arange(4 * 512, device="cuda", dtype=torch.float32).to(
        torch.bfloat16
    ).reshape(4, 512)
    destination = torch.full_like(source, -1)
    compiled = _copy_kernel_2d_nonsme_raw[(1,)](source, destination, num_warps=8)
    torch.cuda.synchronize()
    raw = torch.arange(4 * 512, device="cuda", dtype=torch.int32)
    expected_row = (raw & 1) | (((raw >> 5) & 1) << 1)
    expected_col = ((raw >> 1) & 15) | (((raw >> 6) & 1) << 4) | ((raw >> 7) << 5)
    expected = source[expected_row, expected_col]
    mismatch = destination.reshape(-1) != expected
    print("nonsme_raw_shared_mismatch", int(mismatch.sum().item()))
    print("nonsme_raw_shared_rows", mismatch.reshape(4, 512).sum(dim=1).tolist())
    assert "llvm.bi.sme.load.4x1b64.rowxfb16" not in compiled.asm["llir"]
    torch.testing.assert_close(destination.reshape(-1), expected, rtol=0, atol=0)


@triton.jit
def _copy_4row_bf16_4w_raw(src, dst):
    rows = tl.arange(0, 4)[:, None]
    cols = tl.arange(0, 512)[None, :]
    src_ptrs = tle.gpu.set_layout(
        src + rows * 512 + cols,
        tle.gpu.IluvatarSmeBlockEncoding(
            [4, 512], tl.bfloat16, [2, 4], [2, 32], [1, 4], [1, 0]
        ),
    )
    storage = tle.gpu.alloc(
        [4, 512],
        dtype=tl.bfloat16,
        layout=tle.gpu.iluvatar_sme_shared_layout(
            [4, 512], tl.bfloat16, [1, 0], [1, 1], [1, 1], [1, 0]
        ),
        nv_mma_shared_layout=False,
    )
    tle.gpu.copy(src_ptrs, storage, [4, 512], is_async=True, input_stride=512)
    tle.gpu.async_commit_group()
    tle.gpu.async_wait_group(0)
    raw = rows * 512 + cols
    logical_row = (raw & 1) | (((raw >> 5) & 1) << 1)
    logical_col = ((raw >> 1) & 15) | (((raw >> 6) & 1) << 4) | ((raw >> 7) << 5)
    values = tl.load(tle.gpu.local_ptr(storage, (logical_row, logical_col), [4, 512]))
    tl.store(dst + rows * 512 + cols, values)


def test_sme_4row_bf16_four_warp_raw_shared_dump():
    source = torch.arange(4 * 512, device="cuda", dtype=torch.float32).to(
        torch.bfloat16
    ).reshape(4, 512)
    destination = torch.full_like(source, -1)
    compiled = _copy_4row_bf16_4w_raw[(1,)](source, destination, num_warps=4)
    torch.cuda.synchronize()
    raw = torch.arange(4 * 512, device="cuda", dtype=torch.int32)
    expected_row = (raw & 1) | (((raw >> 5) & 1) << 1)
    expected_col = ((raw >> 1) & 15) | (((raw >> 6) & 1) << 4) | ((raw >> 7) << 5)
    expected = source[expected_row, expected_col]
    mismatch = destination.reshape(-1) != expected
    print("direct_raw_shared_mismatch", int(mismatch.sum().item()))
    print("direct_raw_shared_rows", mismatch.reshape(4, 512).sum(dim=1).tolist())
    assert "llvm.bi.sme.load.4x1b64.rowxfb16" in compiled.asm["llir"]
    torch.testing.assert_close(destination.reshape(-1), expected, rtol=0, atol=0)


@triton.jit
def _copy_kernel_2d_sme_4w_once(src, dst):
    storage = tle.gpu.alloc(
        [2, 4, 512],
        dtype=tl.bfloat16,
        layout=tle.gpu.iluvatar_sme_shared_layout(
            [2, 4, 512],
            tl.bfloat16,
            [2, 1, 0],
            [1, 1, 1],
            [1, 1, 1],
            [2, 1, 0],
        ),
        nv_mma_shared_layout=False,
    )
    pipe = tle.pipe(capacity=2, scope="cta", name="transport_2d_sme_4w_once", tile=storage)
    tle.gpu.warp_specialize(
        [
            (_consume_2d_sme_4w_once, (pipe.reader(), dst)),
            (_produce_2d_sme_4w_once, (pipe.writer(), src)),
        ],
        [4],
        [64],
    )


@triton.jit(noinline=True)
def _consume_2d_sme_8w_explicit(reader, dst):
    rows = tl.arange(0, 4)[:, None]
    cols = tl.arange(0, 512)[None, :]
    ready = reader.wait(0)
    ptrs = tle.gpu.set_layout(
        tle.gpu.local_ptr(ready.slot.tile),
        tle.gpu.IluvatarSmeBlockEncoding(
            [4, 512], tl.bfloat16, [1, 4], [2, 32], [2, 4], [1, 0]
        ),
    )
    values = tl.load(ptrs)
    tl.store(dst + rows * 512 + cols, values)
    reader.release(0)


@triton.jit
def _copy_kernel_2d_sme_4w_consumer_8w_sme(src, dst):
    storage = tle.gpu.alloc(
        [2, 4, 512],
        dtype=tl.bfloat16,
        layout=tle.gpu.iluvatar_sme_shared_layout(
            [2, 4, 512],
            tl.bfloat16,
            [2, 1, 0],
            [1, 1, 1],
            [1, 1, 1],
            [2, 1, 0],
        ),
        nv_mma_shared_layout=False,
    )
    pipe = tle.pipe(
        capacity=2,
        scope="cta",
        name="transport_2d_sme_4w_consumer_8w_sme",
        tile=storage,
    )
    tle.gpu.warp_specialize(
        [
            (_consume_2d_sme_8w_explicit, (pipe.reader(), dst)),
            (_produce_2d_sme_4w_once, (pipe.writer(), src)),
        ],
        [4],
        [64],
    )


def test_ws_explicit_copy_2d_sme_consumer_explicit_8w_layout():
    """A matching 8-warp SME view must decode the producer's physical tile."""
    source = torch.arange(4 * 512, device="cuda", dtype=torch.float32).to(
        torch.bfloat16
    ).reshape(4, 512)
    destination = torch.full_like(source, -1)
    compiled = _copy_kernel_2d_sme_4w_consumer_8w_sme[(1,)](
        source, destination, num_warps=8
    )
    torch.cuda.synchronize()
    print("explicit_consumer_8w_mismatch", int((destination != source).sum().item()))
    print(
        "explicit_consumer_8w_ttgir_layouts",
        [line for line in compiled.asm["ttgir"].splitlines() if line.startswith("#blocked")],
    )
    llir = compiled.asm["llir"]
    partial_barrier = llir.rfind(
        "atomicrmw add ptr addrspace(3) getelementptr inbounds nuw "
        "(i8, ptr addrspace(3) @__ws_namedbar_state"
    )
    publication_barrier = llir.rfind(
        "tail call void @llvm.bi.sl.barrier.alu"
    )
    assert partial_barrier >= 0
    assert publication_barrier >= 0
    assert partial_barrier < publication_barrier
    torch.testing.assert_close(destination, source, rtol=0, atol=0)


def test_ws_explicit_copy_2d_sme_once_without_drain():
    """Minimal SME+WS payload before any close/drain lifecycle operations."""
    source = torch.randn((4, 512), device="cuda", dtype=torch.bfloat16)
    destination = torch.full_like(source, float("nan"))
    _copy_kernel_2d_sme_4w_once[(1,)](
        source, destination, num_warps=8
    )
    torch.cuda.synchronize()
    torch.testing.assert_close(destination, source, rtol=0, atol=0)


@triton.jit(noinline=True)
def _ws_sme_direct_producer(src, dst, storage):
    rows = tl.arange(0, 4)[:, None]
    cols = tl.arange(0, 512)[None, :]
    src_ptrs = tle.gpu.set_layout(
        src + rows * 512 + cols,
        tle.gpu.IluvatarSmeBlockEncoding(
            [4, 512], tl.bfloat16, [2, 4], [2, 32], [1, 4], [1, 0]
        ),
    )
    tle.gpu.copy(src_ptrs, storage, [4, 512], is_async=True, input_stride=512)
    tle.gpu.async_commit_group()
    tle.gpu.async_wait_group(0)
    values = tl.load(tle.gpu.local_ptr(storage))
    tl.store(dst + rows * 512 + cols, values)


@triton.jit
def _ws_sme_direct_worker():
    return


@triton.jit
def _ws_sme_direct_producer_kernel(src, dst):
    storage = tle.gpu.alloc(
        [2, 4, 512],
        dtype=tl.bfloat16,
        layout=tle.gpu.iluvatar_sme_shared_layout(
            [2, 4, 512], tl.bfloat16, [2, 1, 0], [1, 1, 1], [1, 1, 1], [2, 1, 0]
        ),
        nv_mma_shared_layout=False,
    )
    slot = storage.slot(0)
    tle.gpu.warp_specialize(
        [
            (_ws_sme_direct_producer, (src, dst, slot)),
            (_ws_sme_direct_worker, ()),
        ],
        [4],
        [64],
    )


def test_ws_sme_direct_producer_preserves_payload():
    source = torch.arange(4 * 512, device="cuda", dtype=torch.float32).to(
        torch.bfloat16
    ).reshape(4, 512)
    destination = torch.full_like(source, -1)
    _ws_sme_direct_producer_kernel[(1,)](source, destination, num_warps=4)
    torch.cuda.synchronize()
    torch.testing.assert_close(destination, source, rtol=0, atol=0)


@triton.jit
def _copy_4row_bf16_4w(src, dst):
    rows = tl.arange(0, 4)[:, None]
    cols = tl.arange(0, 512)[None, :]
    src_ptrs = tle.gpu.set_layout(
        src + rows * 512 + cols,
        tle.gpu.IluvatarSmeBlockEncoding(
            [4, 512], tl.bfloat16, [2, 4], [2, 32], [1, 4], [1, 0]
        ),
    )
    storage = tle.gpu.alloc(
        [4, 512],
        dtype=tl.bfloat16,
        layout=tle.gpu.iluvatar_sme_shared_layout(
            [4, 512], tl.bfloat16, [1, 0], [1, 1], [1, 1], [1, 0]
        ),
        nv_mma_shared_layout=False,
    )
    tle.gpu.copy(src_ptrs, storage, [4, 512], is_async=True, input_stride=512)
    tle.gpu.async_commit_group()
    tle.gpu.async_wait_group(0)
    values = tl.load(tle.gpu.local_ptr(storage))
    tl.store(dst + rows * 512 + cols, values)


def test_sme_4row_bf16_four_warp_encoding_preserves_payload():
    source = torch.randn((4, 512), device="cuda", dtype=torch.bfloat16)
    destination = torch.full_like(source, float("nan"))
    _copy_4row_bf16_4w[(1,)](source, destination, num_warps=4)
    torch.cuda.synchronize()
    torch.testing.assert_close(destination, source, rtol=0, atol=0)


@triton.jit
def _copy_kernel_2d_sme_4w_order012(src, dst):
    storage = tle.gpu.alloc(
        [2, 4, 512],
        dtype=tl.bfloat16,
        layout=tle.gpu.iluvatar_sme_shared_layout(
            [2, 4, 512],
            tl.bfloat16,
            [0, 2, 1],
            [1, 1, 1],
            [1, 1, 1],
            [0, 2, 1],
        ),
        nv_mma_shared_layout=False,
    )
    pipe = tle.pipe(capacity=2, scope="cta", name="transport_2d_sme_order012", tile=storage)
    tle.gpu.warp_specialize(
        [
            (_produce_2d_sme_4w_once, (pipe.writer(), src)),
            (_consume_2d_sme_4w_once, (pipe.reader(), dst)),
        ],
        [4],
        [64],
    )


@triton.jit
def _copy_kernel_2d_sme_4w_default_storage(src, dst):
    storage = tle.gpu.alloc(
        [2, 4, 512], dtype=tl.bfloat16, nv_mma_shared_layout=False
    )
    pipe = tle.pipe(capacity=2, scope="cta", name="transport_2d_sme_default", tile=storage)
    tle.gpu.warp_specialize(
        [
            (_produce_2d_sme_4w_once, (pipe.writer(), src)),
            (_consume_2d_sme_4w_once, (pipe.reader(), dst)),
        ],
        [4],
        [64],
    )


def test_ws_explicit_copy_2d_sme_rank3_layout_matrix():
    source = torch.arange(4 * 512, device="cuda", dtype=torch.float32).to(
        torch.bfloat16
    ).reshape(4, 512)
    for name, kernel in (
        ("order012", _copy_kernel_2d_sme_4w_order012),
        ("default", _copy_kernel_2d_sme_4w_default_storage),
    ):
        destination = torch.full_like(source, -1)
        kernel[(1,)](source, destination, num_warps=8)
        torch.cuda.synchronize()
        mismatch = destination != source
        print(name, int(mismatch.sum().item()), mismatch.sum(dim=1).tolist())


@triton.jit
def _copy_kernel_2d_sme_rank3_without_ws(src, dst):
    storage = tle.gpu.alloc(
        [2, 4, 512],
        dtype=tl.bfloat16,
        layout=tle.gpu.iluvatar_sme_shared_layout(
            [2, 4, 512],
            tl.bfloat16,
            [2, 1, 0],
            [1, 1, 1],
            [1, 1, 1],
            [2, 1, 0],
        ),
        nv_mma_shared_layout=False,
    )
    pipe = tle.pipe(capacity=2, scope="cta", name="transport_2d_sme_no_ws", tile=storage)
    writer = pipe.writer()
    reader = pipe.reader()
    rows = tl.arange(0, 4)[:, None]
    cols = tl.arange(0, 512)[None, :]
    slot = writer.acquire(0)
    src_ptrs = tle.gpu.set_layout(
        src + rows * 512 + cols,
        tle.gpu.IluvatarSmeBlockEncoding(
            [4, 512], tl.bfloat16, [2, 4], [2, 32], [1, 4], [1, 0]
        ),
    )
    tle.gpu.copy(src_ptrs, slot.tile, [4, 512], is_async=True, input_stride=512)
    tle.gpu.async_commit_group()
    tle.gpu.async_wait_group(0)
    writer.commit(0)
    ready = reader.wait(0)
    values = tl.load(tle.gpu.local_ptr(ready.slot.tile))
    tl.store(dst + rows * 512 + cols, values)
    reader.release(0)


def test_ws_explicit_copy_2d_sme_rank3_without_ws_preserves_payload():
    source = torch.arange(4 * 512, device="cuda", dtype=torch.float32).to(
        torch.bfloat16
    ).reshape(4, 512)
    destination = torch.full_like(source, -1)
    _copy_kernel_2d_sme_rank3_without_ws[(1,)](source, destination, num_warps=4)
    torch.cuda.synchronize()
    torch.testing.assert_close(destination, source, rtol=0, atol=0)


@triton.jit(noinline=True)
def _produce_2d_pair(writer0, writer1, src, STEPS: tl.constexpr):
    rows = tl.arange(0, 4)[:, None]
    cols = tl.arange(0, 512)[None, :]
    for step in range(STEPS):
        slot0 = writer0.acquire(step)
        tle.gpu.copy(
            src + step * 2048 + rows * 512 + cols,
            slot0.tile,
            [4, 512],
            is_async=True,
        )
        tle.gpu.async_commit_group()
        tle.gpu.async_wait_group(0)
        writer0.commit(step)
    writer0.close(STEPS)
    writer0.pipe.wait_drained()
    for step in range(STEPS):
        slot1 = writer1.acquire(step)
        tle.gpu.copy(
            src + STEPS * 2048 + step * 2048 + rows * 512 + cols,
            slot1.tile,
            [4, 512],
            is_async=True,
        )
        tle.gpu.async_commit_group()
        tle.gpu.async_wait_group(0)
        writer1.commit(step)
    writer1.close(STEPS)
    writer1.pipe.wait_drained()


@triton.jit(noinline=True)
def _consume_2d_pair(reader0, reader1, dst, STEPS: tl.constexpr):
    rows = tl.arange(0, 4)[:, None]
    cols = tl.arange(0, 512)[None, :]
    for step in range(STEPS):
        ready0 = reader0.wait(step)
        values0 = tl.load(tle.gpu.local_ptr(
            ready0.slot.tile,
            (tl.broadcast_to(rows, [4, 512]), tl.broadcast_to(cols, [4, 512])),
            [4, 512],
        ))
        tl.store(dst + step * 2048 + rows * 512 + cols, values0)
        reader0.release(step)
    reader0.pipe.wait_drained()
    for step in range(STEPS):
        ready1 = reader1.wait(step)
        values1 = tl.load(tle.gpu.local_ptr(
            ready1.slot.tile,
            (tl.broadcast_to(rows, [4, 512]), tl.broadcast_to(cols, [4, 512])),
            [4, 512],
        ))
        tl.store(dst + STEPS * 2048 + step * 2048 + rows * 512 + cols, values1)
        reader1.release(step)
    reader1.pipe.wait_drained()


@triton.jit
def _copy_kernel_2d_pair(src, dst, STEPS: tl.constexpr, PW: tl.constexpr):
    arena = tle.gpu.alloc(
        [2 * 4 * 512 * 2],
        dtype=tl.uint8,
        scope=tle.gpu.smem,
        nv_mma_shared_layout=False,
    )
    storage0 = tle.gpu.alloc(
        [2, 4, 512],
        dtype=tl.bfloat16,
        scope=tle.gpu.smem,
        nv_mma_shared_layout=False,
        alias=arena,
        alias_offset_bytes=0,
    )
    storage1 = tle.gpu.alloc(
        [2, 4, 512],
        dtype=tl.bfloat16,
        scope=tle.gpu.smem,
        nv_mma_shared_layout=False,
        alias=arena,
        alias_offset_bytes=0,
    )
    pipe0 = tle.pipe(capacity=2, scope="cta", name="transport_2d_pair_0", tile=storage0)
    pipe1 = tle.pipe(capacity=2, scope="cta", name="transport_2d_pair_1", tile=storage1)
    tle.gpu.warp_specialize([
        (_consume_2d_pair, (pipe0.reader(), pipe1.reader(), dst, STEPS)),
        (_produce_2d_pair, (pipe0.writer(), pipe1.writer(), src, STEPS)),
    ], [PW], [64])


@pytest.mark.parametrize("producer_warps", [4, 8])
def test_ws_explicit_copy_aliased_2d_pipes_are_deterministic(producer_warps):
    """Distinct pipe tokens must protect separately drained aliased slots."""
    steps = 32
    source = torch.randn(2 * steps * 2048, device="cuda", dtype=torch.bfloat16)
    destination = torch.empty_like(source)
    for _ in range(5):
        source.normal_()
        destination.fill_(float("nan"))
        _copy_kernel_2d_pair[(1,)](
            source,
            destination,
            steps,
            producer_warps,
            num_warps=16,
        )
        torch.testing.assert_close(destination, source, rtol=0, atol=0)
