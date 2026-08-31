# flagtree tle
"""
Unit tests for TLE buffered_tensor.slot(stage).
"""

import pytest
import torch
import triton
import triton.language as tl
import triton.experimental.tle.language as tle


def _is_enflame_backend():
    target = triton.runtime.driver.active.get_current_target()
    return target.backend == "gcu"


def _require_cuda():
    try:
        if _is_enflame_backend():
            pass
        else:
            torch.cuda.init()
    except Exception as exc:
        pytest.skip(f"CUDA init failed: {exc}")


@pytest.fixture(scope="module", autouse=True)
def _cuda_guard():
    _require_cuda()


@triton.jit
def _slot_local_ptr_store_kernel(out_ptr, BLOCK: tl.constexpr):
    idx = tl.arange(0, BLOCK)
    smem = tle.gpu.alloc([2, BLOCK], dtype=tl.int32, layout=None, scope=tle.gpu.smem, nv_mma_shared_layout=False)
    slot = smem.slot(0)
    ptrs = tle.gpu.local_ptr(slot, (idx, ))
    tl.store(ptrs, idx + 7)
    vals = tl.load(ptrs)
    tl.store(out_ptr + idx, vals)


@triton.jit
def _three_stage_pipe_producer(writer, src, BLOCK: tl.constexpr, TILES: tl.constexpr):
    offsets = tl.arange(0, BLOCK)
    for sequence in tl.range(0, TILES):
        slot = writer.acquire(sequence)
        tle.gpu.copy(
            src + sequence * BLOCK + offsets,
            slot.value,
            [BLOCK],
        )
        writer.commit(sequence)


@triton.jit
def _three_stage_pipe_consumer(reader, dst, BLOCK: tl.constexpr, TILES: tl.constexpr):
    offsets = tl.arange(0, BLOCK)
    for sequence in tl.range(0, TILES):
        ready = reader.wait(sequence)
        ptrs = tle.gpu.local_ptr(ready.slot.value, (offsets, ))
        tl.store(dst + sequence * BLOCK + offsets, tl.load(ptrs))
        reader.release(sequence)


@triton.jit(noinline=True)
def _rank5_tma_pipe_producer(writer, descriptor, tiles: tl.constexpr):
    for sequence in tl.range(0, tiles):
        slot = writer.acquire(sequence)
        tle.gpu.copy(
            descriptor,
            slot.value,
            [1, 1, 64, 1, 128],
            [0, 0, 0, 0, 0],
            eviction_policy="evict_first",
        )
        writer.commit(sequence)


@triton.jit
def _rank5_tma_pipe_consumer(reader, dst, tiles: tl.constexpr):
    offsets = tl.arange(0, 64)[:, None] * 128 + tl.arange(0, 128)[None, :]
    for sequence in tl.range(0, tiles):
        ready = reader.wait(sequence)
        values = tl.load(tle.gpu.local_ptr(ready.slot.value)).reshape((64, 128))
        tl.store(dst + sequence * 64 * 128 + offsets, values)
        reader.release(sequence)


@triton.jit
def _rank5_tma_pipe_kernel(descriptor, dst, tiles: tl.constexpr):
    stages: tl.constexpr = 2
    values = tle.gpu.alloc(
        [stages, 1, 1, 64, 1, 128],
        dtype=tl.bfloat16,
        layout=None,
        scope=tle.gpu.smem,
        nv_mma_shared_layout=True,
    )
    pipe = tle.pipe(
        capacity=stages,
        scope="cta",
        name="rank5_tma_pipe",
        value=values,
    )
    tle.gpu.warp_specialize(
        [
            (_rank5_tma_pipe_consumer, (pipe.reader(), dst, tiles)),
            (_rank5_tma_pipe_producer, (pipe.writer(), descriptor, tiles)),
        ],
        [1],
        [48],
    )


@triton.jit(noinline=True)
def _rank3_tma_pipe_producer(writer, descriptor, tiles: tl.constexpr):
    for sequence in tl.range(0, tiles):
        slot = writer.acquire(sequence)
        tle.gpu.copy(
            descriptor,
            slot.value,
            [2, 64, 64],
            [0, 0, 0],
            eviction_policy="evict_first",
        )
        writer.commit(sequence)


@triton.jit
def _rank3_tma_pipe_consumer(reader, dst, tiles: tl.constexpr):
    offsets = tl.arange(0, 128)[:, None] * 64 + tl.arange(0, 64)[None, :]
    for sequence in tl.range(0, tiles):
        ready = reader.wait(sequence)
        values = tl.load(tle.gpu.local_ptr(ready.slot.value)).reshape((128, 64))
        tl.store(dst + sequence * 128 * 64 + offsets, values)
        reader.release(sequence)


@triton.jit
def _rank3_tma_pipe_kernel(
    descriptor, dst, stages: tl.constexpr, tiles: tl.constexpr
):
    values = tle.gpu.alloc(
        [stages, 2, 64, 64],
        dtype=tl.bfloat16,
        layout=None,
        scope=tle.gpu.smem,
        nv_mma_shared_layout=True,
    )
    pipe = tle.pipe(
        capacity=stages,
        scope="cta",
        name="rank3_tma_pipe",
        value=values,
    )
    tle.gpu.warp_specialize(
        [
            (_rank3_tma_pipe_consumer, (pipe.reader(), dst, tiles)),
            (_rank3_tma_pipe_producer, (pipe.writer(), descriptor, tiles)),
        ],
        [1],
        [48],
    )


_rank3_mma_layout = tl.constexpr(
    tle.gpu.MmaEncoding([2, 0], [4, 1], [16, 8])
)
_rank3_mma_weight_layout = tl.constexpr(
    tle.gpu.DotOperandEncoding(0, _rank3_mma_layout, 4)
)
_rank3_mma_activation_layout = tl.constexpr(
    tle.gpu.DotOperandEncoding(1, _rank3_mma_layout, 4)
)
_rank3_mma_output_layout = tl.constexpr(
    tle.gpu.SlicedEncoding(1, _rank3_mma_layout)
)


@triton.jit
def _rank3_tma_mma_pipe_producer(
    writer, descriptor, tiles: tl.constexpr, k_tiles: tl.constexpr
):
    for sequence in tl.range(0, tiles):
        slot = writer.acquire(sequence)
        tle.gpu.copy(
            descriptor,
            slot.value,
            [2, 64, 128],
            [(sequence % k_tiles) * 2, 0, 0],
            eviction_policy="evict_first",
        )
        writer.commit(sequence)
    writer.close(tiles)


@triton.jit
def _rank3_tma_mma_pipe_consumer(reader, dst, tiles: tl.constexpr):
    activation = tle.encoding(
        tl.full((128, 8), 1.0, tl.float32).to(tl.float8e4nv),
        _rank3_mma_activation_layout,
    )
    local_n = tle.encoding(tl.arange(0, 64), _rank3_mma_output_layout)
    local_column = tle.encoding(
        tl.arange(0, 8)[None, :], _rank3_mma_layout
    )
    for sequence in tl.range(0, tiles):
        ready = reader.wait(sequence)
        result = tle.encoding(
            tl.zeros((64, 8), tl.float32), _rank3_mma_layout
        )
        for group in tl.static_range(0, 2):
            weight = tle.encoding(
                tl.load(
                    tle.gpu.local_ptr(
                        ready.slot.value.subslice(
                            [group, 0, 0], [1, 64, 128]
                        )
                    )
                ).reshape((64, 128)),
                _rank3_mma_weight_layout,
            )
            result += tl.dot(
                weight,
                activation,
                acc=tle.encoding(
                    tl.zeros((64, 8), tl.float32), _rank3_mma_layout
                ),
                out_dtype=tl.float32,
            )
        reduced = tle.encoding(
            tl.sum(tl.where(local_column == 0, result, 0.0), axis=1),
            _rank3_mma_output_layout,
        )
        tl.store(dst + sequence * 64 + local_n, reduced)
        reader.release(sequence)


@triton.jit
def _rank3_tma_mma_pipe_kernel(
    descriptor,
    dst,
    stages: tl.constexpr,
    tiles: tl.constexpr,
    k_tiles: tl.constexpr,
):
    values = tle.gpu.alloc(
        [stages, 2, 64, 128],
        dtype=tl.float8e4nv,
        layout=None,
        scope=tle.gpu.smem,
        nv_mma_shared_layout=True,
    )
    pipe = tle.pipe(
        capacity=stages,
        scope="cta",
        name="rank3_tma_mma_pipe",
        value=values,
    )
    tle.gpu.warp_specialize(
        [
            (_rank3_tma_mma_pipe_consumer, (pipe.reader(), dst, tiles)),
            (
                _rank3_tma_mma_pipe_producer,
                (pipe.writer(), descriptor, tiles, k_tiles),
            ),
        ],
        [1],
        [48],
    )


@triton.jit
def _three_stage_pipe_copy_kernel(src, dst, BLOCK: tl.constexpr, TILES: tl.constexpr):
    stages: tl.constexpr = 3
    values = tle.gpu.alloc(
        [stages, BLOCK],
        dtype=src.dtype.element_ty,
        layout=None,
        scope=tle.gpu.smem,
        nv_mma_shared_layout=False,
    )
    pipe = tle.pipe(
        capacity=stages,
        scope="cta",
        name="three_stage_copy",
        value=values,
    )
    tle.gpu.warp_specialize(
        [
            (_three_stage_pipe_consumer, (pipe.reader(), dst, BLOCK, TILES)),
            (_three_stage_pipe_producer, (pipe.writer(), src, BLOCK, TILES)),
        ],
        [1],
        [48],
    )


@triton.jit
def _one_shot_handoff_owner(writer, owned_storage, BLOCK: tl.constexpr):
    offsets = tl.arange(0, BLOCK)
    pointers = tle.gpu.local_ptr(owned_storage, (offsets, ))
    tl.store(pointers, offsets + 11)
    writer.commit(0)


@triton.jit
def _one_shot_handoff_next(reader, owned_storage, dst, BLOCK: tl.constexpr):
    reader.wait(0)
    offsets = tl.arange(0, BLOCK)
    pointers = tle.gpu.local_ptr(owned_storage, (offsets, ))
    tl.store(dst + offsets, tl.load(pointers))


@triton.jit
def _one_shot_shared_alias_handoff_kernel(dst, BLOCK: tl.constexpr):
    arena = tle.gpu.alloc(
        [BLOCK],
        dtype=tl.int32,
        layout=None,
        scope=tle.gpu.smem,
        nv_mma_shared_layout=False,
    )
    owned_storage = tle.gpu.alloc(
        [BLOCK],
        dtype=tl.int32,
        layout=None,
        scope=tle.gpu.smem,
        alias=arena,
        alias_offset_bytes=0,
        nv_mma_shared_layout=False,
    )
    handoff = tle.pipe(
        capacity=1,
        scope="cta",
        name="one_shot_shared_alias_handoff",
        one_shot=True,
    )
    tle.gpu.warp_specialize(
        [
            (
                _one_shot_handoff_owner,
                (handoff.writer(), owned_storage, BLOCK),
            ),
            (
                _one_shot_handoff_next,
                (handoff.reader(), owned_storage, dst, BLOCK),
            ),
        ],
        [1],
        [48],
    )


def test_buffered_tensor_slot_lowers_to_memdesc_index_and_executes():
    block = 64
    out = torch.empty((block, ), device="cuda", dtype=torch.int32)

    compiled = _slot_local_ptr_store_kernel.warmup(out, BLOCK=block, grid=(1, ), num_warps=4)
    ttgir = compiled.asm["ttgir"]
    assert "ttg.memdesc_index" in ttgir
    assert "!ttg.memdesc<2x64xi32" in ttgir
    assert "!ttg.memdesc<64xi32" in ttgir

    _slot_local_ptr_store_kernel[(1, )](out, BLOCK=block, num_warps=4)
    expected = torch.arange(0, block, device="cuda", dtype=torch.int32) + 7
    torch.testing.assert_close(out, expected, atol=0, rtol=0)


def test_three_stage_pipe_uses_non_power_of_two_payload_and_padded_control_state():
    block = 64
    tiles = 6
    src = torch.arange(block * tiles, device="cuda", dtype=torch.int32)
    dst = torch.empty_like(src)

    compiled = _three_stage_pipe_copy_kernel.warmup(
        src,
        dst,
        BLOCK=block,
        TILES=tiles,
        grid=(1, ),
        num_warps=8,
    )
    ttgir = compiled.asm["ttgir"]
    assert "!ttg.memdesc<3x64xi32" in ttgir
    assert "!ttg.memdesc<4x1xi64" in ttgir

    _three_stage_pipe_copy_kernel[(1, )](
        src,
        dst,
        BLOCK=block,
        TILES=tiles,
        num_warps=8,
    )
    torch.testing.assert_close(dst, src, atol=0, rtol=0)


def test_noinline_rank5_tma_pipe_producer_executes():
    from triton.tools.tensor_descriptor import TensorDescriptor

    tiles = 3
    shape = (1, 1, 64, 1, 128)
    src = torch.arange(
        64 * 128, device="cuda", dtype=torch.float32
    ).to(torch.bfloat16).reshape(shape)
    dst = torch.empty((tiles, 64, 128), device="cuda", dtype=torch.bfloat16)
    descriptor = TensorDescriptor.from_tensor(src, block_shape=list(shape))

    compiled = _rank5_tma_pipe_kernel[(1, )](
        descriptor,
        dst,
        tiles=tiles,
        num_warps=4,
    )

    torch.testing.assert_close(
        dst, src.reshape(1, 64, 128).expand_as(dst), atol=0, rtol=0
    )
    ttgir = compiled.asm["ttgir"]
    assert "_rank5_tma_pipe_producer" in ttgir
    assert "noinline = true" in ttgir
    assert "ttng.async_tma_copy_global_to_local" in ttgir
    assert compiled.asm["ptx"].count("cp.async.bulk.tensor.5d") == 2 * tiles


@pytest.mark.parametrize("stages", [4, 8])
def test_rank3_tma_pipe_large_shared_ring_executes(stages):
    from triton.tools.tensor_descriptor import TensorDescriptor

    tiles = stages + 3
    shape = (2, 64, 64)
    src = torch.arange(
        2 * 64 * 64, device="cuda", dtype=torch.float32
    ).to(torch.bfloat16).reshape(shape)
    dst = torch.empty((tiles, 128, 64), device="cuda", dtype=torch.bfloat16)
    descriptor = TensorDescriptor.from_tensor(src, block_shape=list(shape))

    compiled = _rank3_tma_pipe_kernel[(1, )](
        descriptor,
        dst,
        stages=stages,
        tiles=tiles,
        num_warps=4,
    )

    torch.testing.assert_close(
        dst, src.reshape(1, 128, 64).expand_as(dst), atol=0, rtol=0
    )
    assert "ttng.async_tma_copy_global_to_local" in compiled.asm["ttgir"]


@pytest.mark.parametrize("stages", [4, 8])
def test_rank3_tma_mma_pipe_large_shared_ring_executes(stages):
    from triton.tools.tensor_descriptor import TensorDescriptor

    k_tiles = 9
    n_tiles = 5
    tiles = n_tiles * k_tiles
    tile_values = torch.arange(
        1, k_tiles * 2, device="cuda", dtype=torch.float32
    )
    src = tile_values[:, None, None].expand(k_tiles * 2 - 1, 64, 128)
    src = src.to(torch.float8_e4m3fn).contiguous()
    dst = torch.empty((tiles, 64), device="cuda", dtype=torch.float32)
    descriptor = TensorDescriptor.from_tensor(
        src, block_shape=[2, 64, 128]
    )

    compiled = _rank3_tma_mma_pipe_kernel[(1, )](
        descriptor,
        dst,
        stages=stages,
        tiles=tiles,
        k_tiles=k_tiles,
        num_warps=4,
    )

    quantized_values = tile_values.to(torch.float8_e4m3fn).to(torch.float32)
    paired_values = torch.cat(
        (quantized_values, torch.zeros(1, device="cuda"))
    ).reshape(k_tiles, 2).sum(dim=1)
    expected = (
        paired_values.repeat(n_tiles) * 128
    )[:, None].expand_as(dst)
    torch.testing.assert_close(dst, expected, atol=0, rtol=0)
    assert "tt.dot" in compiled.asm["ttgir"]
    assert "mma.sync" in compiled.asm["ptx"]


def test_one_shot_pipe_hands_aliased_shared_storage_to_next_task():
    block = 64
    dst = torch.empty((block, ), device="cuda", dtype=torch.int32)

    compiled = _one_shot_shared_alias_handoff_kernel.warmup(
        dst,
        BLOCK=block,
        grid=(1, ),
        num_warps=8,
    )
    ttgir = compiled.asm["ttgir"]
    assert "ttng.arrive_barrier" in ttgir
    assert "ttng.wait_barrier" in ttgir
    assert "nvws.producer_acquire" not in ttgir
    assert "nvws.consumer_release" not in ttgir

    _one_shot_shared_alias_handoff_kernel[(1, )](
        dst,
        BLOCK=block,
        num_warps=8,
    )
    expected = torch.arange(0, block, device="cuda", dtype=torch.int32) + 11
    torch.testing.assert_close(dst, expected, atol=0, rtol=0)
