"""Compile and runtime coverage for the MUSA TLE pipe contract."""

import re

import pytest
import torch
import triton
import triton.language as tl
import triton.experimental.tle.language as tle
from triton._C import libtriton
from triton._C.libtriton import ir
from triton.backends.compiler import Language
from triton.compiler import ASTSource
from triton.compiler.errors import CompilationError
from triton.tools.tensor_descriptor import TensorDescriptor

from test_tle_utils import mthreads_backend, require_mthreads_libtriton, tme_descriptor_attrs

require_mthreads_libtriton()

_LOCAL_STORE_WRITER_LAYOUT = tl.constexpr(tle.gpu.BlockEncoding([1], [32], [4], [0]))
_LOCAL_STORE_WRITER_LAYOUT_2D = tl.constexpr(tle.gpu.BlockEncoding([1, 1], [1, 32], [4, 1], [1, 0]))
_LOCAL_STORE_READER_LAYOUT_2D_16 = tl.constexpr(tle.gpu.BlockEncoding([1, 1], [1, 32], [16, 1], [1, 0]))

_PIPE_REUSE_STAGES = (1, 2, 3)
_EXTERNAL_BARRIER_STAGES = (1, 2, 4)


def _phase_reuse_iterations(stages):
    # Every physical slot is reused through two complete phase transitions.
    return 4 * stages + 1


@triton.jit
def _pipe_consumer(reader, out, ITERATIONS: tl.constexpr):
    for iteration in tl.static_range(0, ITERATIONS):
        wait = reader.wait(iteration)
        tl.store(out + iteration, iteration + tl.where(wait.is_closed, 1000, 0))
        reader.release(iteration)


@triton.jit
def _pipe_producer(writer, desc, BLOCK: tl.constexpr, ITERATIONS: tl.constexpr):
    for iteration in tl.static_range(0, ITERATIONS):
        slot = writer.acquire(iteration)
        tle.gpu.copy(desc, slot.data, (BLOCK, ), (iteration * BLOCK, ))
        writer.commit(iteration)


@triton.jit
def _pipe_tme_store_consumer(reader, desc, BLOCK: tl.constexpr, ITERATIONS: tl.constexpr):
    for iteration in tl.static_range(0, ITERATIONS):
        wait = reader.wait(iteration)
        tle.gpu.copy(wait.slot.data, desc, (BLOCK, ), (iteration * BLOCK, ))
        reader.release(iteration)


@triton.jit
def _non_ws_pipe_mm_kernel(
    a_desc,
    b_desc,
    out,
    K_TILES: tl.constexpr,
    STAGES: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    a_smem = tle.gpu.alloc(
        (STAGES, BLOCK_M, BLOCK_K),
        dtype=tl.float16,
        layout=None,
        nv_mma_shared_layout=True,
    )
    b_smem = tle.gpu.alloc(
        (STAGES, BLOCK_K, BLOCK_N),
        dtype=tl.float16,
        layout=None,
        nv_mma_shared_layout=True,
    )
    a_pipe = tle.pipe(capacity=STAGES, name="runtime_a", a=a_smem)
    b_pipe = tle.pipe(capacity=STAGES, name="runtime_b", b=b_smem)
    a_writer = a_pipe.writer()
    b_writer = b_pipe.writer()
    a_reader = a_pipe.reader()
    b_reader = b_pipe.reader()

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k_iter in tl.static_range(0, K_TILES):
        a_slot = a_writer.acquire(k_iter)
        b_slot = b_writer.acquire(k_iter)
        k_offset = k_iter * BLOCK_K
        tle.gpu.copy(a_desc, a_slot.a, (BLOCK_M, BLOCK_K), (0, k_offset))
        tle.gpu.copy(b_desc, b_slot.b, (BLOCK_K, BLOCK_N), (k_offset, 0))
        a_writer.commit(k_iter)
        b_writer.commit(k_iter)

        a_wait = a_reader.wait(k_iter)
        b_wait = b_reader.wait(k_iter)
        acc = tle.gpu.wgmma(a_wait.slot.a, b_wait.slot.b, acc)
        acc = tle.gpu.wgmma_wait(0, acc)
        a_reader.release(k_iter)
        b_reader.release(k_iter)

    offsets = tl.arange(0, BLOCK_M)[:, None] * BLOCK_N + tl.arange(0, BLOCK_N)[None, :]
    tl.store(out + offsets, acc.to(tl.float16))


@triton.jit
def _ws_pipe_mm_consumer(
    a_reader,
    b_reader,
    out,
    K_TILES: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k_iter in tl.range(0, K_TILES, num_stages=1):
        a_wait = a_reader.wait(k_iter)
        b_wait = b_reader.wait(k_iter)
        acc = tle.gpu.wgmma(a_wait.slot.a, b_wait.slot.b, acc)
        acc = tle.gpu.wgmma_wait(0, acc)
        a_reader.release(k_iter)
        b_reader.release(k_iter)

    offsets = tl.arange(0, BLOCK_M)[:, None] * BLOCK_N + tl.arange(0, BLOCK_N)[None, :]
    tl.store(out + offsets, acc.to(tl.float16))


@triton.jit
def _ws_pipe_mm_producer(
    a_writer,
    b_writer,
    a_desc,
    b_desc,
    K_TILES: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    for k_iter in tl.range(0, K_TILES, num_stages=1):
        a_slot = a_writer.acquire(k_iter)
        b_slot = b_writer.acquire(k_iter)
        k_offset = k_iter * BLOCK_K
        tle.gpu.copy(a_desc, a_slot.a, (BLOCK_M, BLOCK_K), (0, k_offset))
        tle.gpu.copy(b_desc, b_slot.b, (BLOCK_K, BLOCK_N), (k_offset, 0))
        a_writer.commit(k_iter)
        b_writer.commit(k_iter)


@triton.jit
def _ws_pipe_mm_kernel(
    a_desc,
    b_desc,
    out,
    K_TILES: tl.constexpr,
    STAGES: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    a_smem = tle.gpu.alloc(
        (STAGES, BLOCK_M, BLOCK_K),
        dtype=tl.float16,
        nv_mma_shared_layout=True,
    )
    b_smem = tle.gpu.alloc(
        (STAGES, BLOCK_K, BLOCK_N),
        dtype=tl.float16,
        nv_mma_shared_layout=True,
    )
    a_pipe = tle.pipe(capacity=STAGES, name="ws_runtime_a", a=a_smem)
    b_pipe = tle.pipe(capacity=STAGES, name="ws_runtime_b", b=b_smem)
    tle.gpu.warp_specialize(
        [
            (
                _ws_pipe_mm_consumer,
                (a_pipe.reader(), b_pipe.reader(), out, K_TILES, BLOCK_M, BLOCK_N),
            ),
            (
                _ws_pipe_mm_producer,
                (
                    a_pipe.writer(),
                    b_pipe.writer(),
                    a_desc,
                    b_desc,
                    K_TILES,
                    BLOCK_M,
                    BLOCK_N,
                    BLOCK_K,
                ),
            ),
        ],
        worker_num_warps=[4],
        worker_num_regs=[24],
    )


@triton.jit
def _ws_multi_field_pipe_mm_consumer(
    reader,
    out,
    K_TILES: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k_iter in tl.range(0, K_TILES, num_stages=1):
        wait = reader.wait(k_iter)
        acc = tle.gpu.wgmma(wait.slot.a, wait.slot.b, acc)
        acc = tle.gpu.wgmma_wait(0, acc)
        reader.release(k_iter)

    offsets = tl.arange(0, BLOCK_M)[:, None] * BLOCK_N + tl.arange(0, BLOCK_N)[None, :]
    tl.store(out + offsets, acc.to(tl.float16))


@triton.jit
def _ws_multi_field_pipe_mm_producer(
    writer,
    a_desc,
    b_desc,
    K_TILES: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    for k_iter in tl.range(0, K_TILES, num_stages=1):
        slot = writer.acquire(k_iter)
        k_offset = k_iter * BLOCK_K
        tle.gpu.copy(a_desc, slot.a, (BLOCK_M, BLOCK_K), (0, k_offset))
        tle.gpu.copy(b_desc, slot.b, (BLOCK_K, BLOCK_N), (k_offset, 0))
        writer.commit(k_iter)


@triton.jit
def _ws_multi_field_pipe_mm_kernel(
    a_desc,
    b_desc,
    out,
    K_TILES: tl.constexpr,
    STAGES: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    a_smem = tle.gpu.alloc(
        (STAGES, BLOCK_M, BLOCK_K),
        dtype=tl.float16,
        nv_mma_shared_layout=True,
    )
    b_smem = tle.gpu.alloc(
        (STAGES, BLOCK_K, BLOCK_N),
        dtype=tl.float16,
        nv_mma_shared_layout=True,
    )
    pipe = tle.pipe(capacity=STAGES, name="ws_multi_field", a=a_smem, b=b_smem)
    tle.gpu.warp_specialize(
        [
            (
                _ws_multi_field_pipe_mm_consumer,
                (pipe.reader(), out, K_TILES, BLOCK_M, BLOCK_N),
            ),
            (
                _ws_multi_field_pipe_mm_producer,
                (
                    pipe.writer(),
                    a_desc,
                    b_desc,
                    K_TILES,
                    BLOCK_M,
                    BLOCK_N,
                    BLOCK_K,
                ),
            ),
        ],
        worker_num_warps=[4],
        worker_num_regs=[24],
    )


@triton.jit
def _heterogeneous_multi_field_roundtrip_kernel(
    half_src_desc,
    float_src_desc,
    half_dst_desc,
    float_dst_desc,
    STAGES: tl.constexpr,
    ITERATIONS: tl.constexpr,
    HALF_M: tl.constexpr,
    HALF_N: tl.constexpr,
    FLOAT_M: tl.constexpr,
    FLOAT_N: tl.constexpr,
):
    half_smem = tle.gpu.alloc(
        (STAGES, HALF_M, HALF_N),
        dtype=tl.float16,
        nv_mma_shared_layout=False,
    )
    float_smem = tle.gpu.alloc(
        (STAGES, FLOAT_M, FLOAT_N),
        dtype=tl.float32,
        nv_mma_shared_layout=False,
    )
    pipe = tle.pipe(capacity=STAGES, name="heterogeneous", half=half_smem, float_data=float_smem)
    writer = pipe.writer()
    reader = pipe.reader()

    for iteration in tl.static_range(0, ITERATIONS):
        slot = writer.acquire(iteration)
        tle.gpu.copy(
            half_src_desc,
            slot.half,
            (HALF_M, HALF_N),
            (iteration * HALF_M, 0),
        )
        tle.gpu.copy(
            float_src_desc,
            slot.float_data,
            (FLOAT_M, FLOAT_N),
            (iteration * FLOAT_M, 0),
        )
        writer.commit(iteration)

        wait = reader.wait(iteration)
        tle.gpu.copy(
            wait.slot.half,
            half_dst_desc,
            (HALF_M, HALF_N),
            (iteration * HALF_M, 0),
        )
        tle.gpu.copy(
            wait.slot.float_data,
            float_dst_desc,
            (FLOAT_M, FLOAT_N),
            (iteration * FLOAT_M, 0),
        )
        reader.release(iteration)


@triton.jit
def _mixed_multi_local_store_roundtrip_kernel(
    tme_src_desc,
    tme2_src_desc,
    tme_out,
    tme2_out,
    local_i32_out,
    local_f32_out,
    STAGES: tl.constexpr,
    ITERATIONS: tl.constexpr,
    M: tl.constexpr,
    N: tl.constexpr,
):
    tme_smem = tle.gpu.alloc((STAGES, M, N), dtype=tl.float16, nv_mma_shared_layout=False)
    tme2_smem = tle.gpu.alloc((STAGES, M, N), dtype=tl.float32, nv_mma_shared_layout=False)
    local_i32_smem = tle.gpu.alloc((STAGES, M, N), dtype=tl.int32, nv_mma_shared_layout=False)
    local_f32_smem = tle.gpu.alloc((STAGES, M, N), dtype=tl.float32, nv_mma_shared_layout=False)
    pipe = tle.pipe(
        capacity=STAGES,
        name="mixed_multiple_local",
        tme=tme_smem,
        tme2=tme2_smem,
        local_i32=local_i32_smem,
        local_f32=local_f32_smem,
    )
    writer = pipe.writer()
    reader = pipe.reader()
    rows = tl.arange(0, M)[:, None]
    cols = tl.arange(0, N)[None, :]
    row_indices = tl.broadcast_to(rows, (M, N))
    col_indices = tl.broadcast_to(cols, (M, N))
    row_indices = tle.gpu.set_layout(row_indices, _LOCAL_STORE_WRITER_LAYOUT_2D)
    col_indices = tle.gpu.set_layout(col_indices, _LOCAL_STORE_WRITER_LAYOUT_2D)
    for iteration in tl.static_range(0, ITERATIONS):
        slot = writer.acquire(iteration)
        tle.gpu.copy(tme_src_desc, slot.tme, (M, N), (iteration * M, 0))
        i32_values = (iteration * M * N + rows * N + cols).to(tl.int32)
        i32_values = tle.gpu.set_layout(i32_values, _LOCAL_STORE_WRITER_LAYOUT_2D)
        tl.store(tle.gpu.local_ptr(slot.local_i32, (row_indices, col_indices)), i32_values)
        tle.gpu.copy(tme2_src_desc, slot.tme2, (M, N), (iteration * M, 0))
        f32_values = i32_values.to(tl.float32) + 0.5
        f32_values = tle.gpu.set_layout(f32_values, _LOCAL_STORE_WRITER_LAYOUT_2D)
        tl.store(tle.gpu.local_ptr(slot.local_f32, (row_indices, col_indices)), f32_values)
        writer.commit(iteration)

        wait = reader.wait(iteration)
        tme_values = tl.load(tle.gpu.local_ptr(wait.slot.tme))
        tme2_values = tl.load(tle.gpu.local_ptr(wait.slot.tme2))
        i32_values = tl.load(tle.gpu.local_ptr(wait.slot.local_i32))
        f32_values = tl.load(tle.gpu.local_ptr(wait.slot.local_f32))
        tl.store(local_i32_out + iteration * M * N + rows * N + cols, i32_values)
        tl.store(local_f32_out + iteration * M * N + rows * N + cols, f32_values)
        tl.store(tme_out + iteration * M * N + rows * N + cols, tme_values)
        tl.store(tme2_out + iteration * M * N + rows * N + cols, tme2_values)
        reader.release(iteration)


@triton.jit
def _mixed_tme_local_store_consumer(
    reader,
    tme_out,
    local_out,
    STAGES: tl.constexpr,
    ITERATIONS: tl.constexpr,
    M: tl.constexpr,
    N: tl.constexpr,
):
    rows = tl.arange(0, M)[:, None]
    cols = tl.arange(0, N)[None, :]
    for iteration in tl.range(0, ITERATIONS, num_stages=1):
        wait = reader.wait(iteration)
        tme_values = tl.load(tle.gpu.local_ptr(wait.slot.tme))
        local_values = tl.load(tle.gpu.local_ptr(wait.slot.local))
        tl.store(tme_out + iteration * M * N + rows * N + cols, tme_values)
        tl.store(local_out + iteration * M * N + rows * N + cols, local_values)
        reader.release(iteration)


@triton.jit
def _mixed_tme_local_store_producer_ws(
    writer,
    tme_src_desc,
    STAGES: tl.constexpr,
    ITERATIONS: tl.constexpr,
    M: tl.constexpr,
    N: tl.constexpr,
):
    rows = tl.arange(0, M)[:, None]
    cols = tl.arange(0, N)[None, :]
    row_indices = tl.broadcast_to(rows, (M, N))
    col_indices = tl.broadcast_to(cols, (M, N))
    row_indices = tle.gpu.set_layout(row_indices, _LOCAL_STORE_WRITER_LAYOUT_2D)
    col_indices = tle.gpu.set_layout(col_indices, _LOCAL_STORE_WRITER_LAYOUT_2D)
    for iteration in tl.range(0, ITERATIONS, num_stages=1):
        slot = writer.acquire(iteration)
        tle.gpu.copy(tme_src_desc, slot.tme, (M, N), (iteration * M, 0))
        local_values = (iteration * M * N + rows * N + cols).to(tl.int32)
        local_values = tle.gpu.set_layout(local_values, _LOCAL_STORE_WRITER_LAYOUT_2D)
        tl.store(tle.gpu.local_ptr(slot.local, (row_indices, col_indices)), local_values)
        writer.commit(iteration)


@triton.jit
def _ws_mixed_tme_local_store_roundtrip_kernel(
    tme_src_desc,
    tme_out,
    local_out,
    STAGES: tl.constexpr,
    ITERATIONS: tl.constexpr,
    M: tl.constexpr,
    N: tl.constexpr,
):
    tme_smem = tle.gpu.alloc((STAGES, M, N), dtype=tl.float16, nv_mma_shared_layout=False)
    local_smem = tle.gpu.alloc((STAGES, M, N), dtype=tl.int32, nv_mma_shared_layout=False)
    pipe = tle.pipe(capacity=STAGES, name="ws_mixed_roundtrip", tme=tme_smem, local=local_smem)
    tle.gpu.warp_specialize(
        [
            (_mixed_tme_local_store_consumer, (pipe.reader(), tme_out, local_out, STAGES, ITERATIONS, M, N)),
            (_mixed_tme_local_store_producer_ws, (pipe.writer(), tme_src_desc, STAGES, ITERATIONS, M, N)),
        ],
        worker_num_warps=[4],
        worker_num_regs=[24],
    )


@triton.jit
def _local_store_producer(writer, BLOCK: tl.constexpr, ITERATIONS: tl.constexpr):
    for iteration in tl.static_range(0, ITERATIONS):
        slot = writer.acquire(iteration)
        tl.store(tle.gpu.local_ptr(slot.data, (0, )), 0.0)
        writer.commit(iteration)


@triton.jit
def _whole_field_local_store_producer(writer, BLOCK: tl.constexpr, ITERATIONS: tl.constexpr):
    offsets = tle.gpu.set_layout(tl.arange(0, BLOCK), _LOCAL_STORE_WRITER_LAYOUT)
    for iteration in tl.static_range(0, ITERATIONS):
        slot = writer.acquire(iteration)
        values = (iteration * BLOCK + offsets).to(tl.int32)
        tl.store(tle.gpu.local_ptr(slot.data, (offsets, )), values)
        writer.commit(iteration)


@triton.jit
def _whole_field_local_store_consumer(reader, out, BLOCK: tl.constexpr, ITERATIONS: tl.constexpr):
    offsets = tl.arange(0, BLOCK)
    for iteration in tl.static_range(0, ITERATIONS):
        wait = reader.wait(iteration)
        values = tl.load(tle.gpu.local_ptr(wait.slot.data, (offsets, )))
        tl.store(out + iteration * BLOCK + offsets, values)
        reader.release(iteration)


@triton.jit
def _non_ws_local_store_pipe_kernel(out, STAGES: tl.constexpr, BLOCK: tl.constexpr, ITERATIONS: tl.constexpr):
    smem = tle.gpu.alloc(
        (STAGES, BLOCK),
        dtype=tl.int32,
        layout=None,
        scope=tle.gpu.smem,
        nv_mma_shared_layout=False,
    )
    pipe = tle.pipe(capacity=STAGES, scope="cta", name="non_ws_local_store", data=smem)
    writer = pipe.writer()
    reader = pipe.reader()
    offsets = tl.arange(0, BLOCK)
    for iteration in tl.static_range(0, ITERATIONS):
        slot = writer.acquire(iteration)
        values = (iteration * BLOCK + offsets).to(tl.int32)
        tl.store(tle.gpu.local_ptr(slot.data, (offsets, )), values)
        writer.commit(iteration)

        wait = reader.wait(iteration)
        loaded = tl.load(tle.gpu.local_ptr(wait.slot.data, (offsets, )))
        tl.store(out + iteration * BLOCK + offsets, loaded)
        reader.release(iteration)


@triton.jit
def _non_ws_local_store_pipe_with_independent_buffer_kernel(
    out,
    independent_out,
    STAGES: tl.constexpr,
    BLOCK: tl.constexpr,
    ITERATIONS: tl.constexpr,
):
    smem = tle.gpu.alloc(
        (STAGES, BLOCK),
        dtype=tl.int32,
        layout=None,
        scope=tle.gpu.smem,
        nv_mma_shared_layout=False,
    )
    independent = tle.gpu.alloc(
        (BLOCK, ),
        dtype=tl.int32,
        layout=None,
        scope=tle.gpu.smem,
        nv_mma_shared_layout=False,
    )
    pipe = tle.pipe(capacity=STAGES, scope="cta", name="local_store_isolation", data=smem)
    writer = pipe.writer()
    reader = pipe.reader()
    offsets = tl.arange(0, BLOCK)
    for iteration in tl.static_range(0, ITERATIONS):
        slot = writer.acquire(iteration)
        values = (iteration * BLOCK + offsets).to(tl.int32)
        tl.store(tle.gpu.local_ptr(slot.data, (offsets, )), values)
        tl.store(tle.gpu.local_ptr(independent, (offsets, )), values + 1000)
        writer.commit(iteration)

        wait = reader.wait(iteration)
        loaded = tl.load(tle.gpu.local_ptr(wait.slot.data, (offsets, )))
        independent_loaded = tl.load(tle.gpu.local_ptr(independent, (offsets, )))
        tl.store(out + iteration * BLOCK + offsets, loaded)
        tl.store(independent_out + iteration * BLOCK + offsets, independent_loaded)
        reader.release(iteration)


@triton.jit
def _ws_local_store_pipe_kernel(out, STAGES: tl.constexpr, BLOCK: tl.constexpr, ITERATIONS: tl.constexpr):
    smem = tle.gpu.alloc(
        (STAGES, BLOCK),
        dtype=tl.int32,
        layout=None,
        scope=tle.gpu.smem,
        nv_mma_shared_layout=False,
    )
    pipe = tle.pipe(capacity=STAGES, scope="cta", name="ws_local_store", data=smem)
    tle.gpu.warp_specialize(
        [
            (_whole_field_local_store_consumer, (pipe.reader(), out, BLOCK, ITERATIONS)),
            (_whole_field_local_store_producer, (pipe.writer(), BLOCK, ITERATIONS)),
        ],
        worker_num_warps=[4],
        worker_num_regs=[24],
    )


@triton.jit
def _mixed_tme_local_store_producer(writer, desc, BLOCK: tl.constexpr, ITERATIONS: tl.constexpr):
    offsets = tle.gpu.set_layout(tl.arange(0, BLOCK), _LOCAL_STORE_WRITER_LAYOUT)
    for iteration in tl.static_range(0, ITERATIONS):
        slot = writer.acquire(iteration)
        tle.gpu.copy(desc, slot.data, (BLOCK, ), (iteration * BLOCK, ))
        tl.store(tle.gpu.local_ptr(slot.data, (offsets, )), offsets.to(tl.float16))
        writer.commit(iteration)


@triton.jit
def _multi_local_store_only_producer(writer, BLOCK: tl.constexpr, ITERATIONS: tl.constexpr):
    offsets = tle.gpu.set_layout(tl.arange(0, BLOCK), _LOCAL_STORE_WRITER_LAYOUT)
    for iteration in tl.static_range(0, ITERATIONS):
        slot = writer.acquire(iteration)
        values = (iteration * BLOCK + offsets).to(tl.int32)
        tl.store(tle.gpu.local_ptr(slot.first, (offsets, )), values)
        tl.store(tle.gpu.local_ptr(slot.second, (offsets, )), values + 1)
        writer.commit(iteration)


@triton.jit
def _multi_local_store_only_consumer(reader, out, BLOCK: tl.constexpr, ITERATIONS: tl.constexpr):
    offsets = tl.arange(0, BLOCK)
    for iteration in tl.static_range(0, ITERATIONS):
        wait = reader.wait(iteration)
        first = tl.load(tle.gpu.local_ptr(wait.slot.first, (offsets, )))
        second = tl.load(tle.gpu.local_ptr(wait.slot.second, (offsets, )))
        tl.store(out + iteration * BLOCK + offsets, first + second)
        reader.release(iteration)


@triton.jit
def _multi_local_store_only_pipe_kernel(
    desc,
    out,
    STAGES: tl.constexpr,
    BLOCK: tl.constexpr,
    ITERATIONS: tl.constexpr,
):
    first_smem = tle.gpu.alloc((STAGES, BLOCK), dtype=tl.int32, nv_mma_shared_layout=False)
    second_smem = tle.gpu.alloc((STAGES, BLOCK), dtype=tl.int32, nv_mma_shared_layout=False)
    pipe = tle.pipe(capacity=STAGES, name="invalid_multi_local", first=first_smem, second=second_smem)
    tle.gpu.warp_specialize(
        [
            (_multi_local_store_only_consumer, (pipe.reader(), out, BLOCK, ITERATIONS)),
            (_multi_local_store_only_producer, (pipe.writer(), BLOCK, ITERATIONS)),
        ],
        worker_num_warps=[4],
        worker_num_regs=[24],
    )


@triton.jit
def _double_tme_producer(writer, desc, BLOCK: tl.constexpr, ITERATIONS: tl.constexpr):
    for iteration in tl.static_range(0, ITERATIONS):
        slot = writer.acquire(iteration)
        tle.gpu.copy(desc, slot.data, (BLOCK, ), (iteration * BLOCK, ))
        tle.gpu.copy(desc, slot.data, (BLOCK, ), (iteration * BLOCK, ))
        writer.commit(iteration)


@triton.jit
def _ws_same_field_tme_fragments_consumer(
    reader,
    top_destination_desc,
    bottom_destination_desc,
    ITERATIONS: tl.constexpr,
):
    for iteration in tl.range(0, ITERATIONS, num_stages=1):
        wait = reader.wait(iteration)
        top_view = tle.gpu.alloc(
            (8, 32),
            dtype=tl.float16,
            alias=wait.slot.data,
            alias_offset_bytes=0,
            nv_mma_shared_layout=False,
        )
        bottom_view = tle.gpu.alloc(
            (8, 32),
            dtype=tl.float16,
            alias=wait.slot.data,
            alias_offset_bytes=8 * 32 * 2,
            nv_mma_shared_layout=False,
        )
        tle.gpu.copy(top_view, top_destination_desc, (8, 32), (iteration * 8, 0))
        tle.gpu.copy(bottom_view, bottom_destination_desc, (8, 32), (iteration * 8, 0))
        reader.release(iteration)


@triton.jit
def _ws_same_field_tme_fragments_producer(
    writer,
    top_source_desc,
    bottom_source_desc,
    ITERATIONS: tl.constexpr,
):
    for iteration in tl.range(0, ITERATIONS, num_stages=1):
        slot = writer.acquire(iteration)
        top_view = tle.gpu.alloc(
            (8, 32),
            dtype=tl.float16,
            alias=slot.data,
            alias_offset_bytes=0,
            nv_mma_shared_layout=False,
        )
        bottom_view = tle.gpu.alloc(
            (8, 32),
            dtype=tl.float16,
            alias=slot.data,
            alias_offset_bytes=8 * 32 * 2,
            nv_mma_shared_layout=False,
        )
        tle.gpu.copy(top_source_desc, top_view, (8, 32), (iteration * 8, 0))
        tle.gpu.copy(bottom_source_desc, bottom_view, (8, 32), (iteration * 8, 0))
        writer.commit(iteration)


@triton.jit
def _ws_same_field_tme_fragments_kernel(
    top_source_desc,
    bottom_source_desc,
    top_destination_desc,
    bottom_destination_desc,
    STAGES: tl.constexpr,
    ITERATIONS: tl.constexpr,
):
    smem = tle.gpu.alloc((STAGES, 16, 32), dtype=tl.float16, nv_mma_shared_layout=False)
    pipe = tle.pipe(capacity=STAGES, name="ws_same_field_tme_fragments", data=smem)
    tle.gpu.warp_specialize(
        [
            (
                _ws_same_field_tme_fragments_consumer,
                (pipe.reader(), top_destination_desc, bottom_destination_desc, ITERATIONS),
            ),
            (
                _ws_same_field_tme_fragments_producer,
                (pipe.writer(), top_source_desc, bottom_source_desc, ITERATIONS),
            ),
        ],
        worker_num_warps=[4],
        worker_num_regs=[24],
    )


@triton.jit
def _external_same_field_tme_fragments_kernel(
    top_source_desc,
    bottom_source_desc,
    top_destination_desc,
    bottom_destination_desc,
    STAGES: tl.constexpr,
    ITERATIONS: tl.constexpr,
):
    """Use one external full ring for two disjoint TME fragments."""
    smem = tle.gpu.alloc((STAGES, 16, 32), dtype=tl.float16, nv_mma_shared_layout=False)
    full = tle.gpu.alloc_barriers(
        STAGES,
        arrive_count=1,
        init=tle.gpu.PENDING,
        expect_bytes=16 * 32 * 2,
    )
    pipe = tle.pipe(capacity=STAGES, name="external_same_field_tme_fragments", data=smem)
    writer = pipe.writer()
    reader = pipe.reader()
    for iteration in tl.static_range(0, ITERATIONS):
        slot = writer.acquire(iteration)
        stage = iteration % STAGES
        top = tle.gpu.alloc(
            (8, 32),
            dtype=tl.float16,
            alias=slot.data,
            alias_offset_bytes=0,
            nv_mma_shared_layout=False,
        )
        bottom = tle.gpu.alloc(
            (8, 32),
            dtype=tl.float16,
            alias=slot.data,
            alias_offset_bytes=8 * 32 * 2,
            nv_mma_shared_layout=False,
        )
        tle.gpu.copy(top_source_desc, top, (8, 32), (iteration * 8, 0), barrier=full[stage])
        tle.gpu.copy(bottom_source_desc, bottom, (8, 32), (iteration * 8, 0), barrier=full[stage])
        writer.commit(iteration)

        wait = reader.wait(iteration)
        top_view = tle.gpu.alloc(
            (8, 32),
            dtype=tl.float16,
            alias=wait.slot.data,
            alias_offset_bytes=0,
            nv_mma_shared_layout=False,
        )
        bottom_view = tle.gpu.alloc(
            (8, 32),
            dtype=tl.float16,
            alias=wait.slot.data,
            alias_offset_bytes=8 * 32 * 2,
            nv_mma_shared_layout=False,
        )
        tle.gpu.copy(top_view, top_destination_desc, (8, 32), (iteration * 8, 0))
        tle.gpu.copy(bottom_view, bottom_destination_desc, (8, 32), (iteration * 8, 0))
        reader.release(iteration)


@triton.jit
def _non_ws_close_only_kernel(out):
    smem = tle.gpu.alloc((1, 128), dtype=tl.int32, nv_mma_shared_layout=False)
    pipe = tle.pipe(capacity=1, scope="cta", name="close_only", data=smem)
    writer = pipe.writer()
    reader = pipe.reader()
    writer.close(0)
    wait = reader.wait(0)
    tl.store(out, tl.where(wait.is_closed, 1, 0))


@triton.jit
def _non_ws_tme_close_kernel(
    src_desc,
    dst_desc,
    flags,
    STAGES: tl.constexpr,
    BLOCK: tl.constexpr,
    PAYLOADS: tl.constexpr,
    RELEASE_CLOSE: tl.constexpr,
):
    smem = tle.gpu.alloc((STAGES, BLOCK), dtype=tl.float16, nv_mma_shared_layout=False)
    pipe = tle.pipe(capacity=STAGES, scope="cta", name="tme_close", data=smem)
    writer = pipe.writer()
    reader = pipe.reader()
    for iteration in tl.static_range(0, PAYLOADS):
        slot = writer.acquire(iteration)
        tle.gpu.copy(src_desc, slot.data, (BLOCK, ), (iteration * BLOCK, ))
        writer.commit(iteration)
        wait = reader.wait(iteration)
        tle.gpu.copy(wait.slot.data, dst_desc, (BLOCK, ), (iteration * BLOCK, ))
        tl.store(flags + iteration, tl.where(wait.is_closed, 1, 0))
        reader.release(iteration)
    writer.close(PAYLOADS)
    wait = reader.wait(PAYLOADS)
    tl.store(flags + PAYLOADS, tl.where(wait.is_closed, 1, 0))
    if RELEASE_CLOSE:
        reader.release(PAYLOADS)


@triton.jit
def _ws_close_tme_producer(writer, desc, BLOCK: tl.constexpr, PAYLOADS: tl.constexpr):
    for iteration in tl.static_range(0, PAYLOADS):
        slot = writer.acquire(iteration)
        tle.gpu.copy(desc, slot.data, (BLOCK, ), (iteration * BLOCK, ))
        writer.commit(iteration)
    writer.close(PAYLOADS)


@triton.jit
def _ws_close_tme_store_consumer(reader, dst_desc, flags, BLOCK: tl.constexpr, PAYLOADS: tl.constexpr):
    for iteration in tl.static_range(0, PAYLOADS):
        wait = reader.wait(iteration)
        tle.gpu.copy(wait.slot.data, dst_desc, (BLOCK, ), (iteration * BLOCK, ))
        tl.store(flags + iteration, tl.where(wait.is_closed, 1, 0))
        reader.release(iteration)
    wait = reader.wait(PAYLOADS)
    tl.store(flags + PAYLOADS, tl.where(wait.is_closed, 1, 0))


@triton.jit
def _ws_tme_store_close_kernel(
    src_desc,
    dst_desc,
    flags,
    STAGES: tl.constexpr,
    BLOCK: tl.constexpr,
    PAYLOADS: tl.constexpr,
):
    smem = tle.gpu.alloc((STAGES, BLOCK), dtype=tl.float16, nv_mma_shared_layout=False)
    pipe = tle.pipe(capacity=STAGES, name="ws_tme_store_close", data=smem)
    tle.gpu.warp_specialize(
        [
            (_ws_close_tme_store_consumer, (pipe.reader(), dst_desc, flags, BLOCK, PAYLOADS)),
            (_ws_close_tme_producer, (pipe.writer(), src_desc, BLOCK, PAYLOADS)),
        ],
        worker_num_warps=[4],
        worker_num_regs=[24],
    )


@triton.jit
def _invalid_pipe_kernel(
    desc,
    out,
    STAGES: tl.constexpr,
    BLOCK: tl.constexpr,
    ITERATIONS: tl.constexpr,
    KIND: tl.constexpr,
):
    smem = tle.gpu.alloc(
        (STAGES, BLOCK),
        dtype=tl.float16,
        layout=None,
        scope=tle.gpu.smem,
        nv_mma_shared_layout=False,
    )
    pipe = tle.pipe(capacity=STAGES, scope="cta", name="invalid_pipe", data=smem)
    if KIND == 0:
        tle.gpu.warp_specialize(
            [
                (_pipe_consumer, (pipe.reader(), out, ITERATIONS)),
                (_local_store_producer, (pipe.writer(), BLOCK, ITERATIONS)),
            ],
            worker_num_warps=[4],
            worker_num_regs=[24],
        )
    elif KIND == 1:
        tle.gpu.warp_specialize(
            [
                (_pipe_consumer, (pipe.reader(), out, ITERATIONS)),
                (_double_tme_producer, (pipe.writer(), desc, BLOCK, ITERATIONS)),
            ],
            worker_num_warps=[4],
            worker_num_regs=[24],
        )
    else:
        tle.gpu.warp_specialize(
            [
                (_pipe_consumer, (pipe.reader(), out, ITERATIONS)),
                (_mixed_tme_local_store_producer, (pipe.writer(), desc, BLOCK, ITERATIONS)),
            ],
            worker_num_warps=[4],
            worker_num_regs=[24],
        )


@triton.jit
def _async_copy_pipe_kernel(desc, out, STAGES: tl.constexpr, BLOCK: tl.constexpr, ITERATIONS: tl.constexpr):
    smem = tle.gpu.alloc(
        (STAGES, BLOCK),
        dtype=tl.float32,
        layout=None,
        scope=tle.gpu.smem,
        nv_mma_shared_layout=False,
    )
    pipe = tle.pipe(capacity=STAGES, scope="cta", name="async_copy_pipe", data=smem)
    writer = pipe.writer()
    reader = pipe.reader()
    offsets = tl.arange(0, BLOCK)
    for iteration in tl.static_range(0, ITERATIONS):
        slot = writer.acquire(iteration)
        values = tl.load(desc + iteration * BLOCK + offsets)
        tl.store(tle.gpu.local_ptr(slot.data), values)
        writer.commit(iteration)
        wait = reader.wait(iteration)
        tl.store(out + iteration, tl.where(wait.is_closed, 1, 0))
        reader.release(iteration)


@triton.jit
def _non_ws_async_copy_pipe_roundtrip_kernel(
    desc,
    out,
    STAGES: tl.constexpr,
    BLOCK: tl.constexpr,
    ITERATIONS: tl.constexpr,
):
    smem = tle.gpu.alloc(
        (STAGES, BLOCK),
        dtype=tl.float32,
        layout=None,
        scope=tle.gpu.smem,
        nv_mma_shared_layout=False,
    )
    pipe = tle.pipe(capacity=STAGES, scope="cta", name="non_ws_async_copy", data=smem)
    writer = pipe.writer()
    reader = pipe.reader()
    offsets = tl.arange(0, BLOCK)
    for iteration in tl.static_range(0, ITERATIONS):
        slot = writer.acquire(iteration)
        values = tl.load(desc + iteration * BLOCK + offsets)
        tl.store(tle.gpu.local_ptr(slot.data), values)
        writer.commit(iteration)

        wait = reader.wait(iteration)
        loaded = tl.load(tle.gpu.local_ptr(wait.slot.data, (offsets, )))
        tl.store(out + iteration * BLOCK + offsets, loaded)
        reader.release(iteration)


@triton.jit
def _async_copy_producer(writer, desc, BLOCK: tl.constexpr, ITERATIONS: tl.constexpr):
    offsets = tl.arange(0, BLOCK)
    for iteration in tl.static_range(0, ITERATIONS):
        slot = writer.acquire(iteration)
        values = tl.load(desc + iteration * BLOCK + offsets)
        tl.store(tle.gpu.local_ptr(slot.data), values)
        writer.commit(iteration)


@triton.jit
def _async_copy_consumer(reader, out, BLOCK: tl.constexpr, ITERATIONS: tl.constexpr):
    offsets = tle.gpu.set_layout(tl.arange(0, BLOCK), _LOCAL_STORE_WRITER_LAYOUT)
    for iteration in tl.static_range(0, ITERATIONS):
        wait = reader.wait(iteration)
        loaded = tl.load(tle.gpu.local_ptr(wait.slot.data, (offsets, )))
        tl.store(out + iteration * BLOCK + offsets, loaded)
        reader.release(iteration)


@triton.jit
def _ws_default_writer_async_copy_pipe_kernel(
    desc,
    out,
    STAGES: tl.constexpr,
    BLOCK: tl.constexpr,
    ITERATIONS: tl.constexpr,
):
    smem = tle.gpu.alloc(
        (STAGES, BLOCK),
        dtype=tl.float32,
        layout=None,
        scope=tle.gpu.smem,
        nv_mma_shared_layout=False,
    )
    pipe = tle.pipe(capacity=STAGES, scope="cta", name="ws_async_copy", data=smem)
    tle.gpu.warp_specialize(
        [
            (_async_copy_producer, (pipe.writer(), desc, BLOCK, ITERATIONS)),
            (_async_copy_consumer, (pipe.reader(), out, BLOCK, ITERATIONS)),
        ],
        worker_num_warps=[4],
        worker_num_regs=[24],
    )


@triton.jit
def _one_shot_async_copy_pipe_kernel(desc, out, STAGES: tl.constexpr, BLOCK: tl.constexpr):
    smem = tle.gpu.alloc(
        (STAGES, BLOCK),
        dtype=tl.float32,
        layout=None,
        scope=tle.gpu.smem,
        nv_mma_shared_layout=False,
    )
    pipe = tle.pipe(
        capacity=STAGES,
        scope="cta",
        one_shot=True,
        name="one_shot_async_copy",
        data=smem,
    )
    writer = pipe.writer()
    reader = pipe.reader()
    offsets = tl.arange(0, BLOCK)
    for stage in tl.static_range(0, STAGES):
        slot = writer.acquire(stage)
        values = tl.load(desc + stage * BLOCK + offsets)
        tl.store(tle.gpu.local_ptr(slot.data), values)
        writer.commit(stage)
    for stage in tl.static_range(0, STAGES):
        wait = reader.wait(stage)
        loaded = tl.load(tle.gpu.local_ptr(wait.slot.data, (offsets, )))
        tl.store(out + stage * BLOCK + offsets, loaded)


@triton.jit
def _non_ws_async_copy_mixed_local_store_kernel(
    desc,
    async_out,
    local_out,
    STAGES: tl.constexpr,
    BLOCK: tl.constexpr,
    ITERATIONS: tl.constexpr,
):
    async_smem = tle.gpu.alloc(
        (STAGES, BLOCK),
        dtype=tl.float32,
        layout=None,
        scope=tle.gpu.smem,
        nv_mma_shared_layout=False,
    )
    local_smem = tle.gpu.alloc(
        (STAGES, BLOCK),
        dtype=tl.float32,
        layout=None,
        scope=tle.gpu.smem,
        nv_mma_shared_layout=False,
    )
    pipe = tle.pipe(
        capacity=STAGES,
        scope="cta",
        name="mixed_async_local",
        async_data=async_smem,
        local_data=local_smem,
    )
    writer = pipe.writer()
    reader = pipe.reader()
    offsets = tl.arange(0, BLOCK)
    for iteration in tl.static_range(0, ITERATIONS):
        slot = writer.acquire(iteration)
        async_values = tl.load(desc + iteration * BLOCK + offsets)
        tl.store(tle.gpu.local_ptr(slot.async_data), async_values)
        local_values = (iteration * BLOCK + offsets).to(tl.float32) + 0.5
        tl.store(tle.gpu.local_ptr(slot.local_data, (offsets, )), local_values)
        writer.commit(iteration)

        wait = reader.wait(iteration)
        async_loaded = tl.load(tle.gpu.local_ptr(wait.slot.async_data, (offsets, )))
        local_loaded = tl.load(tle.gpu.local_ptr(wait.slot.local_data, (offsets, )))
        tl.store(async_out + iteration * BLOCK + offsets, async_loaded)
        tl.store(local_out + iteration * BLOCK + offsets, local_loaded)
        reader.release(iteration)


@triton.jit
def _non_ws_async_copy_mixed_tme_kernel(
    desc,
    src,
    async_out,
    tme_out,
    STAGES: tl.constexpr,
    BLOCK: tl.constexpr,
    ITERATIONS: tl.constexpr,
):
    async_smem = tle.gpu.alloc(
        (STAGES, BLOCK),
        dtype=tl.float32,
        layout=None,
        scope=tle.gpu.smem,
        nv_mma_shared_layout=False,
    )
    tme_smem = tle.gpu.alloc(
        (STAGES, BLOCK),
        dtype=tl.float32,
        layout=None,
        scope=tle.gpu.smem,
        nv_mma_shared_layout=False,
    )
    pipe = tle.pipe(
        capacity=STAGES,
        scope="cta",
        name="mixed_async_tme",
        async_data=async_smem,
        tme_data=tme_smem,
    )
    writer = pipe.writer()
    reader = pipe.reader()
    offsets = tl.arange(0, BLOCK)
    for iteration in tl.static_range(0, ITERATIONS):
        slot = writer.acquire(iteration)
        async_values = tl.load(src + iteration * BLOCK + offsets)
        tl.store(tle.gpu.local_ptr(slot.async_data), async_values)
        tle.gpu.copy(desc, slot.tme_data, (BLOCK, ), (iteration * BLOCK, ))
        writer.commit(iteration)

        wait = reader.wait(iteration)
        async_loaded = tl.load(tle.gpu.local_ptr(wait.slot.async_data, (offsets, )))
        tme_loaded = tl.load(tle.gpu.local_ptr(wait.slot.tme_data, (offsets, )))
        tl.store(async_out + iteration * BLOCK + offsets, async_loaded)
        tl.store(tme_out + iteration * BLOCK + offsets, tme_loaded)
        reader.release(iteration)


@triton.jit
def _mixed_tme_async_copy_same_field_kernel(
    desc,
    src,
    out,
    STAGES: tl.constexpr,
    BLOCK: tl.constexpr,
    ITERATIONS: tl.constexpr,
):
    smem = tle.gpu.alloc(
        (STAGES, BLOCK),
        dtype=tl.float32,
        layout=None,
        scope=tle.gpu.smem,
        nv_mma_shared_layout=False,
    )
    pipe = tle.pipe(capacity=STAGES, scope="cta", name="mixed_same_field", data=smem)
    writer = pipe.writer()
    reader = pipe.reader()
    offsets = tl.arange(0, BLOCK)
    for iteration in tl.static_range(0, ITERATIONS):
        slot = writer.acquire(iteration)
        tle.gpu.copy(desc, slot.data, (BLOCK, ), (iteration * BLOCK, ))
        values = tl.load(src + iteration * BLOCK + offsets)
        tl.store(tle.gpu.local_ptr(slot.data), values)
        writer.commit(iteration)
        wait = reader.wait(iteration)
        tl.store(out + iteration, tl.where(wait.is_closed, 1, 0))
        reader.release(iteration)


@triton.jit
def _ws_idle_worker(marker, source_desc):
    tl.store(marker, 0)


@triton.jit
def _pipe_same_partition_endpoints(writer, reader, source_desc, out, BLOCK: tl.constexpr):
    slot = writer.acquire(0)
    tle.gpu.copy(source_desc, slot.data, (BLOCK, ), (0, ))
    writer.commit(0)
    reader.wait(0)
    reader.release(0)
    tl.store(out, 1)


@triton.jit
def _pipe_writer_acquire_only(writer):
    writer.acquire(0)


@triton.jit
def _pipe_writer_commit_only(writer):
    writer.commit(0)


@triton.jit
def _ws_invalid_endpoint_placement_kernel(
    desc,
    out,
    STAGES: tl.constexpr,
    BLOCK: tl.constexpr,
    ITERATIONS: tl.constexpr,
    KIND: tl.constexpr,
):
    smem = tle.gpu.alloc(
        (1, BLOCK),
        dtype=tl.float16,
        layout=None,
        scope=tle.gpu.smem,
        nv_mma_shared_layout=False,
    )
    pipe = tle.pipe(capacity=1, name="invalid_endpoint_placement", data=smem)
    if KIND == 0:
        tle.gpu.warp_specialize(
            [
                (_pipe_same_partition_endpoints, (pipe.writer(), pipe.reader(), desc, out, BLOCK)),
                (_ws_idle_worker, (out, desc)),
            ],
            worker_num_warps=[4],
            worker_num_regs=[24],
        )
    else:
        tle.gpu.warp_specialize(
            [
                (_pipe_writer_acquire_only, (pipe.writer(), )),
                (_pipe_writer_commit_only, (pipe.writer(), )),
            ],
            worker_num_warps=[4],
            worker_num_regs=[24],
        )


@triton.jit
def _pipe_tme_store_roundtrip_kernel(
    src_desc,
    dst_desc,
    STAGES: tl.constexpr,
    BLOCK: tl.constexpr,
    ITERATIONS: tl.constexpr,
):
    smem = tle.gpu.alloc(
        (STAGES, BLOCK),
        dtype=tl.float16,
        layout=None,
        scope=tle.gpu.smem,
        nv_mma_shared_layout=False,
    )
    pipe = tle.pipe(capacity=STAGES, scope="cta", name="reader_tme_store_roundtrip", data=smem)
    writer = pipe.writer()
    reader = pipe.reader()
    for iteration in tl.static_range(0, ITERATIONS):
        slot = writer.acquire(iteration)
        tle.gpu.copy(src_desc, slot.data, (BLOCK, ), (iteration * BLOCK, ))
        writer.commit(iteration)

        wait = reader.wait(iteration)
        tle.gpu.copy(wait.slot.data, dst_desc, (BLOCK, ), (iteration * BLOCK, ))
        reader.release(iteration)


@triton.jit
def _pipe_external_full_roundtrip_kernel(
    src_desc,
    dst_desc,
    STAGES: tl.constexpr,
    BLOCK: tl.constexpr,
    ITERATIONS: tl.constexpr,
):
    smem = tle.gpu.alloc(
        (STAGES, BLOCK),
        dtype=tl.float16,
        layout=None,
        scope=tle.gpu.smem,
        nv_mma_shared_layout=False,
    )
    full = tle.gpu.alloc_barriers(
        STAGES,
        arrive_count=1,
        init=tle.gpu.PENDING,
        expect_bytes=BLOCK * 2,
    )
    pipe = tle.pipe(capacity=STAGES, scope="cta", name="external_full_roundtrip", data=smem)
    writer = pipe.writer()
    reader = pipe.reader()
    for iteration in tl.static_range(0, ITERATIONS):
        slot = writer.acquire(iteration)
        stage = iteration % STAGES
        tle.gpu.copy(src_desc, slot.data, (BLOCK, ), (iteration * BLOCK, ), barrier=full[stage])
        writer.commit(iteration)
        wait = reader.wait(iteration)
        tle.gpu.copy(wait.slot.data, dst_desc, (BLOCK, ), (iteration * BLOCK, ))
        reader.release(iteration)


@triton.jit
def _pipe_external_producer_ws(
    writer,
    desc,
    full,
    STAGES: tl.constexpr,
    BLOCK: tl.constexpr,
    ITERATIONS: tl.constexpr,
):
    for iteration in tl.static_range(0, ITERATIONS):
        slot = writer.acquire(iteration)
        stage = iteration % STAGES
        tle.gpu.copy(desc, slot.data, (BLOCK, ), (iteration * BLOCK, ), barrier=full[stage])
        writer.commit(iteration)


@triton.jit
def _pipe_external_full_roundtrip_ws_kernel(
    src_desc,
    dst_desc,
    STAGES: tl.constexpr,
    BLOCK: tl.constexpr,
    ITERATIONS: tl.constexpr,
):
    smem = tle.gpu.alloc(
        (STAGES, BLOCK),
        dtype=tl.float16,
        layout=None,
        scope=tle.gpu.smem,
        nv_mma_shared_layout=False,
    )
    full = tle.gpu.alloc_barriers(
        STAGES,
        arrive_count=1,
        init=tle.gpu.PENDING,
        expect_bytes=BLOCK * 2,
    )
    pipe = tle.pipe(capacity=STAGES, scope="cta", name="ws_external_full_roundtrip", data=smem)
    tle.gpu.warp_specialize(
        [
            (_pipe_tme_store_consumer, (pipe.reader(), dst_desc, BLOCK, ITERATIONS)),
            (_pipe_external_producer_ws, (pipe.writer(), src_desc, full, STAGES, BLOCK, ITERATIONS)),
        ],
        worker_num_warps=[4],
        worker_num_regs=[24],
    )


@triton.jit
def _pipe_tme_store_roundtrip_ws_kernel(
    src_desc,
    dst_desc,
    STAGES: tl.constexpr,
    BLOCK: tl.constexpr,
    ITERATIONS: tl.constexpr,
):
    smem = tle.gpu.alloc(
        (STAGES, BLOCK),
        dtype=tl.float16,
        layout=None,
        scope=tle.gpu.smem,
        nv_mma_shared_layout=False,
    )
    pipe = tle.pipe(capacity=STAGES, scope="cta", name="ws_reader_tme_store_roundtrip", data=smem)
    tle.gpu.warp_specialize(
        [
            (_pipe_tme_store_consumer, (pipe.reader(), dst_desc, BLOCK, ITERATIONS)),
            (_pipe_producer, (pipe.writer(), src_desc, BLOCK, ITERATIONS)),
        ],
        worker_num_warps=[4],
        worker_num_regs=[24],
    )


@triton.jit
def _ws_heterogeneous_tme_store_consumer(
    reader,
    half_dst_desc,
    float_dst_desc,
    HALF_M: tl.constexpr,
    HALF_N: tl.constexpr,
    FLOAT_M: tl.constexpr,
    FLOAT_N: tl.constexpr,
    ITERATIONS: tl.constexpr,
):
    for iteration in tl.static_range(0, ITERATIONS):
        wait = reader.wait(iteration)
        tle.gpu.copy(
            wait.slot.half,
            half_dst_desc,
            (HALF_M, HALF_N),
            (iteration * HALF_M, 0),
        )
        tle.gpu.copy(
            wait.slot.float_data,
            float_dst_desc,
            (FLOAT_M, FLOAT_N),
            (iteration * FLOAT_M, 0),
        )
        reader.release(iteration)


@triton.jit
def _ws_heterogeneous_tme_store_producer(
    writer,
    half_src_desc,
    float_src_desc,
    HALF_M: tl.constexpr,
    HALF_N: tl.constexpr,
    FLOAT_M: tl.constexpr,
    FLOAT_N: tl.constexpr,
    ITERATIONS: tl.constexpr,
):
    for iteration in tl.static_range(0, ITERATIONS):
        slot = writer.acquire(iteration)
        tle.gpu.copy(
            half_src_desc,
            slot.half,
            (HALF_M, HALF_N),
            (iteration * HALF_M, 0),
        )
        tle.gpu.copy(
            float_src_desc,
            slot.float_data,
            (FLOAT_M, FLOAT_N),
            (iteration * FLOAT_M, 0),
        )
        writer.commit(iteration)


@triton.jit
def _ws_heterogeneous_tme_store_roundtrip_kernel(
    half_src_desc,
    float_src_desc,
    half_dst_desc,
    float_dst_desc,
    STAGES: tl.constexpr,
    ITERATIONS: tl.constexpr,
    HALF_M: tl.constexpr,
    HALF_N: tl.constexpr,
    FLOAT_M: tl.constexpr,
    FLOAT_N: tl.constexpr,
):
    half_smem = tle.gpu.alloc(
        (STAGES, HALF_M, HALF_N),
        dtype=tl.float16,
        nv_mma_shared_layout=False,
    )
    float_smem = tle.gpu.alloc(
        (STAGES, FLOAT_M, FLOAT_N),
        dtype=tl.float32,
        nv_mma_shared_layout=False,
    )
    pipe = tle.pipe(capacity=STAGES, name="ws_heterogeneous_tme_store", half=half_smem, float_data=float_smem)
    tle.gpu.warp_specialize(
        [
            (
                _ws_heterogeneous_tme_store_consumer,
                (
                    pipe.reader(),
                    half_dst_desc,
                    float_dst_desc,
                    HALF_M,
                    HALF_N,
                    FLOAT_M,
                    FLOAT_N,
                    ITERATIONS,
                ),
            ),
            (
                _ws_heterogeneous_tme_store_producer,
                (
                    pipe.writer(),
                    half_src_desc,
                    float_src_desc,
                    HALF_M,
                    HALF_N,
                    FLOAT_M,
                    FLOAT_N,
                    ITERATIONS,
                ),
            ),
        ],
        worker_num_warps=[4],
        worker_num_regs=[24],
    )


@triton.jit
def _pipe_tme_store_source_mutation_kernel(
    desc,
    out,
    STAGES: tl.constexpr,
    BLOCK: tl.constexpr,
    ITERATIONS: tl.constexpr,
):
    smem = tle.gpu.alloc(
        (STAGES, BLOCK),
        dtype=tl.float16,
        layout=None,
        scope=tle.gpu.smem,
        nv_mma_shared_layout=False,
    )
    pipe = tle.pipe(capacity=STAGES, scope="cta", name="reader_tme_store_mutation", data=smem)
    writer = pipe.writer()
    reader = pipe.reader()
    offsets = tl.arange(0, BLOCK)
    for iteration in tl.static_range(0, ITERATIONS):
        slot = writer.acquire(iteration)
        tle.gpu.copy(desc, slot.data, (BLOCK, ), (iteration * BLOCK, ))
        writer.commit(iteration)

        wait = reader.wait(iteration)
        modified = (offsets + iteration * BLOCK).to(tl.float16)
        tl.store(tle.gpu.local_ptr(wait.slot.data, (offsets, )), modified)
        tle.gpu.copy(wait.slot.data, desc, (BLOCK, ), (iteration * BLOCK, ))
        tl.store(out + iteration, 1)
        reader.release(iteration)


@triton.jit
def _pipe_tme_store_source_mutation_after_kernel(
    desc,
    out,
    STAGES: tl.constexpr,
    BLOCK: tl.constexpr,
    ITERATIONS: tl.constexpr,
):
    smem = tle.gpu.alloc(
        (STAGES, BLOCK),
        dtype=tl.float16,
        layout=None,
        scope=tle.gpu.smem,
        nv_mma_shared_layout=False,
    )
    pipe = tle.pipe(capacity=STAGES, scope="cta", name="reader_tme_store_mutation_after", data=smem)
    writer = pipe.writer()
    reader = pipe.reader()
    offsets = tl.arange(0, BLOCK)
    for iteration in tl.static_range(0, ITERATIONS):
        slot = writer.acquire(iteration)
        tle.gpu.copy(desc, slot.data, (BLOCK, ), (iteration * BLOCK, ))
        writer.commit(iteration)

        wait = reader.wait(iteration)
        tle.gpu.copy(wait.slot.data, desc, (BLOCK, ), (iteration * BLOCK, ))
        modified = (offsets + iteration * BLOCK).to(tl.float16)
        tl.store(tle.gpu.local_ptr(wait.slot.data, (offsets, )), modified, mask=offsets < BLOCK - 1)
        tl.store(out + iteration, 1)
        reader.release(iteration)


@triton.jit
def _named_reader_kernel(desc, out, STAGES: tl.constexpr, BLOCK: tl.constexpr, ITERATIONS: tl.constexpr):
    smem = tle.gpu.alloc((STAGES, BLOCK), dtype=tl.float16, nv_mma_shared_layout=False)
    tle.pipe(capacity=STAGES, readers=("consumer", ), data=smem)


@triton.jit
def _named_reader_tme_roundtrip_kernel(
    source_desc,
    local_output,
    drain_destination_desc,
    closed_flags,
    STAGES: tl.constexpr,
    BLOCK: tl.constexpr,
    ITERATIONS: tl.constexpr,
):
    smem = tle.gpu.alloc((STAGES, BLOCK), dtype=tl.float16, nv_mma_shared_layout=False)
    pipe = tle.pipe(
        capacity=STAGES,
        scope="cta",
        name="named_reader_tme_roundtrip",
        readers=("local", "drain"),
        data=smem,
    )
    writer = pipe.writer()
    local_reader = pipe.reader("local")
    drain_reader = pipe.reader("drain")
    offsets = tl.arange(0, BLOCK)
    for iteration in tl.static_range(0, ITERATIONS):
        slot = writer.acquire(iteration)
        tle.gpu.copy(source_desc, slot.data, (BLOCK, ), (iteration * BLOCK, ))
        writer.commit(iteration)

        local_wait = local_reader.wait(iteration)
        values = tl.load(tle.gpu.local_ptr(local_wait.slot.data, (offsets, )))
        tl.store(local_output + iteration * BLOCK + offsets, values)
        tl.store(closed_flags + 2 * iteration, tl.where(local_wait.is_closed, 1, 0))
        local_reader.release(iteration)

        drain_wait = drain_reader.wait(iteration)
        tle.gpu.copy(drain_wait.slot.data, drain_destination_desc, (BLOCK, ), (iteration * BLOCK, ))
        tl.store(closed_flags + 2 * iteration + 1, tl.where(drain_wait.is_closed, 1, 0))
        drain_reader.release(iteration)


@triton.jit
def _named_reader_interleaved_order_kernel(
    a_source_desc,
    b_source_desc,
    all_local_output,
    subset_local_output,
    all_drain_destination_desc,
    subset_drain_destination_desc,
    closed_flags,
    STAGES: tl.constexpr,
    BLOCK: tl.constexpr,
):
    a_smem = tle.gpu.alloc((STAGES, BLOCK), dtype=tl.float16, nv_mma_shared_layout=False)
    b_smem = tle.gpu.alloc((STAGES, BLOCK), dtype=tl.float32, nv_mma_shared_layout=False)
    pipe = tle.pipe(
        capacity=STAGES,
        scope="cta",
        name="named_reader_interleaved_order",
        readers=("all", "subset"),
        a=a_smem,
        b=b_smem,
    )
    writer = pipe.writer()
    all_reader = pipe.reader("all")
    subset_reader = pipe.reader("subset", fields=("b", ))
    offsets = tl.arange(0, BLOCK)

    # Each physical stage is published three times.  The static schedule
    # changes which endpoint releases early and which endpoint releases last,
    # while keeping exactly one reader-side TME store in every generation.
    for cycle in tl.static_range(0, 3):
        for stage in tl.static_range(0, STAGES):
            iteration = cycle * STAGES + stage
            slot = writer.acquire(iteration)
            tle.gpu.copy(a_source_desc, slot.a, (BLOCK, ), (iteration * BLOCK, ))
            tle.gpu.copy(b_source_desc, slot.b, (BLOCK, ), (iteration * BLOCK, ))
            writer.commit(iteration)

            if cycle == 0:
                all_wait = all_reader.wait(iteration)
                a_values = tl.load(tle.gpu.local_ptr(all_wait.slot.a, (offsets, )))
                tl.store(all_local_output + iteration * BLOCK + offsets, a_values)
                tl.store(closed_flags + 2 * iteration, tl.where(all_wait.is_closed, 1, 0))
                all_reader.release(iteration)

                subset_wait = subset_reader.wait(iteration)
                tle.gpu.copy(
                    subset_wait.slot.b,
                    subset_drain_destination_desc,
                    (BLOCK, ),
                    (iteration * BLOCK, ),
                )
                tl.store(closed_flags + 2 * iteration + 1, tl.where(subset_wait.is_closed, 1, 0))
                subset_reader.release(iteration)
            elif cycle == 1:
                subset_wait = subset_reader.wait(iteration)
                b_values = tl.load(tle.gpu.local_ptr(subset_wait.slot.b, (offsets, )))
                tl.store(subset_local_output + iteration * BLOCK + offsets, b_values)
                tl.store(closed_flags + 2 * iteration + 1, tl.where(subset_wait.is_closed, 1, 0))
                subset_reader.release(iteration)

                all_wait = all_reader.wait(iteration)
                tle.gpu.copy(
                    all_wait.slot.a,
                    all_drain_destination_desc,
                    (BLOCK, ),
                    (iteration * BLOCK, ),
                )
                tl.store(closed_flags + 2 * iteration, tl.where(all_wait.is_closed, 1, 0))
                all_reader.release(iteration)
            else:
                all_wait = all_reader.wait(iteration)
                subset_wait = subset_reader.wait(iteration)
                b_values = tl.load(tle.gpu.local_ptr(subset_wait.slot.b, (offsets, )))
                tl.store(subset_local_output + iteration * BLOCK + offsets, b_values)
                tl.store(closed_flags + 2 * iteration + 1, tl.where(subset_wait.is_closed, 1, 0))
                subset_reader.release(iteration)

                tle.gpu.copy(
                    all_wait.slot.a,
                    all_drain_destination_desc,
                    (BLOCK, ),
                    (iteration * BLOCK, ),
                )
                tl.store(closed_flags + 2 * iteration, tl.where(all_wait.is_closed, 1, 0))
                all_reader.release(iteration)


@triton.jit
def _invalid_named_reader_lifecycle_kernel(
    desc,
    out,
    STAGES: tl.constexpr,
    BLOCK: tl.constexpr,
    ITERATIONS: tl.constexpr,
    KIND: tl.constexpr,
):
    smem = tle.gpu.alloc((STAGES, BLOCK), dtype=tl.float16, nv_mma_shared_layout=False)
    pipe = tle.pipe(
        capacity=STAGES,
        name="invalid_named_reader_lifecycle",
        readers=("left", "right"),
        data=smem,
    )
    slot = pipe.writer().acquire(0)
    tle.gpu.copy(desc, slot.data, (BLOCK, ), (0, ))
    pipe.writer().commit(0)
    left = pipe.reader("left")
    right = pipe.reader("right")
    if KIND == 0:
        left_wait = left.wait(0)
        right_wait = right.wait(0)
        tl.store(out, tl.where(left_wait.is_closed | right_wait.is_closed, 1, 0))
        left.release(0)
    elif KIND == 1:
        left.wait(0)
        right.wait(0)
        left.release(0)
        right.release(0)
        left.release(0)
    elif KIND == 2:
        left.wait(0)
        right.release(0)
    else:
        left.wait(STAGES)
        right.wait(STAGES)
        left.release(STAGES)
        right.release(STAGES)


@triton.jit
def _invalid_named_reader_python_api_kernel(
    desc,
    out,
    STAGES: tl.constexpr,
    BLOCK: tl.constexpr,
    ITERATIONS: tl.constexpr,
    KIND: tl.constexpr,
):
    smem = tle.gpu.alloc((STAGES, BLOCK), dtype=tl.float16, nv_mma_shared_layout=False)
    if KIND == 0:
        tle.pipe(capacity=STAGES, readers=("left", "left"), data=smem)
    elif KIND == 1:
        pipe = tle.pipe(capacity=STAGES, readers=("left", "right"), data=smem)
        pipe.reader()
    elif KIND == 2:
        pipe = tle.pipe(capacity=STAGES, readers=("left", "right"), data=smem)
        pipe.reader("missing")
    elif KIND == 3:
        pipe = tle.pipe(capacity=STAGES, readers=("left", "right"), data=smem)
        pipe.reader("left", fields=("missing", ))
    else:
        pipe = tle.pipe(capacity=STAGES, readers=("left", "right"), data=smem)
        pipe.reader("left", fields=("data", "data"))


@triton.jit
def _named_reader_partial_heterogeneous_kernel(
    half_source_desc,
    float_source_desc,
    half_output,
    float_destination_desc,
    closed_flags,
    STAGES: tl.constexpr,
    ITERATIONS: tl.constexpr,
    HALF_M: tl.constexpr,
    HALF_N: tl.constexpr,
    FLOAT_M: tl.constexpr,
    FLOAT_N: tl.constexpr,
):
    half_smem = tle.gpu.alloc((STAGES, HALF_M, HALF_N), dtype=tl.float16, nv_mma_shared_layout=False)
    float_smem = tle.gpu.alloc((STAGES, FLOAT_M, FLOAT_N), dtype=tl.float32, nv_mma_shared_layout=False)
    pipe = tle.pipe(
        capacity=STAGES,
        name="named_reader_partial_heterogeneous",
        readers=("left", "right"),
        half=half_smem,
        float_data=float_smem,
    )
    writer = pipe.writer()
    left = pipe.reader("left", fields=("half", ))
    right = pipe.reader("right", fields=("float_data", ))
    half_rows = tl.arange(0, HALF_M)[:, None]
    half_cols = tl.arange(0, HALF_N)[None, :]

    for iteration in tl.static_range(0, ITERATIONS):
        slot = writer.acquire(iteration)
        tle.gpu.copy(half_source_desc, slot.half, (HALF_M, HALF_N), (iteration * HALF_M, 0))
        tle.gpu.copy(float_source_desc, slot.float_data, (FLOAT_M, FLOAT_N), (iteration * FLOAT_M, 0))
        writer.commit(iteration)

        left_wait = left.wait(iteration)
        half_values = tl.load(tle.gpu.local_ptr(left_wait.slot.half))
        half_offsets = (iteration * HALF_M + half_rows) * HALF_N + half_cols
        tl.store(half_output + half_offsets, half_values)
        tl.store(closed_flags + 2 * iteration, tl.where(left_wait.is_closed, 1, 0))
        left.release(iteration)

        right_wait = right.wait(iteration)
        tle.gpu.copy(
            right_wait.slot.float_data,
            float_destination_desc,
            (FLOAT_M, FLOAT_N),
            (iteration * FLOAT_M, 0),
        )
        tl.store(closed_flags + 2 * iteration + 1, tl.where(right_wait.is_closed, 1, 0))
        right.release(iteration)


@triton.jit
def _named_reader_partial_mixed_kernel(
    tme_source_desc,
    tme_output,
    left_local_output,
    right_local_output,
    STAGES: tl.constexpr,
    ITERATIONS: tl.constexpr,
    M: tl.constexpr,
    N: tl.constexpr,
):
    tme_smem = tle.gpu.alloc((STAGES, M, N), dtype=tl.float16, nv_mma_shared_layout=False)
    left_local_smem = tle.gpu.alloc((STAGES, M, N), dtype=tl.int32, nv_mma_shared_layout=False)
    right_local_smem = tle.gpu.alloc((STAGES, M, N), dtype=tl.int32, nv_mma_shared_layout=False)
    pipe = tle.pipe(
        capacity=STAGES,
        name="named_reader_partial_mixed",
        readers=("left", "right"),
        tme=tme_smem,
        left_local=left_local_smem,
        right_local=right_local_smem,
    )
    writer = pipe.writer()
    left = pipe.reader("left", fields=("tme", "left_local"))
    right = pipe.reader("right", fields=("tme", "right_local"))
    rows = tl.arange(0, M)[:, None]
    cols = tl.arange(0, N)[None, :]
    writer_rows = tle.gpu.set_layout(tl.broadcast_to(rows, (M, N)), _LOCAL_STORE_WRITER_LAYOUT_2D)
    writer_cols = tle.gpu.set_layout(tl.broadcast_to(cols, (M, N)), _LOCAL_STORE_WRITER_LAYOUT_2D)

    for iteration in tl.static_range(0, ITERATIONS):
        slot = writer.acquire(iteration)
        tle.gpu.copy(tme_source_desc, slot.tme, (M, N), (iteration * M, 0))
        left_values = (iteration * M * N + rows * N + cols).to(tl.int32)
        left_values = tle.gpu.set_layout(left_values, _LOCAL_STORE_WRITER_LAYOUT_2D)
        right_values = left_values + 100000
        tl.store(tle.gpu.local_ptr(slot.left_local, (writer_rows, writer_cols)), left_values)
        tl.store(tle.gpu.local_ptr(slot.right_local, (writer_rows, writer_cols)), right_values)
        writer.commit(iteration)

        left_wait = left.wait(iteration)
        tme_values = tl.load(tle.gpu.local_ptr(left_wait.slot.tme))
        left_values = tl.load(tle.gpu.local_ptr(left_wait.slot.left_local))
        offsets = iteration * M * N + rows * N + cols
        tl.store(tme_output + offsets, tme_values)
        tl.store(left_local_output + offsets, left_values)
        left.release(iteration)

        right_wait = right.wait(iteration)
        right_values = tl.load(tle.gpu.local_ptr(right_wait.slot.right_local))
        tl.store(right_local_output + offsets, right_values)
        right.release(iteration)


@triton.jit
def _named_reader_same_field_tme_fragments_kernel(
    top_source_desc,
    bottom_source_desc,
    top_destination_desc,
    bottom_destination_desc,
    STAGES: tl.constexpr,
    ITERATIONS: tl.constexpr,
):
    """Two named readers drain disjoint fragments of the same pipe field."""
    smem = tle.gpu.alloc((STAGES, 16, 32), dtype=tl.float16, nv_mma_shared_layout=False)
    pipe = tle.pipe(
        capacity=STAGES,
        scope="cta",
        name="named_reader_same_field_tme_fragments",
        readers=("top", "bottom"),
        data=smem,
    )
    writer = pipe.writer()
    top_reader = pipe.reader("top", fields=("data", ))
    bottom_reader = pipe.reader("bottom", fields=("data", ))
    for iteration in tl.static_range(0, ITERATIONS):
        slot = writer.acquire(iteration)
        top = tle.gpu.alloc(
            (8, 32),
            dtype=tl.float16,
            alias=slot.data,
            alias_offset_bytes=0,
            nv_mma_shared_layout=False,
        )
        bottom = tle.gpu.alloc(
            (8, 32),
            dtype=tl.float16,
            alias=slot.data,
            alias_offset_bytes=8 * 32 * 2,
            nv_mma_shared_layout=False,
        )
        tle.gpu.copy(top_source_desc, top, (8, 32), (iteration * 8, 0))
        tle.gpu.copy(bottom_source_desc, bottom, (8, 32), (iteration * 8, 0))
        writer.commit(iteration)

        top_wait = top_reader.wait(iteration)
        top_view = tle.gpu.alloc(
            (8, 32),
            dtype=tl.float16,
            alias=top_wait.slot.data,
            alias_offset_bytes=0,
            nv_mma_shared_layout=False,
        )
        tle.gpu.copy(top_view, top_destination_desc, (8, 32), (iteration * 8, 0))
        top_reader.release(iteration)

        bottom_wait = bottom_reader.wait(iteration)
        bottom_view = tle.gpu.alloc(
            (8, 32),
            dtype=tl.float16,
            alias=bottom_wait.slot.data,
            alias_offset_bytes=8 * 32 * 2,
            nv_mma_shared_layout=False,
        )
        tle.gpu.copy(bottom_view, bottom_destination_desc, (8, 32), (iteration * 8, 0))
        bottom_reader.release(iteration)


@triton.jit
def _named_reader_unsubscribed_tme_store_kernel(
    desc,
    out,
    STAGES: tl.constexpr,
    BLOCK: tl.constexpr,
    ITERATIONS: tl.constexpr,
):
    first_smem = tle.gpu.alloc((STAGES, BLOCK), dtype=tl.float16, nv_mma_shared_layout=False)
    second_smem = tle.gpu.alloc((STAGES, BLOCK), dtype=tl.float16, nv_mma_shared_layout=False)
    pipe = tle.pipe(
        capacity=STAGES,
        name="named_reader_unsubscribed_tme_store",
        readers=("left", "right"),
        first=first_smem,
        second=second_smem,
    )
    slot = pipe.writer().acquire(0)
    tle.gpu.copy(desc, slot.first, (BLOCK, ), (0, ))
    tle.gpu.copy(desc, slot.second, (BLOCK, ), (0, ))
    pipe.writer().commit(0)

    left = pipe.reader("left", fields=("first", ))
    right = pipe.reader("right", fields=("second", ))
    left.wait(0)
    tle.gpu.copy(second_smem.slot(0), desc, (BLOCK, ), (0, ))
    left.release(0)
    right.wait(0)
    right.release(0)


@triton.jit
def _named_reader_subscription_mismatch_kernel(
    desc,
    out,
    STAGES: tl.constexpr,
    BLOCK: tl.constexpr,
    ITERATIONS: tl.constexpr,
):
    first_smem = tle.gpu.alloc((STAGES, BLOCK), dtype=tl.float16, nv_mma_shared_layout=False)
    second_smem = tle.gpu.alloc((STAGES, BLOCK), dtype=tl.float16, nv_mma_shared_layout=False)
    pipe = tle.pipe(
        capacity=STAGES,
        name="named_reader_subscription_mismatch",
        readers=("left", "right"),
        first=first_smem,
        second=second_smem,
    )
    slot = pipe.writer().acquire(0)
    tle.gpu.copy(desc, slot.first, (BLOCK, ), (0, ))
    tle.gpu.copy(desc, slot.second, (BLOCK, ), (0, ))
    pipe.writer().commit(0)
    pipe.reader("left", fields=("first", )).wait(0)
    pipe.reader("left", fields=("second", )).release(0)
    pipe.reader("right").wait(0)
    pipe.reader("right").release(0)


@triton.jit
def _named_reader_close_kernel(desc, out, STAGES: tl.constexpr, BLOCK: tl.constexpr, ITERATIONS: tl.constexpr):
    smem = tle.gpu.alloc((STAGES, BLOCK), dtype=tl.float16, nv_mma_shared_layout=False)
    pipe = tle.pipe(capacity=STAGES, name="named_reader_close", readers=("left", "right"), data=smem)
    pipe.writer().close(0)
    left_wait = pipe.reader("left").wait(0)
    pipe.reader("left").release(0)
    right_wait = pipe.reader("right").wait(0)
    tl.store(out, tl.where(left_wait.is_closed & right_wait.is_closed, 1, 0))


@triton.jit
def _named_reader_partial_tme_close_kernel(
    half_source_desc,
    float_source_desc,
    half_output,
    float_destination_desc,
    closed_flags,
    STAGES: tl.constexpr,
    PAYLOADS: tl.constexpr,
    HALF_M: tl.constexpr,
    HALF_N: tl.constexpr,
    FLOAT_M: tl.constexpr,
    FLOAT_N: tl.constexpr,
):
    half_smem = tle.gpu.alloc((STAGES, HALF_M, HALF_N), dtype=tl.float16, nv_mma_shared_layout=False)
    float_smem = tle.gpu.alloc((STAGES, FLOAT_M, FLOAT_N), dtype=tl.float32, nv_mma_shared_layout=False)
    pipe = tle.pipe(
        capacity=STAGES,
        name="named_reader_partial_tme_close",
        readers=("left", "right"),
        half=half_smem,
        float_data=float_smem,
    )
    writer = pipe.writer()
    left = pipe.reader("left", fields=("half", ))
    right = pipe.reader("right", fields=("float_data", ))
    half_rows = tl.arange(0, HALF_M)[:, None]
    half_cols = tl.arange(0, HALF_N)[None, :]

    for iteration in tl.static_range(0, PAYLOADS):
        slot = writer.acquire(iteration)
        tle.gpu.copy(half_source_desc, slot.half, (HALF_M, HALF_N), (iteration * HALF_M, 0))
        tle.gpu.copy(float_source_desc, slot.float_data, (FLOAT_M, FLOAT_N), (iteration * FLOAT_M, 0))
        writer.commit(iteration)

        left_wait = left.wait(iteration)
        half_values = tl.load(tle.gpu.local_ptr(left_wait.slot.half))
        half_offsets = (iteration * HALF_M + half_rows) * HALF_N + half_cols
        tl.store(half_output + half_offsets, half_values)
        tl.store(closed_flags + 2 * iteration, tl.where(left_wait.is_closed, 1, 0))
        left.release(iteration)

        right_wait = right.wait(iteration)
        tle.gpu.copy(
            right_wait.slot.float_data,
            float_destination_desc,
            (FLOAT_M, FLOAT_N),
            (iteration * FLOAT_M, 0),
        )
        tl.store(closed_flags + 2 * iteration + 1, tl.where(right_wait.is_closed, 1, 0))
        right.release(iteration)

    writer.close(PAYLOADS)
    left_close = left.wait(PAYLOADS)
    tl.store(closed_flags + 2 * PAYLOADS, tl.where(left_close.is_closed, 1, 0))
    left.release(PAYLOADS)
    right_close = right.wait(PAYLOADS)
    tl.store(closed_flags + 2 * PAYLOADS + 1, tl.where(right_close.is_closed, 1, 0))


@triton.jit
def _named_reader_mixed_close_kernel(
    source_desc,
    tme_output,
    local_output,
    closed_flags,
    STAGES: tl.constexpr,
    BLOCK: tl.constexpr,
    PAYLOADS: tl.constexpr,
):
    tme_smem = tle.gpu.alloc((STAGES, BLOCK), dtype=tl.float16, nv_mma_shared_layout=False)
    local_smem = tle.gpu.alloc((STAGES, BLOCK), dtype=tl.int32, nv_mma_shared_layout=False)
    pipe = tle.pipe(
        capacity=STAGES,
        name="named_reader_mixed_close",
        readers=("tme_reader", "local_reader"),
        tme=tme_smem,
        local=local_smem,
    )
    writer = pipe.writer()
    tme_reader = pipe.reader("tme_reader", fields=("tme", ))
    local_reader = pipe.reader("local_reader", fields=("local", ))
    writer_offsets = tle.gpu.set_layout(tl.arange(0, BLOCK), _LOCAL_STORE_WRITER_LAYOUT)
    offsets = tl.arange(0, BLOCK)
    for iteration in tl.static_range(0, PAYLOADS):
        slot = writer.acquire(iteration)
        tle.gpu.copy(source_desc, slot.tme, (BLOCK, ), (iteration * BLOCK, ))
        local_values = (iteration * BLOCK + writer_offsets).to(tl.int32)
        tl.store(tle.gpu.local_ptr(slot.local, (writer_offsets, )), local_values)
        writer.commit(iteration)

        tme_wait = tme_reader.wait(iteration)
        tme_values = tl.load(tle.gpu.local_ptr(tme_wait.slot.tme, (offsets, )))
        tl.store(tme_output + iteration * BLOCK + offsets, tme_values)
        tl.store(closed_flags + 2 * iteration, tl.where(tme_wait.is_closed, 1, 0))
        tme_reader.release(iteration)

        local_wait = local_reader.wait(iteration)
        values = tl.load(tle.gpu.local_ptr(local_wait.slot.local, (offsets, )))
        tl.store(local_output + iteration * BLOCK + offsets, values)
        tl.store(closed_flags + 2 * iteration + 1, tl.where(local_wait.is_closed, 1, 0))
        local_reader.release(iteration)

    writer.close(PAYLOADS)
    tme_close = tme_reader.wait(PAYLOADS)
    tl.store(closed_flags + 2 * PAYLOADS, tl.where(tme_close.is_closed, 1, 0))
    tme_reader.release(PAYLOADS)
    local_close = local_reader.wait(PAYLOADS)
    tl.store(closed_flags + 2 * PAYLOADS + 1, tl.where(local_close.is_closed, 1, 0))


@triton.jit
def _named_reader_one_shot_heterogeneous_kernel(
    half_source_desc,
    float_source_desc,
    half_output,
    float_destination_desc,
    closed_flags,
    STAGES: tl.constexpr,
    HALF_M: tl.constexpr,
    HALF_N: tl.constexpr,
    FLOAT_M: tl.constexpr,
    FLOAT_N: tl.constexpr,
):
    half_smem = tle.gpu.alloc((STAGES, HALF_M, HALF_N), dtype=tl.float16, nv_mma_shared_layout=False)
    float_smem = tle.gpu.alloc((STAGES, FLOAT_M, FLOAT_N), dtype=tl.float32, nv_mma_shared_layout=False)
    pipe = tle.pipe(
        capacity=STAGES,
        one_shot=True,
        name="named_reader_one_shot_heterogeneous",
        readers=("left", "right"),
        half=half_smem,
        float_data=float_smem,
    )
    writer = pipe.writer()
    left = pipe.reader("left", fields=("half", ))
    right = pipe.reader("right", fields=("float_data", ))
    half_rows = tl.arange(0, HALF_M)[:, None]
    half_cols = tl.arange(0, HALF_N)[None, :]

    for stage in tl.static_range(0, STAGES):
        slot = writer.acquire(stage)
        tle.gpu.copy(half_source_desc, slot.half, (HALF_M, HALF_N), (stage * HALF_M, 0))
        tle.gpu.copy(float_source_desc, slot.float_data, (FLOAT_M, FLOAT_N), (stage * FLOAT_M, 0))
        writer.commit(stage)

    for stage in tl.static_range(0, STAGES):
        left_wait = left.wait(stage)
        half_values = tl.load(tle.gpu.local_ptr(left_wait.slot.half))
        half_offsets = (stage * HALF_M + half_rows) * HALF_N + half_cols
        tl.store(half_output + half_offsets, half_values)
        tl.store(closed_flags + 4 * stage, tl.where(left_wait.is_closed, 1, 0))
        left.release(stage)
        left.release(stage)

        left_again = left.wait(stage)
        tl.store(closed_flags + 4 * stage + 1, tl.where(left_again.is_closed, 1, 0))

        right_wait = right.wait(stage)
        tle.gpu.copy(
            right_wait.slot.float_data,
            float_destination_desc,
            (FLOAT_M, FLOAT_N),
            (stage * FLOAT_M, 0),
        )
        tl.store(closed_flags + 4 * stage + 2, tl.where(right_wait.is_closed, 1, 0))
        right.release(stage)

        right_again = right.wait(stage)
        tle.gpu.copy(
            right_again.slot.float_data,
            float_destination_desc,
            (FLOAT_M, FLOAT_N),
            (stage * FLOAT_M, 0),
        )
        tl.store(closed_flags + 4 * stage + 3, tl.where(right_again.is_closed, 1, 0))


@triton.jit
def _invalid_named_reader_close_lifecycle_kernel(
    desc,
    out,
    STAGES: tl.constexpr,
    BLOCK: tl.constexpr,
    ITERATIONS: tl.constexpr,
    KIND: tl.constexpr,
):
    smem = tle.gpu.alloc((STAGES, BLOCK), dtype=tl.float16, nv_mma_shared_layout=False)
    pipe = tle.pipe(
        capacity=STAGES,
        name="invalid_named_reader_close_lifecycle",
        readers=("left", "right"),
        data=smem,
    )
    left = pipe.reader("left")
    right = pipe.reader("right")

    if KIND == 0:
        slot = pipe.writer().acquire(0)
        tle.gpu.copy(desc, slot.data, (BLOCK, ), (0, ))
        pipe.writer().commit(0)
        left.wait(0)
        left.release(0)
        right.wait(0)
        right.release(0)
        pipe.writer().close(1)
        left.wait(1)
    elif KIND == 1:
        pipe.writer().close(0)
        left_wait = left.wait(0)
        tle.gpu.copy(left_wait.slot.data, desc, (BLOCK, ), (0, ))
        left.release(0)
        right.wait(0)
    elif KIND == 2:
        pipe.writer().close(0)
        left.wait(1)
        right.wait(0)
    else:
        pipe.writer().close(0)
        left.wait(0)
        left.release(0)
        left.wait(0)
        right.wait(0)


@triton.jit
def _invalid_named_reader_one_shot_drain_kernel(
    desc,
    out,
    STAGES: tl.constexpr,
    BLOCK: tl.constexpr,
    ITERATIONS: tl.constexpr,
    KIND: tl.constexpr,
):
    smem = tle.gpu.alloc((STAGES, BLOCK), dtype=tl.float16, nv_mma_shared_layout=False)
    pipe = tle.pipe(
        capacity=STAGES,
        one_shot=True,
        name="invalid_named_reader_one_shot_drain",
        readers=("left", "right"),
        data=smem,
    )
    slot = pipe.writer().acquire(0)
    tle.gpu.copy(desc, slot.data, (BLOCK, ), (0, ))
    pipe.writer().commit(0)
    left = pipe.reader("left")
    right = pipe.reader("right")
    left_wait = left.wait(0)
    if KIND == 0:
        left.release(0)
        tle.gpu.copy(left_wait.slot.data, desc, (BLOCK, ), (0, ))
        right.wait(0)
    else:
        right.wait(0)
        tle.gpu.copy(left_wait.slot.data, desc, (BLOCK, ), (0, ))


@triton.jit
def _ws_named_partition_writer(
    writer,
    half_src_desc,
    float_src_desc,
    STAGES: tl.constexpr,
    ITERATIONS: tl.constexpr,
    HALF_M: tl.constexpr,
    HALF_N: tl.constexpr,
    FLOAT_M: tl.constexpr,
    FLOAT_N: tl.constexpr,
):
    for iteration in tl.static_range(0, ITERATIONS):
        slot = writer.acquire(iteration)
        tle.gpu.copy(half_src_desc, slot.a, (HALF_M, HALF_N), (iteration * HALF_M, 0))
        tle.gpu.copy(float_src_desc, slot.b, (FLOAT_M, FLOAT_N), (iteration * FLOAT_M, 0))
        writer.commit(iteration)


@triton.jit
def _ws_named_partition_default_left_reader(
    reader,
    output,
    flags,
    ITERATIONS: tl.constexpr,
    HALF_M: tl.constexpr,
    HALF_N: tl.constexpr,
):
    rows = tl.arange(0, HALF_M)[:, None]
    cols = tl.arange(0, HALF_N)[None, :]
    row_indices = tl.broadcast_to(rows, (HALF_M, HALF_N))
    col_indices = tl.broadcast_to(cols, (HALF_M, HALF_N))
    row_indices = tle.gpu.set_layout(row_indices, _LOCAL_STORE_READER_LAYOUT_2D_16)
    col_indices = tle.gpu.set_layout(col_indices, _LOCAL_STORE_READER_LAYOUT_2D_16)
    for iteration in tl.static_range(0, ITERATIONS):
        wait = reader.wait(iteration)
        values = tl.load(tle.gpu.local_ptr(wait.slot.a, (row_indices, col_indices)))
        offsets = (iteration * HALF_M + rows) * HALF_N + cols
        tl.store(output + offsets, values)
        tl.store(flags + iteration, tl.where(wait.is_closed, 1, 0))
        reader.release(iteration)


@triton.jit
def _ws_named_partition_right_reader(
    reader,
    destination_desc,
    flags,
    ITERATIONS: tl.constexpr,
    FLOAT_M: tl.constexpr,
    FLOAT_N: tl.constexpr,
):
    for iteration in tl.static_range(0, ITERATIONS):
        wait = reader.wait(iteration)
        tle.gpu.copy(
            wait.slot.b,
            destination_desc,
            (FLOAT_M, FLOAT_N),
            (iteration * FLOAT_M, 0),
        )
        tl.store(flags + iteration, tl.where(wait.is_closed, 1, 0))
        reader.release(iteration)


@triton.jit
def _ws_named_mixed_writer(
    writer,
    source_desc,
    STAGES: tl.constexpr,
    ITERATIONS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    offsets = tle.gpu.set_layout(tl.arange(0, BLOCK), _LOCAL_STORE_WRITER_LAYOUT)
    for iteration in tl.static_range(0, ITERATIONS):
        slot = writer.acquire(iteration)
        tle.gpu.copy(source_desc, slot.b, (BLOCK, ), (iteration * BLOCK, ))
        values = (iteration * BLOCK + offsets).to(tl.int32)
        tl.store(tle.gpu.local_ptr(slot.a, (offsets, )), values)
        writer.commit(iteration)


@triton.jit
def _ws_named_mixed_local_reader(
    reader,
    output,
    flags,
    ITERATIONS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    offsets = tle.gpu.set_layout(tl.arange(0, BLOCK), _LOCAL_STORE_WRITER_LAYOUT)
    for iteration in tl.static_range(0, ITERATIONS):
        wait = reader.wait(iteration)
        values = tl.load(tle.gpu.local_ptr(wait.slot.a, (offsets, )))
        tl.store(output + iteration * BLOCK + offsets, values)
        tl.store(flags + iteration, tl.where(wait.is_closed, 1, 0))
        reader.release(iteration)


@triton.jit
def _ws_named_mixed_tme_reader(
    reader,
    destination_desc,
    flags,
    ITERATIONS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    for iteration in tl.static_range(0, ITERATIONS):
        wait = reader.wait(iteration)
        tle.gpu.copy(wait.slot.b, destination_desc, (BLOCK, ), (iteration * BLOCK, ))
        tl.store(flags + iteration, tl.where(wait.is_closed, 1, 0))
        reader.release(iteration)


@triton.jit
def _ws_named_partition_mixed_worker_writer_kernel(
    source_desc,
    local_output,
    tme_output_desc,
    flags,
    STAGES: tl.constexpr,
    ITERATIONS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    local_smem = tle.gpu.alloc((STAGES, BLOCK), dtype=tl.int32, nv_mma_shared_layout=False)
    tme_smem = tle.gpu.alloc((STAGES, BLOCK), dtype=tl.float16, nv_mma_shared_layout=False)
    pipe = tle.pipe(
        capacity=STAGES,
        name="ws_named_partition_mixed_worker_writer",
        readers=("left", "right"),
        a=local_smem,
        b=tme_smem,
    )
    tle.gpu.warp_specialize(
        [
            (
                _ws_named_mixed_tme_reader,
                (
                    pipe.reader("left", fields=("b", )),
                    tme_output_desc,
                    flags,
                    ITERATIONS,
                    BLOCK,
                ),
            ),
            (
                _ws_named_mixed_writer,
                (pipe.writer(), source_desc, STAGES, ITERATIONS, BLOCK),
            ),
            (
                _ws_named_mixed_local_reader,
                (
                    pipe.reader("right", fields=("a", )),
                    local_output,
                    flags + ITERATIONS,
                    ITERATIONS,
                    BLOCK,
                ),
            ),
        ],
        worker_num_warps=[4, 4],
        worker_num_regs=[24, 24],
    )


@triton.jit
def _ws_named_partition_worker1_writer_kernel(
    half_src_desc,
    float_src_desc,
    half_output,
    float_dst_desc,
    idle_marker,
    flags,
    STAGES: tl.constexpr,
    ITERATIONS: tl.constexpr,
    HALF_M: tl.constexpr,
    HALF_N: tl.constexpr,
    FLOAT_M: tl.constexpr,
    FLOAT_N: tl.constexpr,
):
    half_smem = tle.gpu.alloc((STAGES, HALF_M, HALF_N), dtype=tl.float16, nv_mma_shared_layout=False)
    float_smem = tle.gpu.alloc((STAGES, FLOAT_M, FLOAT_N), dtype=tl.float32, nv_mma_shared_layout=False)
    pipe = tle.pipe(
        capacity=STAGES,
        name="ws_named_partition_worker1_writer",
        readers=("left", "right"),
        a=half_smem,
        b=float_smem,
    )
    tle.gpu.warp_specialize(
        [
            (
                _ws_named_partition_default_left_reader,
                (pipe.reader("left", fields=("a", )), half_output, flags, ITERATIONS, HALF_M, HALF_N),
            ),
            (_ws_idle_worker, (idle_marker, float_src_desc)),
            (
                _ws_named_partition_writer,
                (
                    pipe.writer(),
                    half_src_desc,
                    float_src_desc,
                    STAGES,
                    ITERATIONS,
                    HALF_M,
                    HALF_N,
                    FLOAT_M,
                    FLOAT_N,
                ),
            ),
            (
                _ws_named_partition_right_reader,
                (
                    pipe.reader("right", fields=("b", )),
                    float_dst_desc,
                    flags + ITERATIONS,
                    ITERATIONS,
                    FLOAT_M,
                    FLOAT_N,
                ),
            ),
        ],
        worker_num_warps=[1, 4, 4],
        worker_num_regs=[24, 24, 24],
    )


@triton.jit
def _ws_named_payload_close_writer(
    writer,
    half_src_desc,
    float_src_desc,
    STAGES: tl.constexpr,
    PAYLOADS: tl.constexpr,
    HALF_M: tl.constexpr,
    HALF_N: tl.constexpr,
    FLOAT_M: tl.constexpr,
    FLOAT_N: tl.constexpr,
):
    for iteration in tl.static_range(0, PAYLOADS):
        slot = writer.acquire(iteration)
        tle.gpu.copy(half_src_desc, slot.a, (HALF_M, HALF_N), (iteration * HALF_M, 0))
        tle.gpu.copy(float_src_desc, slot.b, (FLOAT_M, FLOAT_N), (iteration * FLOAT_M, 0))
        writer.commit(iteration)
    writer.close(PAYLOADS)


@triton.jit
def _ws_named_payload_close_left_reader(
    reader,
    output,
    flags,
    PAYLOADS: tl.constexpr,
    HALF_M: tl.constexpr,
    HALF_N: tl.constexpr,
):
    rows = tl.arange(0, HALF_M)[:, None]
    cols = tl.arange(0, HALF_N)[None, :]
    row_indices = tle.gpu.set_layout(tl.broadcast_to(rows, (HALF_M, HALF_N)), _LOCAL_STORE_WRITER_LAYOUT_2D)
    col_indices = tle.gpu.set_layout(tl.broadcast_to(cols, (HALF_M, HALF_N)), _LOCAL_STORE_WRITER_LAYOUT_2D)
    for iteration in tl.static_range(0, PAYLOADS):
        wait = reader.wait(iteration)
        values = tl.load(tle.gpu.local_ptr(wait.slot.a, (row_indices, col_indices)))
        offsets = (iteration * HALF_M + rows) * HALF_N + cols
        tl.store(output + offsets, values)
        tl.store(flags + iteration, tl.where(wait.is_closed, 1, 0))
        reader.release(iteration)
    close = reader.wait(PAYLOADS)
    tl.store(flags + PAYLOADS, tl.where(close.is_closed, 1, 0))
    reader.release(PAYLOADS)


@triton.jit
def _ws_named_payload_close_right_reader(
    reader,
    destination_desc,
    flags,
    PAYLOADS: tl.constexpr,
    FLOAT_M: tl.constexpr,
    FLOAT_N: tl.constexpr,
):
    for iteration in tl.static_range(0, PAYLOADS):
        wait = reader.wait(iteration)
        tle.gpu.copy(
            wait.slot.b,
            destination_desc,
            (FLOAT_M, FLOAT_N),
            (iteration * FLOAT_M, 0),
        )
        tl.store(flags + iteration, tl.where(wait.is_closed, 1, 0))
        reader.release(iteration)
    close = reader.wait(PAYLOADS)
    tl.store(flags + PAYLOADS, tl.where(close.is_closed, 1, 0))


@triton.jit
def _ws_named_payload_close_kernel(
    half_src_desc,
    float_src_desc,
    half_output,
    float_dst_desc,
    left_flags,
    right_flags,
    STAGES: tl.constexpr,
    PAYLOADS: tl.constexpr,
    HALF_M: tl.constexpr,
    HALF_N: tl.constexpr,
    FLOAT_M: tl.constexpr,
    FLOAT_N: tl.constexpr,
):
    half_smem = tle.gpu.alloc((STAGES, HALF_M, HALF_N), dtype=tl.float16, nv_mma_shared_layout=False)
    float_smem = tle.gpu.alloc((STAGES, FLOAT_M, FLOAT_N), dtype=tl.float32, nv_mma_shared_layout=False)
    pipe = tle.pipe(
        capacity=STAGES,
        name="ws_named_payload_close",
        readers=("left", "right"),
        a=half_smem,
        b=float_smem,
    )
    tle.gpu.warp_specialize(
        [
            (
                _ws_named_payload_close_writer,
                (
                    pipe.writer(),
                    half_src_desc,
                    float_src_desc,
                    STAGES,
                    PAYLOADS,
                    HALF_M,
                    HALF_N,
                    FLOAT_M,
                    FLOAT_N,
                ),
            ),
            (
                _ws_named_payload_close_left_reader,
                (pipe.reader("left", fields=("a", )), half_output, left_flags, PAYLOADS, HALF_M, HALF_N),
            ),
            (
                _ws_named_payload_close_right_reader,
                (pipe.reader("right", fields=("b", )), float_dst_desc, right_flags, PAYLOADS, FLOAT_M, FLOAT_N),
            ),
        ],
        worker_num_warps=[4, 4],
        worker_num_regs=[24, 24],
    )


@triton.jit
def _one_shot_tme_roundtrip_kernel(
    src_desc,
    first_dst_desc,
    second_dst_desc,
    flags,
    STAGES: tl.constexpr,
    BLOCK: tl.constexpr,
):
    smem = tle.gpu.alloc((STAGES, BLOCK), dtype=tl.float16, nv_mma_shared_layout=False)
    pipe = tle.pipe(capacity=STAGES, one_shot=True, name="one_shot_tme", data=smem)
    writer = pipe.writer()
    reader = pipe.reader()
    for stage in tl.static_range(0, STAGES):
        writer.acquire(stage)
        slot = writer.acquire(stage)
        tle.gpu.copy(src_desc, slot.data, (BLOCK, ), (stage * BLOCK, ))
        writer.commit(stage)
    for stage in tl.static_range(0, STAGES):
        first = reader.wait(stage)
        tle.gpu.copy(first.slot.data, first_dst_desc, (BLOCK, ), (stage * BLOCK, ))
        tl.store(flags + 2 * stage, tl.where(first.is_closed, 1, 0))
        reader.release(stage)
        reader.release(stage)
        second = reader.wait(stage)
        tle.gpu.copy(second.slot.data, second_dst_desc, (BLOCK, ), (stage * BLOCK, ))
        tl.store(flags + 2 * stage + 1, tl.where(second.is_closed, 1, 0))


@triton.jit
def _one_shot_mixed_kernel(
    tme_src_desc,
    tme_dst_desc,
    local_out,
    flags,
    STAGES: tl.constexpr,
    BLOCK: tl.constexpr,
):
    tme_smem = tle.gpu.alloc((STAGES, BLOCK), dtype=tl.float16, nv_mma_shared_layout=False)
    local_smem = tle.gpu.alloc((STAGES, BLOCK), dtype=tl.int32, nv_mma_shared_layout=False)
    pipe = tle.pipe(capacity=STAGES, one_shot=True, name="one_shot_mixed", tme=tme_smem, local=local_smem)
    writer = pipe.writer()
    reader = pipe.reader()
    offsets = tle.gpu.set_layout(tl.arange(0, BLOCK), _LOCAL_STORE_WRITER_LAYOUT)
    for stage in tl.static_range(0, STAGES):
        slot = writer.acquire(stage)
        tle.gpu.copy(tme_src_desc, slot.tme, (BLOCK, ), (stage * BLOCK, ))
        values = (stage * BLOCK + offsets).to(tl.int32)
        tl.store(tle.gpu.local_ptr(slot.local, (offsets, )), values)
        writer.commit(stage)
    for stage in tl.static_range(0, STAGES):
        wait = reader.wait(stage)
        tle.gpu.copy(wait.slot.tme, tme_dst_desc, (BLOCK, ), (stage * BLOCK, ))
        values = tl.load(tle.gpu.local_ptr(wait.slot.local))
        tl.store(local_out + stage * BLOCK + tl.arange(0, BLOCK), values)
        tl.store(flags + stage, tl.where(wait.is_closed, 1, 0))


@triton.jit
def _one_shot_heterogeneous_tme_kernel(
    half_src_desc,
    float_src_desc,
    half_dst_desc,
    float_dst_desc,
    flags,
    STAGES: tl.constexpr,
    HALF_M: tl.constexpr,
    HALF_N: tl.constexpr,
    FLOAT_M: tl.constexpr,
    FLOAT_N: tl.constexpr,
):
    half_smem = tle.gpu.alloc((STAGES, HALF_M, HALF_N), dtype=tl.float16, nv_mma_shared_layout=False)
    float_smem = tle.gpu.alloc((STAGES, FLOAT_M, FLOAT_N), dtype=tl.float32, nv_mma_shared_layout=False)
    pipe = tle.pipe(
        capacity=STAGES,
        one_shot=True,
        name="one_shot_heterogeneous",
        half=half_smem,
        float_data=float_smem,
    )
    writer = pipe.writer()
    reader = pipe.reader()
    for stage in tl.static_range(0, STAGES):
        slot = writer.acquire(stage)
        tle.gpu.copy(half_src_desc, slot.half, (HALF_M, HALF_N), (stage * HALF_M, 0))
        tle.gpu.copy(float_src_desc, slot.float_data, (FLOAT_M, FLOAT_N), (stage * FLOAT_M, 0))
        writer.commit(stage)
    for stage in tl.static_range(0, STAGES):
        wait = reader.wait(stage)
        tle.gpu.copy(wait.slot.half, half_dst_desc, (HALF_M, HALF_N), (stage * HALF_M, 0))
        tle.gpu.copy(wait.slot.float_data, float_dst_desc, (FLOAT_M, FLOAT_N), (stage * FLOAT_M, 0))
        tl.store(flags + stage, tl.where(wait.is_closed, 1, 0))


@triton.jit
def _duplicate_close_kernel(
    desc,
    out,
    STAGES: tl.constexpr,
    BLOCK: tl.constexpr,
    ITERATIONS: tl.constexpr,
):
    smem = tle.gpu.alloc((STAGES, BLOCK), dtype=tl.float16, nv_mma_shared_layout=False)
    writer = tle.pipe(capacity=STAGES, name="duplicate_close", data=smem).writer()
    writer.close(0)
    writer.close(1)


@triton.jit
def _close_with_open_writer_kernel(
    desc,
    out,
    STAGES: tl.constexpr,
    BLOCK: tl.constexpr,
    ITERATIONS: tl.constexpr,
):
    smem = tle.gpu.alloc((STAGES, BLOCK), dtype=tl.float16, nv_mma_shared_layout=False)
    pipe = tle.pipe(capacity=STAGES, name="close_with_open_writer", data=smem)
    slot = pipe.writer().acquire(0)
    tle.gpu.copy(desc, slot.data, (BLOCK, ), (0, ))
    pipe.writer().close(1)


@triton.jit
def _acquire_after_close_kernel(
    desc,
    out,
    STAGES: tl.constexpr,
    BLOCK: tl.constexpr,
    ITERATIONS: tl.constexpr,
):
    smem = tle.gpu.alloc((STAGES, BLOCK), dtype=tl.float16, nv_mma_shared_layout=False)
    writer = tle.pipe(capacity=STAGES, name="acquire_after_close", data=smem).writer()
    writer.close(0)
    writer.acquire(1)


@triton.jit
def _commit_after_close_kernel(
    desc,
    out,
    STAGES: tl.constexpr,
    BLOCK: tl.constexpr,
    ITERATIONS: tl.constexpr,
):
    smem = tle.gpu.alloc((STAGES, BLOCK), dtype=tl.float16, nv_mma_shared_layout=False)
    pipe = tle.pipe(capacity=STAGES, name="commit_after_close", data=smem)
    writer = pipe.writer()
    slot = writer.acquire(0)
    tle.gpu.copy(desc, slot.data, (BLOCK, ), (0, ))
    writer.commit(0)
    writer.close(1)
    writer.commit(0)


@triton.jit
def _close_terminal_mismatch_kernel(
    desc,
    out,
    STAGES: tl.constexpr,
    BLOCK: tl.constexpr,
    ITERATIONS: tl.constexpr,
    KIND: tl.constexpr,
):
    smem = tle.gpu.alloc((STAGES, BLOCK), dtype=tl.float16, nv_mma_shared_layout=False)
    pipe = tle.pipe(capacity=STAGES, name="close_terminal_mismatch", data=smem)
    pipe.writer().close(0)
    if KIND == 0:
        pipe.reader().wait(1)
    else:
        pipe.reader().wait(STAGES)
        pipe.reader().release(STAGES)


@triton.jit
def _close_payload_tme_store_kernel(
    desc,
    out,
    STAGES: tl.constexpr,
    BLOCK: tl.constexpr,
    ITERATIONS: tl.constexpr,
):
    smem = tle.gpu.alloc((STAGES, BLOCK), dtype=tl.float16, nv_mma_shared_layout=False)
    pipe = tle.pipe(capacity=STAGES, name="close_payload_tme_store", data=smem)
    pipe.writer().close(0)
    wait = pipe.reader().wait(0)
    tle.gpu.copy(wait.slot.data, desc, (BLOCK, ), (0, ))
    pipe.reader().release(0)


@triton.jit
def _one_shot_duplicate_commit_kernel(
    desc,
    out,
    STAGES: tl.constexpr,
    BLOCK: tl.constexpr,
    ITERATIONS: tl.constexpr,
):
    smem = tle.gpu.alloc((STAGES, BLOCK), dtype=tl.float16, nv_mma_shared_layout=False)
    pipe = tle.pipe(capacity=STAGES, one_shot=True, name="one_shot_duplicate", data=smem)
    writer = pipe.writer()
    reader = pipe.reader()
    slot = writer.acquire(0)
    tle.gpu.copy(desc, slot.data, (BLOCK, ), (0, ))
    writer.commit(0)
    slot = writer.acquire(0)
    tle.gpu.copy(desc, slot.data, (BLOCK, ), (0, ))
    writer.commit(0)
    wait = reader.wait(0)
    tl.store(out, tl.where(wait.is_closed, 1, 0))


@triton.jit
def _one_shot_phase_reuse_kernel(
    desc,
    out,
    STAGES: tl.constexpr,
    BLOCK: tl.constexpr,
    ITERATIONS: tl.constexpr,
):
    smem = tle.gpu.alloc((STAGES, BLOCK), dtype=tl.float16, nv_mma_shared_layout=False)
    pipe = tle.pipe(capacity=STAGES, one_shot=True, name="one_shot_phase", data=smem)
    writer = pipe.writer()
    reader = pipe.reader()
    slot = writer.acquire(STAGES)
    tle.gpu.copy(desc, slot.data, (BLOCK, ), (0, ))
    writer.commit(STAGES)
    wait = reader.wait(0)
    tl.store(out, tl.where(wait.is_closed, 1, 0))


@triton.jit
def _one_shot_mutation_kernel(
    desc,
    out,
    STAGES: tl.constexpr,
    BLOCK: tl.constexpr,
    ITERATIONS: tl.constexpr,
):
    smem = tle.gpu.alloc((STAGES, BLOCK), dtype=tl.float16, nv_mma_shared_layout=False)
    pipe = tle.pipe(capacity=STAGES, one_shot=True, name="one_shot_mutation", data=smem)
    writer = pipe.writer()
    reader = pipe.reader()
    slot = writer.acquire(0)
    tle.gpu.copy(desc, slot.data, (BLOCK, ), (0, ))
    writer.commit(0)
    wait = reader.wait(0)
    offsets = tl.arange(0, BLOCK)
    tl.store(tle.gpu.local_ptr(wait.slot.data, (offsets, )), offsets.to(tl.float16))
    tl.store(out, tl.where(wait.is_closed, 1, 0))


@triton.jit
def _one_shot_uncommitted_rewrite_kernel(
    desc,
    out,
    STAGES: tl.constexpr,
    BLOCK: tl.constexpr,
    ITERATIONS: tl.constexpr,
):
    smem = tle.gpu.alloc((STAGES, BLOCK), dtype=tl.float16, nv_mma_shared_layout=False)
    pipe = tle.pipe(capacity=STAGES, one_shot=True, name="one_shot_uncommitted_rewrite", data=smem)
    writer = pipe.writer()
    slot = writer.acquire(0)
    tle.gpu.copy(desc, slot.data, (BLOCK, ), (0, ))
    writer.commit(0)
    slot = writer.acquire(0)
    tle.gpu.copy(desc, slot.data, (BLOCK, ), (0, ))
    wait = pipe.reader().wait(0)
    tl.store(out, tl.where(wait.is_closed, 1, 0))


@triton.jit
def _one_shot_close_kernel(
    desc,
    out,
    STAGES: tl.constexpr,
    BLOCK: tl.constexpr,
    ITERATIONS: tl.constexpr,
):
    smem = tle.gpu.alloc((STAGES, BLOCK), dtype=tl.float16, nv_mma_shared_layout=False)
    pipe = tle.pipe(capacity=STAGES, one_shot=True, name="one_shot_close", data=smem)
    pipe.writer().close(0)


@triton.jit
def _one_shot_dynamic_stage_kernel(
    desc,
    out,
    STAGES: tl.constexpr,
    BLOCK: tl.constexpr,
    ITERATIONS: tl.constexpr,
):
    smem = tle.gpu.alloc((STAGES, BLOCK), dtype=tl.float16, nv_mma_shared_layout=False)
    pipe = tle.pipe(capacity=STAGES, one_shot=True, name="one_shot_dynamic_stage", data=smem)
    stage = tl.program_id(0)
    slot = pipe.writer().acquire(stage)
    tle.gpu.copy(desc, slot.data, (BLOCK, ), (0, ))
    pipe.writer().commit(stage)
    wait = pipe.reader().wait(stage)
    tl.store(out, tl.where(wait.is_closed, 1, 0))


def _compile_invalid_pipeline(
    fn,
    kind=None,
    desc_type="tensordesc<fp16[128]>",
    signature=None,
):
    target, backend = mthreads_backend()
    options = backend.parse_options({"num_warps": 16, "num_stages": 1})
    context = ir.context()
    ir.load_dialects(context)
    backend.load_dialects(context)
    if signature is None:
        signature = {
            "desc": desc_type,
            "out": "*i32",
            "STAGES": "constexpr",
            "BLOCK": "constexpr",
            "ITERATIONS": "constexpr",
        }
    constexprs = {"STAGES": 2, "BLOCK": 128, "ITERATIONS": 2}
    if kind is not None:
        signature["KIND"] = "constexpr"
        constexprs["KIND"] = kind
    source = ASTSource(
        fn=fn,
        signature=signature,
        constexprs=constexprs,
        attrs=tme_descriptor_attrs(signature),
    )
    module = source.make_ir(
        target,
        options,
        backend.get_codegen_implementation(options),
        backend.get_module_map(),
        context,
    )
    compiler_stages = {}
    backend.add_stages(compiler_stages, options, Language.TRITON)
    metadata = {}
    module = compiler_stages["ttir"](module, metadata)
    return compiler_stages["ttgir"](module, metadata)


def _i32_constants(ir_text):
    return {
        name: int(value)
        for name, value in re.findall(r"(%[-\w.]+)\s*=\s*arith\.constant\s+(-?\d+)\s*:\s*i32", ir_text)
    }


def _assert_pipe_lowering_clean(compiled):
    """Check only final-IR hygiene that device execution cannot observe."""
    for ir_text in (compiled.asm["ttgir"], compiled.asm["llir"]):
        for marker in (
                "musa_tle.pipe.",
                "musa_tle.completion_group",
                "musa_tle.expect_bytes",
                "musa_tle.pipe_reader_tme_store",
        ):
            assert marker not in ir_text, ir_text


def _assert_local_store_pipe_artifacts(compiled, stages, iterations, writer_warps, reader_warps):
    ttgir = compiled.asm["ttgir"]
    llir = compiled.asm["llir"]
    _assert_pipe_lowering_clean(compiled)
    assert "ttmg.barrier_add_trans" not in ttgir, ttgir
    assert "ttmg.async_tme_copy_global_to_local" not in ttgir, ttgir
    assert "llvm.musa.async.add.trans" not in llir, llir
    assert "llvm.musa.tme.ld" not in llir, llir


def _assert_async_copy_pipe_artifacts(
    compiled,
    iterations,
    async_copies_per_iteration=1,
    static_ws=False,
    expect_warp_arrives=2,
    expect_tme_transactions=False,
):
    """Async-copy transport lowers to per-thread g2s + wait and a warp arrive.

    ``async_copies_per_iteration`` counts the fused async copies the
    OptimizeLocalPointerAsyncStores pass emits per pipe generation.
    ``expect_warp_arrives`` counts warp-collective arrivals per iteration
    (writer full arrive + reader empty arrive).
    """
    ttgir = compiled.asm["ttgir"]
    llir = compiled.asm["llir"]
    _assert_pipe_lowering_clean(compiled)
    assert ttgir.count("ttg.async_copy_global_to_local") == async_copies_per_iteration * iterations, ttgir
    if expect_tme_transactions:
        assert "ttmg.barrier_add_trans" in ttgir, ttgir
        assert "llvm.musa.async.add.trans" in llir, llir
    else:
        assert "ttmg.barrier_add_trans" not in ttgir, ttgir
        assert "llvm.musa.async.add.trans" not in llir, llir
    assert ttgir.count("ttmg.warp_arrive_barrier") == expect_warp_arrives * iterations, ttgir
    # "llvm.musa.memcpy.g2s" is a prefix of "llvm.musa.memcpy.g2s.wait", so
    # count the copy calls through their call syntax.
    assert llir.count("call void @llvm.musa.memcpy.g2s(") == async_copies_per_iteration * iterations, llir
    assert llir.count("call void @llvm.musa.memcpy.g2s.wait()") >= iterations, llir
    if static_ws:
        assert "llvm.musa.barrier0" not in llir, llir


def _assert_close_artifacts(
    compiled,
    stages,
    payloads,
    full_arrival_count,
    reader_warps,
    control_arrivals,
    warp_arrivals,
    payload_tme=False,
    reader_tme_stores=0,
    static_ws=False,
    wait_count=None,
    static_total_warps=20,
):
    ttgir = compiled.asm["ttgir"]
    llir = compiled.asm["llir"]
    _assert_pipe_lowering_clean(compiled)
    assert f"!ttg.memdesc<{stages}x1xi32" in ttgir, ttgir
    assert "musa_tle.pipe.writer_close" not in ttgir, ttgir
    assert ttgir.count("arith.cmpi ne") >= 1, ttgir

    if static_ws:
        assert compiled.metadata.num_warps == static_total_warps
        assert f'"ttg.total-num-warps" = {static_total_warps} : i32' in ttgir, ttgir
        assert "llvm.musa.barrier0" not in llir, llir


def _assert_one_shot_artifacts(
    compiled,
    stages,
    publications,
    full_arrival_count,
    reader_waits,
    tme_fields=0,
    reader_tme_stores=0,
    transaction_bytes=None,
    local_transport=False,
    static_ws=False,
    static_total_warps=20,
):
    ttgir = compiled.asm["ttgir"]
    llir = compiled.asm["llir"]
    _assert_pipe_lowering_clean(compiled)
    assert f"!ttg.memdesc<{stages}x1xi32" not in ttgir, ttgir

    if static_ws:
        assert compiled.metadata.num_warps == static_total_warps
        assert f'"ttg.total-num-warps" = {static_total_warps} : i32' in ttgir, ttgir
        assert "llvm.musa.barrier0" not in llir, llir


def _assert_ws_named_close_artifacts(compiled, stages, payloads=0, reader_tme_stores=0, reader_warps=20):
    """Check the static-WS named close protocol without snapshotting IR."""
    ttgir = compiled.asm["ttgir"]
    llir = compiled.asm["llir"]
    _assert_close_artifacts(
        compiled,
        stages,
        payloads,
        1,
        reader_warps,
        0,
        0,
        payload_tme=bool(payloads),
        reader_tme_stores=reader_tme_stores,
        static_ws=True,
        static_total_warps=24,
    )
    assert 'readers = ["left", "right"]' in compiled.asm["ttir"]
    assert "musa.max_bar_id = %d" % (2 * stages) in ttgir
    assert "llvm.musa.barrier0" not in llir


@pytest.mark.parametrize("stages", [1, 2, 3], ids=["stage1", "stage2", "stage3"])
def test_musa_ws_named_reader_payload_close_runtime(stages):
    payloads = 2 * stages + 1
    half_m, half_n, float_m, float_n = 16, 32, 8, 16
    half_source = torch.arange(payloads * half_m * half_n, dtype=torch.float16,
                               device="musa").reshape(payloads * half_m, half_n)
    float_source = torch.arange(payloads * float_m * float_n, dtype=torch.float32,
                                device="musa").reshape(payloads * float_m, float_n)
    half_output = torch.empty_like(half_source)
    float_output = torch.empty_like(float_source)
    left_flags = torch.empty((payloads + 1, ), dtype=torch.int32, device="musa")
    right_flags = torch.empty_like(left_flags)
    half_desc = TensorDescriptor.from_tensor(half_source, [half_m, half_n])
    float_desc = TensorDescriptor.from_tensor(float_source, [float_m, float_n])
    float_output_desc = TensorDescriptor.from_tensor(float_output, [float_m, float_n])
    compiled = _ws_named_payload_close_kernel.warmup(
        half_desc,
        float_desc,
        half_output,
        float_output_desc,
        left_flags,
        right_flags,
        STAGES=stages,
        PAYLOADS=payloads,
        HALF_M=half_m,
        HALF_N=half_n,
        FLOAT_M=float_m,
        FLOAT_N=float_n,
        grid=(1, ),
        num_warps=16,
        num_stages=1,
    )
    _assert_ws_named_close_artifacts(compiled, stages, payloads, payloads, reader_warps=8)
    for _ in range(4):
        half_output.fill_(float("nan"))
        float_output.fill_(float("nan"))
        left_flags.fill_(-1)
        right_flags.fill_(-1)
        _ws_named_payload_close_kernel[(1, )](
            half_desc,
            float_desc,
            half_output,
            float_output_desc,
            left_flags,
            right_flags,
            STAGES=stages,
            PAYLOADS=payloads,
            HALF_M=half_m,
            HALF_N=half_n,
            FLOAT_M=float_m,
            FLOAT_N=float_n,
            num_warps=16,
            num_stages=1,
        )
        torch.musa.synchronize()
        assert torch.equal(half_output, half_source)
        assert torch.equal(float_output, float_source)
        assert torch.equal(left_flags[:payloads], torch.zeros_like(left_flags[:payloads]))
        assert torch.equal(right_flags[:payloads], torch.zeros_like(right_flags[:payloads]))
        assert left_flags[payloads].item() == 1
        assert right_flags[payloads].item() == 1


@pytest.mark.parametrize("stages", [1, 2, 3], ids=["stage1", "stage2", "stage3"])
def test_musa_non_ws_one_shot_tme_roundtrip_runtime(stages):
    block = 128
    source = torch.arange(stages * block, dtype=torch.float16, device="musa")
    first_destination = torch.empty_like(source)
    second_destination = torch.empty_like(source)
    flags = torch.empty((2 * stages, ), dtype=torch.int32, device="musa")
    source_desc = TensorDescriptor.from_tensor(source, [block])
    first_destination_desc = TensorDescriptor.from_tensor(first_destination, [block])
    second_destination_desc = TensorDescriptor.from_tensor(second_destination, [block])

    compiled = _one_shot_tme_roundtrip_kernel.warmup(
        source_desc,
        first_destination_desc,
        second_destination_desc,
        flags,
        STAGES=stages,
        BLOCK=block,
        grid=(1, ),
        num_warps=4,
        num_stages=1,
    )
    _assert_one_shot_artifacts(
        compiled,
        stages,
        stages,
        1,
        2 * stages,
        tme_fields=1,
        reader_tme_stores=2 * stages,
        transaction_bytes=block * 2,
    )

    for _ in range(2):
        first_destination.fill_(float("nan"))
        second_destination.fill_(float("nan"))
        flags.fill_(-1)
        _one_shot_tme_roundtrip_kernel[(1, )](
            source_desc,
            first_destination_desc,
            second_destination_desc,
            flags,
            STAGES=stages,
            BLOCK=block,
            num_warps=4,
            num_stages=1,
        )
        torch.musa.synchronize()
        assert torch.equal(first_destination, source)
        assert torch.equal(second_destination, source)
        assert torch.equal(flags, torch.zeros_like(flags))


def test_musa_non_ws_one_shot_heterogeneous_tme_runtime():
    stages = 2
    half_m, half_n = 16, 32
    float_m, float_n = 8, 16
    half_source = torch.randn((stages * half_m, half_n), dtype=torch.float16, device="musa")
    float_source = torch.randn((stages * float_m, float_n), dtype=torch.float32, device="musa")
    half_destination = torch.empty_like(half_source)
    float_destination = torch.empty_like(float_source)
    flags = torch.empty((stages, ), dtype=torch.int32, device="musa")
    half_source_desc = TensorDescriptor.from_tensor(half_source, [half_m, half_n])
    float_source_desc = TensorDescriptor.from_tensor(float_source, [float_m, float_n])
    half_destination_desc = TensorDescriptor.from_tensor(half_destination, [half_m, half_n])
    float_destination_desc = TensorDescriptor.from_tensor(float_destination, [float_m, float_n])

    compiled = _one_shot_heterogeneous_tme_kernel.warmup(
        half_source_desc,
        float_source_desc,
        half_destination_desc,
        float_destination_desc,
        flags,
        STAGES=stages,
        HALF_M=half_m,
        HALF_N=half_n,
        FLOAT_M=float_m,
        FLOAT_N=float_n,
        grid=(1, ),
        num_warps=4,
        num_stages=1,
    )
    transaction_bytes = half_m * half_n * 2 + float_m * float_n * 4
    _assert_one_shot_artifacts(
        compiled,
        stages,
        stages,
        1,
        stages,
        tme_fields=2,
        reader_tme_stores=2 * stages,
        transaction_bytes=transaction_bytes,
    )

    for _ in range(2):
        half_destination.fill_(float("nan"))
        float_destination.fill_(float("nan"))
        flags.fill_(-1)
        _one_shot_heterogeneous_tme_kernel[(1, )](
            half_source_desc,
            float_source_desc,
            half_destination_desc,
            float_destination_desc,
            flags,
            STAGES=stages,
            HALF_M=half_m,
            HALF_N=half_n,
            FLOAT_M=float_m,
            FLOAT_N=float_n,
            num_warps=4,
            num_stages=1,
        )
        torch.musa.synchronize()
        assert torch.equal(half_destination, half_source)
        assert torch.equal(float_destination, float_source)
        assert torch.equal(flags, torch.zeros_like(flags))


def test_musa_non_ws_one_shot_mixed_runtime():
    stages = 2
    block = 128
    tme_source = torch.arange(stages * block, dtype=torch.float16, device="musa")
    tme_destination = torch.empty_like(tme_source)
    local_output = torch.empty((stages * block, ), dtype=torch.int32, device="musa")
    flags = torch.empty((stages, ), dtype=torch.int32, device="musa")
    source_desc = TensorDescriptor.from_tensor(tme_source, [block])
    destination_desc = TensorDescriptor.from_tensor(tme_destination, [block])
    compiled = _one_shot_mixed_kernel.warmup(
        source_desc,
        destination_desc,
        local_output,
        flags,
        STAGES=stages,
        BLOCK=block,
        grid=(1, ),
        num_warps=4,
        num_stages=1,
    )
    _assert_one_shot_artifacts(
        compiled,
        stages,
        stages,
        5,
        stages,
        tme_fields=1,
        reader_tme_stores=stages,
        transaction_bytes=block * 2,
        local_transport=True,
    )
    expected_local = torch.arange(stages * block, dtype=torch.int32, device="musa")
    for _ in range(2):
        tme_destination.fill_(float("nan"))
        local_output.fill_(-1)
        flags.fill_(-1)
        _one_shot_mixed_kernel[(1, )](
            source_desc,
            destination_desc,
            local_output,
            flags,
            STAGES=stages,
            BLOCK=block,
            num_warps=4,
            num_stages=1,
        )
        torch.musa.synchronize()
        assert torch.equal(tme_destination, tme_source)
        assert torch.equal(local_output, expected_local)
        assert torch.equal(flags, torch.zeros_like(flags))


def test_musa_non_ws_pipe_close_only_runtime():
    output = torch.empty((1, ), dtype=torch.int32, device="musa")
    compiled = _non_ws_close_only_kernel.warmup(
        output,
        grid=(1, ),
        num_warps=4,
        num_stages=1,
    )
    _assert_close_artifacts(compiled, 1, 0, 1, 4, 1, 0, wait_count=2)
    for _ in range(2):
        output.fill_(-1)
        _non_ws_close_only_kernel[(1, )](output, num_warps=4, num_stages=1)
        torch.musa.synchronize()
        assert torch.equal(output, torch.ones_like(output))


@pytest.mark.parametrize("stages", [1, 2, 3], ids=["stage1", "stage2", "stage3"])
def test_musa_non_ws_pipe_tme_close_runtime(stages):
    payloads = 2 * stages + 1
    block = 128
    source = torch.arange(payloads * block, dtype=torch.float16, device="musa")
    output = torch.empty_like(source)
    flags = torch.empty((payloads + 1, ), dtype=torch.int32, device="musa")
    source_desc = TensorDescriptor.from_tensor(source, [block])
    output_desc = TensorDescriptor.from_tensor(output, [block])
    compiled = _non_ws_tme_close_kernel.warmup(
        source_desc,
        output_desc,
        flags,
        STAGES=stages,
        BLOCK=block,
        PAYLOADS=payloads,
        RELEASE_CLOSE=False,
        grid=(1, ),
        num_warps=4,
        num_stages=1,
    )
    _assert_close_artifacts(
        compiled,
        stages,
        payloads,
        1,
        4,
        payloads + 1,
        payloads,
        payload_tme=True,
        reader_tme_stores=payloads,
        wait_count=2 * (payloads + 1),
    )
    for _ in range(2):
        output.fill_(float("nan"))
        flags.fill_(-1)
        _non_ws_tme_close_kernel[(1, )](
            source_desc,
            output_desc,
            flags,
            STAGES=stages,
            BLOCK=block,
            PAYLOADS=payloads,
            RELEASE_CLOSE=False,
            num_warps=4,
            num_stages=1,
        )
        torch.musa.synchronize()
        assert torch.equal(output, source)
        assert torch.equal(flags[:payloads], torch.zeros_like(flags[:payloads]))
        assert flags[payloads].item() == 1


def test_musa_non_ws_pipe_closed_wait_release_runtime():
    stages = 2
    payloads = 5
    block = 128
    source = torch.arange(payloads * block, dtype=torch.float16, device="musa")
    output = torch.empty_like(source)
    flags = torch.empty((payloads + 1, ), dtype=torch.int32, device="musa")
    source_desc = TensorDescriptor.from_tensor(source, [block])
    output_desc = TensorDescriptor.from_tensor(output, [block])
    compiled = _non_ws_tme_close_kernel.warmup(
        source_desc,
        output_desc,
        flags,
        STAGES=stages,
        BLOCK=block,
        PAYLOADS=payloads,
        RELEASE_CLOSE=True,
        grid=(1, ),
        num_warps=4,
        num_stages=1,
    )
    _assert_close_artifacts(
        compiled,
        stages,
        payloads,
        1,
        4,
        payloads + 1,
        payloads + 1,
        payload_tme=True,
        reader_tme_stores=payloads,
        wait_count=2 * (payloads + 1),
    )
    for _ in range(2):
        output.fill_(float("nan"))
        flags.fill_(-1)
        _non_ws_tme_close_kernel[(1, )](
            source_desc,
            output_desc,
            flags,
            STAGES=stages,
            BLOCK=block,
            PAYLOADS=payloads,
            RELEASE_CLOSE=True,
            num_warps=4,
            num_stages=1,
        )
        torch.musa.synchronize()
        assert torch.equal(output, source)
        assert torch.equal(flags[:payloads], torch.zeros_like(flags[:payloads]))
        assert flags[payloads].item() == 1


@pytest.mark.parametrize("stages", [1, 2, 3], ids=["stage1", "stage2", "stage3"])
def test_musa_ws_pipe_tme_store_drain_close_runtime(stages):
    payloads = 2 * stages + 1
    block = 128
    source = torch.arange(payloads * block, dtype=torch.float16, device="musa")
    destination = torch.empty_like(source)
    flags = torch.empty((payloads + 1, ), dtype=torch.int32, device="musa")
    source_desc = TensorDescriptor.from_tensor(source, [block])
    destination_desc = TensorDescriptor.from_tensor(destination, [block])
    compiled = _ws_tme_store_close_kernel.warmup(
        source_desc,
        destination_desc,
        flags,
        STAGES=stages,
        BLOCK=block,
        PAYLOADS=payloads,
        grid=(1, ),
        num_warps=16,
        num_stages=1,
    )
    _assert_close_artifacts(
        compiled,
        stages,
        payloads,
        1,
        16,
        payloads + 1,
        payloads,
        payload_tme=True,
        reader_tme_stores=payloads,
        static_ws=True,
        wait_count=2 * (payloads + 1),
    )
    for _ in range(2):
        destination.fill_(float("nan"))
        flags.fill_(-1)
        _ws_tme_store_close_kernel[(1, )](
            source_desc,
            destination_desc,
            flags,
            STAGES=stages,
            BLOCK=block,
            PAYLOADS=payloads,
            num_warps=16,
            num_stages=1,
        )
        torch.musa.synchronize()
        assert torch.equal(destination, source)
        assert torch.equal(flags[:payloads], torch.zeros_like(flags[:payloads]))
        assert flags[payloads].item() == 1


def test_musa_pipe_rejects_declared_reader_without_lifecycle(capfd):
    with pytest.raises(RuntimeError, match="PassManager::run failed"):
        _compile_invalid_pipeline(_named_reader_kernel)
    assert "MUSA TLE pipe declared reader has no lifecycle operations" in capfd.readouterr().err


@pytest.mark.parametrize(
    "kind,diagnostic",
    [
        (0, "tle.pipe readers must be unique, got duplicate 'left'"),
        (1, "tle.pipe.reader requires a reader name when pipe readers are declared"),
        (2, "tle.pipe.reader name 'missing' is not declared"),
        (3, "tle.pipe.reader field 'missing' is not a pipe field"),
        (4, "tle.pipe.reader fields must be unique, got duplicate 'data'"),
    ],
)
def test_musa_pipe_python_api_rejects_invalid_named_reader(kind, diagnostic):
    with pytest.raises(CompilationError, match=re.escape(diagnostic)):
        _compile_invalid_pipeline(_invalid_named_reader_python_api_kernel, kind)


@pytest.mark.parametrize(
    "kind,diagnostic",
    [
        (0, "cyclic named reader must wait and release each payload generation exactly once"),
        (1, "MUSA TLE pipe reader.release requires a same-endpoint, same-block, same-stage reader.wait"),
        (2, "MUSA TLE pipe reader.release requires a same-endpoint, same-block, same-stage reader.wait"),
        (3, "cyclic named reader generation must match writer stage and phase"),
    ],
)
def test_musa_pipe_rejects_invalid_named_reader_lifecycle(capfd, kind, diagnostic):
    with pytest.raises(RuntimeError, match="PassManager::run failed"):
        _compile_invalid_pipeline(_invalid_named_reader_lifecycle_kernel, kind)
    assert diagnostic in capfd.readouterr().err


@pytest.mark.parametrize(
    "kind,diagnostic",
    [
        (0, "cyclic named reader must observe writer.close exactly once"),
        (1, "close generation does not carry payload"),
        (2, "terminal reader.wait must match writer.close stage and phase"),
        (3, "cyclic named reader must wait and release each payload generation exactly once"),
    ],
)
def test_musa_pipe_rejects_invalid_named_reader_close_lifecycle(capfd, kind, diagnostic):
    with pytest.raises(RuntimeError, match="PassManager::run failed"):
        _compile_invalid_pipeline(_invalid_named_reader_close_lifecycle_kernel, kind)
    assert diagnostic in capfd.readouterr().err


@pytest.mark.parametrize(
    "kind,diagnostic",
    [
        (0, "pipe TME store requires an open same-block reader generation with the same stage"),
        (1, "cannot uniquely associate TME store with a pipe field"),
    ],
)
def test_musa_pipe_rejects_invalid_named_reader_one_shot_drain(capfd, kind, diagnostic):
    with pytest.raises(RuntimeError, match="PassManager::run failed"):
        _compile_invalid_pipeline(_invalid_named_reader_one_shot_drain_kernel, kind)
    assert diagnostic in capfd.readouterr().err


@pytest.mark.parametrize(
    "kernel,diagnostic",
    [
        (
            _named_reader_unsubscribed_tme_store_kernel,
            "reader TME store source is not included in the reader field subscription",
        ),
        (
            _named_reader_subscription_mismatch_kernel,
            "endpoint operations require one stable field subscription",
        ),
    ],
)
def test_musa_pipe_rejects_invalid_named_reader_subscription(capfd, kernel, diagnostic):
    with pytest.raises(RuntimeError, match="PassManager::run failed"):
        _compile_invalid_pipeline(kernel)
    assert diagnostic in capfd.readouterr().err


def _assert_ws_named_partition_artifacts(
    compiled,
    stages,
    iterations,
    total_warps,
    empty_arrival,
    right_issue_thread,
    writer_issue_thread=0,
    tme_fields=2,
    full_arrival=1,
    writer_local_transport=False,
):
    ttir = compiled.asm["ttir"]
    ttgir = compiled.asm["ttgir"]
    llir = compiled.asm["llir"]
    _assert_pipe_lowering_clean(compiled)

    assert compiled.metadata.num_warps == total_warps
    assert f'"ttg.total-num-warps" = {total_warps} : i32' in ttgir, ttgir
    assert f"musa.max_bar_id = {2 * stages}" in ttgir, ttgir
    assert 'readers = ["left", "right"]' in ttir, ttir
    assert 'reader_fields = ["a"]' in ttir, ttir
    assert 'reader_fields = ["b"]' in ttir, ttir
    # Partition issue-thread placement is a low-level ownership contract; the
    # runtime test covers iteration/phase reuse without pinning textual counts.
    assert any(f"musa.tme.issue_thread = {writer_issue_thread} : i32" in line for line in ttgir.splitlines()), ttgir
    assert any(f"musa.tme.issue_thread = {right_issue_thread} : i32" in line for line in ttgir.splitlines()), ttgir
    assert "llvm.musa.barrier0" not in llir, llir


@pytest.mark.parametrize("stages", [1, 2, 3], ids=["stage1", "stage2", "stage3"])
def test_musa_ws_named_reader_worker1_writer_partition_runtime(stages):
    iterations = 2 * stages + 1
    half_m, half_n = 16, 32
    float_m, float_n = 8, 16
    half_source = torch.arange(iterations * half_m * half_n, dtype=torch.float16,
                               device="musa").reshape(iterations * half_m, half_n)
    float_source = torch.arange(iterations * float_m * float_n, dtype=torch.float32,
                                device="musa").reshape(iterations * float_m, float_n)
    half_output = torch.empty_like(half_source)
    float_output = torch.empty_like(float_source)
    idle_marker = torch.full((1, ), -1, dtype=torch.int32, device="musa")
    flags = torch.empty((2 * iterations, ), dtype=torch.int32, device="musa")
    half_source_desc = TensorDescriptor.from_tensor(half_source, [half_m, half_n])
    float_source_desc = TensorDescriptor.from_tensor(float_source, [float_m, float_n])
    float_destination_desc = TensorDescriptor.from_tensor(float_output, [float_m, float_n])

    compiled = _ws_named_partition_worker1_writer_kernel.warmup(
        half_source_desc,
        float_source_desc,
        half_output,
        float_destination_desc,
        idle_marker,
        flags,
        STAGES=stages,
        ITERATIONS=iterations,
        HALF_M=half_m,
        HALF_N=half_n,
        FLOAT_M=float_m,
        FLOAT_N=float_n,
        grid=(1, ),
        num_warps=16,
        num_stages=1,
    )
    _assert_ws_named_partition_artifacts(
        compiled,
        stages,
        iterations,
        total_warps=25,
        empty_arrival=20,
        right_issue_thread=672,
        writer_issue_thread=544,
    )

    for _ in range(4):
        half_output.fill_(float("nan"))
        float_output.fill_(float("nan"))
        idle_marker.fill_(-1)
        flags.fill_(-1)
        _ws_named_partition_worker1_writer_kernel[(1, )](
            half_source_desc,
            float_source_desc,
            half_output,
            float_destination_desc,
            idle_marker,
            flags,
            STAGES=stages,
            ITERATIONS=iterations,
            HALF_M=half_m,
            HALF_N=half_n,
            FLOAT_M=float_m,
            FLOAT_N=float_n,
            num_warps=16,
            num_stages=1,
        )
        torch.musa.synchronize()
        assert torch.equal(half_output, half_source)
        assert torch.equal(float_output, float_source)
        assert torch.equal(idle_marker, torch.zeros_like(idle_marker))
        assert torch.equal(flags, torch.zeros_like(flags))


@pytest.mark.parametrize("stages", [1, 2, 3], ids=["stage1", "stage2", "stage3"])
def test_musa_ws_named_reader_mixed_worker_partition_runtime(stages):
    iterations = 2 * stages + 1
    block = 128
    tme_source = torch.arange(iterations * block, dtype=torch.float16, device="musa")
    tme_output = torch.empty_like(tme_source)
    local_output = torch.empty((iterations * block, ), dtype=torch.int32, device="musa")
    flags = torch.empty((2 * iterations, ), dtype=torch.int32, device="musa")
    source_desc = TensorDescriptor.from_tensor(tme_source, [block])
    tme_output_desc = TensorDescriptor.from_tensor(tme_output, [block])

    compiled = _ws_named_partition_mixed_worker_writer_kernel.warmup(
        source_desc,
        local_output,
        tme_output_desc,
        flags,
        STAGES=stages,
        ITERATIONS=iterations,
        BLOCK=block,
        grid=(1, ),
        num_warps=16,
        num_stages=1,
    )
    _assert_ws_named_partition_artifacts(
        compiled,
        stages,
        iterations,
        total_warps=24,
        empty_arrival=20,
        right_issue_thread=0,
        writer_issue_thread=512,
        tme_fields=1,
        full_arrival=5,
        writer_local_transport=True,
    )

    expected_local = torch.arange(iterations * block, dtype=torch.int32, device="musa")
    for _ in range(4):
        tme_output.fill_(float("nan"))
        local_output.fill_(-1)
        flags.fill_(-1)
        _ws_named_partition_mixed_worker_writer_kernel[(1, )](
            source_desc,
            local_output,
            tme_output_desc,
            flags,
            STAGES=stages,
            ITERATIONS=iterations,
            BLOCK=block,
            num_warps=16,
            num_stages=1,
        )
        torch.musa.synchronize()
        assert torch.equal(tme_output, tme_source)
        assert torch.equal(local_output, expected_local)
        assert torch.equal(flags, torch.zeros_like(flags))


@pytest.mark.parametrize(
    "kind,diagnostic",
    [
        (0, "local-store transport requires one unmasked whole-field store"),
        (1, "MUSA TLE pipe completion sources for one field must not overlap"),
        (2, "MUSA TLE pipe does not support mixed transport sources for one payload field"),
    ],
)
def test_mthreads_pipe_rejects_unsupported_producer_protocol(capfd, kind, diagnostic):
    with pytest.raises(RuntimeError, match="PassManager::run failed"):
        _compile_invalid_pipeline(_invalid_pipe_kernel, kind)
    assert diagnostic in capfd.readouterr().err


def test_musa_pipe_rejects_multi_field_local_store_only(capfd):
    with pytest.raises(RuntimeError, match="PassManager::run failed"):
        _compile_invalid_pipeline(_multi_local_store_only_pipe_kernel)
    assert "MUSA TLE pipe local-store-only transport currently requires one payload field" in capfd.readouterr().err


def test_musa_async_copy_pipe_compiles():
    module = _compile_invalid_pipeline(_async_copy_pipe_kernel, desc_type="*fp32")
    ttgir = str(module)
    # The fused load+store pair now lowers to an async-copy pipe transport.
    assert "ttg.async_copy_global_to_local" in ttgir, ttgir
    assert "musa_tle.pipe." not in ttgir, ttgir


def test_musa_pipe_rejects_mixed_tme_async_copy_same_field(capfd):
    with pytest.raises(RuntimeError, match="PassManager::run failed"):
        _compile_invalid_pipeline(
            _mixed_tme_async_copy_same_field_kernel,
            signature={
                "desc": "tensordesc<fp32[128]>",
                "src": "*fp32",
                "out": "*i32",
                "STAGES": "constexpr",
                "BLOCK": "constexpr",
                "ITERATIONS": "constexpr",
            },
        )
    assert "MUSA TLE pipe does not support mixed transport sources for one payload field" in capfd.readouterr().err


@pytest.mark.parametrize(
    "kernel,diagnostic",
    [
        (_duplicate_close_kernel, "MUSA TLE pipe supports at most one writer.close per pipe"),
        (_close_with_open_writer_kernel, "MUSA TLE pipe close requires all writer payload generations to commit"),
        (_acquire_after_close_kernel, "MUSA TLE pipe writer operations are not allowed after writer.close"),
        (_commit_after_close_kernel, "MUSA TLE pipe writer operations are not allowed after writer.close"),
        (_close_payload_tme_store_kernel, "MUSA TLE pipe close generation does not carry payload"),
    ],
)
def test_musa_pipe_rejects_invalid_close_lifecycle(capfd, kernel, diagnostic):
    with pytest.raises(RuntimeError, match="PassManager::run failed"):
        _compile_invalid_pipeline(kernel)
    assert diagnostic in capfd.readouterr().err


@pytest.mark.parametrize("kind", [0, 1], ids=["stage", "phase_with_release"])
def test_musa_pipe_rejects_terminal_close_coordinate_mismatch(capfd, kind):
    with pytest.raises(RuntimeError, match="PassManager::run failed"):
        _compile_invalid_pipeline(_close_terminal_mismatch_kernel, kind)
    assert ("MUSA TLE pipe terminal reader.wait must match writer.close stage and phase" in capfd.readouterr().err)


@pytest.mark.parametrize(
    "kernel,diagnostic",
    [
        (_one_shot_duplicate_commit_kernel, "MUSA TLE one-shot pipe stage may be published at most once"),
        (_one_shot_phase_reuse_kernel, "MUSA TLE one-shot pipe does not support phase changes or stage reuse"),
        (_one_shot_dynamic_stage_kernel, "MUSA TLE one-shot pipe requires a statically known stage within capacity"),
        (_one_shot_mutation_kernel, "MUSA TLE one-shot pipe payload is immutable after publication"),
        (_one_shot_uncommitted_rewrite_kernel, "MUSA TLE one-shot pipe payload is immutable after publication"),
    ],
)
def test_musa_one_shot_pipe_rejects_invalid_lifecycle(capfd, kernel, diagnostic):
    with pytest.raises(RuntimeError, match="PassManager::run failed"):
        _compile_invalid_pipeline(kernel)
    assert diagnostic in capfd.readouterr().err


def test_musa_one_shot_pipe_rejects_close():
    with pytest.raises(CompilationError, match="one_shot pipes do not support close"):
        _compile_invalid_pipeline(_one_shot_close_kernel)


@pytest.mark.parametrize(
    "kind,diagnostic",
    [
        (0, "MUSA TLE static warp-specialized pipe partitions must host at most one pipe endpoint"),
        (1, "MUSA TLE pipe endpoint operations must remain in one static warp-specialize partition"),
    ],
)
def test_musa_pipe_rejects_invalid_static_endpoint_placement(capfd, kind, diagnostic):
    with pytest.raises(RuntimeError, match="PassManager::run failed"):
        _compile_invalid_pipeline(
            _ws_invalid_endpoint_placement_kernel,
            kind,
        )
    assert diagnostic in capfd.readouterr().err


def _assert_reader_tme_store_artifacts(
    compiled,
    stages,
    iterations,
    writer_warps,
    reader_warps,
    static_ws=False,
    tme_fields=1,
    transaction_bytes=None,
):
    ttgir = compiled.asm["ttgir"]
    llir = compiled.asm["llir"]
    _assert_pipe_lowering_clean(compiled)
    if static_ws:
        # Static WS uses the full barrier wait as the acquire publication edge;
        # the per-store CTA-wide issue barrier would otherwise wait for the
        # producer partition and deadlock.
        assert "llvm.musa.barrier0" not in llir, llir


def _assert_external_barrier_pipe_artifacts(compiled, stages, iterations, static_ws=False, tme_sources_per_generation=1,
                                            transaction_bytes=256):
    ttgir = compiled.asm["ttgir"]
    llir = compiled.asm["llir"]
    _assert_pipe_lowering_clean(compiled)
    assert "musa_tle.pipe_barrier_ring" not in ttgir, ttgir
    assert f"llvm.musa.async.add.trans(i32 1, i32 {transaction_bytes})" in llir, llir
    if static_ws:
        assert "llvm.musa.barrier0" not in llir, llir


def _assert_named_reader_tme_artifacts(compiled, stages, iterations, reader_warps):
    ttir = compiled.asm["ttir"]
    ttgir = compiled.asm["ttgir"]
    llir = compiled.asm["llir"]
    _assert_pipe_lowering_clean(compiled)
    assert 'readers = ["local", "drain"]' in ttir, ttir
    assert "reader_fields" not in ttir, ttir
    assert f"musa.max_bar_id = {2 * stages}" in ttgir, ttgir


@pytest.mark.parametrize("stages", [1, 2, 3], ids=["stage1", "stage2", "stage3"])
def test_musa_non_ws_named_reader_tme_roundtrip_runtime(stages):
    iterations = 2 * stages + 1
    block = 128
    source = torch.arange(iterations * block, dtype=torch.float16, device="musa")
    local_output = torch.empty_like(source)
    drain_output = torch.empty_like(source)
    closed_flags = torch.empty((2 * iterations, ), dtype=torch.int32, device="musa")
    source_desc = TensorDescriptor.from_tensor(source, [block])
    drain_destination_desc = TensorDescriptor.from_tensor(drain_output, [block])

    compiled = _named_reader_tme_roundtrip_kernel.warmup(
        source_desc,
        local_output,
        drain_destination_desc,
        closed_flags,
        STAGES=stages,
        BLOCK=block,
        ITERATIONS=iterations,
        grid=(1, ),
        num_warps=4,
        num_stages=1,
    )
    _assert_named_reader_tme_artifacts(compiled, stages, iterations, 4)

    for _ in range(4):
        local_output.fill_(float("nan"))
        drain_output.fill_(float("nan"))
        closed_flags.fill_(-1)
        _named_reader_tme_roundtrip_kernel[(1, )](
            source_desc,
            local_output,
            drain_destination_desc,
            closed_flags,
            STAGES=stages,
            BLOCK=block,
            ITERATIONS=iterations,
            num_warps=4,
            num_stages=1,
        )
        torch.musa.synchronize()
        assert torch.equal(local_output, source)
        assert torch.equal(drain_output, source)
        assert torch.equal(closed_flags, torch.zeros_like(closed_flags))


def _assert_named_reader_interleaved_order_artifacts(compiled, stages, block):
    ttir = compiled.asm["ttir"]
    _assert_pipe_lowering_clean(compiled)
    readers = set()
    for line in ttir.splitlines():
        if "musa_tle.pipe.reader_wait" in line:
            pass
        elif "musa_tle.pipe.reader_release" in line:
            pass
        else:
            continue
        reader_name = re.search(r'reader_name = "([^"]+)"', line)
        assert reader_name, line
        readers.add(reader_name.group(1))
        if reader_name.group(1) == "all":
            assert "reader_fields" not in line, line
        else:
            assert reader_name.group(1) == "subset", line
            assert 'reader_fields = ["b"]' in line, line
    assert 'readers = ["all", "subset"]' in ttir, ttir
    assert readers == {"all", "subset"}


@pytest.mark.parametrize("stages", [1, 2, 3], ids=["stage1", "stage2", "stage3"])
def test_musa_non_ws_named_reader_interleaved_order_runtime(stages):
    block = 128
    iterations = 3 * stages
    cycle_elements = stages * block
    a_source = torch.arange(iterations * block, dtype=torch.float16, device="musa")
    b_source = torch.arange(iterations * block, dtype=torch.float32, device="musa") + 1000
    all_local_output = torch.empty_like(a_source)
    subset_local_output = torch.empty_like(b_source)
    all_drain_output = torch.empty_like(a_source)
    subset_drain_output = torch.empty_like(b_source)
    closed_flags = torch.empty((2 * iterations, ), dtype=torch.int32, device="musa")
    a_source_desc = TensorDescriptor.from_tensor(a_source, [block])
    b_source_desc = TensorDescriptor.from_tensor(b_source, [block])
    all_drain_destination_desc = TensorDescriptor.from_tensor(all_drain_output, [block])
    subset_drain_destination_desc = TensorDescriptor.from_tensor(subset_drain_output, [block])

    expected_all_local = torch.full_like(a_source, -1)
    expected_subset_local = torch.full_like(b_source, -1)
    expected_all_drain = torch.full_like(a_source, -1)
    expected_subset_drain = torch.full_like(b_source, -1)
    expected_all_local[:cycle_elements] = a_source[:cycle_elements]
    expected_subset_local[cycle_elements:] = b_source[cycle_elements:]
    expected_all_drain[cycle_elements:] = a_source[cycle_elements:]
    expected_subset_drain[:cycle_elements] = b_source[:cycle_elements]

    compiled = _named_reader_interleaved_order_kernel.warmup(
        a_source_desc,
        b_source_desc,
        all_local_output,
        subset_local_output,
        all_drain_destination_desc,
        subset_drain_destination_desc,
        closed_flags,
        STAGES=stages,
        BLOCK=block,
        grid=(1, ),
        num_warps=4,
        num_stages=1,
    )
    _assert_named_reader_interleaved_order_artifacts(compiled, stages, block)

    for _ in range(4):
        all_local_output.fill_(-1)
        subset_local_output.fill_(-1)
        all_drain_output.fill_(-1)
        subset_drain_output.fill_(-1)
        closed_flags.fill_(-1)
        _named_reader_interleaved_order_kernel[(1, )](
            a_source_desc,
            b_source_desc,
            all_local_output,
            subset_local_output,
            all_drain_destination_desc,
            subset_drain_destination_desc,
            closed_flags,
            STAGES=stages,
            BLOCK=block,
            num_warps=4,
            num_stages=1,
        )
        torch.musa.synchronize()
        assert torch.equal(all_local_output, expected_all_local)
        assert torch.equal(subset_local_output, expected_subset_local)
        assert torch.equal(all_drain_output, expected_all_drain)
        assert torch.equal(subset_drain_output, expected_subset_drain)
        assert torch.equal(closed_flags, torch.zeros_like(closed_flags))


def _assert_named_reader_partial_tme_artifacts(compiled, stages, iterations, transaction_bytes):
    ttir = compiled.asm["ttir"]
    ttgir = compiled.asm["ttgir"]
    _assert_pipe_lowering_clean(compiled)
    assert 'reader_fields = ["half"]' in ttir, ttir
    assert 'reader_fields = ["float_data"]' in ttir, ttir
    assert "ttmg.barrier_add_trans" in ttgir, ttgir
    assert "ttmg.async_tme_copy_local_to_global" in ttgir, ttgir


@pytest.mark.parametrize("stages", [1, 2, 3], ids=["stage1", "stage2", "stage3"])
def test_musa_non_ws_named_reader_partial_heterogeneous_runtime(stages):
    iterations = 2 * stages + 1
    half_m, half_n = 16, 32
    float_m, float_n = 8, 16
    half_source = torch.arange(iterations * half_m * half_n, dtype=torch.float16,
                               device="musa").reshape(iterations * half_m, half_n)
    float_source = torch.arange(iterations * float_m * float_n, dtype=torch.float32,
                                device="musa").reshape(iterations * float_m, float_n)
    half_output = torch.empty_like(half_source)
    float_output = torch.empty_like(float_source)
    closed_flags = torch.empty((2 * iterations, ), dtype=torch.int32, device="musa")
    half_source_desc = TensorDescriptor.from_tensor(half_source, [half_m, half_n])
    float_source_desc = TensorDescriptor.from_tensor(float_source, [float_m, float_n])
    float_destination_desc = TensorDescriptor.from_tensor(float_output, [float_m, float_n])

    compiled = _named_reader_partial_heterogeneous_kernel.warmup(
        half_source_desc,
        float_source_desc,
        half_output,
        float_destination_desc,
        closed_flags,
        STAGES=stages,
        ITERATIONS=iterations,
        HALF_M=half_m,
        HALF_N=half_n,
        FLOAT_M=float_m,
        FLOAT_N=float_n,
        grid=(1, ),
        num_warps=4,
        num_stages=1,
    )
    _assert_named_reader_partial_tme_artifacts(
        compiled,
        stages,
        iterations,
        half_m * half_n * 2 + float_m * float_n * 4,
    )

    for _ in range(4):
        half_output.fill_(float("nan"))
        float_output.fill_(float("nan"))
        closed_flags.fill_(-1)
        _named_reader_partial_heterogeneous_kernel[(1, )](
            half_source_desc,
            float_source_desc,
            half_output,
            float_destination_desc,
            closed_flags,
            STAGES=stages,
            ITERATIONS=iterations,
            HALF_M=half_m,
            HALF_N=half_n,
            FLOAT_M=float_m,
            FLOAT_N=float_n,
            num_warps=4,
            num_stages=1,
        )
        torch.musa.synchronize()
        assert torch.equal(half_output, half_source)
        assert torch.equal(float_output, float_source)
        assert torch.equal(closed_flags, torch.zeros_like(closed_flags))


def _assert_named_reader_partial_mixed_artifacts(compiled, stages, iterations, transaction_bytes):
    ttir = compiled.asm["ttir"]
    ttgir = compiled.asm["ttgir"]
    _assert_pipe_lowering_clean(compiled)
    assert 'reader_fields = ["tme", "left_local"]' in ttir, ttir
    assert 'reader_fields = ["tme", "right_local"]' in ttir, ttir
    assert "musa_tle.pipe_local_store_group" in ttgir, ttgir


def test_musa_non_ws_named_reader_partial_mixed_runtime():
    stages = 2
    iterations = 5
    m, n = 16, 32
    tme_source = torch.arange(iterations * m * n, dtype=torch.float16, device="musa").reshape(iterations * m, n)
    tme_output = torch.empty_like(tme_source)
    left_local_output = torch.empty((iterations * m * n, ), dtype=torch.int32, device="musa")
    right_local_output = torch.empty_like(left_local_output)
    expected_left = torch.arange(iterations * m * n, dtype=torch.int32, device="musa")
    expected_right = expected_left + 100000
    tme_source_desc = TensorDescriptor.from_tensor(tme_source, [m, n])

    compiled = _named_reader_partial_mixed_kernel.warmup(
        tme_source_desc,
        tme_output,
        left_local_output,
        right_local_output,
        STAGES=stages,
        ITERATIONS=iterations,
        M=m,
        N=n,
        grid=(1, ),
        num_warps=4,
        num_stages=1,
    )
    _assert_named_reader_partial_mixed_artifacts(compiled, stages, iterations, m * n * 2)

    for _ in range(4):
        tme_output.fill_(float("nan"))
        left_local_output.fill_(-1)
        right_local_output.fill_(-1)
        _named_reader_partial_mixed_kernel[(1, )](
            tme_source_desc,
            tme_output,
            left_local_output,
            right_local_output,
            STAGES=stages,
            ITERATIONS=iterations,
            M=m,
            N=n,
            num_warps=4,
            num_stages=1,
        )
        torch.musa.synchronize()
        assert torch.equal(tme_output, tme_source)
        assert torch.equal(left_local_output, expected_left)
        assert torch.equal(right_local_output, expected_right)


def _assert_same_field_tme_fragment_artifacts(compiled, iterations, external=False):
    ttgir = compiled.asm["ttgir"]
    llir = compiled.asm["llir"]
    _assert_pipe_lowering_clean(compiled)
    if external:
        assert "musa_tle.pipe_barrier_ring" not in ttgir, ttgir
    assert f"llvm.musa.async.add.trans(i32 1, i32 {16 * 32 * 2})" in llir, llir


def _assert_named_reader_same_field_tme_fragment_artifacts(compiled, stages, iterations):
    ttir = compiled.asm["ttir"]
    ttgir = compiled.asm["ttgir"]
    llir = compiled.asm["llir"]
    _assert_pipe_lowering_clean(compiled)
    assert 'readers = ["top", "bottom"]' in ttir, ttir


@pytest.mark.parametrize("stages", _PIPE_REUSE_STAGES, ids=[f"stage{stage}" for stage in _PIPE_REUSE_STAGES])
def test_musa_non_ws_named_reader_same_field_tme_fragments_runtime(stages):
    iterations = _phase_reuse_iterations(stages)
    half_m, half_n = 8, 32
    source_top = torch.arange(iterations * half_m * half_n, dtype=torch.float16,
                              device="musa").reshape(iterations * half_m, half_n)
    source_bottom = source_top + 10000
    output_top = torch.empty_like(source_top)
    output_bottom = torch.empty_like(source_bottom)
    top_source_desc = TensorDescriptor.from_tensor(source_top, [half_m, half_n])
    bottom_source_desc = TensorDescriptor.from_tensor(source_bottom, [half_m, half_n])
    top_destination_desc = TensorDescriptor.from_tensor(output_top, [half_m, half_n])
    bottom_destination_desc = TensorDescriptor.from_tensor(output_bottom, [half_m, half_n])

    compiled = _named_reader_same_field_tme_fragments_kernel.warmup(
        top_source_desc,
        bottom_source_desc,
        top_destination_desc,
        bottom_destination_desc,
        STAGES=stages,
        ITERATIONS=iterations,
        grid=(1, ),
        num_warps=4,
        num_stages=1,
    )
    _assert_named_reader_same_field_tme_fragment_artifacts(compiled, stages, iterations)

    for _ in range(4):
        output_top.fill_(float("nan"))
        output_bottom.fill_(float("nan"))
        _named_reader_same_field_tme_fragments_kernel[(1, )](
            top_source_desc,
            bottom_source_desc,
            top_destination_desc,
            bottom_destination_desc,
            STAGES=stages,
            ITERATIONS=iterations,
            num_warps=4,
            num_stages=1,
        )
        torch.musa.synchronize()
        assert torch.equal(output_top, source_top)
        assert torch.equal(output_bottom, source_bottom)


def test_musa_non_ws_external_same_field_tme_fragments_runtime():
    stages = 2
    iterations = _phase_reuse_iterations(stages)
    half_m, half_n = 8, 32
    source_top = torch.arange(iterations * half_m * half_n, dtype=torch.float16,
                              device="musa").reshape(iterations * half_m, half_n)
    source_bottom = source_top + 10000
    output_top = torch.empty_like(source_top)
    output_bottom = torch.empty_like(source_bottom)
    top_source_desc = TensorDescriptor.from_tensor(source_top, [half_m, half_n])
    bottom_source_desc = TensorDescriptor.from_tensor(source_bottom, [half_m, half_n])
    top_destination_desc = TensorDescriptor.from_tensor(output_top, [half_m, half_n])
    bottom_destination_desc = TensorDescriptor.from_tensor(output_bottom, [half_m, half_n])

    compiled = _external_same_field_tme_fragments_kernel.warmup(
        top_source_desc,
        bottom_source_desc,
        top_destination_desc,
        bottom_destination_desc,
        STAGES=stages,
        ITERATIONS=iterations,
        grid=(1, ),
        num_warps=4,
        num_stages=1,
    )
    _assert_same_field_tme_fragment_artifacts(compiled, iterations, external=True)

    for _ in range(4):
        output_top.fill_(float("nan"))
        output_bottom.fill_(float("nan"))
        _external_same_field_tme_fragments_kernel[(1, )](
            top_source_desc,
            bottom_source_desc,
            top_destination_desc,
            bottom_destination_desc,
            STAGES=stages,
            ITERATIONS=iterations,
            num_warps=4,
            num_stages=1,
        )
        torch.musa.synchronize()
        assert torch.equal(output_top, source_top)
        assert torch.equal(output_bottom, source_bottom)


@pytest.mark.parametrize("stages", _PIPE_REUSE_STAGES, ids=[f"stage{stage}" for stage in _PIPE_REUSE_STAGES])
def test_musa_ws_same_field_tme_fragments_runtime(stages):
    iterations = _phase_reuse_iterations(stages)
    half_m, half_n = 8, 32
    source_top = torch.arange(iterations * half_m * half_n, dtype=torch.float16,
                              device="musa").reshape(iterations * half_m, half_n)
    source_bottom = source_top + 10000
    output_top = torch.empty_like(source_top)
    output_bottom = torch.empty_like(source_bottom)
    top_source_desc = TensorDescriptor.from_tensor(source_top, [half_m, half_n])
    bottom_source_desc = TensorDescriptor.from_tensor(source_bottom, [half_m, half_n])
    top_destination_desc = TensorDescriptor.from_tensor(output_top, [half_m, half_n])
    bottom_destination_desc = TensorDescriptor.from_tensor(output_bottom, [half_m, half_n])

    compiled = _ws_same_field_tme_fragments_kernel.warmup(
        top_source_desc,
        bottom_source_desc,
        top_destination_desc,
        bottom_destination_desc,
        STAGES=stages,
        ITERATIONS=iterations,
        grid=(1, ),
        num_warps=16,
        num_stages=1,
    )
    ttir = compiled.asm["ttir"]
    ttgir = compiled.asm["ttgir"]
    llir = compiled.asm["llir"]
    _assert_pipe_lowering_clean(compiled)
    assert compiled.metadata.num_warps == 20
    assert '"ttg.total-num-warps" = 20 : i32' in ttgir, ttgir
    assert 'reader_fields' not in ttir
    assert "musa.tme.issue_thread = 512 : i32" in ttgir, ttgir
    assert "musa.tme.issue_thread = 0 : i32" in ttgir, ttgir
    assert "llvm.musa.barrier0" not in llir, llir

    for _ in range(2):
        output_top.fill_(float("nan"))
        output_bottom.fill_(float("nan"))
        _ws_same_field_tme_fragments_kernel[(1, )](
            top_source_desc,
            bottom_source_desc,
            top_destination_desc,
            bottom_destination_desc,
            STAGES=stages,
            ITERATIONS=iterations,
            num_warps=16,
            num_stages=1,
        )
        torch.musa.synchronize()
        assert torch.equal(output_top, source_top)
        assert torch.equal(output_bottom, source_bottom)


def test_musa_non_ws_named_reader_close_only_runtime():
    stages = 1
    block = 128
    source = torch.zeros((block, ), dtype=torch.float16, device="musa")
    output = torch.empty((1, ), dtype=torch.int32, device="musa")
    source_desc = TensorDescriptor.from_tensor(source, [block])
    compiled = _named_reader_close_kernel.warmup(
        source_desc,
        output,
        STAGES=stages,
        BLOCK=block,
        ITERATIONS=0,
        grid=(1, ),
        num_warps=4,
        num_stages=1,
    )
    _assert_close_artifacts(
        compiled,
        stages,
        payloads=0,
        full_arrival_count=1,
        reader_warps=8,
        control_arrivals=1,
        warp_arrivals=1,
        wait_count=3,
    )
    for _ in range(4):
        output.fill_(-1)
        _named_reader_close_kernel[(1, )](
            source_desc,
            output,
            STAGES=stages,
            BLOCK=block,
            ITERATIONS=0,
            num_warps=4,
            num_stages=1,
        )
        torch.musa.synchronize()
        assert torch.equal(output, torch.ones_like(output))


@pytest.mark.parametrize("stages", [1, 2, 3], ids=["stage1", "stage2", "stage3"])
def test_musa_non_ws_named_reader_partial_tme_close_runtime(stages):
    payloads = 2 * stages + 1
    half_m, half_n = 16, 32
    float_m, float_n = 8, 16
    half_source = torch.arange(payloads * half_m * half_n, dtype=torch.float16,
                               device="musa").reshape(payloads * half_m, half_n)
    float_source = torch.arange(payloads * float_m * float_n, dtype=torch.float32,
                                device="musa").reshape(payloads * float_m, float_n)
    half_output = torch.empty_like(half_source)
    float_output = torch.empty_like(float_source)
    closed_flags = torch.empty((2 * (payloads + 1), ), dtype=torch.int32, device="musa")
    half_source_desc = TensorDescriptor.from_tensor(half_source, [half_m, half_n])
    float_source_desc = TensorDescriptor.from_tensor(float_source, [float_m, float_n])
    float_destination_desc = TensorDescriptor.from_tensor(float_output, [float_m, float_n])

    compiled = _named_reader_partial_tme_close_kernel.warmup(
        half_source_desc,
        float_source_desc,
        half_output,
        float_destination_desc,
        closed_flags,
        STAGES=stages,
        PAYLOADS=payloads,
        HALF_M=half_m,
        HALF_N=half_n,
        FLOAT_M=float_m,
        FLOAT_N=float_n,
        grid=(1, ),
        num_warps=4,
        num_stages=1,
    )
    _assert_close_artifacts(
        compiled,
        stages,
        payloads,
        full_arrival_count=1,
        reader_warps=8,
        control_arrivals=payloads + 1,
        warp_arrivals=2 * payloads + 1,
        payload_tme=True,
        reader_tme_stores=payloads,
        wait_count=3 * (payloads + 1),
    )

    for _ in range(4):
        half_output.fill_(float("nan"))
        float_output.fill_(float("nan"))
        closed_flags.fill_(-1)
        _named_reader_partial_tme_close_kernel[(1, )](
            half_source_desc,
            float_source_desc,
            half_output,
            float_destination_desc,
            closed_flags,
            STAGES=stages,
            PAYLOADS=payloads,
            HALF_M=half_m,
            HALF_N=half_n,
            FLOAT_M=float_m,
            FLOAT_N=float_n,
            num_warps=4,
            num_stages=1,
        )
        torch.musa.synchronize()
        assert torch.equal(half_output, half_source)
        assert torch.equal(float_output, float_source)
        assert torch.equal(closed_flags[:2 * payloads], torch.zeros_like(closed_flags[:2 * payloads]))
        assert torch.equal(closed_flags[2 * payloads:], torch.ones_like(closed_flags[2 * payloads:]))


def test_musa_non_ws_named_reader_mixed_close_runtime():
    stages = 2
    payloads = 5
    block = 128
    source = torch.arange(payloads * block, dtype=torch.float16, device="musa")
    tme_output = torch.empty_like(source)
    local_output = torch.empty((payloads * block, ), dtype=torch.int32, device="musa")
    closed_flags = torch.empty((2 * (payloads + 1), ), dtype=torch.int32, device="musa")
    source_desc = TensorDescriptor.from_tensor(source, [block])
    compiled = _named_reader_mixed_close_kernel.warmup(
        source_desc,
        tme_output,
        local_output,
        closed_flags,
        STAGES=stages,
        BLOCK=block,
        PAYLOADS=payloads,
        grid=(1, ),
        num_warps=4,
        num_stages=1,
    )
    _assert_close_artifacts(
        compiled,
        stages,
        payloads,
        full_arrival_count=5,
        reader_warps=8,
        control_arrivals=payloads + 1,
        warp_arrivals=3 * payloads + 2,
        payload_tme=True,
        wait_count=3 * (payloads + 1),
    )
    expected_local = torch.arange(payloads * block, dtype=torch.int32, device="musa")
    for _ in range(4):
        tme_output.fill_(float("nan"))
        local_output.fill_(-1)
        closed_flags.fill_(-1)
        _named_reader_mixed_close_kernel[(1, )](
            source_desc,
            tme_output,
            local_output,
            closed_flags,
            STAGES=stages,
            BLOCK=block,
            PAYLOADS=payloads,
            num_warps=4,
            num_stages=1,
        )
        torch.musa.synchronize()
        assert torch.equal(tme_output, source)
        assert torch.equal(local_output, expected_local)
        assert torch.equal(closed_flags[:2 * payloads], torch.zeros_like(closed_flags[:2 * payloads]))
        assert torch.equal(closed_flags[2 * payloads:], torch.ones_like(closed_flags[2 * payloads:]))


@pytest.mark.parametrize("stages", [1, 2, 3], ids=["stage1", "stage2", "stage3"])
def test_musa_non_ws_named_reader_one_shot_heterogeneous_runtime(stages):
    half_m, half_n = 16, 32
    float_m, float_n = 8, 16
    half_source = torch.arange(stages * half_m * half_n, dtype=torch.float16,
                               device="musa").reshape(stages * half_m, half_n)
    float_source = torch.arange(stages * float_m * float_n, dtype=torch.float32,
                                device="musa").reshape(stages * float_m, float_n)
    half_output = torch.empty_like(half_source)
    float_output = torch.empty_like(float_source)
    closed_flags = torch.empty((4 * stages, ), dtype=torch.int32, device="musa")
    half_source_desc = TensorDescriptor.from_tensor(half_source, [half_m, half_n])
    float_source_desc = TensorDescriptor.from_tensor(float_source, [float_m, float_n])
    float_destination_desc = TensorDescriptor.from_tensor(float_output, [float_m, float_n])

    compiled = _named_reader_one_shot_heterogeneous_kernel.warmup(
        half_source_desc,
        float_source_desc,
        half_output,
        float_destination_desc,
        closed_flags,
        STAGES=stages,
        HALF_M=half_m,
        HALF_N=half_n,
        FLOAT_M=float_m,
        FLOAT_N=float_n,
        grid=(1, ),
        num_warps=4,
        num_stages=1,
    )
    _assert_one_shot_artifacts(
        compiled,
        stages,
        publications=stages,
        full_arrival_count=1,
        reader_waits=4 * stages,
        tme_fields=2,
        reader_tme_stores=2 * stages,
        transaction_bytes=half_m * half_n * 2 + float_m * float_n * 4,
    )

    for _ in range(4):
        half_output.fill_(float("nan"))
        float_output.fill_(float("nan"))
        closed_flags.fill_(-1)
        _named_reader_one_shot_heterogeneous_kernel[(1, )](
            half_source_desc,
            float_source_desc,
            half_output,
            float_destination_desc,
            closed_flags,
            STAGES=stages,
            HALF_M=half_m,
            HALF_N=half_n,
            FLOAT_M=float_m,
            FLOAT_N=float_n,
            num_warps=4,
            num_stages=1,
        )
        torch.musa.synchronize()
        assert torch.equal(half_output, half_source)
        assert torch.equal(float_output, float_source)
        assert torch.equal(closed_flags, torch.zeros_like(closed_flags))


def test_musa_non_ws_pipe_reader_tme_store_roundtrip_runtime():
    stages = 2
    iterations = 5
    block = 128
    source = torch.arange(iterations * block, dtype=torch.float16, device="musa")
    destination = torch.empty_like(source)
    source_desc = TensorDescriptor.from_tensor(source, [block])
    destination_desc = TensorDescriptor.from_tensor(destination, [block])

    compiled = _pipe_tme_store_roundtrip_kernel.warmup(
        source_desc,
        destination_desc,
        STAGES=stages,
        BLOCK=block,
        ITERATIONS=iterations,
        grid=(1, ),
        num_warps=4,
        num_stages=1,
    )
    _assert_reader_tme_store_artifacts(
        compiled,
        stages,
        iterations,
        4,
        4,
        transaction_bytes=block * 2,
    )

    for _ in range(4):
        destination.fill_(float("nan"))
        _pipe_tme_store_roundtrip_kernel[(1, )](
            source_desc,
            destination_desc,
            STAGES=stages,
            BLOCK=block,
            ITERATIONS=iterations,
            num_warps=4,
            num_stages=1,
        )
        torch.musa.synchronize()
        assert torch.equal(destination, source)


@pytest.mark.parametrize("stages", _EXTERNAL_BARRIER_STAGES,
                         ids=[f"stage{stage}" for stage in _EXTERNAL_BARRIER_STAGES])
def test_musa_non_ws_pipe_external_full_roundtrip_runtime(stages):
    iterations = _phase_reuse_iterations(stages)
    block = 128
    source = torch.arange(iterations * block, dtype=torch.float16, device="musa")
    destination = torch.empty_like(source)
    source_desc = TensorDescriptor.from_tensor(source, [block])
    destination_desc = TensorDescriptor.from_tensor(destination, [block])

    compiled = _pipe_external_full_roundtrip_kernel.warmup(
        source_desc,
        destination_desc,
        STAGES=stages,
        BLOCK=block,
        ITERATIONS=iterations,
        grid=(1, ),
        num_warps=4,
        num_stages=1,
    )
    _assert_external_barrier_pipe_artifacts(compiled, stages, iterations, transaction_bytes=block * 2)

    for _ in range(4):
        destination.fill_(float("nan"))
        _pipe_external_full_roundtrip_kernel[(1, )](
            source_desc,
            destination_desc,
            STAGES=stages,
            BLOCK=block,
            ITERATIONS=iterations,
            num_warps=4,
            num_stages=1,
        )
        torch.musa.synchronize()
        assert torch.equal(destination, source)


@pytest.mark.parametrize("stages", _EXTERNAL_BARRIER_STAGES,
                         ids=[f"stage{stage}" for stage in _EXTERNAL_BARRIER_STAGES])
def test_musa_ws_pipe_external_full_roundtrip_runtime(stages):
    iterations = _phase_reuse_iterations(stages)
    block = 128
    source = torch.arange(iterations * block, dtype=torch.float16, device="musa")
    destination = torch.empty_like(source)
    source_desc = TensorDescriptor.from_tensor(source, [block])
    destination_desc = TensorDescriptor.from_tensor(destination, [block])

    compiled = _pipe_external_full_roundtrip_ws_kernel.warmup(
        source_desc,
        destination_desc,
        STAGES=stages,
        BLOCK=block,
        ITERATIONS=iterations,
        grid=(1, ),
        num_warps=16,
        num_stages=1,
    )
    assert compiled.metadata.num_warps == 20
    assert '"ttg.total-num-warps" = 20 : i32' in compiled.asm["ttgir"]
    _assert_external_barrier_pipe_artifacts(
        compiled,
        stages,
        iterations,
        static_ws=True,
        transaction_bytes=block * 2,
    )

    for _ in range(2):
        destination.fill_(float("nan"))
        _pipe_external_full_roundtrip_ws_kernel[(1, )](
            source_desc,
            destination_desc,
            STAGES=stages,
            BLOCK=block,
            ITERATIONS=iterations,
            num_warps=16,
            num_stages=1,
        )
        torch.musa.synchronize()
        assert torch.equal(destination, source)


@pytest.mark.parametrize("stages", [1, 2, 3], ids=["stage1", "stage2", "stage3"])
def test_musa_ws_heterogeneous_pipe_reader_tme_store_roundtrip_runtime(stages):
    iterations = 2 * stages + 1
    half_m, half_n = 16, 32
    float_m, float_n = 8, 16
    half_source = torch.arange(iterations * half_m * half_n, dtype=torch.float16,
                               device="musa").reshape(iterations * half_m, half_n)
    float_source = torch.arange(iterations * float_m * float_n, dtype=torch.float32,
                                device="musa").reshape(iterations * float_m, float_n)
    half_destination = torch.empty_like(half_source)
    float_destination = torch.empty_like(float_source)
    half_source_desc = TensorDescriptor.from_tensor(half_source, [half_m, half_n])
    float_source_desc = TensorDescriptor.from_tensor(float_source, [float_m, float_n])
    half_destination_desc = TensorDescriptor.from_tensor(half_destination, [half_m, half_n])
    float_destination_desc = TensorDescriptor.from_tensor(float_destination, [float_m, float_n])

    compiled = _ws_heterogeneous_tme_store_roundtrip_kernel.warmup(
        half_source_desc,
        float_source_desc,
        half_destination_desc,
        float_destination_desc,
        STAGES=stages,
        ITERATIONS=iterations,
        HALF_M=half_m,
        HALF_N=half_n,
        FLOAT_M=float_m,
        FLOAT_N=float_n,
        grid=(1, ),
        num_warps=16,
        num_stages=1,
    )
    assert compiled.metadata.num_warps == 20
    ttgir = compiled.asm["ttgir"]
    assert '"ttg.total-num-warps" = 20 : i32' in ttgir
    assert ttgir.count("ttg.warp_specialize") == 1, ttgir
    transaction_bytes = half_m * half_n * 2 + float_m * float_n * 4
    _assert_reader_tme_store_artifacts(
        compiled,
        stages,
        iterations,
        4,
        16,
        static_ws=True,
        tme_fields=2,
        transaction_bytes=transaction_bytes,
    )

    for _ in range(2):
        half_destination.fill_(float("nan"))
        float_destination.fill_(float("nan"))
        _ws_heterogeneous_tme_store_roundtrip_kernel[(1, )](
            half_source_desc,
            float_source_desc,
            half_destination_desc,
            float_destination_desc,
            STAGES=stages,
            ITERATIONS=iterations,
            HALF_M=half_m,
            HALF_N=half_n,
            FLOAT_M=float_m,
            FLOAT_N=float_n,
            num_warps=16,
            num_stages=1,
        )
        torch.musa.synchronize()
        assert torch.equal(half_destination, half_source)
        assert torch.equal(float_destination, float_source)


@pytest.mark.parametrize(
    "kernel",
    [_pipe_tme_store_source_mutation_kernel, _pipe_tme_store_source_mutation_after_kernel],
    ids=["before-store", "after-store"],
)
def test_musa_pipe_reader_tme_store_rejects_source_mutation(kernel, capfd):
    with pytest.raises(RuntimeError, match="PassManager::run failed"):
        _compile_invalid_pipeline(kernel)
    assert ("MUSA TLE pipe reader TME store source must not be modified after reader.wait" in capfd.readouterr().err)


def test_musa_non_ws_local_store_pipe_roundtrip_runtime():
    stages = 2
    iterations = 5
    block = 128
    out = torch.empty((iterations * block, ), dtype=torch.int32, device="musa")
    expected = torch.arange(iterations * block, dtype=torch.int32, device="musa")

    compiled = _non_ws_local_store_pipe_kernel.warmup(
        out,
        STAGES=stages,
        BLOCK=block,
        ITERATIONS=iterations,
        grid=(1, ),
        num_warps=4,
        num_stages=1,
    )
    assert compiled.metadata.num_warps == 4
    _assert_local_store_pipe_artifacts(compiled, stages, iterations, 4, 4)

    for _ in range(4):
        out.fill_(-1)
        _non_ws_local_store_pipe_kernel[(1, )](
            out,
            STAGES=stages,
            BLOCK=block,
            ITERATIONS=iterations,
            num_warps=4,
            num_stages=1,
        )
        torch.musa.synchronize()
        assert torch.equal(out, expected)


def test_musa_local_store_pipe_keeps_independent_shared_hazard_runtime():
    stages = 2
    iterations = 3
    block = 128
    out = torch.empty((iterations * block, ), dtype=torch.int32, device="musa")
    independent_out = torch.empty_like(out)
    expected = torch.arange(iterations * block, dtype=torch.int32, device="musa")

    compiled = _non_ws_local_store_pipe_with_independent_buffer_kernel.warmup(
        out,
        independent_out,
        STAGES=stages,
        BLOCK=block,
        ITERATIONS=iterations,
        grid=(1, ),
        num_warps=4,
        num_stages=1,
    )
    llir = compiled.asm["llir"]
    # The pipe waits suppress barriers only for the pipe allocation.  The
    # independent buffer still needs one RAW barrier per load and one WAR
    # barrier before every store after the first iteration.
    assert llir.count("call void @llvm.musa.syncthreads.lm()") == 2 * iterations, llir
    assert compiled.asm["ttgir"].count("ttmg.warp_arrive_barrier") == 2 * iterations

    out.fill_(-1)
    independent_out.fill_(-1)
    _non_ws_local_store_pipe_with_independent_buffer_kernel[(1, )](
        out,
        independent_out,
        STAGES=stages,
        BLOCK=block,
        ITERATIONS=iterations,
        num_warps=4,
        num_stages=1,
    )
    torch.musa.synchronize()
    assert torch.equal(out, expected)
    assert torch.equal(independent_out, expected + 1000)


@pytest.mark.parametrize("stages", [1, 2, 3], ids=["stage1", "stage2", "stage3"])
def test_musa_ws_local_store_pipe_roundtrip_runtime(stages):
    iterations = 2 * stages + 1
    block = 128
    out = torch.empty((iterations * block, ), dtype=torch.int32, device="musa")
    expected = torch.arange(iterations * block, dtype=torch.int32, device="musa")

    compiled = _ws_local_store_pipe_kernel.warmup(
        out,
        STAGES=stages,
        BLOCK=block,
        ITERATIONS=iterations,
        grid=(1, ),
        num_warps=16,
        num_stages=1,
    )
    assert compiled.metadata.num_warps == 20
    assert '"ttg.total-num-warps" = 20 : i32' in compiled.asm["ttgir"]
    _assert_local_store_pipe_artifacts(compiled, stages, iterations, 4, 16)

    for _ in range(2):
        out.fill_(-1)
        _ws_local_store_pipe_kernel[(1, )](
            out,
            STAGES=stages,
            BLOCK=block,
            ITERATIONS=iterations,
            num_warps=16,
            num_stages=1,
        )
        torch.musa.synchronize()
        assert torch.equal(out, expected)


@pytest.mark.parametrize("stages", [1, 2, 3], ids=["stage1", "stage2", "stage3"])
def test_musa_non_ws_async_copy_pipe_roundtrip_runtime(stages):
    iterations = _phase_reuse_iterations(stages)
    block = 128
    source = torch.arange(iterations * block, dtype=torch.float32, device="musa")
    out = torch.empty_like(source)

    compiled = _non_ws_async_copy_pipe_roundtrip_kernel.warmup(
        source,
        out,
        STAGES=stages,
        BLOCK=block,
        ITERATIONS=iterations,
        grid=(1, ),
        num_warps=16,
        num_stages=1,
    )
    _assert_async_copy_pipe_artifacts(compiled, iterations)

    for _ in range(4):
        out.fill_(float("nan"))
        _non_ws_async_copy_pipe_roundtrip_kernel[(1, )](
            source,
            out,
            STAGES=stages,
            BLOCK=block,
            ITERATIONS=iterations,
            num_warps=16,
            num_stages=1,
        )
        torch.musa.synchronize()
        assert torch.equal(out, source)


@pytest.mark.parametrize("stages", [1, 2, 3], ids=["stage1", "stage2", "stage3"])
def test_musa_ws_default_writer_async_copy_pipe_roundtrip_runtime(stages):
    iterations = _phase_reuse_iterations(stages)
    block = 128
    source = torch.arange(iterations * block, dtype=torch.float32, device="musa")
    out = torch.empty_like(source)

    compiled = _ws_default_writer_async_copy_pipe_kernel.warmup(
        source,
        out,
        STAGES=stages,
        BLOCK=block,
        ITERATIONS=iterations,
        grid=(1, ),
        num_warps=16,
        num_stages=1,
    )
    assert compiled.metadata.num_warps == 20
    assert '"ttg.total-num-warps" = 20 : i32' in compiled.asm["ttgir"]
    _assert_async_copy_pipe_artifacts(compiled, iterations, static_ws=True)

    for _ in range(4):
        out.fill_(float("nan"))
        _ws_default_writer_async_copy_pipe_kernel[(1, )](
            source,
            out,
            STAGES=stages,
            BLOCK=block,
            ITERATIONS=iterations,
            num_warps=16,
            num_stages=1,
        )
        torch.musa.synchronize()
        assert torch.equal(out, source)


def test_musa_one_shot_async_copy_pipe_runtime():
    stages = 3
    block = 128
    source = torch.arange(stages * block, dtype=torch.float32, device="musa")
    out = torch.empty_like(source)

    compiled = _one_shot_async_copy_pipe_kernel.warmup(
        source,
        out,
        STAGES=stages,
        BLOCK=block,
        grid=(1, ),
        num_warps=16,
        num_stages=1,
    )
    # One-shot pipes publish once per stage; the writer arrival is the only
    # warp arrive (no reader release).
    _assert_async_copy_pipe_artifacts(
        compiled,
        stages,
        expect_warp_arrives=1,
    )

    for _ in range(4):
        out.fill_(float("nan"))
        _one_shot_async_copy_pipe_kernel[(1, )](
            source,
            out,
            STAGES=stages,
            BLOCK=block,
            num_warps=16,
            num_stages=1,
        )
        torch.musa.synchronize()
        assert torch.equal(out, source)


def test_musa_non_ws_async_copy_mixed_local_store_pipe_runtime():
    stages = 2
    iterations = _phase_reuse_iterations(stages)
    block = 128
    source = torch.arange(iterations * block, dtype=torch.float32, device="musa")
    async_out = torch.empty_like(source)
    local_out = torch.empty_like(source)
    expected_local = source + 0.5

    compiled = _non_ws_async_copy_mixed_local_store_kernel.warmup(
        source,
        async_out,
        local_out,
        STAGES=stages,
        BLOCK=block,
        ITERATIONS=iterations,
        grid=(1, ),
        num_warps=16,
        num_stages=1,
    )
    _assert_async_copy_pipe_artifacts(compiled, iterations)

    for _ in range(4):
        async_out.fill_(float("nan"))
        local_out.fill_(float("nan"))
        _non_ws_async_copy_mixed_local_store_kernel[(1, )](
            source,
            async_out,
            local_out,
            STAGES=stages,
            BLOCK=block,
            ITERATIONS=iterations,
            num_warps=16,
            num_stages=1,
        )
        torch.musa.synchronize()
        assert torch.equal(async_out, source)
        assert torch.equal(local_out, expected_local)


def test_musa_non_ws_async_copy_mixed_tme_pipe_runtime():
    stages = 2
    iterations = _phase_reuse_iterations(stages)
    block = 128
    source = torch.arange(iterations * block, dtype=torch.float32, device="musa")
    async_out = torch.empty_like(source)
    tme_out = torch.empty_like(source)
    source_desc = TensorDescriptor.from_tensor(source, [block])

    compiled = _non_ws_async_copy_mixed_tme_kernel.warmup(
        source_desc,
        source,
        async_out,
        tme_out,
        STAGES=stages,
        BLOCK=block,
        ITERATIONS=iterations,
        grid=(1, ),
        num_warps=16,
        num_stages=1,
    )
    # Mixed async-copy + TME transport: one TME transaction barrier plus the
    # writer warp arrivals for the async-copy field.
    _assert_async_copy_pipe_artifacts(compiled, iterations, expect_tme_transactions=True)

    for _ in range(4):
        async_out.fill_(float("nan"))
        tme_out.fill_(float("nan"))
        _non_ws_async_copy_mixed_tme_kernel[(1, )](
            source_desc,
            source,
            async_out,
            tme_out,
            STAGES=stages,
            BLOCK=block,
            ITERATIONS=iterations,
            num_warps=16,
            num_stages=1,
        )
        torch.musa.synchronize()
        assert torch.equal(async_out, source)
        assert torch.equal(tme_out, source)


@pytest.mark.parametrize(
    "block_m,block_n,k_tiles,stages",
    [
        pytest.param(128, 128, 4, 1, id="m128-n128-k256-stage1"),
        pytest.param(128, 128, 4, 2, id="m128-n128-k256-stage2"),
        pytest.param(128, 128, 7, 3, id="m128-n128-k448-stage3"),
        pytest.param(256, 256, 1, 1, id="m256-n256-k64-stage1"),
    ],
)
def test_mthreads_non_ws_pipe_mm_runtime(block_m, block_n, k_tiles, stages):
    torch.manual_seed(42)
    block_k = 64
    k = k_tiles * block_k
    a = torch.randn((block_m, k), dtype=torch.float16, device="musa")
    b = torch.randn((k, block_n), dtype=torch.float16, device="musa")
    out = torch.empty((block_m, block_n), dtype=torch.float16, device="musa")
    a_desc = TensorDescriptor.from_tensor(a, [block_m, block_k])
    b_desc = TensorDescriptor.from_tensor(b, [block_k, block_n])
    reference = torch.matmul(a.to(torch.float32), b.to(torch.float32))

    for _ in range(2):
        out.fill_(float("nan"))
        _non_ws_pipe_mm_kernel[(1, )](
            a_desc,
            b_desc,
            out,
            K_TILES=k_tiles,
            STAGES=stages,
            BLOCK_M=block_m,
            BLOCK_N=block_n,
            BLOCK_K=block_k,
            num_warps=4,
            num_stages=1,
        )
        torch.musa.synchronize()
        torch.testing.assert_close(out.to(torch.float32), reference, rtol=1.25e-1, atol=1.25e-1)


@pytest.mark.parametrize("stages", [1, 2, 3], ids=["stage1", "stage2", "stage3"])
def test_mthreads_ws_pipe_mm_runtime(stages):
    torch.manual_seed(42)
    block_m = block_n = 256
    block_k = 64
    # Seven tiles exercise a full two-phase cycle and then reuse slot zero for
    # the third generation when capacity is three.
    k_tiles = 7
    k = k_tiles * block_k
    a = torch.randn((block_m, k), dtype=torch.float16, device="musa")
    b = torch.randn((k, block_n), dtype=torch.float16, device="musa")
    out = torch.empty((block_m, block_n), dtype=torch.float16, device="musa")
    a_desc = TensorDescriptor.from_tensor(a, [block_m, block_k])
    b_desc = TensorDescriptor.from_tensor(b, [block_k, block_n])
    reference = torch.matmul(a.to(torch.float32), b.to(torch.float32))

    compiled = _ws_pipe_mm_kernel.warmup(
        a_desc,
        b_desc,
        out,
        K_TILES=k_tiles,
        STAGES=stages,
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        BLOCK_K=block_k,
        grid=(1, ),
        num_warps=16,
        num_stages=1,
    )
    assert compiled.metadata.num_warps == 20
    assert compiled.metadata.shared == stages * 65536
    assert '"ttg.total-num-warps" = 20 : i32' in compiled.asm["ttgir"]
    assert compiled.asm["ttgir"].count("ttg.warp_specialize") == 1
    assert "musa_tle.static_ws." not in compiled.asm["ttgir"]
    assert "ttg.warp_specialize" not in compiled.asm["llir"]
    assert "ttg.convert_layout" not in compiled.asm["ttgir"]
    assert "swizzleGranularity = 1 : i32" in compiled.asm["ttgir"]
    assert "swizzleGranularity = 2 : i32" in compiled.asm["ttgir"]
    assert "builtin.unrealized_conversion_cast" not in compiled.asm["llir"]
    assert compiled.asm["llir"].count("call void @llvm.musa.syncthreads.lm()") == 1
    assert f"llvm.musa.async.bar.record(i32 {4 * stages})" in compiled.asm["llir"]

    for _ in range(2):
        out.fill_(float("nan"))
        _ws_pipe_mm_kernel[(1, )](
            a_desc,
            b_desc,
            out,
            K_TILES=k_tiles,
            STAGES=stages,
            BLOCK_M=block_m,
            BLOCK_N=block_n,
            BLOCK_K=block_k,
            num_warps=16,
            num_stages=1,
        )
        torch.musa.synchronize()
        torch.testing.assert_close(out.to(torch.float32), reference, rtol=1.25e-1, atol=1.25e-1)


@pytest.mark.parametrize("stages", [1, 2, 3], ids=["stage1", "stage2", "stage3"])
def test_musa_ws_multi_field_pipe_mm_runtime(stages):
    torch.manual_seed(42)
    block_m = block_n = 256
    block_k = 64
    k_tiles = 7
    k = k_tiles * block_k
    a = torch.randn((block_m, k), dtype=torch.float16, device="musa")
    b = torch.randn((k, block_n), dtype=torch.float16, device="musa")
    out = torch.empty((block_m, block_n), dtype=torch.float16, device="musa")
    a_desc = TensorDescriptor.from_tensor(a, [block_m, block_k])
    b_desc = TensorDescriptor.from_tensor(b, [block_k, block_n])
    reference = torch.matmul(a.to(torch.float32), b.to(torch.float32))

    compiled = _ws_multi_field_pipe_mm_kernel.warmup(
        a_desc,
        b_desc,
        out,
        K_TILES=k_tiles,
        STAGES=stages,
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        BLOCK_K=block_k,
        grid=(1, ),
        num_warps=16,
        num_stages=1,
    )
    assert compiled.metadata.num_warps == 20
    assert compiled.metadata.shared == stages * 65536
    assert '"ttg.total-num-warps" = 20 : i32' in compiled.asm["ttgir"]
    assert f"llvm.musa.async.bar.record(i32 {2 * stages})" in compiled.asm["llir"]
    _assert_pipe_lowering_clean(compiled)

    for _ in range(2):
        out.fill_(float("nan"))
        _ws_multi_field_pipe_mm_kernel[(1, )](
            a_desc,
            b_desc,
            out,
            K_TILES=k_tiles,
            STAGES=stages,
            BLOCK_M=block_m,
            BLOCK_N=block_n,
            BLOCK_K=block_k,
            num_warps=16,
            num_stages=1,
        )
        torch.musa.synchronize()
        torch.testing.assert_close(out.to(torch.float32), reference, rtol=1.25e-1, atol=1.25e-1)


def test_musa_heterogeneous_multi_field_pipe_roundtrip_runtime():
    torch.manual_seed(42)
    stages = 2
    iterations = 5
    half_m, half_n = 16, 32
    float_m, float_n = 8, 16
    half_src = torch.randn((iterations * half_m, half_n), dtype=torch.float16, device="musa")
    float_src = torch.randn((iterations * float_m, float_n), dtype=torch.float32, device="musa")
    half_dst = torch.empty_like(half_src)
    float_dst = torch.empty_like(float_src)
    half_src_desc = TensorDescriptor.from_tensor(half_src, [half_m, half_n])
    float_src_desc = TensorDescriptor.from_tensor(float_src, [float_m, float_n])
    half_dst_desc = TensorDescriptor.from_tensor(half_dst, [half_m, half_n])
    float_dst_desc = TensorDescriptor.from_tensor(float_dst, [float_m, float_n])

    compiled = _heterogeneous_multi_field_roundtrip_kernel.warmup(
        half_src_desc,
        float_src_desc,
        half_dst_desc,
        float_dst_desc,
        STAGES=stages,
        ITERATIONS=iterations,
        HALF_M=half_m,
        HALF_N=half_n,
        FLOAT_M=float_m,
        FLOAT_N=float_n,
        grid=(1, ),
        num_warps=4,
        num_stages=1,
    )
    ttgir = compiled.asm["ttgir"]
    llir = compiled.asm["llir"]
    transaction_bytes = half_m * half_n * 2 + float_m * float_n * 4
    _assert_pipe_lowering_clean(compiled)
    constants = _i32_constants(ttgir)
    add_records = re.findall(r"ttmg\.barrier_add_trans\s+(%[-\w.]+),\s*(%[-\w.]+)", ttgir)

    assert add_records, ttgir
    assert all(constants[byte_value] == transaction_bytes for _, byte_value in add_records), ttgir

    hardware_add_bytes = re.findall(r"llvm\.musa\.async\.add\.trans\(i32 \d+, i32 (\d+)\)", llir)
    assert hardware_add_bytes and all(value == str(transaction_bytes) for value in hardware_add_bytes), llir

    for _ in range(4):
        half_dst.fill_(float("nan"))
        float_dst.fill_(float("nan"))
        _heterogeneous_multi_field_roundtrip_kernel[(1, )](
            half_src_desc,
            float_src_desc,
            half_dst_desc,
            float_dst_desc,
            STAGES=stages,
            ITERATIONS=iterations,
            HALF_M=half_m,
            HALF_N=half_n,
            FLOAT_M=float_m,
            FLOAT_N=float_n,
            num_warps=4,
            num_stages=1,
        )
        torch.musa.synchronize()
        assert torch.equal(half_dst, half_src)
        assert torch.equal(float_dst, float_src)


def _assert_mixed_pipe_artifacts(compiled, stages, iterations, writer_warps, reader_warps, tme_sources_per_generation,
                                 transaction_bytes, ir_generations=None):
    ttgir = compiled.asm["ttgir"]
    llir = compiled.asm["llir"]
    _assert_pipe_lowering_clean(compiled)
    assert "ttmg.barrier_add_trans" in ttgir, ttgir
    assert "ttmg.async_tme_copy_global_to_local" in ttgir, ttgir
    assert "ttmg.async_tme_copy_local_to_global" not in ttgir or "ttmg.tme_store_read_wait" in ttgir, ttgir


def test_musa_non_ws_mixed_pipe_supports_multiple_local_fields_runtime():
    stages = 2
    iterations = 5
    m, n = 16, 32
    half_source = torch.arange(iterations * m * n, dtype=torch.float16, device="musa").reshape(iterations * m, n)
    float_source = torch.arange(iterations * m * n, dtype=torch.float32, device="musa").reshape(iterations * m,
                                                                                                n) + 0.25
    half_out = torch.empty_like(half_source)
    float_out = torch.empty_like(float_source)
    local_i32_out = torch.empty((iterations * m * n, ), dtype=torch.int32, device="musa")
    local_f32_out = torch.empty((iterations * m * n, ), dtype=torch.float32, device="musa")
    half_desc = TensorDescriptor.from_tensor(half_source, [m, n])
    float_desc = TensorDescriptor.from_tensor(float_source, [m, n])
    expected_i32 = torch.arange(iterations * m * n, dtype=torch.int32, device="musa")
    expected_f32 = expected_i32.to(torch.float32) + 0.5

    compiled = _mixed_multi_local_store_roundtrip_kernel.warmup(
        half_desc,
        float_desc,
        half_out,
        float_out,
        local_i32_out,
        local_f32_out,
        STAGES=stages,
        ITERATIONS=iterations,
        M=m,
        N=n,
        grid=(1, ),
        num_warps=4,
        num_stages=1,
    )
    transaction_bytes = m * n * (2 + 4)
    _assert_mixed_pipe_artifacts(compiled, stages, iterations, 4, 4, 2, transaction_bytes)

    for _ in range(4):
        half_out.fill_(float("nan"))
        float_out.fill_(float("nan"))
        local_i32_out.fill_(-1)
        local_f32_out.fill_(float("nan"))
        _mixed_multi_local_store_roundtrip_kernel[(1, )](
            half_desc,
            float_desc,
            half_out,
            float_out,
            local_i32_out,
            local_f32_out,
            STAGES=stages,
            ITERATIONS=iterations,
            M=m,
            N=n,
            num_warps=4,
            num_stages=1,
        )
        torch.musa.synchronize()
        assert torch.equal(half_out, half_source)
        assert torch.equal(float_out, float_source)
        assert torch.equal(local_i32_out, expected_i32)
        assert torch.equal(local_f32_out, expected_f32)


@pytest.mark.parametrize("stages", [1, 2, 3], ids=["stage1", "stage2", "stage3"])
def test_musa_ws_mixed_tme_local_store_pipe_roundtrip_runtime(stages):
    iterations = 2 * stages + 1
    m, n = 16, 32
    source = torch.arange(iterations * m * n, dtype=torch.float16, device="musa").reshape(iterations * m, n)
    tme_out = torch.empty_like(source)
    local_out = torch.empty((iterations * m * n, ), dtype=torch.int32, device="musa")
    source_desc = TensorDescriptor.from_tensor(source, [m, n])
    expected_local = torch.arange(iterations * m * n, dtype=torch.int32, device="musa")

    compiled = _ws_mixed_tme_local_store_roundtrip_kernel.warmup(
        source_desc,
        tme_out,
        local_out,
        STAGES=stages,
        ITERATIONS=iterations,
        M=m,
        N=n,
        grid=(1, ),
        num_warps=16,
        num_stages=1,
    )
    assert compiled.metadata.num_warps == 20
    assert '"ttg.total-num-warps" = 20 : i32' in compiled.asm["ttgir"]
    _assert_mixed_pipe_artifacts(compiled, stages, iterations, 4, 16, 1, m * n * 2, ir_generations=1)
    # The only CTA rendezvous is barrier initialization; payload publication
    # uses the full wait plus TME/group and local warp arrivals.
    assert compiled.asm["ttgir"].count("ttg.barrier ") == 1
    assert "llvm.musa.barrier0" not in compiled.asm["llir"]

    for _ in range(2):
        tme_out.fill_(float("nan"))
        local_out.fill_(-1)
        _ws_mixed_tme_local_store_roundtrip_kernel[(1, )](
            source_desc,
            tme_out,
            local_out,
            STAGES=stages,
            ITERATIONS=iterations,
            M=m,
            N=n,
            num_warps=16,
            num_stages=1,
        )
        torch.musa.synchronize()
        assert torch.equal(tme_out, source)
        assert torch.equal(local_out, expected_local)


def test_mthreads_pipe_bindings_are_optional_and_backend_local():
    assert hasattr(libtriton.ir.builder, "create_pipe_create")
    assert hasattr(libtriton.ir.builder, "create_pipe_writer_acquire")
    assert hasattr(libtriton.ir.builder, "create_pipe_writer_commit")
    assert hasattr(libtriton.ir.builder, "create_pipe_writer_close")
    assert hasattr(libtriton.ir.builder, "create_pipe_reader_wait")
    assert hasattr(libtriton.ir.builder, "create_pipe_reader_release")
    # Alias views are emitted through the MUSA-local builder binding.  The
    # community TLE builder remains untouched.
    assert hasattr(libtriton.ir.builder, "create_memdesc_alias")
    assert hasattr(libtriton.mthreads.passes.ttgpuir, "add_tle_lower_pipe")


def _assert_structured_pipe_artifacts(compiled):
    ttgir = compiled.asm["ttgir"]
    _assert_pipe_lowering_clean(compiled)
    assert "musa_tle.pipe_barrier_ring" not in ttgir, ttgir


@triton.jit
def _structured_if_pipe_kernel(
    desc,
    out_desc,
    flag_ptr,
    STAGES: tl.constexpr,
    ITERATIONS: tl.constexpr,
):
    """Exercise equivalent writer and reader paths across stage reuse."""
    smem = tle.gpu.alloc((STAGES, 128), dtype=tl.float16, nv_mma_shared_layout=False)
    pipe = tle.pipe(capacity=STAGES, scope="cta", name="structured_if", data=smem)
    writer = pipe.writer()
    reader = pipe.reader()
    flag = tl.load(flag_ptr)

    for iteration in tl.range(0, ITERATIONS, num_stages=1):
        slot = writer.acquire(iteration)
        if flag:
            tle.gpu.copy(desc, slot.data, (128, ), (iteration * 128, ))
            writer.commit(iteration)
        else:
            tle.gpu.copy(desc, slot.data, (128, ), (iteration * 128, ))
            writer.commit(iteration)

        wait = reader.wait(iteration)
        if flag:
            tle.gpu.copy(wait.slot.data, out_desc, (128, ), (iteration * 128, ))
        else:
            tle.gpu.copy(wait.slot.data, out_desc, (128, ), (iteration * 128, ))
        reader.release(iteration)


@pytest.mark.parametrize("stages", _PIPE_REUSE_STAGES, ids=[f"stage{stage}" for stage in _PIPE_REUSE_STAGES])
@pytest.mark.parametrize("flag_value", [0, 1], ids=["else", "then"])
def test_musa_pipe_structured_if_generation_runtime(stages, flag_value):
    size = 128
    iterations = _phase_reuse_iterations(stages)
    source = torch.arange(iterations * size, dtype=torch.float16, device="musa")
    output = torch.empty_like(source)
    flag = torch.full((1, ), flag_value, dtype=torch.int32, device="musa")
    source_desc = TensorDescriptor.from_tensor(source, [size])
    output_desc = TensorDescriptor.from_tensor(output, [size])

    compiled = _structured_if_pipe_kernel.warmup(
        source_desc,
        output_desc,
        flag,
        STAGES=stages,
        ITERATIONS=iterations,
        grid=(1, ),
        num_warps=4,
        num_stages=1,
    )
    _assert_structured_pipe_artifacts(compiled)

    for _ in range(4):
        output.fill_(float("nan"))
        _structured_if_pipe_kernel[(1, )](
            source_desc,
            output_desc,
            flag,
            STAGES=stages,
            ITERATIONS=iterations,
            num_warps=4,
            num_stages=1,
        )
        torch.musa.synchronize()
        assert torch.equal(output, source)


@triton.jit
def _structured_for_pipe_kernel(desc, out_desc, STAGES: tl.constexpr, ITERS: tl.constexpr):
    """Exercise loop-carried stage/phase values through stage reuse."""
    smem = tle.gpu.alloc((STAGES, 128), dtype=tl.float16, nv_mma_shared_layout=False)
    pipe = tle.pipe(capacity=STAGES, scope="cta", name="structured_for", data=smem)
    writer = pipe.writer()
    reader = pipe.reader()
    for iteration in tl.range(0, ITERS, num_stages=1):
        slot = writer.acquire(iteration)
        tle.gpu.copy(desc, slot.data, (128, ), (iteration * 128, ))
        writer.commit(iteration)
        wait = reader.wait(iteration)
        tle.gpu.copy(wait.slot.data, out_desc, (128, ), (iteration * 128, ))
        reader.release(iteration)


@pytest.mark.parametrize("stages", _PIPE_REUSE_STAGES, ids=[f"stage{stage}" for stage in _PIPE_REUSE_STAGES])
def test_musa_pipe_structured_for_stage_phase_runtime(stages):
    size = 128
    iterations = _phase_reuse_iterations(stages)
    source = torch.arange(iterations * size, dtype=torch.float16, device="musa")
    output = torch.empty_like(source)
    source_desc = TensorDescriptor.from_tensor(source, [size])
    output_desc = TensorDescriptor.from_tensor(output, [size])

    compiled = _structured_for_pipe_kernel.warmup(
        source_desc,
        output_desc,
        STAGES=stages,
        ITERS=iterations,
        grid=(1, ),
        num_warps=4,
        num_stages=1,
    )
    _assert_structured_pipe_artifacts(compiled)
    for _ in range(4):
        output.fill_(float("nan"))
        _structured_for_pipe_kernel[(1, )](
            source_desc,
            output_desc,
            STAGES=stages,
            ITERS=iterations,
            num_warps=4,
            num_stages=1,
        )
        torch.musa.synchronize()
        assert torch.equal(output, source)


# M7.4 resource-boundary and long-reuse coverage.  Keep these probes separate
# from the shorter M7.3 matrix so that a failure identifies long-lived phase or
# barrier-resource behavior rather than ordinary payload functionality.
def _extended_phase_reuse_iterations(stages):
    # Six complete phase transitions exercise substantially more reuse than
    # _phase_reuse_iterations (which covers two transitions).
    return 12 * stages + 1


@pytest.mark.parametrize("stages", _PIPE_REUSE_STAGES, ids=[f"stage{stage}" for stage in _PIPE_REUSE_STAGES])
def test_musa_non_ws_pipe_tme_store_survives_extended_phase_reuse_runtime(stages):
    block = 32
    iterations = _extended_phase_reuse_iterations(stages)
    source = torch.arange(iterations * block, dtype=torch.float16, device="musa")
    destination = torch.empty_like(source)
    source_desc = TensorDescriptor.from_tensor(source, [block])
    destination_desc = TensorDescriptor.from_tensor(destination, [block])

    compiled = _pipe_tme_store_roundtrip_kernel.warmup(
        source_desc,
        destination_desc,
        STAGES=stages,
        BLOCK=block,
        ITERATIONS=iterations,
        grid=(1, ),
        num_warps=4,
        num_stages=1,
    )
    _assert_reader_tme_store_artifacts(
        compiled,
        stages,
        iterations,
        4,
        4,
        transaction_bytes=block * 2,
    )

    for _ in range(3):
        destination.fill_(float("nan"))
        _pipe_tme_store_roundtrip_kernel[(1, )](
            source_desc,
            destination_desc,
            STAGES=stages,
            BLOCK=block,
            ITERATIONS=iterations,
            num_warps=4,
            num_stages=1,
        )
        torch.musa.synchronize()
        assert torch.equal(destination, source)


@pytest.mark.parametrize("stages", _PIPE_REUSE_STAGES, ids=[f"stage{stage}" for stage in _PIPE_REUSE_STAGES])
def test_musa_ws_pipe_tme_store_survives_extended_phase_reuse_runtime(stages):
    block = 32
    iterations = _extended_phase_reuse_iterations(stages)
    source = torch.arange(iterations * block, dtype=torch.float16, device="musa")
    destination = torch.empty_like(source)
    source_desc = TensorDescriptor.from_tensor(source, [block])
    destination_desc = TensorDescriptor.from_tensor(destination, [block])

    compiled = _pipe_tme_store_roundtrip_ws_kernel.warmup(
        source_desc,
        destination_desc,
        STAGES=stages,
        BLOCK=block,
        ITERATIONS=iterations,
        grid=(1, ),
        num_warps=16,
        num_stages=1,
    )
    assert compiled.metadata.num_warps == 20
    _assert_reader_tme_store_artifacts(
        compiled,
        stages,
        iterations,
        4,
        16,
        static_ws=True,
        transaction_bytes=block * 2,
    )

    for _ in range(3):
        destination.fill_(float("nan"))
        _pipe_tme_store_roundtrip_ws_kernel[(1, )](
            source_desc,
            destination_desc,
            STAGES=stages,
            BLOCK=block,
            ITERATIONS=iterations,
            num_warps=16,
            num_stages=1,
        )
        torch.musa.synchronize()
        assert torch.equal(destination, source)


@triton.jit
def _multiple_pipes_long_reuse_roundtrip_kernel(
    a_source_desc,
    b_source_desc,
    a_destination_desc,
    b_destination_desc,
    STAGES: tl.constexpr,
    BLOCK: tl.constexpr,
    ITERATIONS: tl.constexpr,
):
    a_smem = tle.gpu.alloc((STAGES, BLOCK), dtype=tl.float16, nv_mma_shared_layout=False)
    b_smem = tle.gpu.alloc((STAGES, BLOCK), dtype=tl.float16, nv_mma_shared_layout=False)
    a_pipe = tle.pipe(capacity=STAGES, scope="cta", name="multiple_long_reuse_a", data=a_smem)
    b_pipe = tle.pipe(capacity=STAGES, scope="cta", name="multiple_long_reuse_b", data=b_smem)
    a_writer = a_pipe.writer()
    b_writer = b_pipe.writer()
    a_reader = a_pipe.reader()
    b_reader = b_pipe.reader()
    for iteration in tl.range(0, ITERATIONS, num_stages=1):
        a_slot = a_writer.acquire(iteration)
        b_slot = b_writer.acquire(iteration)
        offset = iteration * BLOCK
        tle.gpu.copy(a_source_desc, a_slot.data, (BLOCK, ), (offset, ))
        tle.gpu.copy(b_source_desc, b_slot.data, (BLOCK, ), (offset, ))
        a_writer.commit(iteration)
        b_writer.commit(iteration)

        a_wait = a_reader.wait(iteration)
        b_wait = b_reader.wait(iteration)
        tle.gpu.copy(a_wait.slot.data, a_destination_desc, (BLOCK, ), (offset, ))
        tle.gpu.copy(b_wait.slot.data, b_destination_desc, (BLOCK, ), (offset, ))
        a_reader.release(iteration)
        b_reader.release(iteration)


@pytest.mark.parametrize("stages", [1, 3], ids=["stage1", "stage3"])
def test_musa_non_ws_multiple_pipes_survive_extended_phase_reuse_runtime(stages):
    block = 32
    iterations = _extended_phase_reuse_iterations(stages)
    a_source = torch.arange(iterations * block, dtype=torch.float16, device="musa")
    b_source = a_source + 10000
    a_destination = torch.empty_like(a_source)
    b_destination = torch.empty_like(b_source)
    a_source_desc = TensorDescriptor.from_tensor(a_source, [block])
    b_source_desc = TensorDescriptor.from_tensor(b_source, [block])
    a_destination_desc = TensorDescriptor.from_tensor(a_destination, [block])
    b_destination_desc = TensorDescriptor.from_tensor(b_destination, [block])

    compiled = _multiple_pipes_long_reuse_roundtrip_kernel.warmup(
        a_source_desc,
        b_source_desc,
        a_destination_desc,
        b_destination_desc,
        STAGES=stages,
        BLOCK=block,
        ITERATIONS=iterations,
        grid=(1, ),
        num_warps=4,
        num_stages=1,
    )
    ttgir = compiled.asm["ttgir"]
    _assert_pipe_lowering_clean(compiled)
    # The loop is represented once in TTGIR; there is one ingress completion
    # source per pipe, independent of the runtime iteration count.
    assert ttgir.count("ttmg.barrier_add_trans") == 2
    assert "musa_tle.pipe_barrier_ring" not in ttgir

    for _ in range(3):
        a_destination.fill_(float("nan"))
        b_destination.fill_(float("nan"))
        _multiple_pipes_long_reuse_roundtrip_kernel[(1, )](
            a_source_desc,
            b_source_desc,
            a_destination_desc,
            b_destination_desc,
            STAGES=stages,
            BLOCK=block,
            ITERATIONS=iterations,
            num_warps=4,
            num_stages=1,
        )
        torch.musa.synchronize()
        assert torch.equal(a_destination, a_source)
        assert torch.equal(b_destination, b_source)


@pytest.mark.parametrize("stages", [31], ids=["stage31-max-ring"])
def test_musa_pipe_capacity31_barrier_ring_runs_at_hardware_limit(stages):
    # _structured_for_pipe_kernel intentionally uses a fixed 128-element
    # descriptor block; keep the payload small enough for a 31-stage ring while
    # matching that frontend contract.
    block = 128
    iterations = 7 * stages + 1
    source = torch.arange(iterations * block, dtype=torch.float16, device="musa")
    destination = torch.empty_like(source)
    source_desc = TensorDescriptor.from_tensor(source, [block])
    destination_desc = TensorDescriptor.from_tensor(destination, [block])

    compiled = _structured_for_pipe_kernel.warmup(
        source_desc,
        destination_desc,
        STAGES=stages,
        ITERS=iterations,
        grid=(1, ),
        num_warps=4,
        num_stages=1,
    )
    ttgir = compiled.asm["ttgir"]
    _assert_structured_pipe_artifacts(compiled)
    assert "musa.max_bar_id = 62" in ttgir, ttgir

    for _ in range(2):
        destination.fill_(float("nan"))
        _structured_for_pipe_kernel[(1, )](
            source_desc,
            destination_desc,
            STAGES=stages,
            ITERS=iterations,
            num_warps=4,
            num_stages=1,
        )
        torch.musa.synchronize()
        assert torch.equal(destination, source)


@triton.jit
def _capacity32_pipe_barrier_overflow_kernel(
    desc,
    out,
    STAGES: tl.constexpr,
    BLOCK: tl.constexpr,
    ITERATIONS: tl.constexpr,
):
    # Keep the frontend signature compatible with _compile_invalid_pipeline;
    # the literal capacity intentionally requests 64 pipe barriers.
    smem = tle.gpu.alloc((32, BLOCK), dtype=tl.int32, nv_mma_shared_layout=False)
    pipe = tle.pipe(capacity=32, scope="cta", name="capacity32_overflow", data=smem)
    writer = pipe.writer()
    reader = pipe.reader()
    # _compile_invalid_pipeline uses a 16-warp compiler option; leave the
    # range in the native blocked layout instead of importing the 4-warp
    # writer layout used by runtime kernels.
    offsets = tl.arange(0, BLOCK)
    slot = writer.acquire(0)
    tl.store(tle.gpu.local_ptr(slot.data, (offsets, )), offsets.to(tl.int32))
    writer.commit(0)
    wait = reader.wait(0)
    values = tl.load(tle.gpu.local_ptr(wait.slot.data, (offsets, )))
    tl.store(out + offsets, values)
    reader.release(0)


def test_musa_pipe_barrier_overflow_has_stable_diagnostic(capfd):
    with pytest.raises(RuntimeError, match="PassManager::run failed"):
        _compile_invalid_pipeline(_capacity32_pipe_barrier_overflow_kernel)
    stderr = capfd.readouterr().err
    assert "MUSA TLE pipe barrier allocation exceeds hardware barrier id limit" in stderr


@triton.jit
def _local_store_fp16_producer_for_slow_reader(writer, BLOCK: tl.constexpr, ITERATIONS: tl.constexpr):
    offsets = tle.gpu.set_layout(tl.arange(0, BLOCK), _LOCAL_STORE_WRITER_LAYOUT)
    for iteration in tl.static_range(0, ITERATIONS):
        slot = writer.acquire(iteration)
        values = (iteration * BLOCK + offsets).to(tl.float16)
        tl.store(tle.gpu.local_ptr(slot.data, (offsets, )), values)
        writer.commit(iteration)


@triton.jit
def _double_tme_store_named_reader(reader, destination_a_desc, destination_b_desc, flags, BLOCK: tl.constexpr,
                                   ITERATIONS: tl.constexpr):
    for iteration in tl.static_range(0, ITERATIONS):
        wait = reader.wait(iteration)
        offset = iteration * BLOCK
        tle.gpu.copy(wait.slot.data, destination_a_desc, (BLOCK, ), (offset, ))
        tle.gpu.copy(wait.slot.data, destination_b_desc, (BLOCK, ), (offset, ))
        tl.store(flags + iteration, tl.where(wait.is_closed, 1, 0))
        reader.release(iteration)


@triton.jit
def _named_reader_slowest_consumer_kernel(
    fast_output,
    slow_destination_a_desc,
    slow_destination_b_desc,
    flags,
    STAGES: tl.constexpr,
    BLOCK: tl.constexpr,
    ITERATIONS: tl.constexpr,
):
    smem = tle.gpu.alloc((STAGES, BLOCK), dtype=tl.float16, nv_mma_shared_layout=False)
    pipe = tle.pipe(
        capacity=STAGES,
        scope="cta",
        name="slowest_named_reader",
        readers=("fast", "slow"),
        data=smem,
    )
    tle.gpu.warp_specialize(
        [
            (
                _local_store_fp16_producer_for_slow_reader,
                (pipe.writer(), BLOCK, ITERATIONS),
            ),
            (
                _whole_field_local_store_consumer,
                (pipe.reader("fast"), fast_output, BLOCK, ITERATIONS),
            ),
            (
                _double_tme_store_named_reader,
                (pipe.reader("slow"), slow_destination_a_desc, slow_destination_b_desc, flags, BLOCK, ITERATIONS),
            ),
        ],
        worker_num_warps=[4, 4],
        worker_num_regs=[24, 24],
    )


def test_musa_named_reader_slowest_consumer_controls_stage_reuse_runtime():
    stages = 2
    block = 32
    iterations = _extended_phase_reuse_iterations(stages)
    fast_output = torch.empty((iterations * block, ), dtype=torch.float16, device="musa")
    slow_destination_a = torch.empty((iterations * block, ), dtype=torch.float16, device="musa")
    slow_destination_b = torch.empty_like(slow_destination_a)
    flags = torch.empty((iterations, ), dtype=torch.int32, device="musa")
    expected = torch.arange(iterations * block, dtype=torch.float16, device="musa")
    slow_destination_a_desc = TensorDescriptor.from_tensor(slow_destination_a, [block])
    slow_destination_b_desc = TensorDescriptor.from_tensor(slow_destination_b, [block])

    compiled = _named_reader_slowest_consumer_kernel.warmup(
        fast_output,
        slow_destination_a_desc,
        slow_destination_b_desc,
        flags,
        STAGES=stages,
        BLOCK=block,
        ITERATIONS=iterations,
        grid=(1, ),
        num_warps=4,
        num_stages=1,
    )
    assert compiled.metadata.num_warps == 12
    assert 'readers = ["fast", "slow"]' in compiled.asm["ttir"]
    assert "musa_tle.pipe_barrier_ring" not in compiled.asm["ttgir"]

    for _ in range(2):
        fast_output.fill_(float("nan"))
        slow_destination_a.fill_(float("nan"))
        slow_destination_b.fill_(float("nan"))
        flags.fill_(-1)
        _named_reader_slowest_consumer_kernel[(1, )](
            fast_output,
            slow_destination_a_desc,
            slow_destination_b_desc,
            flags,
            STAGES=stages,
            BLOCK=block,
            ITERATIONS=iterations,
            num_warps=4,
            num_stages=1,
        )
        torch.musa.synchronize()
        assert torch.equal(fast_output, expected)
        assert torch.equal(slow_destination_a, expected)
        assert torch.equal(slow_destination_b, expected)
        assert torch.equal(flags, torch.zeros_like(flags))


# Negative structured-CFG coverage. These kernels intentionally stop at
# LowerPipe diagnostics; none is launched because an incomplete lifecycle can
# deadlock a real device.
@triton.jit
def _m64_invalid_writer_missing_commit_kernel(
    desc,
    out,
    STAGES: tl.constexpr,
    BLOCK: tl.constexpr,
    ITERATIONS: tl.constexpr,
):
    smem = tle.gpu.alloc((1, BLOCK), dtype=tl.float16, nv_mma_shared_layout=False)
    pipe = tle.pipe(capacity=1, scope="cta", name="m64_writer_missing_commit", data=smem)
    writer = pipe.writer()
    reader = pipe.reader()
    flag = tl.load(out)
    slot = writer.acquire(0)
    if flag:
        tle.gpu.copy(desc, slot.data, (BLOCK, ), (0, ))
        writer.commit(0)
    else:
        tle.gpu.copy(desc, slot.data, (BLOCK, ), (0, ))
    wait = reader.wait(0)
    reader.release(0)


@triton.jit
def _m64_invalid_writer_stage_merge_kernel(
    desc,
    out,
    STAGES: tl.constexpr,
    BLOCK: tl.constexpr,
    ITERATIONS: tl.constexpr,
):
    smem = tle.gpu.alloc((2, BLOCK), dtype=tl.float16, nv_mma_shared_layout=False)
    pipe = tle.pipe(capacity=2, scope="cta", name="m64_writer_stage_merge", data=smem)
    writer = pipe.writer()
    reader = pipe.reader()
    slot = writer.acquire(0)
    tle.gpu.copy(desc, slot.data, (BLOCK, ), (0, ))
    writer.commit(0)
    flag = tl.load(out)
    if flag:
        reader.wait(0)
        reader.release(0)
    else:
        reader.wait(1)
        reader.release(1)


@triton.jit
def _m64_invalid_reader_missing_release_kernel(
    desc,
    out,
    STAGES: tl.constexpr,
    BLOCK: tl.constexpr,
    ITERATIONS: tl.constexpr,
):
    smem = tle.gpu.alloc((1, BLOCK), dtype=tl.float16, nv_mma_shared_layout=False)
    pipe = tle.pipe(capacity=1, scope="cta", name="m64_reader_missing_release", data=smem)
    writer = pipe.writer()
    reader = pipe.reader()
    slot = writer.acquire(0)
    tle.gpu.copy(desc, slot.data, (BLOCK, ), (0, ))
    writer.commit(0)
    wait = reader.wait(0)
    flag = tl.load(out)
    if flag:
        reader.release(0)


@triton.jit
def _m64_invalid_reader_drain_after_release_kernel(
    desc,
    out,
    STAGES: tl.constexpr,
    BLOCK: tl.constexpr,
    ITERATIONS: tl.constexpr,
):
    smem = tle.gpu.alloc((1, BLOCK), dtype=tl.float16, nv_mma_shared_layout=False)
    pipe = tle.pipe(capacity=1, scope="cta", name="m64_reader_drain_after_release", data=smem)
    writer = pipe.writer()
    reader = pipe.reader()
    slot = writer.acquire(0)
    tle.gpu.copy(desc, slot.data, (BLOCK, ), (0, ))
    writer.commit(0)
    wait = reader.wait(0)
    reader.release(0)
    tle.gpu.copy(wait.slot.data, desc, (BLOCK, ), (0, ))


@triton.jit
def _m64_invalid_writer_no_else_kernel(
    desc,
    out,
    STAGES: tl.constexpr,
    BLOCK: tl.constexpr,
    ITERATIONS: tl.constexpr,
):
    smem = tle.gpu.alloc((1, BLOCK), dtype=tl.float16, nv_mma_shared_layout=False)
    pipe = tle.pipe(capacity=1, scope="cta", name="m64_writer_no_else", data=smem)
    writer = pipe.writer()
    reader = pipe.reader()
    slot = writer.acquire(0)
    flag = tl.load(out)
    if flag:
        tle.gpu.copy(desc, slot.data, (BLOCK, ), (0, ))
        writer.commit(0)
    wait = reader.wait(0)
    reader.release(0)


@triton.jit
def _m64_invalid_reader_no_else_kernel(
    desc,
    out,
    STAGES: tl.constexpr,
    BLOCK: tl.constexpr,
    ITERATIONS: tl.constexpr,
):
    smem = tle.gpu.alloc((1, BLOCK), dtype=tl.float16, nv_mma_shared_layout=False)
    pipe = tle.pipe(capacity=1, scope="cta", name="m64_reader_no_else", data=smem)
    writer = pipe.writer()
    reader = pipe.reader()
    slot = writer.acquire(0)
    tle.gpu.copy(desc, slot.data, (BLOCK, ), (0, ))
    writer.commit(0)
    flag = tl.load(out)
    if flag:
        reader.wait(0)
        reader.release(0)


@pytest.mark.parametrize(
    "kernel,diagnostic",
    [
        (
            _m64_invalid_writer_missing_commit_kernel,
            "MUSA TLE pipe writer generation must commit on every reachable path",
        ),
        (
            _m64_invalid_writer_stage_merge_kernel,
            "MUSA TLE pipe lifecycle stage and phase are not equivalent at control-flow merge",
        ),
        (
            _m64_invalid_reader_missing_release_kernel,
            "MUSA TLE pipe reader release must post-dominate the wait on all normal paths",
        ),
        (
            _m64_invalid_reader_drain_after_release_kernel,
            "MUSA TLE pipe reader TME store must complete before every release or lifecycle exit",
        ),
        (
            _m64_invalid_writer_no_else_kernel,
            "MUSA TLE pipe writer generation must commit on every reachable path",
        ),
        (
            _m64_invalid_reader_no_else_kernel,
            "MUSA TLE pipe reader lifecycle generation is not path complete",
        ),
    ],
)
def test_musa_pipe_structured_cfg_rejects_incomplete_lifecycle(capfd, kernel, diagnostic):
    with pytest.raises(RuntimeError, match="PassManager::run failed"):
        _compile_invalid_pipeline(kernel)
    assert diagnostic in capfd.readouterr().err
