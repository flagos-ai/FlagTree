# flagtree tle
"""Runtime coverage for fieldless one-shot control handoffs."""

import pytest
import torch
import triton
import triton.language as tl
import triton.experimental.tle.language as tle


def _has_nvidia_hopper_gpu() -> bool:
    target = triton.runtime.driver.active.get_current_target()
    return (
        target.backend == "cuda"
        and torch.cuda.is_available()
        and torch.cuda.get_device_capability()[0] >= 9
    )


pytestmark = pytest.mark.skipif(
    not _has_nvidia_hopper_gpu(),
    reason="warp-specialized fieldless handoffs require NVIDIA Hopper (sm90+)",
)


_MESH_2D_8X16_VALUE = tle.device_mesh(
    {"block": [("block_y", 8), ("block_x", 16)]}
)
_MESH_2D_8X16 = tl.constexpr(_MESH_2D_8X16_VALUE)


@triton.jit(noinline=True)
def _fieldless_handoff_consumer(first_writer, second_writer, output):
    tl.store(output, 1)
    first_writer.commit(0)
    tl.store(output + 1, 2)
    second_writer.commit(0)


@triton.jit(noinline=True)
def _fieldless_handoff_producer(first_reader, second_reader, output):
    first_reader.wait(0)
    tl.store(output + 2, 3)
    second_reader.wait(0)
    tl.store(output + 3, 4)


@triton.jit
def _fieldless_handoff_kernel(output):
    first = tle.pipe(
        capacity=1,
        scope="cta",
        name="first_handoff",
        one_shot=True,
    )
    second = tle.pipe(
        capacity=1,
        scope="cta",
        name="second_handoff",
        one_shot=True,
    )
    tle.gpu.warp_specialize(
        [
            (
                _fieldless_handoff_consumer,
                (first.writer(), second.writer(), output),
            ),
            (
                _fieldless_handoff_producer,
                (first.reader(), second.reader(), output),
            ),
        ],
        [1],
        [32],
    )


@triton.jit(noinline=True)
def _grid_handoff_consumer(handoff_writer, output):
    pid = tl.program_id(0)
    tl.store(output + pid, 1)
    tle.distributed_barrier(_MESH_2D_8X16)
    handoff_writer.commit(0)
    tle.distributed_barrier(_MESH_2D_8X16)
    tl.store(output + 128 + pid, 2)


@triton.jit(noinline=True)
def _grid_handoff_producer(handoff_reader, output):
    pid = tl.program_id(0)
    handoff_reader.wait(0)
    tl.store(output + 256 + pid, 3)


@triton.jit
def _grid_handoff_kernel(output):
    handoff = tle.pipe(
        capacity=1,
        scope="cta",
        name="grid_handoff",
        one_shot=True,
    )
    tle.gpu.warp_specialize(
        [
            (_grid_handoff_consumer, (handoff.writer(), output)),
            (_grid_handoff_producer, (handoff.reader(), output)),
        ],
        [1],
        [32],
    )


@triton.jit(noinline=True)
def _fieldless_cyclic_consumer(handoff_writer, output):
    for iteration in tl.range(0, 4, loop_unroll_factor=1):
        tl.store(output + iteration, iteration + 1)
        handoff_writer.acquire(iteration)
        handoff_writer.commit(iteration)


@triton.jit(noinline=True)
def _fieldless_cyclic_producer(handoff_reader, output):
    for iteration in tl.range(0, 4, loop_unroll_factor=1):
        handoff_reader.wait(iteration)
        tl.store(output + 4 + iteration, iteration + 5)
        handoff_reader.release(iteration)


@triton.jit
def _fieldless_cyclic_kernel(output):
    handoff = tle.pipe(
        capacity=1,
        scope="cta",
        name="cyclic_handoff",
    )
    tle.gpu.warp_specialize(
        [
            (_fieldless_cyclic_consumer, (handoff.writer(), output)),
            (_fieldless_cyclic_producer, (handoff.reader(), output)),
        ],
        [1],
        [32],
    )


@triton.jit(noinline=True)
def _named_rendezvous_consumer(rendezvous, output):
    for iteration in tl.range(0, 4, loop_unroll_factor=1):
        tl.store(output + iteration, iteration + 1)
        tle.gpu.barrier_wait(rendezvous)


@triton.jit(noinline=True)
def _named_rendezvous_producer(rendezvous, output):
    for iteration in tl.range(0, 4, loop_unroll_factor=1):
        tle.gpu.barrier_wait(rendezvous)
        tl.store(output + 4 + iteration, iteration + 5)


@triton.jit
def _named_rendezvous_kernel(output):
    rendezvous = tle.gpu.alloc_barrier(arrive_count=9 * 32)
    tle.gpu.warp_specialize(
        [
            (_named_rendezvous_consumer, (rendezvous, output)),
            (_named_rendezvous_producer, (rendezvous, output)),
        ],
        [1],
        [32],
    )


@pytest.mark.require_tle("pipe", "gpu.warp_specialize")
def test_two_fieldless_one_shot_handoffs_execute_in_order():
    output = torch.zeros(4, dtype=torch.int32, device="cuda")
    _fieldless_handoff_kernel[(1,)](output, num_warps=8)
    torch.cuda.synchronize()
    torch.testing.assert_close(
        output.cpu(),
        torch.tensor([1, 2, 3, 4], dtype=torch.int32),
    )


@pytest.mark.require_tle("pipe", "gpu.warp_specialize")
def test_fieldless_cyclic_handoff_reuses_token_without_payload_storage(with_allocator):
    output = torch.zeros(8, dtype=torch.int32, device="cuda")
    _fieldless_cyclic_kernel.warmup(output, grid=(1,), num_warps=8)

    _fieldless_cyclic_kernel[(1,)](output, num_warps=8)
    torch.cuda.synchronize()

    torch.testing.assert_close(
        output.cpu(),
        torch.arange(1, 9, dtype=torch.int32),
    )


@pytest.mark.require_tle("gpu.alloc_barrier", "gpu.barrier_wait", "gpu.warp_specialize")
def test_named_barrier_rendezvous_is_reusable_across_warp_groups(with_allocator):
    output = torch.zeros(8, dtype=torch.int32, device="cuda")
    compiled = _named_rendezvous_kernel.warmup(output, grid=(1,), num_warps=8)

    _named_rendezvous_kernel[(1,)](output, num_warps=8)
    torch.cuda.synchronize()

    assert compiled.n_spills == 0
    torch.testing.assert_close(
        output.cpu(),
        torch.arange(1, 9, dtype=torch.int32),
    )


@pytest.mark.require_tle("distributed_barrier", "pipe", "gpu.warp_specialize")
def test_2d_grid_barriers_compose_with_fieldless_handoff(with_allocator):
    output = torch.zeros(128 * 3, dtype=torch.int32, device="cuda")
    compiled = _grid_handoff_kernel.warmup(output, grid=(128,), num_warps=8)
    assert compiled.metadata.launch_cooperative_grid is True
    # Static barrier call sites must own distinct phase counters.  Sharing a
    # counter permits a fast CTA from the next call site to flip the phase
    # before a slow CTA has observed the previous phase and eventually
    # deadlocks repeated CUDA-graph replays.
    assert compiled.metadata.global_scratch_size == 8
    _grid_handoff_kernel[(128,)](output, num_warps=8)
    torch.cuda.synchronize()
    torch.testing.assert_close(
        output.cpu(),
        torch.cat(
            [
                torch.ones(128, dtype=torch.int32),
                torch.full((128,), 2, dtype=torch.int32),
                torch.full((128,), 3, dtype=torch.int32),
            ]
        ),
    )
