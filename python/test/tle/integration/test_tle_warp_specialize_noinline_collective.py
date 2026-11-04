# flagtree tle
"""Runtime coverage for collectives in a noinline warp-group call graph."""

import pytest
import torch
import triton
import triton.language as tl
import triton.experimental.tle.language as tle


_MESH_VALUE = tle.device_mesh(
    {"block": [("block_y", 8), ("block_x", 16)]}
)
_MESH = tl.constexpr(_MESH_VALUE)


def _has_nvidia_hopper_gpu() -> bool:
    target = triton.runtime.driver.active.get_current_target()
    return (
        target.backend == "cuda"
        and torch.cuda.is_available()
        and torch.cuda.get_device_capability()[0] >= 9
    )


pytestmark = pytest.mark.skipif(
    not _has_nvidia_hopper_gpu(),
    reason="warp-specialized collectives require NVIDIA Hopper (sm90+)",
)


@triton.jit(noinline=True)
def _noinline_partial(source, partials, shard_y, shard_x):
    offsets = tl.arange(0, 128)
    if shard_x == 0:
        total = 0.0
        for start in tl.static_range(0, 2):
            values = tl.load(source + shard_y * 128 + start * 1024 + offsets)
            total += tl.sum(values, axis=0)
        tl.store(partials + shard_y, total)


@triton.jit(noinline=True)
def _noinline_collective_consumer(source, partials, output):
    shard_y = tle.shard_id(_MESH, "block_y")
    shard_x = tle.shard_id(_MESH, "block_x")
    pid = tl.program_id(0)
    _noinline_partial(source, partials, shard_y, shard_x)
    if shard_x == 0:
        tl.store(output + pid, tl.load(partials + shard_y))


@triton.jit(noinline=True)
def _noinline_collective_worker():
    pass


@triton.jit
def _noinline_collective_kernel(source, partials, output):
    tle.gpu.warp_specialize(
        [
            (_noinline_collective_consumer, (source, partials, output)),
            (_noinline_collective_worker, ()),
        ],
        [1],
        [32],
    )


@pytest.mark.require_tle("gpu.warp_specialize")
def test_noinline_reduction_inherits_default_warp_group_barrier_scope(with_allocator):
    source = torch.arange(2048, dtype=torch.float32, device="cuda")
    partials = torch.zeros(8, dtype=torch.float32, device="cuda")
    output = torch.zeros(128, dtype=torch.float32, device="cuda")

    compiled = _noinline_collective_kernel.warmup(
        source,
        partials,
        output,
        grid=(128,),
        num_warps=8,
    )
    _noinline_collective_kernel[(128,)](
        source,
        partials,
        output,
        num_warps=8,
    )
    torch.cuda.synchronize()

    expected = torch.zeros(128, dtype=torch.float32)
    source_cpu = source.cpu()
    for shard_y in range(8):
        expected[shard_y * 16] = (
            source_cpu[shard_y * 128 : (shard_y + 1) * 128].sum()
            + source_cpu[1024 + shard_y * 128 : 1024 + (shard_y + 1) * 128].sum()
        )
    torch.testing.assert_close(output.cpu(), expected)
