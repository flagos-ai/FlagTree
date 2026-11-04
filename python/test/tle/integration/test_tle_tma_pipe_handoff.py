# flagtree tle
"""Runtime composition test for a large TMA pipe and a control handoff."""

import pytest
import torch
import triton
import triton.language as tl
import triton.experimental.tle.language as tle


_MESH_VALUE = tle.device_mesh(
    {"block": [("block_y", 8), ("block_x", 16)]}
)
_MESH = tl.constexpr(_MESH_VALUE)
_WEIGHT_LAYOUT = tl.constexpr(
    tle.gpu.BlockEncoding([1, 4], [4, 8], [8, 1], [1, 0])
)
_SOURCE_LAYOUT = tl.constexpr(
    tle.gpu.SlicedEncoding(0, _WEIGHT_LAYOUT.value)
)
_OUTPUT_LAYOUT = tl.constexpr(
    tle.gpu.SlicedEncoding(1, _WEIGHT_LAYOUT.value)
)


def _has_nvidia_hopper_gpu() -> bool:
    target = triton.runtime.driver.active.get_current_target()
    return (
        target.backend == "cuda"
        and torch.cuda.is_available()
        and torch.cuda.get_device_capability()[0] >= 9
    )


pytestmark = pytest.mark.skipif(
    not _has_nvidia_hopper_gpu(),
    reason="warp-specialized TMA pipes require NVIDIA Hopper (sm90+)",
)


@triton.jit(noinline=True)
def _large_tma_pipe_consumer(reader, handoff_writer, source, output):
    pid = tl.program_id(0)
    # The real layer performs embedding/RMS work behind grid barriers while
    # the producer fills QKV stages ahead of the first pipe wait.
    for _ in tl.static_range(0, 5):
        tle.distributed_barrier(_MESH)
    local_n_source = tl.arange(0, 32)
    local_n = tle.gpu.set_layout(
        tle.gpu.set_layout(local_n_source, _OUTPUT_LAYOUT)[:, None],
        _WEIGHT_LAYOUT,
    )
    local_k_vector = tle.gpu.set_layout(tl.arange(0, 32), _SOURCE_LAYOUT)
    local_k = tle.gpu.set_layout(local_k_vector[None, :], _WEIGHT_LAYOUT)
    partial = tle.gpu.set_layout(tl.zeros((32, 32), tl.float32), _WEIGHT_LAYOUT)
    for tile in tl.static_range(0, 2):
        ready = reader.wait(tile)
        for k_group in tl.static_range(0, 32):
            shared_k = k_group * 32 + local_k
            weight_ptr = tle.gpu.local_ptr(
                ready.slot.weight,
                (local_n, shared_k),
                [32, 32],
            )
            weight_ptr = tl.max_contiguous(weight_ptr, [1, 8])
            weight_ptr = tl.multiple_of(weight_ptr, [1, 16])
            weight_value = tl.load(weight_ptr)
            source_offset = tile * 1024 + k_group * 32 + local_k_vector
            source_value = tle.gpu.set_layout(tl.load(source + source_offset), _SOURCE_LAYOUT)
            source_value = tle.gpu.set_layout(source_value[None, :], _WEIGHT_LAYOUT)
            partial += weight_value.to(tl.float32) * source_value.to(tl.float32)
        reader.release(tile)
    result = tl.sum(partial, axis=1)
    tl.store(output + pid * 32 + tle.gpu.set_layout(local_n_source, _OUTPUT_LAYOUT), result)
    handoff_writer.commit(0)
    tle.distributed_barrier(_MESH)


@triton.jit(noinline=True)
def _large_tma_pipe_producer(
    writer,
    handoff_reader,
    q_descriptor,
    k_descriptor,
    v_descriptor,
    status,
):
    pid = tl.program_id(0)
    for tile in tl.static_range(0, 2):
        slot = writer.acquire(tile)
        q_weight = slot.weight.subslice([0, 0], [16, 1024])
        k_weight = slot.weight.subslice([16, 0], [8, 1024])
        v_weight = slot.weight.subslice([24, 0], [8, 1024])
        tle.gpu.copy(
            q_descriptor,
            q_weight,
            [16, 1024],
            [pid * 16, tile * 1024],
        )
        tle.gpu.copy(
            k_descriptor,
            k_weight,
            [8, 1024],
            [pid * 8, tile * 1024],
        )
        tle.gpu.copy(
            v_descriptor,
            v_weight,
            [8, 1024],
            [pid * 8, tile * 1024],
        )
        writer.commit(tile)
    writer.close(2)
    handoff_reader.wait(0)
    tl.store(status + pid, 1)


@triton.jit
def _large_tma_pipe_handoff_kernel(
    q_descriptor,
    k_descriptor,
    v_descriptor,
    source,
    output,
    status,
    PRODUCER_WARPS: tl.constexpr,
):
    arena = tle.gpu.alloc(
        [131072],
        dtype=tl.uint8,
        layout=None,
        scope=tle.gpu.smem,
        nv_mma_shared_layout=False,
    )
    stages = tle.gpu.alloc(
        [2, 32, 1024],
        dtype=tl.bfloat16,
        layout=None,
        scope=tle.gpu.smem,
        alias=arena,
        alias_offset_bytes=0,
        nv_mma_shared_layout=True,
    )
    weight_pipe = tle.pipe(
        capacity=2,
        scope="cta",
        name="large_weight",
        weight=stages,
    )
    handoff = tle.pipe(
        capacity=1,
        scope="cta",
        name="large_weight_handoff",
        one_shot=True,
    )
    tle.gpu.warp_specialize(
        [
            (
                _large_tma_pipe_consumer,
                (weight_pipe.reader(), handoff.writer(), source, output),
            ),
            (
                _large_tma_pipe_producer,
                (
                    weight_pipe.writer(),
                    handoff.reader(),
                    q_descriptor,
                    k_descriptor,
                    v_descriptor,
                    status,
                ),
            ),
        ],
        [PRODUCER_WARPS],
        [32],
    )


@pytest.mark.require_tle(
    "distributed_barrier",
    "pipe",
    "gpu.alloc",
    "gpu.copy",
    "gpu.warp_specialize",
)
@pytest.mark.parametrize("producer_warps", [1, 2, 4])
def test_large_tma_pipe_drains_before_fieldless_handoff(with_allocator, producer_warps):
    from triton.tools.tensor_descriptor import TensorDescriptor

    q_weights = torch.randn((128 * 16, 2048), dtype=torch.bfloat16, device="cuda")
    k_weights = torch.randn((128 * 8, 2048), dtype=torch.bfloat16, device="cuda")
    v_weights = torch.randn((128 * 8, 2048), dtype=torch.bfloat16, device="cuda")
    source = torch.randn((2048,), dtype=torch.bfloat16, device="cuda")
    q_descriptor = TensorDescriptor.from_tensor(q_weights, block_shape=[16, 1024])
    k_descriptor = TensorDescriptor.from_tensor(k_weights, block_shape=[8, 1024])
    v_descriptor = TensorDescriptor.from_tensor(v_weights, block_shape=[8, 1024])
    output = torch.empty(128 * 32, dtype=torch.float32, device="cuda")
    status = torch.zeros(128, dtype=torch.int32, device="cuda")
    compiled = _large_tma_pipe_handoff_kernel.warmup(
        q_descriptor,
        k_descriptor,
        v_descriptor,
        source,
        output,
        status,
        producer_warps,
        grid=(128,),
        num_warps=8,
    )
    assert compiled.metadata.shared <= torch.cuda.get_device_properties(0).shared_memory_per_block_optin
    assert compiled.metadata.launch_cooperative_grid is True

    _large_tma_pipe_handoff_kernel[(128,)](
        q_descriptor,
        k_descriptor,
        v_descriptor,
        source,
        output,
        status,
        producer_warps,
        num_warps=8,
    )
    torch.cuda.synchronize()
    combined_weights = torch.cat(
        [
            q_weights.reshape(128, 16, 2048),
            k_weights.reshape(128, 8, 2048),
            v_weights.reshape(128, 8, 2048),
        ],
        dim=1,
    ).reshape(128 * 32, 2048)
    torch.testing.assert_close(
        output,
        torch.mv(combined_weights.to(torch.float32), source.to(torch.float32)),
        atol=0.25,
        rtol=0.01,
    )
    torch.testing.assert_close(status.cpu(), torch.ones(128, dtype=torch.int32))
