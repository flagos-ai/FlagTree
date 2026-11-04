# flagtree tle
"""TMA regression tests for the paged-attention KV cache contract."""

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
    reason="warp-specialized TMA pipes require NVIDIA Hopper (sm90+)",
)


@triton.jit(noinline=True)
def _attention_tile_consumer(reader, row_major, transposed):
    ready = reader.wait(0)
    linear = tl.arange(0, 8192)
    token = linear // 128
    dimension = linear % 128
    zero = tl.full((8192,), 0, tl.int32)
    row_pointer = tle.gpu.local_ptr(
        ready.slot.value,
        (zero, zero, token, zero, dimension),
        shape=(8192,),
    )
    tl.store(row_major + linear, tl.load(row_pointer))

    transposed_linear = tl.arange(0, 8192)
    transposed_dimension = transposed_linear // 64
    transposed_token = transposed_linear % 64
    transposed_zero = tl.full((8192,), 0, tl.int32)
    transposed_pointer = tle.gpu.local_ptr(
        ready.slot.value,
        (
            transposed_zero,
            transposed_zero,
            transposed_token,
            transposed_zero,
            transposed_dimension,
        ),
        shape=(8192,),
    )
    tl.store(transposed + transposed_linear, tl.load(transposed_pointer))
    reader.release(0)


@triton.jit(noinline=True)
def _attention_tile_producer(writer, descriptor):
    slot = writer.acquire(0)
    tle.gpu.copy(
        descriptor,
        slot.value,
        [1, 1, 64, 1, 128],
        offsets=[1, 2, 64, 3, 0],
    )
    writer.commit(0)
    writer.close(1)


@triton.jit
def _attention_tile_kernel(descriptor, row_major, transposed):
    stages = tle.gpu.alloc(
        [2, 1, 1, 64, 1, 128],
        dtype=tl.bfloat16,
        layout=None,
        scope=tle.gpu.smem,
        nv_mma_shared_layout=True,
    )
    pipe = tle.pipe(capacity=2, scope="cta", name="attention_value", value=stages)
    tle.gpu.warp_specialize(
        [
            (_attention_tile_consumer, (pipe.reader(), row_major, transposed)),
            (_attention_tile_producer, (pipe.writer(), descriptor)),
        ],
        [1],
        [32],
    )


@pytest.mark.require_tle("pipe", "gpu.alloc", "gpu.copy", "gpu.local_ptr", "gpu.warp_specialize")
def test_noncontiguous_paged_attention_descriptor_preserves_tile_axes(with_allocator):
    from triton.tools.tensor_descriptor import TensorDescriptor

    cache = torch.arange(
        2 * 4 * 2 * 256 * 4 * 128,
        dtype=torch.float32,
        device="cuda",
    ).reshape(2, 4, 2, 256, 4, 128).to(torch.bfloat16)
    value = cache[:, :, 1]
    assert value.stride() == (1048576, 262144, 512, 128, 1)
    descriptor = TensorDescriptor.from_tensor(value, [1, 1, 64, 1, 128])
    row_major = torch.empty((64, 128), dtype=torch.bfloat16, device="cuda")
    transposed = torch.empty((128, 64), dtype=torch.bfloat16, device="cuda")

    _attention_tile_kernel[(1,)](
        descriptor,
        row_major,
        transposed,
        num_warps=8,
    )
    torch.cuda.synchronize()
    expected = value[1, 2, 64:128, 3]
    torch.testing.assert_close(row_major, expected)
    torch.testing.assert_close(transposed, expected.T)
