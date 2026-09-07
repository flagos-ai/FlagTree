# flagtree tle
"""A byte-addressable Shared arena is not a Triton register tensor shape."""

import pytest
import torch
import triton
import triton.language as tl
import triton.experimental.tle.language as tle


def _has_nvidia_gpu() -> bool:
    target = triton.runtime.driver.active.get_current_target()
    return target.backend == "cuda" and torch.cuda.is_available()


pytestmark = pytest.mark.skipif(
    not _has_nvidia_gpu(),
    reason="Shared-memory allocation regression requires CUDA",
)


@triton.jit
def _non_power_two_shared_arena_kernel(output):
    arena = tle.gpu.alloc(
        [135168],
        dtype=tl.uint8,
        layout=None,
        scope=tle.gpu.smem,
        nv_mma_shared_layout=False,
    )
    tail = tle.gpu.alloc(
        [128],
        dtype=tl.int32,
        layout=None,
        scope=tle.gpu.smem,
        alias=arena,
        alias_offset_bytes=134656,
        nv_mma_shared_layout=False,
    )
    offsets = tl.arange(0, 128)
    pointers = tle.gpu.local_ptr(tail, (offsets,), [128])
    tl.store(pointers, offsets)
    tl.debug_barrier()
    tl.store(output + offsets, tl.load(pointers))


@pytest.mark.require_tle("gpu.alloc", "gpu.local_ptr")
def test_non_power_two_shared_arena_compiles_and_aliases_tail(with_allocator):
    output = torch.empty((128,), dtype=torch.int32, device="cuda")
    compiled = _non_power_two_shared_arena_kernel.warmup(
        output,
        grid=(1,),
        num_warps=4,
    )
    assert compiled.metadata.shared >= 135168
    assert (
        compiled.metadata.shared
        <= torch.cuda.get_device_properties(0).shared_memory_per_block_optin
    )

    _non_power_two_shared_arena_kernel[(1,)](output, num_warps=4)
    torch.testing.assert_close(
        output,
        torch.arange(128, dtype=torch.int32, device="cuda"),
        rtol=0,
        atol=0,
    )
