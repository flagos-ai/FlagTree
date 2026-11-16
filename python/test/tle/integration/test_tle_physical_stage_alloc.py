"""Rank-5 physical NVMMA buffers retain three distinct, unpadded slots."""
import pytest
import torch
import triton
import triton.language as tl
import triton.experimental.tle.language as tle


_PHYSICAL_LAYOUT = tl.constexpr(
    tle.gpu.nv_mma_shared_layout.make_default([3, 64, 2, 2, 64], tl.bfloat16))


@triton.jit
def _three_physical_slots(output, EXPLICIT: tl.constexpr):
    arena = tle.gpu.alloc([98304], tl.uint8, nv_mma_shared_layout=False, alignment_bytes=1024)
    if EXPLICIT:
        buffer = tle.gpu.alloc([3, 64, 2, 2, 64], tl.bfloat16, layout=_PHYSICAL_LAYOUT,
                               alias=arena, nv_mma_shared_layout=True)
    else:
        buffer = tle.gpu.alloc([3, 64, 2, 2, 64], tl.bfloat16, alias=arena,
                               nv_mma_shared_layout=True)
    i = tl.arange(0, 16384)
    coordinates = (i // 256, (i // 128) % 2, (i // 64) % 2, i % 64)
    # Fill all three slots before reading any slot: an incorrect stage stride
    # or padded/aliased stage index cannot pass through immediate forwarding.
    for stage in tl.static_range(3):
        pointer = tle.gpu.local_ptr(buffer.slot(stage), coordinates)
        tl.store(pointer, (stage * 32 + i % 16).to(tl.bfloat16))
    tl.debug_barrier()
    for stage in tl.static_range(3):
        pointer = tle.gpu.local_ptr(buffer.slot(stage), coordinates)
        tl.store(output + stage * 16384 + i, tl.load(pointer).to(tl.float32))


@pytest.mark.parametrize("explicit", [False, True])
def test_three_physical_nvmma_slots_and_last_element(explicit):
    if not torch.cuda.is_available() or torch.version.cuda is None:
        pytest.skip("NVIDIA CUDA is required")
    if torch.cuda.get_device_capability()[0] < 9:
        pytest.skip("NVIDIA Hopper shared layout is required")
    output = torch.empty((3, 16384), dtype=torch.float32, device="cuda")
    _three_physical_slots[(1,)](output, explicit, num_warps=4)
    expected = (torch.arange(16384, dtype=torch.int32, device="cuda")[None, :] % 16
                + 32 * torch.arange(3, dtype=torch.int32, device="cuda")[:, None]).float()
    assert torch.equal(output, expected)
    assert output[2, -1].item() == 79
