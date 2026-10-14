"""GPU coverage for dynamic shared-buffer slot addressing."""
import pytest
import torch
import triton
import triton.language as tl
import triton.experimental.tle.language as tle


@triton.jit
def _slot_roundtrip(source, output, SLOT: tl.constexpr):
    stages = tle.gpu.alloc([2, 128], tl.int32, scope=tle.gpu.smem, nv_mma_shared_layout=False)
    offsets = tl.arange(0, 128)
    for stage in tl.static_range(2):
        pointer = tle.gpu.local_ptr(stages.slot(stage), (offsets, ))
        tl.store(pointer, tl.load(source + stage * 128 + offsets))
    tl.debug_barrier()
    slot = tl.program_id(0) % 2 if SLOT < 0 else SLOT
    selected = tl.load(tle.gpu.local_ptr(stages.slot(slot), (offsets, )))
    tl.store(output + tl.program_id(0) * 128 + offsets, selected)


@pytest.mark.skipif(not torch.cuda.is_available(), reason='requires CUDA')
@pytest.mark.require_tle('gpu.alloc', 'gpu.local_ptr')
@pytest.mark.parametrize('slot', [-1, 0, 1])
def test_buffered_tensor_slot_selects_distinct_shared_payloads(slot):
    source = torch.arange(256, device='cuda', dtype=torch.int32)
    output = torch.empty((2, 128), device='cuda', dtype=torch.int32)
    _slot_roundtrip[(2, )](source, output, slot, num_warps=4)
    expected = source.reshape(2, 128)
    if slot >= 0:
        expected = expected[slot].expand_as(output)
    torch.testing.assert_close(output, expected, rtol=0, atol=0)
