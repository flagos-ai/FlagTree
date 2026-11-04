"""Every warp must publish global writes before its CTA announces arrival."""
import pytest
import torch
import triton
import triton.language as tl
import triton.experimental.tle.language as tle

MESH = tl.constexpr(tle.device_mesh({'block': [('block_y', 4), ('block_x', 4)]}))
ROWS = tl.constexpr(MESH.value.axis_group(('block_x',), group_shape=(4,)))

@triton.jit
def _exchange(X, Y, Errors, N: tl.constexpr, STEPS: tl.constexpr, ROW_ONLY: tl.constexpr):
    pid = tl.program_id(0)
    lane = tl.arange(0, N)
    if ROW_ONLY:
        next_id = pid // 4 * 4 + (pid + 1) % 4
        previous_id = pid // 4 * 4 + (pid + 3) % 4
    else:
        next_id = (pid + 1) % 16
        previous_id = (pid + 15) % 16
    for step in range(STEPS):
        expected = step * 100000 + pid * N + lane
        tl.store(X + pid * N + lane, expected)
        if ROW_ONLY: tle.distributed_barrier(ROWS)
        else: tle.distributed_barrier(MESH)
        received = tl.load(X + next_id * N + lane)
        tl.atomic_add(Errors + pid, tl.sum((received != step * 100000 + next_id * N + lane).to(tl.int32), 0))
        tl.store(Y + pid * N + lane, received)
        if ROW_ONLY: tle.distributed_barrier(ROWS)
        else: tle.distributed_barrier(MESH)
        echoed = tl.load(Y + previous_id * N + lane)
        tl.atomic_add(Errors + pid, tl.sum((echoed != expected).to(tl.int32), 0))
        if ROW_ONLY: tle.distributed_barrier(ROWS)
        else: tle.distributed_barrier(MESH)

@pytest.mark.parametrize('n', [128, 512, 2048, 8192])
@pytest.mark.parametrize('warps', [4, 16])
@pytest.mark.parametrize('row_only', [False, True])
def test_grid_barrier_publishes_all_warp_writes(n, warps, row_only):
    if not torch.cuda.is_available(): pytest.skip('requires GPU')
    if 'Iluvatar' not in torch.cuda.get_device_name(): pytest.skip('Iluvatar publication protocol')
    if torch.cuda.get_device_properties(0).multi_processor_count < 16: pytest.skip('requires 16 resident CTAs')
    triton.set_allocator(lambda size, alignment, stream: torch.zeros(max(size,16), device='cuda', dtype=torch.uint8))
    x = torch.zeros(16*n, device='cuda', dtype=torch.int32)
    y = torch.zeros_like(x)
    errors = torch.zeros(16, device='cuda', dtype=torch.int32)
    for _ in range(3):
        _exchange[(16,)](x,y,errors,n,500,row_only,num_warps=warps)
        torch.cuda.synchronize()
        assert errors.sum().item() == 0, errors.cpu().tolist()
