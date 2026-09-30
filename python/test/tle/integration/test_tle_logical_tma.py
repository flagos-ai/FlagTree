import pytest
import torch
import triton
import triton.language as tl
import triton.experimental.tle.language as tle

from triton._flagtree_backend import get_active_backend_name

pytestmark = pytest.mark.skipif(
    get_active_backend_name() != "nvidia" or not torch.cuda.is_available()
    or torch.cuda.get_device_capability()[0] != 9,
    reason="logical TMA and WGMMA require NVIDIA Hopper",
)


@triton.jit
def _logical_tma_gemm_kernel(A, B, C, STAGES: tl.constexpr, POINTER_COPY: tl.constexpr):
    a_desc = tl.make_tensor_descriptor(A, [64, 256], [256, 1], [64, 256])
    a_smem = tle.gpu.alloc([64, 256], tl.float16, scope=tle.gpu.smem)
    if STAGES:
        b_buffers = tle.gpu.alloc([STAGES, 80, 256], tl.float16, scope=tle.gpu.smem)
        tl.static_assert(b_buffers.shape[0] == STAGES)
        tl.static_assert(b_buffers.shape[1] == 128)
        b_smem = b_buffers.slot(STAGES - 1)
    else:
        b_smem = tle.gpu.alloc([80, 256], tl.float16, scope=tle.gpu.smem)
    tl.static_assert(b_smem.shape[0] == 128)
    tl.static_assert(b_smem.shape[1] == 256)
    tl.static_assert(b_smem.type.nbytes == 128 * 256 * 2)
    a_full = tle.gpu.alloc_barrier(expect_bytes=64 * 256 * 2)
    tle.gpu.copy(a_desc, a_smem, [64, 256], [0, 0], barrier=a_full)
    tle.gpu.barrier_wait(a_full, phaseIdx=0)
    if POINTER_COPY:
        rows = tl.arange(0, 128)
        cols = tl.arange(0, 256)
        ptrs = B + rows[:, None] * 256 + cols[None, :]
        tle.gpu.copy(ptrs, b_smem, [80, 256], mask=rows[:, None] < 80)
    else:
        # The transaction covers the logical payload, not its [128, 256] carrier.
        b_desc = tl.make_tensor_descriptor(B, [80, 256], [256, 1], [80, 256])
        b_full = tle.gpu.alloc_barrier(expect_bytes=80 * 256 * 2)
        tle.gpu.copy(b_desc, b_smem, [80, 256], [0, 0], barrier=b_full)
        tle.gpu.barrier_wait(b_full, phaseIdx=0)

    acc = tle.gpu.wgmma(a_smem, b_smem, trans_b=True, out_dtype=tl.float32)
    acc = tle.gpu.wgmma_wait(0, acc)
    rows = tl.arange(0, 64)
    cols = tl.arange(0, 128)
    tl.store(C + rows[:, None] * 80 + cols[None, :], acc, cols[None, :] < 80)


@pytest.mark.require_tle(
    "gpu.alloc",
    "gpu.alloc_barrier",
    "gpu.barrier_wait",
    "gpu.buffered_tensor.slot",
    "gpu.copy",
    "gpu.wgmma",
    "gpu.wgmma_wait",
)
@pytest.mark.parametrize("stages", [0, 1, 2, 3])
@pytest.mark.parametrize("pointer_copy", [False, True])
def test_logical_tma_80x256(stages, pointer_copy, with_allocator):
    torch.manual_seed(42)
    a = torch.randn((64, 256), device="cuda", dtype=torch.float16)
    b = torch.randn((80, 256), device="cuda", dtype=torch.float16)
    c = torch.empty((64, 80), device="cuda", dtype=torch.float32)

    kernel = _logical_tma_gemm_kernel[(1, )](a, b, c, stages, pointer_copy, num_warps=4)
    expected = a.cpu().float() @ b.cpu().float().T
    torch.testing.assert_close(c.cpu(), expected, atol=1e-3, rtol=1e-3)

    # Padding B to 128 rows would already need this much SMEM, before barriers.
    assert kernel.metadata.shared < (64 + max(1, stages) * 128) * 256 * 2


@triton.jit
def _logical_initialized_gemm_kernel(C, STAGES: tl.constexpr, PER_STAGE: tl.constexpr):
    a = tle.gpu.alloc([64, 256], tl.float16, init_value=tl.full([64, 256], 1, tl.float16))
    if STAGES:
        if PER_STAGE:
            stage_ids = tl.arange(0, triton.next_power_of_2(STAGES))
            values = tl.broadcast_to((stage_ids[:, None, None] + 1).to(tl.float16),
                                     [triton.next_power_of_2(STAGES), 128, 256])
        else:
            values = tl.full([128, 256], 2, tl.float16)
        buffers = tle.gpu.alloc([STAGES, 80, 256], tl.float16, init_value=values)
        b = buffers.slot(STAGES - 1)
    else:
        b = tle.gpu.alloc([80, 256], tl.float16, init_value=tl.full([128, 256], 2, tl.float16))
    tl.static_assert(b.shape[0] == 128)
    acc = tle.gpu.wgmma(a, b, trans_b=True)
    acc = tle.gpu.wgmma_wait(0, acc)
    rows = tl.arange(0, 64)
    cols = tl.arange(0, 128)
    tl.store(C + rows[:, None] * 80 + cols[None, :], acc, cols[None, :] < 80)


@pytest.mark.require_tle("gpu.alloc", "gpu.buffered_tensor.slot", "gpu.wgmma", "gpu.wgmma_wait")
@pytest.mark.parametrize("stages,per_stage", [(0, False), (2, False), (2, True), (3, False), (3, True)])
def test_logical_alloc_initializer(stages, per_stage):
    c = torch.empty((64, 80), device="cuda", dtype=torch.float32)
    _logical_initialized_gemm_kernel[(1, )](c, stages, per_stage, num_warps=4)
    expected = 256 * (stages if per_stage else 2)
    torch.testing.assert_close(c, torch.full_like(c, expected), atol=0, rtol=0)
