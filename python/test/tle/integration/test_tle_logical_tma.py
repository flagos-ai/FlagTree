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
def _logical_tma_gemm_kernel(A, B, C):
    a_desc = tl.make_tensor_descriptor(A, [64, 256], [256, 1], [64, 256])
    b_desc = tl.make_tensor_descriptor(B, [80, 256], [256, 1], [80, 256])
    a_smem = tle.gpu.alloc([64, 256], tl.float16, scope=tle.gpu.smem, capacity=1)
    b_smem = tle.gpu.alloc([80, 256], tl.float16, scope=tle.gpu.smem, capacity=1)
    a_full = tle.gpu.alloc_barrier(expect_bytes=64 * 256 * 2)
    # The transaction covers the logical payload, not its [128, 256] carrier.
    b_full = tle.gpu.alloc_barrier(expect_bytes=80 * 256 * 2)

    tle.gpu.copy(a_desc, a_smem.slot(0), [64, 256], [0, 0], barrier=a_full)
    tle.gpu.copy(b_desc, b_smem.slot(0), [80, 256], [0, 0], barrier=b_full)
    tle.gpu.barrier_wait(a_full, phaseIdx=0)
    tle.gpu.barrier_wait(b_full, phaseIdx=0)

    acc = tle.gpu.wgmma(a_smem.slot(0), b_smem.slot(0), trans_b=True, out_dtype=tl.float32)
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
def test_logical_tma_80x256(with_allocator):
    torch.manual_seed(42)
    a = torch.randn((64, 256), device="cuda", dtype=torch.float16)
    b = torch.randn((80, 256), device="cuda", dtype=torch.float16)
    c = torch.empty((64, 80), device="cuda", dtype=torch.float32)

    kernel = _logical_tma_gemm_kernel[(1, )](a, b, c, num_warps=4)
    expected = a.cpu().float() @ b.cpu().float().T
    torch.testing.assert_close(c.cpu(), expected, atol=1e-3, rtol=1e-3)

    # Padding B to 128 rows would already need this much SMEM, before barriers.
    assert kernel.metadata.shared < (64 + 128) * 256 * 2
