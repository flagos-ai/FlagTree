# flagtree tle
"""
Smoke tests for TLE warp_specialize GEMM with staged TMA completion barriers.

This intentionally keeps the GEMM to a single output tile while splitting K
across multiple shared-memory slots. The goal is to validate the basic
producer/consumer protocol:

- default partition waits on per-slot "empty" barriers and issues TMA loads
- TMA completion signals "full" barriers
- worker partition waits on per-slot "full" barriers, computes WGMMA, stores C
- worker arrives on "empty" barriers to release the smem buffers
"""

import pytest
import torch
import triton
import triton.language as tl
import triton.experimental.tle.language as tle

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False


def _is_nvidia_backend() -> bool:
    target = triton.runtime.driver.active.get_current_target()
    return target.backend == "cuda"


def _has_nvidia_hopper_gpu() -> bool:
    return _is_nvidia_backend() and torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 9


pytestmark = pytest.mark.skipif(
    not _has_nvidia_hopper_gpu(),
    reason="warp_specialize TMA WGMMA GEMM requires NVIDIA Hopper (sm90+)",
)


@triton.jit
def _slot_phase(k_iter, num_slots: tl.constexpr):
    slot = k_iter % num_slots
    phase = k_iter // num_slots
    return slot, phase


@triton.jit
def _ws_tma_multi_slot_producer(
    a_desc,
    b_desc,
    a_smem,
    b_smem,
    a_empty,
    b_empty,
    a_full,
    b_full,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    K_TILES: tl.constexpr,
    NUM_SLOTS: tl.constexpr,
):
    for k_iter in range(0, K_TILES):
        slot, phase = _slot_phase(k_iter, NUM_SLOTS)
        k_offset = k_iter * BLOCK_K

        tle.gpu.barrier_wait(a_empty[slot], phaseIdx=phase)
        tle.gpu.barrier_wait(b_empty[slot], phaseIdx=phase)

        tle.gpu.copy(a_desc, a_smem.slot(slot), [BLOCK_M, BLOCK_K], [0, k_offset], barrier=a_full[slot])
        tle.gpu.copy(b_desc, b_smem.slot(slot), [BLOCK_K, BLOCK_N], [k_offset, 0], barrier=b_full[slot])

    for k_iter in range(K_TILES - NUM_SLOTS, K_TILES):
        slot, phase = _slot_phase(k_iter, NUM_SLOTS)
        release_phase = phase + 1

        tle.gpu.barrier_wait(a_empty[slot], phaseIdx=release_phase)
        tle.gpu.barrier_wait(b_empty[slot], phaseIdx=release_phase)


@triton.jit
def _ws_tma_multi_slot_consumer(
    a_smem,
    b_smem,
    a_empty,
    b_empty,
    a_full,
    b_full,
    c_ptr,
    stride_cm: tl.constexpr,
    stride_cn: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    K_TILES: tl.constexpr,
    NUM_SLOTS: tl.constexpr,
):
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    slot, phase = _slot_phase(0, NUM_SLOTS)
    tle.gpu.barrier_wait(a_full[slot], phaseIdx=phase)
    tle.gpu.barrier_wait(b_full[slot], phaseIdx=phase)
    acc = tle.gpu.wgmma(a_smem.slot(slot), b_smem.slot(slot), acc)
    last_slot = slot
    last_phase = phase

    for k_iter in range(1, K_TILES):
        slot, phase = _slot_phase(k_iter, NUM_SLOTS)

        tle.gpu.barrier_wait(a_full[slot], phaseIdx=phase)
        tle.gpu.barrier_wait(b_full[slot], phaseIdx=phase)

        acc = tle.gpu.wgmma(a_smem.slot(slot), b_smem.slot(slot), acc)
        acc = tle.gpu.wgmma_wait(1, acc)
        tle.gpu.barrier_arrive(a_empty[last_slot], phaseIdx=last_phase)
        tle.gpu.barrier_arrive(b_empty[last_slot], phaseIdx=last_phase)

        last_slot = slot
        last_phase = phase

    acc = tle.gpu.wgmma_wait(0, acc)
    tle.gpu.barrier_arrive(a_empty[last_slot], phaseIdx=last_phase)
    tle.gpu.barrier_arrive(b_empty[last_slot], phaseIdx=last_phase)

    offs_m = tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)

    c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    tl.store(c_ptrs, acc)


@triton.jit
def ws_tma_multi_slot_gemm_kernel(
    a_desc,
    b_desc,
    c_ptr,
    stride_cm: tl.constexpr,
    stride_cn: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    A_TILE_BYTES: tl.constexpr,
    B_TILE_BYTES: tl.constexpr,
    K_TILES: tl.constexpr,
    NUM_SLOTS: tl.constexpr,
):
    a_smem = tle.gpu.alloc(
        [NUM_SLOTS, BLOCK_M, BLOCK_K],
        dtype=tl.float16,
        layout=None,
        scope=tle.gpu.smem,
    )
    b_smem = tle.gpu.alloc(
        [NUM_SLOTS, BLOCK_K, BLOCK_N],
        dtype=tl.float16,
        layout=None,
        scope=tle.gpu.smem,
    )

    a_empty = tle.gpu.alloc_barriers(num_barriers=NUM_SLOTS, init=tle.gpu.READY)
    b_empty = tle.gpu.alloc_barriers(num_barriers=NUM_SLOTS, init=tle.gpu.READY)
    a_full = tle.gpu.alloc_barriers(num_barriers=NUM_SLOTS, expect_bytes=A_TILE_BYTES)
    b_full = tle.gpu.alloc_barriers(num_barriers=NUM_SLOTS, expect_bytes=B_TILE_BYTES)

    tle.gpu.warp_specialize(
        [
            (
                _ws_tma_multi_slot_producer,
                (
                    a_desc,
                    b_desc,
                    a_smem,
                    b_smem,
                    a_empty,
                    b_empty,
                    a_full,
                    b_full,
                    BLOCK_M,
                    BLOCK_N,
                    BLOCK_K,
                    K_TILES,
                    NUM_SLOTS,
                ),
            ),
            (
                _ws_tma_multi_slot_consumer,
                (
                    a_smem,
                    b_smem,
                    a_empty,
                    b_empty,
                    a_full,
                    b_full,
                    c_ptr,
                    stride_cm,
                    stride_cn,
                    BLOCK_M,
                    BLOCK_N,
                    K_TILES,
                    NUM_SLOTS,
                ),
            ),
        ],
        [4],
        [168],
    )


def ws_tma_multi_slot_gemm(A, B, C, launch_num_warps, block_k, num_slots):
    assert A.ndim == 2 and B.ndim == 2 and C.ndim == 2
    assert A.shape[1] == B.shape[0]
    assert C.shape == (A.shape[0], B.shape[1])
    assert A.dtype == torch.float16 and B.dtype == torch.float16 and C.dtype == torch.float32

    block_m, total_k = A.shape
    total_k_b, block_n = B.shape
    assert total_k == total_k_b
    assert total_k % block_k == 0
    k_tiles = total_k // block_k
    assert k_tiles >= num_slots

    from triton.tools.tensor_descriptor import TensorDescriptor

    a_desc = TensorDescriptor.from_tensor(A, block_shape=[block_m, block_k])
    b_desc = TensorDescriptor.from_tensor(B, block_shape=[block_k, block_n])
    return ws_tma_multi_slot_gemm_kernel[(1, )](
        a_desc,
        b_desc,
        C,
        C.stride(0),
        C.stride(1),
        block_m,
        block_n,
        block_k,
        block_m * block_k * A.element_size(),
        block_k * block_n * B.element_size(),
        k_tiles,
        num_slots,
        num_warps=launch_num_warps,
    )


class TestTLEWarpSpecializeTmaGemm:

    @pytest.mark.parametrize("launch_num_warps", [4])
    def test_multi_slot_producer_consumer_wgmma(self, launch_num_warps):
        torch.manual_seed(2026 + launch_num_warps)
        block_m, block_n, block_k = 64, 16, 16
        k_tiles, num_slots = 4, 2

        a = torch.randn(block_m, block_k * k_tiles, device="cuda", dtype=torch.float16).contiguous()
        b = torch.randn(block_k * k_tiles, block_n, device="cuda", dtype=torch.float16).contiguous()
        c = torch.empty((block_m, block_n), device="cuda", dtype=torch.float32).contiguous()

        kernel = ws_tma_multi_slot_gemm(a, b, c, launch_num_warps, block_k, num_slots)
        assert "tt.call" not in kernel.asm["ttgir"]

        expected = torch.matmul(a.float(), b.float())
        torch.testing.assert_close(c, expected, atol=5e-2, rtol=5e-2)
