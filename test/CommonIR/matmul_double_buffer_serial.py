import argparse
import os

import torch
import triton
import triton.language as tl

import triton.experimental.tle as tle  # noqa: F401  (registers tile/tle dialects)
from triton.experimental.tle.language.dsa.ascend import L1  # noqa: F401
from triton.experimental.tle.language.dsa import tile_alloc, tile_copy, tile_to_tensor, tile_subview  # noqa: F401

# =============================================================================
#  Compile-time configuration
# =============================================================================
_DEFAULT_M = 1024
_DEFAULT_N = 1024
_DEFAULT_K = 1024
_DEFAULT_NUM_CORES = 24

BLOCK_M = 128
BLOCK_N = 256
BLOCK_K = 128


def get_number_cores():
    """Return the number of AI cores to use as the launch grid size."""
    try:
        import torch_npu  # noqa: F401
        return torch.npu.get_device_properties(0).ai_core_num
    except Exception:
        return _DEFAULT_NUM_CORES


# =============================================================================
#  Matmul kernel: compute-first double buffer
#
#  grid = (NUM_CORES,). Each core handles multiple (M_tile, N_tile) output
#  blocks in round-robin fashion.
#
#  Strategy: "compute-first" ordering -- tl.dot reads from the current buffer
#  BEFORE tile_copy overwrites it with the next K-tile. This allows bishengir's
#  --enable-auto-multi-buffer pass to recognize the pattern and physically
#  pipeline MTE2 (DMA) and Cube (matmul) across iterations.
#
#  Loop body (unrolled x2):
#    Even step: dot(buf0), then copy(next -> buf1)
#    Odd step:  dot(buf1), then copy(next -> buf0)
# =============================================================================
@triton.jit
def matmul_kernel(
    mat_a,
    mat_b,
    mat_c,
    M,
    N: tl.constexpr,
    K: tl.constexpr,
    NUM_CORES: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    NUM_BLOCKS_M = tl.cdiv(M, BLOCK_M)
    NUM_BLOCKS_N = tl.cdiv(N, BLOCK_N)
    NUM_BLOCKS = NUM_BLOCKS_M * NUM_BLOCKS_N
    NUM_K_BLOCKS = tl.cdiv(K, BLOCK_K)

    # On-chip double buffers in L1
    mat_a_l1_0 = tile_alloc([BLOCK_M, BLOCK_K], mat_a.dtype.element_ty, L1)
    mat_a_l1_1 = tile_alloc([BLOCK_M, BLOCK_K], mat_a.dtype.element_ty, L1)
    mat_b_l1_0 = tile_alloc([BLOCK_K, BLOCK_N], mat_b.dtype.element_ty, L1)
    mat_b_l1_1 = tile_alloc([BLOCK_K, BLOCK_N], mat_b.dtype.element_ty, L1)

    # Each core processes output blocks in round-robin
    for block_idx in range(pid, NUM_BLOCKS, NUM_CORES):
        # Compute M/N tile indices
        pid_m = block_idx // NUM_BLOCKS_N
        pid_n = block_idx % NUM_BLOCKS_N
        m_start = pid_m * BLOCK_M
        n_start = pid_n * BLOCK_N

        mat_c_acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

        # -- Prologue: load first K-tile into buffer 0 ---------------------
        a_ptr = tl.make_block_ptr(
            mat_a, (M, K), (K, 1), (m_start, 0),
            (BLOCK_M, BLOCK_K), (1, 0))
        b_ptr = tl.make_block_ptr(
            mat_b, (K, N), (N, 1), (0, n_start),
            (BLOCK_K, BLOCK_N), (1, 0))
        tile_copy(a_ptr, mat_a_l1_0, [BLOCK_M, BLOCK_K])
        tile_copy(b_ptr, mat_b_l1_0, [BLOCK_K, BLOCK_N])

        # -- Main K-loop: compute-first, double-buffered --------------------
        for k_pair in range(0, NUM_K_BLOCKS - 1, 2):
            # Even: compute buf0, then prefetch next -> buf1
            mat_c_acc = tl.dot(
                tile_to_tensor(mat_a_l1_0, writable=False),
                tile_to_tensor(mat_b_l1_0, writable=False),
                mat_c_acc, out_dtype=tl.float32)
            a_ptr = tl.advance(a_ptr, [0, BLOCK_K])
            b_ptr = tl.advance(b_ptr, [BLOCK_K, 0])
            tile_copy(a_ptr, mat_a_l1_1, [BLOCK_M, BLOCK_K])
            tile_copy(b_ptr, mat_b_l1_1, [BLOCK_K, BLOCK_N])

            # Odd: compute buf1, then prefetch next -> buf0
            mat_c_acc = tl.dot(
                tile_to_tensor(mat_a_l1_1, writable=False),
                tile_to_tensor(mat_b_l1_1, writable=False),
                mat_c_acc, out_dtype=tl.float32)
            if k_pair + 2 < NUM_K_BLOCKS:
                a_ptr = tl.advance(a_ptr, [0, BLOCK_K])
                b_ptr = tl.advance(b_ptr, [BLOCK_K, 0])
                tile_copy(a_ptr, mat_a_l1_0, [BLOCK_M, BLOCK_K])
                tile_copy(b_ptr, mat_b_l1_0, [BLOCK_K, BLOCK_N])

        # -- Epilogue: consume the last tile if NUM_K_BLOCKS is odd ---------
        if NUM_K_BLOCKS % 2 == 1:
            mat_c_acc = tl.dot(
                tile_to_tensor(mat_a_l1_0, writable=False),
                tile_to_tensor(mat_b_l1_0, writable=False),
                mat_c_acc, out_dtype=tl.float32)

        # Store result back to GM
        tl.store(
            tl.make_block_ptr(
                mat_c, (M, N), (N, 1),
                (m_start, n_start),
                (BLOCK_M, BLOCK_N), (1, 0)),
            mat_c_acc.to(mat_c.dtype.element_ty))


# =============================================================================
#  Host-side launch
# =============================================================================
def call(mat_a, mat_b, num_cores=_DEFAULT_NUM_CORES):
    m = mat_a.shape[0]
    k = mat_a.shape[1]
    n = mat_b.shape[1]
    mat_c = torch.empty(m, n, dtype=mat_a.dtype, device=mat_a.device)
    matmul_kernel[(num_cores,)](mat_a, mat_b, mat_c, m, n, k, num_cores,
                               BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K)
    return mat_c


# =============================================================================
#  Main
# =============================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Matmul kernel (compute-first double buffer)")
    parser.add_argument("--M", type=int, default=_DEFAULT_M)
    parser.add_argument("--N", type=int, default=_DEFAULT_N)
    parser.add_argument("--K", type=int, default=_DEFAULT_K)
    parser.add_argument("--num-cores", type=int, default=None)
    parser.add_argument("--no-check", action="store_true")
    args = parser.parse_args()

    M, N, K = args.M, args.N, args.K
    num_cores = args.num_cores or get_number_cores()

    # ---- functional test on device ------------------------------------------
    device = "npu" if hasattr(torch, "npu") and torch.npu.is_available() else "cuda"
    torch.manual_seed(0)
    mat_a = torch.randn((M, K), dtype=torch.float16, device=device)
    mat_b = torch.randn((K, N), dtype=torch.float16, device=device)

    mat_c = call(mat_a, mat_b, num_cores)

    if not args.no_check:
        ref = torch.matmul(mat_a.float(), mat_b.float()).to(torch.float16)
        torch.testing.assert_close(ref, mat_c, rtol=1e-2, atol=1e-2)
        print("Test Passed!")
    else:
        print("Reference check skipped.")
