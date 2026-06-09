# Copyright 2026- Xcoresigma Technology Co., Ltd

import torch
import torch_npu
import triton
import triton.language as tl
import triton.experimental.tle as tle
import numpy as np
from triton.backends.ascend.testing import do_bench_npu

np.random.seed(21)
DEVICE = "npu"
torch.manual_seed(20)
torch_npu.npu.set_device(0)
torch.set_printoptions(sci_mode=False, precision=4, linewidth=300)

BLOCK_M = 128
BLOCK_N = 128
BLOCK_K = 128


@triton.jit
def matmul_ref(mat_a, mat_b, mat_c, M: tl.constexpr, N: tl.constexpr, K: tl.constexpr, BLOCK_M: tl.constexpr,
               BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr):
    pid = tl.program_id(0)
    num_blocks_n = triton.cdiv(N, BLOCK_N)
    block_m = pid // num_blocks_n
    block_n = pid % num_blocks_n

    m_start = block_m * BLOCK_M
    n_start = block_n * BLOCK_N

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k_start in range(0, K, BLOCK_K):
        a_offset = (m_start + tl.arange(0, BLOCK_M))[:, None] * K + (k_start + tl.arange(0, BLOCK_K))[None, :]
        a_mask = ((m_start + tl.arange(0, BLOCK_M))[:, None] < M) & ((k_start + tl.arange(0, BLOCK_K))[None, :] < K)
        b_offset = (k_start + tl.arange(0, BLOCK_K))[:, None] * N + (n_start + tl.arange(0, BLOCK_N))[None, :]
        b_mask = ((k_start + tl.arange(0, BLOCK_K))[:, None] < K) & ((n_start + tl.arange(0, BLOCK_N))[None, :] < N)
        a_block = tl.load(mat_a + a_offset, mask=a_mask, other=0.0)
        b_block = tl.load(mat_b + b_offset, mask=b_mask, other=0.0)
        acc = tl.dot(a_block, b_block, acc)

    c_offset = (m_start + tl.arange(0, BLOCK_M))[:, None] * N + (n_start + tl.arange(0, BLOCK_N))[None, :]
    c_mask = ((m_start + tl.arange(0, BLOCK_M))[:, None] < M) & ((n_start + tl.arange(0, BLOCK_N))[None, :] < N)
    tl.store(mat_c + c_offset, acc.to(tl.float16), mask=c_mask)


# ═══════════════════════════════════════════════════════════════════════════════
# mayDiscretememaccess
# ═══════════════════════════════════════════════════════════════════════════════
@triton.jit
def triton_compile_hint_may_discrete(mat_a, mat_b, mat_c, M: tl.constexpr, N: tl.constexpr, K: tl.constexpr,
                                     BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr):
    pid = tl.program_id(0)
    num_blocks_n = triton.cdiv(N, BLOCK_N)
    block_m = pid // num_blocks_n
    block_n = pid % num_blocks_n

    m_start = block_m * BLOCK_M
    n_start = block_n * BLOCK_N

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k_start in range(0, K, BLOCK_K):
        a_offset = (m_start + tl.arange(0, BLOCK_M))[:, None] * K + (k_start + tl.arange(0, BLOCK_K))[None, :]
        a_mask = ((m_start + tl.arange(0, BLOCK_M))[:, None] < M) & ((k_start + tl.arange(0, BLOCK_K))[None, :] < K)

        b_offset = (k_start + tl.arange(0, BLOCK_K))[:, None] * N + (n_start + tl.arange(0, BLOCK_N))[None, :]
        b_mask = ((k_start + tl.arange(0, BLOCK_K))[:, None] < K) & ((n_start + tl.arange(0, BLOCK_N))[None, :] < N)

        a_block = tl.load(mat_a + a_offset, mask=a_mask, other=0.0)
        b_block = tl.load(mat_b + b_offset, mask=b_mask, other=0.0)
        # Degrade tensor→scalar to avoid 32B-alignment axis expansion in UB
        tle.dsa.ascend.compile_hint(a_block, "mayDiscretememaccess")
        tle.dsa.ascend.compile_hint(b_block, "mayDiscretememaccess")
        acc = tl.dot(a_block, b_block, acc)

    c_offset = (m_start + tl.arange(0, BLOCK_M))[:, None] * N + (n_start + tl.arange(0, BLOCK_N))[None, :]
    c_mask = ((m_start + tl.arange(0, BLOCK_M))[:, None] < M) & ((n_start + tl.arange(0, BLOCK_N))[None, :] < N)
    tle.dsa.ascend.compile_hint(acc, "mayDiscretememaccess")
    tl.store(mat_c + c_offset, acc.to(tl.float16), mask=c_mask)


def test_may_discrete_ub_overflow():
    """
    Demonstrate mayDiscretememaccess with <Nx1xf32> block shapes to avoid UB overflow.

    When K=1 and BLOCK_K=1, each loaded block from A has shape (BLOCK_M, 1) x f32.
    Because 1xf32 = 4 bytes < 32-byte hardware alignment, the tensor is expanded
    to (BLOCK_M, 8) x f32 in UB — inflating memory 8× and risking UB overflow.
    mayDiscretememaccess degrades the load to scalar ops, avoiding the expansion.
    """
    M, K, N = 256, 1, 256  # K=1 → A is (M, 1), B is (1, N)
    BLOCK_M, BLOCK_N, BLOCK_K = 64, 64, 1  # BLOCK_K=1 → loaded blocks are Nx1xf32
    dtype = torch.float16

    mat_a = torch.randn((M, K), dtype=dtype).npu()
    mat_b = torch.randn((K, N), dtype=dtype).npu()
    ref = torch.zeros((M, N), dtype=dtype).npu()

    num_blocks_m = (M + BLOCK_M - 1) // BLOCK_M
    num_blocks_n = (N + BLOCK_N - 1) // BLOCK_N
    grid = (num_blocks_m * num_blocks_n, )

    print("\n" + "=" * 60)
    print("Test: mayDiscretememaccess with <Nx1xf32> block shapes")
    print(f"      A({M},{K}) * B({K},{N}), BLOCK_K={BLOCK_K}")
    print("=" * 60)

    # Baseline: reference kernel without any hints
    matmul_ref[grid](mat_a, mat_b, ref, M, N, K, BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K)
    # Hint kernel: uses mayDiscretememaccess on every loaded block
    mat_c = torch.zeros((M, N), dtype=dtype).npu()
    triton_compile_hint_may_discrete[grid](mat_a, mat_b, mat_c, M, N, K, BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N,
                                           BLOCK_K=BLOCK_K)
    # print(mat_c)
    torch.testing.assert_close(mat_c, ref, rtol=1e-2, atol=1e-2)
    print("PASSED result check for mayDiscretememaccess with <Nx1xf32>")

    # Benchmark: compare hint vs no-hint
    ref_time = do_bench_npu(
        lambda: matmul_ref[grid]
        (mat_a, mat_b, ref, M, N, K, BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K, multibuffer=False),
        clear_l2_cache=True, collect_prof=False)
    hint_time = do_bench_npu(
        lambda: triton_compile_hint_may_discrete[grid]
        (mat_a, mat_b, mat_c, M, N, K, BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K, multibuffer=False),
        clear_l2_cache=True, collect_prof=False)
    print(
        f"[BENCH] mayDiscretememaccess (hint): {hint_time:.2f} us, (no hint): {ref_time:.2f} us, ratio={hint_time / ref_time:.3f}"
    )
    print("Note: hint downgrades tensor→scalar, avoiding 8× UB expansion from <Nx1xf32> alignment.")


# ═══════════════════════════════════════════════════════════════════════════════
# dot_pad_only_k
# ═══════════════════════════════════════════════════════════════════════════════
@triton.jit
def triton_dot_pad_only_k(mat_a, mat_b, mat_c, M: tl.constexpr, N: tl.constexpr, K: tl.constexpr, BLOCK_M: tl.constexpr,
                          BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr):
    pid = tl.program_id(0)
    num_blocks_n = triton.cdiv(N, BLOCK_N)
    block_m = pid // num_blocks_n
    block_n = pid % num_blocks_n

    m_start = block_m * BLOCK_M
    n_start = block_n * BLOCK_N

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k_start in range(0, K, BLOCK_K):
        a_offset = (m_start + tl.arange(0, BLOCK_M))[:, None] * K + (k_start + tl.arange(0, BLOCK_K))[None, :]
        a_mask = ((m_start + tl.arange(0, BLOCK_M))[:, None] < M) & ((k_start + tl.arange(0, BLOCK_K))[None, :] < K)

        b_offset = (k_start + tl.arange(0, BLOCK_K))[:, None] * N + (n_start + tl.arange(0, BLOCK_N))[None, :]
        b_mask = ((k_start + tl.arange(0, BLOCK_K))[:, None] < K) & ((n_start + tl.arange(0, BLOCK_N))[None, :] < N)

        a_block = tl.load(mat_a + a_offset, mask=a_mask, other=0.0)
        b_block = tl.load(mat_b + b_offset, mask=b_mask, other=0.0)
        # tle.dsa.ascend.compile_hint(a_block, "hivm.multi_buffer", 2)
        # tle.dsa.ascend.compile_hint(b_block, "hivm.multi_buffer", 2)
        # Hint that we only need to pad the k-dimension for Dot.
        tle.dsa.ascend.compile_hint(a_block, "dot_pad_only_k")
        tle.dsa.ascend.compile_hint(b_block, "dot_pad_only_k")

        acc = tl.dot(a_block, b_block, acc)

    c_offset = (m_start + tl.arange(0, BLOCK_M))[:, None] * N + (n_start + tl.arange(0, BLOCK_N))[None, :]
    c_mask = ((m_start + tl.arange(0, BLOCK_M))[:, None] < M) & ((n_start + tl.arange(0, BLOCK_N))[None, :] < N)
    tl.store(mat_c + c_offset, acc.to(tl.float16), mask=c_mask)


def test_dot_pad_only_k():
    M = 2048
    K = 4
    N = 16384
    BLOCK_M = 128
    BLOCK_N = 128
    BLOCK_K = 4
    dtype = torch.float16

    mat_a = torch.randn((M, K), dtype=dtype).npu()
    mat_b = torch.randn((K, N), dtype=dtype).npu()
    ref = torch.zeros((M, N), dtype=dtype).npu()

    num_blocks_m = (M + BLOCK_M - 1) // BLOCK_M
    num_blocks_n = (N + BLOCK_N - 1) // BLOCK_N
    grid = (num_blocks_m * num_blocks_n, )

    print("\n" + "=" * 60)
    print("Test: test_dot_pad_only_k")
    print(f"      A({M},{K}) * B({K},{N}), BLOCK_K={BLOCK_K}")
    print("=" * 60)

    # Baseline: reference kernel without any hints
    matmul_ref[grid](mat_a, mat_b, ref, M, N, K, BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K)
    # Hint kernel: uses dot_pad_only_k to only pad the k-dimension for Dot.
    mat_c = torch.zeros((M, N), dtype=dtype).npu()
    triton_dot_pad_only_k[grid](mat_a, mat_b, mat_c, M, N, K, BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K)
    # print(mat_c)
    torch.testing.assert_close(mat_c, ref, rtol=1e-2, atol=1e-2)
    print("PASSED result check for test_dot_pad_only_k")

    # Benchmark: compare hint vs no-hint
    ref_time = do_bench_npu(
        lambda: matmul_ref[grid]
        (mat_a, mat_b, ref, M, N, K, BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K, multibuffer=False),
        clear_l2_cache=True, collect_prof=False)
    hint_time = do_bench_npu(
        lambda: triton_dot_pad_only_k[grid]
        (mat_a, mat_b, mat_c, M, N, K, BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K, multibuffer=False),
        clear_l2_cache=True, collect_prof=False)
    print(
        f"[BENCH] dot_pad_only_k (hint): {hint_time:.2f} us, (no hint): {ref_time:.2f} us, ratio={hint_time / ref_time:.3f}"
    )
    print("Note: dot_pad_only_k tells the compiler that only K needs padding (M/N are aligned),")


# ═══════════════════════════════════════════════════════════════════════════════
# hivm.tile_mix_cube_num
# ═══════════════════════════════════════════════════════════════════════════════
@triton.jit
def triton_flash_attn_ref(Q_ptr, K_ptr, V_ptr, O_ptr, stride_q_m, stride_q_k, stride_k_n, stride_k_k, stride_v_n,
                          stride_v_d, stride_o_m, stride_o_d, M: tl.constexpr, N: tl.constexpr, K: tl.constexpr,
                          D: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
                          BLOCK_D: tl.constexpr):
    pid = tl.program_id(0)
    num_blocks_d = triton.cdiv(D, BLOCK_D)
    m_block = pid // num_blocks_d
    d_block = pid % num_blocks_d

    m_start = m_block * BLOCK_M
    d_start = d_block * BLOCK_D
    n_start = 0

    # ── First matmul: Q_block @ K_block^T  ──
    s_acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k_start in range(0, K, BLOCK_K):
        q_off = (m_start + tl.arange(0, BLOCK_M))[:, None] * stride_q_m + \
                (k_start + tl.arange(0, BLOCK_K))[None, :] * stride_q_k
        q_mask = ((m_start + tl.arange(0, BLOCK_M))[:, None] < M) & \
                 ((k_start + tl.arange(0, BLOCK_K))[None, :] < K)
        q_block = tl.load(Q_ptr + q_off, mask=q_mask, other=0.0)

        k_off = (k_start + tl.arange(0, BLOCK_K))[:, None] * stride_k_k + \
                (n_start + tl.arange(0, BLOCK_N))[None, :] * stride_k_n
        k_mask = ((k_start + tl.arange(0, BLOCK_K))[:, None] < K) & \
                 ((n_start + tl.arange(0, BLOCK_N))[None, :] < N)
        k_block = tl.load(K_ptr + k_off, mask=k_mask, other=0.0)

        s_acc = tl.dot(q_block, k_block, s_acc)

    # ── Vector ops between cubes (simulate softmax)  ──
    s_max = tl.max(s_acc, axis=1)[:, None]
    s_exp = tl.exp(s_acc - s_max)
    s_sum = tl.sum(s_exp, axis=1)[:, None]
    p_block = s_exp / s_sum

    # ── Second matmul: P_block @ V_block  (NO hint)  ──
    o_acc = tl.zeros((BLOCK_M, BLOCK_D), dtype=tl.float32)
    for n_start in range(0, N, BLOCK_N):
        v_off = (n_start + tl.arange(0, BLOCK_N))[:, None] * stride_v_n + \
                (d_start + tl.arange(0, BLOCK_D))[None, :] * stride_v_d
        v_mask = ((n_start + tl.arange(0, BLOCK_N))[:, None] < N) & \
                 ((d_start + tl.arange(0, BLOCK_D))[None, :] < D)
        v_block = tl.load(V_ptr + v_off, mask=v_mask, other=0.0)
        o_acc = tl.dot(p_block.to(tl.float16), v_block, o_acc)

    o_off = (m_start + tl.arange(0, BLOCK_M))[:, None] * stride_o_m + \
            (d_start + tl.arange(0, BLOCK_D))[None, :] * stride_o_d
    o_mask = ((m_start + tl.arange(0, BLOCK_M))[:, None] < M) & \
             ((d_start + tl.arange(0, BLOCK_D))[None, :] < D)
    tl.store(O_ptr + o_off, o_acc.to(tl.float16), mask=o_mask)


@triton.jit
def triton_flash_attn_tile_mix_cube_num(Q_ptr, K_ptr, V_ptr, O_ptr, stride_q_m, stride_q_k, stride_k_n, stride_k_k,
                                        stride_v_n, stride_v_d, stride_o_m, stride_o_d, M: tl.constexpr,
                                        N: tl.constexpr, K: tl.constexpr, D: tl.constexpr, BLOCK_M: tl.constexpr,
                                        BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr, BLOCK_D: tl.constexpr):
    pid = tl.program_id(0)
    num_blocks_d = triton.cdiv(D, BLOCK_D)
    m_block = pid // num_blocks_d
    d_block = pid % num_blocks_d

    m_start = m_block * BLOCK_M
    d_start = d_block * BLOCK_D
    n_start = 0

    # ── First matmul: Q_block @ K_block^T  (cube)  ──
    s_acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k_start in range(0, K, BLOCK_K):
        q_off = (m_start + tl.arange(0, BLOCK_M))[:, None] * stride_q_m + \
                (k_start + tl.arange(0, BLOCK_K))[None, :] * stride_q_k
        q_mask = ((m_start + tl.arange(0, BLOCK_M))[:, None] < M) & \
                 ((k_start + tl.arange(0, BLOCK_K))[None, :] < K)
        q_block = tl.load(Q_ptr + q_off, mask=q_mask, other=0.0)

        k_off = (k_start + tl.arange(0, BLOCK_K))[:, None] * stride_k_k + \
                (n_start + tl.arange(0, BLOCK_N))[None, :] * stride_k_n
        k_mask = ((k_start + tl.arange(0, BLOCK_K))[:, None] < K) & \
                 ((n_start + tl.arange(0, BLOCK_N))[None, :] < N)
        k_block = tl.load(K_ptr + k_off, mask=k_mask, other=0.0)

        s_acc = tl.dot(q_block, k_block, s_acc)
        tle.dsa.ascend.compile_hint(s_acc, "hivm.tile_mix_cube_num", 4)

    # ── Vector ops between cubes (simulate softmax)  ──
    s_max = tl.max(s_acc, axis=1)[:, None]
    s_exp = tl.exp(s_acc - s_max)
    s_sum = tl.sum(s_exp, axis=1)[:, None]
    p_block = s_exp / s_sum

    # ── Second matmul: P_block @ V_block  (cube, WITH hint)  ──
    o_acc = tl.zeros((BLOCK_M, BLOCK_D), dtype=tl.float32)
    # Hint: enable sub-tiling on the second cube to avoid L1 overflow

    for n_start in range(0, N, BLOCK_N):
        v_off = (n_start + tl.arange(0, BLOCK_N))[:, None] * stride_v_n + \
                (d_start + tl.arange(0, BLOCK_D))[None, :] * stride_v_d
        v_mask = ((n_start + tl.arange(0, BLOCK_N))[:, None] < N) & \
                 ((d_start + tl.arange(0, BLOCK_D))[None, :] < D)
        v_block = tl.load(V_ptr + v_off, mask=v_mask, other=0.0)

        o_acc = tl.dot(p_block.to(tl.float16), v_block, o_acc)
        tle.dsa.ascend.compile_hint(o_acc, "hivm.tile_mix_cube_num", 4)


    o_off = (m_start + tl.arange(0, BLOCK_M))[:, None] * stride_o_m + \
            (d_start + tl.arange(0, BLOCK_D))[None, :] * stride_o_d
    o_mask = ((m_start + tl.arange(0, BLOCK_M))[:, None] < M) & \
             ((d_start + tl.arange(0, BLOCK_D))[None, :] < D)
    tl.store(O_ptr + o_off, o_acc.to(tl.float16), mask=o_mask)


def test_tile_mix_cube_num():
    M, N, K, D = 2048, 2048, 2048, 2048
    BLOCK_M, BLOCK_N, BLOCK_K, BLOCK_D = 64, 64, 128, 128
    dtype = torch.float16

    q = torch.randn((M, K), dtype=dtype).npu()
    k = torch.randn((N, K), dtype=dtype).npu()
    v = torch.randn((N, D), dtype=dtype).npu()
    ref = torch.zeros((M, D), dtype=dtype).npu()

    num_blocks_m = (M + BLOCK_M - 1) // BLOCK_M
    num_blocks_d = (D + BLOCK_D - 1) // BLOCK_D
    grid = (num_blocks_m * num_blocks_d, )

    print("\n" + "=" * 60)
    print("Test: tile_mix_cube_num (cube->vector->cube pattern)")
    print(f"      Q({M},{K}) @ K^T({K},{N}) → softmax → P @ V({N},{D})")
    print(f"      BLOCK_M={BLOCK_M} BLOCK_N={BLOCK_N} BLOCK_K={BLOCK_K} BLOCK_D={BLOCK_D}")
    print("=" * 60)

    # Reference: no hint (expected to overflow/be slower)
    triton_flash_attn_ref[grid](
        q,
        k,
        v,
        ref,
        q.stride(0),
        q.stride(1),
        k.stride(0),
        k.stride(1),
        v.stride(0),
        v.stride(1),
        ref.stride(0),
        ref.stride(1),
        M,
        N,
        K,
        D,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_K=BLOCK_K,
        BLOCK_D=BLOCK_D,
    )
    # print(ref)

    # Hint kernel: tile_mix_cube_num on second cube
    out = torch.zeros((M, D), dtype=dtype).npu()
    triton_flash_attn_tile_mix_cube_num[grid](
        q,
        k,
        v,
        out,
        q.stride(0),
        q.stride(1),
        k.stride(0),
        k.stride(1),
        v.stride(0),
        v.stride(1),
        out.stride(0),
        out.stride(1),
        M,
        N,
        K,
        D,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_K=BLOCK_K,
        BLOCK_D=BLOCK_D,
    )
    # print(out)
    torch.testing.assert_close(out, ref, rtol=1e-2, atol=1e-2)
    print("PASSED result check for tile_mix_cube_num")

    # Benchmark (multibuffer=True to stress L1 and expose tile_mix_cube_num benefit)
    ref_time = do_bench_npu(
        lambda: triton_flash_attn_ref[grid](
            q,
            k,
            v,
            ref,
            q.stride(0),
            q.stride(1),
            k.stride(0),
            k.stride(1),
            v.stride(0),
            v.stride(1),
            ref.stride(0),
            ref.stride(1),
            M,
            N,
            K,
            D,
            BLOCK_M=BLOCK_M,
            BLOCK_N=BLOCK_N,
            BLOCK_K=BLOCK_K,
            BLOCK_D=BLOCK_D,
            multibuffer=True,
        ),
        clear_l2_cache=True,
        collect_prof=False,
    )
    hint_time = do_bench_npu(
        lambda: triton_flash_attn_tile_mix_cube_num[grid](
            q,
            k,
            v,
            out,
            q.stride(0),
            q.stride(1),
            k.stride(0),
            k.stride(1),
            v.stride(0),
            v.stride(1),
            out.stride(0),
            out.stride(1),
            M,
            N,
            K,
            D,
            BLOCK_M=BLOCK_M,
            BLOCK_N=BLOCK_N,
            BLOCK_K=BLOCK_K,
            BLOCK_D=BLOCK_D,
            multibuffer=True,
        ),
        clear_l2_cache=True,
        collect_prof=False,
    )
    print(
        f"[BENCH] tile_mix_cube_num (hint): {hint_time:.2f} us, (no hint): {ref_time:.2f} us, ratio={hint_time / ref_time:.3f}"
    )


# ═══════════════════════════════════════════════════════════════════════════════
# disable_bubble_up
# ═══════════════════════════════════════════════════════════════════════════════
@triton.jit
def triton_rms_slice_ref(x_ptr, w_ptr, out_ptr, M: tl.constexpr, N: tl.constexpr, BLOCK_SIZE: tl.constexpr,
                         eps: tl.constexpr):
    pid = tl.program_id(0)
    cols = tl.arange(0, BLOCK_SIZE)

    x = tl.load(x_ptr + pid * N + cols)
    var = tl.sum(x * x, axis=0) * (1.0 / N)
    rrms = tl.rsqrt(var + eps)
    w = tl.load(w_ptr + cols)
    y = (x * rrms * w).to(out_ptr.dtype.element_ty)

    # Extract_slice each row → store (simulating KV cache scatter-write)
    for i in tl.static_range(N):
        value_reload = tle.dsa.extract_slice(y, (i, ), (1, ), (1, ))
        offs = pid * N + i + tl.arange(0, 1)
        tl.store(out_ptr + offs, value_reload)


@triton.jit
def triton_rms_slice_disable_bubble_up(x_ptr, w_ptr, out_ptr, M: tl.constexpr, N: tl.constexpr,
                                       BLOCK_SIZE: tl.constexpr, eps: tl.constexpr):
    pid = tl.program_id(0)
    cols = tl.arange(0, BLOCK_SIZE)

    x = tl.load(x_ptr + pid * N + cols)
    var = tl.sum(x * x, axis=0) * (1.0 / N)
    rrms = tl.rsqrt(var + eps)
    w = tl.load(w_ptr + cols)
    y = (x * rrms * w).to(out_ptr.dtype.element_ty)

    # Extract_slice each row with disable_bubble_up hint
    for i in tl.static_range(N):
        value_reload = tle.dsa.extract_slice(y, (i, ), (1, ), (1, ))
        tl.compile_hint(value_reload, "disable_bubble_up")
        offs = pid * N + i + tl.arange(0, 1)
        tl.store(out_ptr + offs, value_reload)


def test_disable_bubble_up():
    M, N = 64, 128
    BLOCK_SIZE = 128
    eps = 1e-5
    dtype = torch.float16

    x = torch.randn((M, N), dtype=dtype).npu()
    w = torch.randn((N, ), dtype=dtype).npu()
    ref = torch.zeros((M, N), dtype=dtype).npu()
    grid = (M, )

    print("\n" + "=" * 60)
    print("Test: disable_bubble_up (vector -> extract_slice -> store)")
    print(f"      x({M},{N}) * w({N}), BLOCK_SIZE={BLOCK_SIZE}")
    print("=" * 60)

    # Reference: no hint
    triton_rms_slice_ref[grid](x, w, ref, M, N, BLOCK_SIZE, eps)

    # Hint kernel: disable_bubble_up on each extract_slice
    out = torch.zeros((M, N), dtype=dtype).npu()
    triton_rms_slice_disable_bubble_up[grid](x, w, out, M, N, BLOCK_SIZE, eps)

    torch.testing.assert_close(out, ref, rtol=1e-2, atol=1e-2)
    print("PASSED result check for disable_bubble_up")

    # Benchmark
    ref_time = do_bench_npu(
        lambda: triton_rms_slice_ref[grid](
            x,
            w,
            ref,
            M,
            N,
            BLOCK_SIZE,
            eps,
            multibuffer=False,
        ),
        clear_l2_cache=True,
        collect_prof=False,
    )
    hint_time = do_bench_npu(
        lambda: triton_rms_slice_disable_bubble_up[grid](
            x,
            w,
            out,
            M,
            N,
            BLOCK_SIZE,
            eps,
            multibuffer=False,
        ),
        clear_l2_cache=True,
        collect_prof=False,
    )
    print(
        f"[BENCH] disable_bubble_up (hint): {hint_time:.2f} us, (no hint): {ref_time:.2f} us, ratio={hint_time / ref_time:.3f}"
    )


# ═══════════════════════════════════════════════════════════════════════════════
# bitwise_mask
# ═══════════════════════════════════════════════════════════════════════════════
@triton.jit
def triton_where_ref(in_ptr0, in_ptr1, cond_ptr, out_ptr0, xnumel, XBLOCK: tl.constexpr, XBLOCK_SUB: tl.constexpr):
    """tl.where without bitwise_mask hint."""
    xoffset = tl.program_id(0) * XBLOCK
    for xoffset_sub in range(0, XBLOCK, XBLOCK_SUB):
        xindex = xoffset + xoffset_sub + tl.arange(0, XBLOCK_SUB)[:]
        xmask = xindex < xnumel
        in0 = tl.load(in_ptr0 + xindex, xmask)
        in1 = tl.load(in_ptr1 + xindex, xmask)
        cond = tl.load(cond_ptr + xindex, xmask)
        res = tl.where(cond, in1, in0)
        tl.store(out_ptr0 + xindex, res, xmask)


@triton.jit
def triton_where_bitwise_mask(in_ptr0, in_ptr1, cond_ptr, out_ptr0, xnumel, XBLOCK: tl.constexpr,
                              XBLOCK_SUB: tl.constexpr):
    """tl.where WITH bitwise_mask hint on the condition."""
    xoffset = tl.program_id(0) * XBLOCK
    for xoffset_sub in range(0, XBLOCK, XBLOCK_SUB):
        xindex = xoffset + xoffset_sub + tl.arange(0, XBLOCK_SUB)[:]
        xmask = xindex < xnumel
        in0 = tl.load(in_ptr0 + xindex, xmask)
        in1 = tl.load(in_ptr1 + xindex, xmask)
        cond = tl.load(cond_ptr + xindex, xmask)
        tl.compile_hint(cond, "bitwise_mask")
        res = tl.where(cond, in1, in0)
        tl.store(out_ptr0 + xindex, res, xmask)


def test_bitwise_mask():
    XBLOCK = 10240
    XBLOCK_SUB = 1024
    xnumel = 40960
    dtype = torch.float16

    in0 = torch.randn((xnumel, ), dtype=dtype).npu()
    in1 = torch.randn((xnumel, ), dtype=dtype).npu()
    cond = torch.randint(0, 2, (xnumel, ), dtype=torch.bool).npu()
    ref = torch.zeros((xnumel, ), dtype=dtype).npu()
    grid = ((xnumel + XBLOCK - 1) // XBLOCK, )

    print("\n" + "=" * 60)
    print("Test: bitwise_mask (tl.where with bitwise mask hint)")
    print(f"      xnumel={xnumel}, XBLOCK={XBLOCK}, XBLOCK_SUB={XBLOCK_SUB}")
    print("=" * 60)

    # Reference: no hint
    triton_where_ref[grid](in0, in1, cond, ref, xnumel, XBLOCK, XBLOCK_SUB)
    # Hint kernel: bitwise_mask on condition
    out = torch.zeros((xnumel, ), dtype=dtype).npu()
    triton_where_bitwise_mask[grid](in0, in1, cond, out, xnumel, XBLOCK, XBLOCK_SUB)

    torch.testing.assert_close(out, ref, rtol=1e-3, atol=1e-3)
    print("PASSED result check for bitwise_mask")

    # Benchmark
    ref_time = do_bench_npu(
        lambda: triton_where_ref[grid](
            in0,
            in1,
            cond,
            ref,
            xnumel,
            XBLOCK,
            XBLOCK_SUB,
            multibuffer=False,
        ),
        clear_l2_cache=True,
        collect_prof=False,
    )
    hint_time = do_bench_npu(
        lambda: triton_where_bitwise_mask[grid](
            in0,
            in1,
            cond,
            out,
            xnumel,
            XBLOCK,
            XBLOCK_SUB,
            multibuffer=False,
        ),
        clear_l2_cache=True,
        collect_prof=False,
    )
    print(
        f"[BENCH] bitwise_mask (hint): {hint_time:.2f} us, (no hint): {ref_time:.2f} us, ratio={hint_time / ref_time:.3f}"
    )


if __name__ == "__main__":
    test_may_discrete_ub_overflow()
    test_dot_pad_only_k()
    test_tile_mix_cube_num()
    test_disable_bubble_up()
    test_bitwise_mask()
