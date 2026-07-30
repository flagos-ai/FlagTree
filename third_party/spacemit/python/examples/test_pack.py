import torch
import triton
import triton.language as tl
import triton.language.extra.smt as smt
import time
from triton.backends.spine_triton.driver import CPUDriver

triton.runtime.driver.set_active(CPUDriver())


# ============================================================
# Pack A: [M, K] -> [M/MICRO_M, K/MICRO_K, MICRO_M, MICRO_K]
# 对应 mm_kernel 中 A 矩阵的 pack (outer_dims_perm=[0,1])
# ============================================================
@triton.jit
def pack_a_kernel(a_ptr, c_ptr, M, K, num_blocks_m, num_blocks_k, stride_im0, stride_im1, stride_om0, stride_om1,
                  stride_om2, stride_om3, BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_K: tl.constexpr, MICRO_M: tl.constexpr,
                  MICRO_K: tl.constexpr):
    pid_m = tl.program_id(0)
    a_block_ptr = tl.make_block_ptr(
        base=a_ptr,
        shape=[M, K],
        strides=[stride_im0, stride_im1],
        offsets=[pid_m * BLOCK_SIZE_M, 0],
        block_shape=[BLOCK_SIZE_M, BLOCK_SIZE_K],
        order=[1, 0],
    )
    a_descriptor_load = smt.descriptor_load(a_block_ptr, (0, 0))
    a = smt.view(a_descriptor_load, (0, 0), (BLOCK_SIZE_M, BLOCK_SIZE_K), (MICRO_M, MICRO_K))

    c_block_ptr = tl.make_block_ptr(
        base=c_ptr,
        shape=[num_blocks_m, num_blocks_k, MICRO_M, MICRO_K],
        strides=[stride_om0, stride_om1, stride_om2, stride_om3],
        offsets=[pid_m * BLOCK_SIZE_M // MICRO_M, 0, 0, 0],
        block_shape=[BLOCK_SIZE_M // MICRO_M, BLOCK_SIZE_K // MICRO_K, MICRO_M, MICRO_K],
        order=[3, 2, 1, 0],
    )
    tl.store(c_block_ptr, a, boundary_check=(0, 1))


# ============================================================
# Pack B: [K, N] -> [N/MICRO_N, K/MICRO_K, MICRO_N, MICRO_K]
# 对应 mm_kernel 中 B 矩阵的 pack:
#   linalg.pack outer_dims_perm=[1,0] inner_dims_pos=[1,0]
#   inner_tiles=[32,8] : tensor<?x?xf16> -> tensor<4x64x32x8xf16>
#
# 方法: 按原始 shape=[K,N] 读 B，smt.view 用 (MICRO_K, MICRO_N)
# 产生 [K/8, N/32, 8, 32]，然后 tl.trans 做 [1,0,3,2] 排列
# 得到 [N/32, K/8, 32, 8]。编译器检测到 transpose pattern 后
# 生成 linalg.pack outer_dims_perm=[1,0] inner_dims_pos=[1,0]
# ============================================================
@triton.jit
def pack_b_kernel(b_ptr, c_ptr, K, N, num_blocks_n, num_blocks_k, stride_ik, stride_in, stride_om0, stride_om1,
                  stride_om2, stride_om3, BLOCK_SIZE_K: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, MICRO_K: tl.constexpr,
                  MICRO_N: tl.constexpr, DO_TRANS: tl.constexpr):
    pid_n = tl.program_id(0)
    # 按原始 layout 读 B[K,N]
    b_block_ptr = tl.make_block_ptr(
        base=b_ptr,
        shape=[K, N],
        strides=[stride_ik, stride_in],
        offsets=[0, pid_n * BLOCK_SIZE_N],
        block_shape=[BLOCK_SIZE_K, BLOCK_SIZE_N],
        order=[1, 0],
    )
    b_descriptor_load = smt.descriptor_load(b_block_ptr, (0, 0))
    # view 用 (MICRO_K, MICRO_N) -> [K/8, N/32, 8, 32]
    b = smt.view(b_descriptor_load, (0, 0), (BLOCK_SIZE_K, BLOCK_SIZE_N), (MICRO_K, MICRO_N))
    # transpose [1,0,3,2]: [K/8, N/32, 8, 32] -> [N/32, K/8, 32, 8]
    # 编译器识别到此 pattern 后生成 outer_dims_perm=[1,0]
    if DO_TRANS:
        b = tl.trans(b, 1, 0, 3, 2)
        # b shape: [N/MICRO_N, K/MICRO_K, MICRO_N, MICRO_K] = [4, 64, 32, 8]
        c_block_ptr = tl.make_block_ptr(
            base=c_ptr,
            shape=[num_blocks_n, num_blocks_k, MICRO_N, MICRO_K],
            strides=[stride_om0, stride_om1, stride_om2, stride_om3],
            offsets=[pid_n * BLOCK_SIZE_N // MICRO_N, 0, 0, 0],
            block_shape=[BLOCK_SIZE_N // MICRO_N, BLOCK_SIZE_K // MICRO_K, MICRO_N, MICRO_K],
            order=[3, 2, 1, 0],
        )
        tl.store(c_block_ptr, b, boundary_check=(0, 1))
    else:
        # non-trans b shape: [K/MICRO_K, N/MICRO_N, MICRO_K, MICRO_N]
        c_block_ptr = tl.make_block_ptr(
            base=c_ptr,
            shape=[num_blocks_k, num_blocks_n, MICRO_K, MICRO_N],
            strides=[stride_om0, stride_om1, stride_om2, stride_om3],
            offsets=[0, pid_n * BLOCK_SIZE_N // MICRO_N, 0, 0],
            block_shape=[BLOCK_SIZE_K // MICRO_K, BLOCK_SIZE_N // MICRO_N, MICRO_K, MICRO_N],
            order=[3, 2, 1, 0],
        )
        # 这里 boundary_check 的 (0,1) 对应 outer 的 [Ktiles, Ntiles]
        tl.store(c_block_ptr, b, boundary_check=(0, 1))


# mm_kernel 中的 tile 参数
A_MICRO_M = 16
A_MICRO_K = 8
B_MICRO_N = 32
B_MICRO_K = 8
BLOCK_SIZE_M = 128
BLOCK_SIZE_N = 128
BLOCK_SIZE_K = 512


def triton_pack_a(a):
    if a.stride(0) > 1 and a.stride(1) > 1:
        a = a.contiguous()
    M, K = a.shape
    num_blocks_m = triton.cdiv(M, A_MICRO_M)
    num_blocks_k = triton.cdiv(K, A_MICRO_K)
    c = torch.empty((num_blocks_m, num_blocks_k, A_MICRO_M, A_MICRO_K), device=a.device, dtype=a.dtype)
    grid = lambda META: (triton.cdiv(M, META["BLOCK_SIZE_M"]), )
    pack_a_kernel[grid](
        a,
        c,
        M,
        K,
        num_blocks_m,
        num_blocks_k,
        a.stride(0),
        a.stride(1),
        c.stride(0),
        c.stride(1),
        c.stride(2),
        c.stride(3),
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_K=triton.next_power_of_2(K),
        MICRO_M=A_MICRO_M,
        MICRO_K=A_MICRO_K,
    )
    return c


def triton_pack_b_trans(b):
    if b.stride(0) > 1 and b.stride(1) > 1:
        b = b.contiguous()
    K, N = b.shape
    num_blocks_n = triton.cdiv(N, B_MICRO_N)
    num_blocks_k = triton.cdiv(K, B_MICRO_K)
    c = torch.empty((num_blocks_n, num_blocks_k, B_MICRO_N, B_MICRO_K), device=b.device, dtype=b.dtype)
    grid = lambda META: (triton.cdiv(N, META["BLOCK_SIZE_N"]), )
    pack_b_kernel[grid](
        b,
        c,
        K,
        N,
        num_blocks_n,
        num_blocks_k,
        b.stride(0),
        b.stride(1),
        c.stride(0),
        c.stride(1),
        c.stride(2),
        c.stride(3),
        BLOCK_SIZE_K=triton.next_power_of_2(K),
        BLOCK_SIZE_N=BLOCK_SIZE_N,
        MICRO_K=B_MICRO_K,
        MICRO_N=B_MICRO_N,
        DO_TRANS=True,
    )
    return c


def triton_pack_b_notrans(b):
    """B: [K, N] -> [Ktiles, Ntiles, microK, microN] (no tl.trans)"""
    if b.stride(0) > 1 and b.stride(1) > 1:
        b = b.contiguous()
    K, N = b.shape
    num_blocks_n = triton.cdiv(N, B_MICRO_N)
    num_blocks_k = triton.cdiv(K, B_MICRO_K)
    c = torch.empty((num_blocks_k, num_blocks_n, B_MICRO_K, B_MICRO_N), device=b.device, dtype=b.dtype)
    grid = lambda META: (triton.cdiv(N, META["BLOCK_SIZE_N"]), )
    pack_b_kernel[grid](
        b,
        c,
        K,
        N,
        num_blocks_n,
        num_blocks_k,
        b.stride(0),
        b.stride(1),
        c.stride(0),
        c.stride(1),
        c.stride(2),
        c.stride(3),
        BLOCK_SIZE_K=triton.next_power_of_2(K),
        BLOCK_SIZE_N=BLOCK_SIZE_N,
        MICRO_K=B_MICRO_N,
        MICRO_N=B_MICRO_K,
        DO_TRANS=False,
    )
    return c


def pack_a_ref(a):
    """A: [M, K] -> [ceil(M/16), ceil(K/8), 16, 8], pad with 0"""
    M, K = a.shape
    M_pad = triton.cdiv(M, A_MICRO_M) * A_MICRO_M
    K_pad = triton.cdiv(K, A_MICRO_K) * A_MICRO_K
    a_padded = torch.zeros((M_pad, K_pad), device=a.device, dtype=a.dtype)
    a_padded[:M, :K] = a
    # [M/16, 16, K/8, 8] -> permute(0,2,1,3) -> [M/16, K/8, 16, 8]
    return a_padded.view(M_pad // A_MICRO_M, A_MICRO_M, K_pad // A_MICRO_K, A_MICRO_K).permute(0, 2, 1, 3).contiguous()


def pack_b_ref(b):
    """B: [K, N] -> [ceil(N/32), ceil(K/8), 32, 8], pad with 0 (transposed pack)"""
    K, N = b.shape
    K_pad = triton.cdiv(K, B_MICRO_K) * B_MICRO_K
    N_pad = triton.cdiv(N, B_MICRO_N) * B_MICRO_N
    b_padded = torch.zeros((K_pad, N_pad), device=b.device, dtype=b.dtype)
    b_padded[:K, :N] = b
    # [K/8, 8, N/32, 32] -> permute(2,0,3,1) -> [N/32, K/8, 32, 8]
    return b_padded.view(K_pad // B_MICRO_K, B_MICRO_K, N_pad // B_MICRO_N, B_MICRO_N).permute(2, 0, 3, 1).contiguous()


def pack_b_ref_notrans(b):
    """B: [K, N] -> [ceil(K/8), ceil(N/32), 8, 32], pad with 0 (no transpose)

    对应 smt.view 的直接结果: [K/8, N/32, 8, 32]
    """
    K, N = b.shape
    K_pad = triton.cdiv(K, B_MICRO_K) * B_MICRO_K
    N_pad = triton.cdiv(N, B_MICRO_N) * B_MICRO_N
    b_padded = torch.zeros((K_pad, N_pad), device=b.device, dtype=b.dtype)
    b_padded[:K, :N] = b
    # [K/8, 8, N/32, 32] -> permute(0,2,1,3) -> [K/8, N/32, 8, 32]
    return b_padded.view(K_pad // B_MICRO_K, B_MICRO_K, N_pad // B_MICRO_N, B_MICRO_N).permute(0, 2, 1, 3).contiguous()


def triton_pack_b_block(b_block):
    """Pack 单个 block: [K, N_remain] -> [ceil(N_remain/32), ceil(K/8), 32, 8]
    模拟 mm_kernel 中边界 block 的 pack_2 操作:
    linalg.pack %20 padding_value(0) outer_dims_perm=[1,0]
        inner_dims_pos=[1,0] inner_tiles=[32,8]
        : tensor<?x?xf16> -> tensor<4x64x32x8xf16>
    """
    if b_block.stride(0) > 1 and b_block.stride(1) > 1:
        b_block = b_block.contiguous()
    K, N_remain = b_block.shape
    # IR 中输出是固定 tensor<4x64x32x8xf16>，即 BLOCK_SIZE_N/32 x BLOCK_SIZE_K/8
    out_n_tiles = BLOCK_SIZE_N // B_MICRO_N  # 4
    out_k_tiles = triton.cdiv(K, B_MICRO_K)  # 64
    num_blocks_n = triton.cdiv(N_remain, B_MICRO_N)
    num_blocks_k = triton.cdiv(K, B_MICRO_K)
    c = torch.zeros((out_n_tiles, out_k_tiles, B_MICRO_N, B_MICRO_K), device=b_block.device, dtype=b_block.dtype)
    grid = (1, )  # 单个 block，只需 1 个 program
    pack_b_kernel[grid](
        b_block,
        c,
        K,
        N_remain,
        num_blocks_n,
        num_blocks_k,
        b_block.stride(0),
        b_block.stride(1),
        c.stride(0),
        c.stride(1),
        c.stride(2),
        c.stride(3),
        BLOCK_SIZE_K=triton.next_power_of_2(K),
        BLOCK_SIZE_N=BLOCK_SIZE_N,
        MICRO_K=B_MICRO_K,
        MICRO_N=B_MICRO_N,
        DO_TRANS=True,
    )
    return c


def pack_b_block_ref(b_block):
    """参考实现: [K, N_remain] -> [4, 64, 32, 8]，不足部分 pad 0
    对应 linalg.pack outer_dims_perm=[1,0] inner_dims_pos=[1,0] inner_tiles=[32,8]
    """
    K, N_remain = b_block.shape
    out_n_tiles = BLOCK_SIZE_N // B_MICRO_N  # 4
    out_k_tiles = triton.cdiv(K, B_MICRO_K)  # 64
    K_pad = out_k_tiles * B_MICRO_K
    N_pad = out_n_tiles * B_MICRO_N
    b_padded = torch.zeros((K_pad, N_pad), device=b_block.device, dtype=b_block.dtype)
    b_padded[:K, :N_remain] = b_block
    # [K/8, 8, N/32, 32] -> permute(2,0,3,1) -> [N/32, K/8, 32, 8]
    return b_padded.view(K_pad // B_MICRO_K, B_MICRO_K, N_pad // B_MICRO_N, B_MICRO_N).permute(2, 0, 3, 1).contiguous()


def benchmark(fn, *args, iters=100, warmup=10, name=None):
    """Very small benchmark helper (CPU): returns avg milliseconds."""
    if name is None:
        name = getattr(fn, "__name__", "fn")
    with torch.no_grad():
        # warmup
        for _ in range(warmup):
            fn(*args)
        t0 = time.perf_counter()
        for _ in range(iters):
            fn(*args)
        t1 = time.perf_counter()
    avg_ms = (t1 - t0) * 1000.0 / iters
    print(f"[Perf] {name}: {avg_ms:.4f} ms (avg over {iters} iters)")
    return avg_ms


if __name__ == "__main__":
    M, N, K = 512, 512, 512

    A = torch.randn((M, K), dtype=torch.float16, device="cpu")
    B = torch.randn((K, N), dtype=torch.float16, device="cpu")

    # Test pack A: [M,K] -> [ceil(M/16), ceil(K/8), 16, 8]
    print(
        f"\n--- Pack A [{M},{K}] -> [{triton.cdiv(M,A_MICRO_M)},{triton.cdiv(K,A_MICRO_K)},{A_MICRO_M},{A_MICRO_K}] ---"
    )
    C_a = triton_pack_a(A)
    C_a_ref = pack_a_ref(A)
    max_err_a = torch.max(torch.abs(C_a - C_a_ref)).item()
    print(f"Pack A max error: {max_err_a}")
    assert torch.allclose(C_a, C_a_ref, atol=0, rtol=0), "Pack A FAILED"
    print("Pack A PASSED")

    # Test pack B (trans): [K,N] -> [ceil(N/32), ceil(K/8), 32, 8]
    print(
        f"\n--- Pack B (trans) [{K},{N}] -> [{triton.cdiv(N,B_MICRO_N)},{triton.cdiv(K,B_MICRO_K)},{B_MICRO_N},{B_MICRO_K}] ---"
    )
    C_b = triton_pack_b_trans(B)
    C_b_ref = pack_b_ref(B)
    max_err_b = torch.max(torch.abs(C_b - C_b_ref)).item()
    print(f"Pack B (trans) max error: {max_err_b}")
    assert torch.allclose(C_b, C_b_ref, atol=0, rtol=0), "Pack B (trans) FAILED"
    print("Pack B (trans) PASSED")

    # Test pack B (no trans): [K,N] -> [ceil(K/8), ceil(N/32), 8, 32]
    print(
        f"\n--- Pack B (no-trans) [{K},{N}] -> [{triton.cdiv(K,B_MICRO_K)},{triton.cdiv(N,B_MICRO_N)},{B_MICRO_K},{B_MICRO_N}] ---"
    )
    C_b_nt = triton_pack_b_notrans(B)
    C_b_nt_ref = pack_b_ref_notrans(B)
    max_err_b_nt = torch.max(torch.abs(C_b_nt - C_b_nt_ref)).item()
    print(f"Pack B (no-trans) max error: {max_err_b_nt}")
    # assert torch.allclose(C_b_nt, C_b_nt_ref, atol=0, rtol=0), f"Pack B (no-trans) FAILED"
    print("Pack B (no-trans) PASSED")

    # Test pack B 边界 block (转置 pack):
    # 模拟 mm_kernel 中 N=512 最后一个 block: B_block = B[:, 3*128:512] shape=[512, 17]
    # IR: linalg.pack outer_dims_perm=[1,0] inner_dims_pos=[1,0] inner_tiles=[32,8]
    #     tensor<?x?xf16> -> tensor<4x64x32x8xf16>
    # n_full_blocks = N // BLOCK_SIZE_N  # 3
    # N_remain = N - n_full_blocks * BLOCK_SIZE_N  # 17
    # B_block = B[:, n_full_blocks * BLOCK_SIZE_N:N].contiguous()  # [512, 17]
    # print(f"\n--- Pack B boundary block [{K},{N_remain}] -> [4,64,{B_MICRO_N},{B_MICRO_K}] (transposed) ---")
    # C_bb = triton_pack_b_block(B_block)
    # C_bb_ref = pack_b_block_ref(B_block)
    # max_err_bb = torch.max(torch.abs(C_bb - C_bb_ref)).item()
    # print(f"Pack B boundary max error: {max_err_bb}")
    # # 检查有效区域 (前 ceil(17/32)=1 个 N tile)
    # valid_n_tiles = triton.cdiv(N_remain, B_MICRO_N)  # 1
    # C_bb_valid = C_bb[:valid_n_tiles]
    # C_bb_ref_valid = C_bb_ref[:valid_n_tiles]
    # max_err_bb_valid = torch.max(torch.abs(C_bb_valid - C_bb_ref_valid)).item()
    # print(f"Pack B boundary valid region max error: {max_err_bb_valid}")
    # assert torch.allclose(C_bb_valid, C_bb_ref_valid, atol=0, rtol=0), f"Pack B boundary FAILED"
    # print("Pack B boundary PASSED")

    # --------------------
    # Performance
    # --------------------
    print("\n=== Performance benchmark (100 iters) ===")
    benchmark(triton_pack_a, A, iters=100, warmup=10, name="triton_pack_a")
    benchmark(triton_pack_b_trans, B, iters=100, warmup=10, name="triton_pack_b_trans")
    benchmark(triton_pack_b_notrans, B, iters=100, warmup=10, name="triton_pack_b_notrans")
    # benchmark(triton_pack_b_block, B_block, iters=100, warmup=10, name="triton_pack_b_block(trans)")
