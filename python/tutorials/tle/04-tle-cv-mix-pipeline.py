"""
CV Mix 手排流水示例
====================
展示Cube与Vector核心之间的流水并行：
- Cube: 矩阵乘法 (tl.dot)
- Vector: 加法操作 + 100次循环 (模拟更长计算时间)

"""

import torch
import torch_npu
import triton
import triton.language as tl
import triton.experimental.tle as tle
from triton.backends.ascend.testing import do_bench_npu

DEVICE = "npu"
DEVICE_ID = 0
torch.manual_seed(20)
torch_npu.npu.set_device(int(DEVICE_ID))
torch.set_printoptions(sci_mode=False, precision=4, linewidth=300)
pipe = tle.dsa.ascend.PIPE


@triton.jit
def cv_mix_matmul_add_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    workspace_ptr,
    stride_am,
    stride_ak,
    stride_bk,
    stride_bn,
    stride_cn,
    M: tl.constexpr,
    N: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    K: tl.constexpr,
    limit_auto_multi_buffer_only_for_local_buffer=False,
):
    """
    CV Mix流水示例 (单核心，M方向切分):
    1. Cube: 计算 C = A @ B (使用tl.dot)，输出f32
    2. Vector: 计算 D = C + 1 + 100*0.001 (模拟更长计算时间)
    """
    num_blocks = M // BLOCK_SIZE_M

    for block_idx in range(num_blocks):
        start_m = block_idx * BLOCK_SIZE_M

        # 偏移量
        offs_am = start_m + tl.arange(0, BLOCK_SIZE_M)
        offs_an = tl.arange(0, BLOCK_SIZE_N)
        offs_k = tl.arange(0, K)

        # ===== Cube计算: 矩阵乘法 =====
        a = tl.load(a_ptr + offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
        b = tl.load(b_ptr + offs_k[:, None] * stride_bk + offs_an[None, :] * stride_bn)

        # Cube矩阵乘法
        accumulator = tl.dot(a, b)

        d = accumulator + 1.0
        # 循环100次模拟更长的Vector计算时间
        for _ in range(100):
            d = d + 0.001

        # 存储最终结果 (使用全局偏移)
        tl.store(c_ptr + offs_am[:, None] * stride_cn + offs_an[None, :], d)


@triton.jit
def cv_mix_matmul_add_double_buffer_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    workspace_ptr,
    stride_am,
    stride_ak,
    stride_bk,
    stride_bn,
    stride_cn,
    M: tl.constexpr,
    N: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    K: tl.constexpr,
):
    """
    双缓冲流水: 单核心，M方向切分，实现Cube和Vector并行
    - num_blocks: M方向的block数量
    - 使用2个缓冲区交替，实现流水并行
    """
    num_blocks = M // BLOCK_SIZE_M

    # 初始化：标记两个缓冲区都可用
    tle.dsa.ascend.sync_block_set('vector', 'cube', 0, pipe.PIPE_MTE2, pipe.PIPE_FIX)
    tle.dsa.ascend.sync_block_set('vector', 'cube', 1, pipe.PIPE_MTE2, pipe.PIPE_FIX)

    for block_idx in range(num_blocks):
        start_m = block_idx * BLOCK_SIZE_M
        buffer_id = block_idx % 2
        ws_offset = buffer_id * BLOCK_SIZE_M * BLOCK_SIZE_N

        # 偏移量
        offs_am = start_m + tl.arange(0, BLOCK_SIZE_M)
        offs_an = tl.arange(0, BLOCK_SIZE_N)
        offs_k = tl.arange(0, K)

        # workspace 内的局部偏移 (始终从0开始)
        offs_m_local = tl.arange(0, BLOCK_SIZE_M)

        # ===== Cube计算: 矩阵乘法 =====
        a = tl.load(a_ptr + offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
        b = tl.load(b_ptr + offs_k[:, None] * stride_bk + offs_an[None, :] * stride_bn)

        accumulator = tl.dot(a, b)

        # 等待对应缓冲区被Vector释放
        tle.dsa.ascend.sync_block_wait("vector", "cube", buffer_id, pipe.PIPE_MTE2, pipe.PIPE_FIX)

        # 存储到workspace (f32)，使用局部偏移 + buffer偏移
        tl.store(workspace_ptr + ws_offset + offs_m_local[:, None] * BLOCK_SIZE_N + offs_an[None, :], accumulator)

        # 通知Vector可以处理
        tle.dsa.ascend.sync_block_set("cube", "vector", buffer_id, pipe.PIPE_FIX, pipe.PIPE_MTE2)

        # 等待Vector完成处理
        tle.dsa.ascend.sync_block_wait("cube", "vector", buffer_id, pipe.PIPE_FIX, pipe.PIPE_MTE2)

        # ===== Vector处理: 加载f32并计算加法（循环100次模拟更长计算时间）=====
        c_reload = tl.load(workspace_ptr + ws_offset + offs_m_local[:, None] * BLOCK_SIZE_N + offs_an[None, :])
        d = c_reload + 1.0
        # 循环100次模拟更长的Vector计算时间
        for _ in range(100):
            d = d + 0.001

        # 存储最终结果
        tl.store(c_ptr + offs_am[:, None] * stride_cn + offs_an[None, :], d)

        # 释放缓冲区
        tle.dsa.ascend.sync_block_set('vector', 'cube', buffer_id, pipe.PIPE_MTE2, pipe.PIPE_FIX)


def cv_mix_matmul_add(a: torch.Tensor, b: torch.Tensor):
    """单缓冲包装函数 (单核心): D = (A @ B) + 1.1"""
    M, K = a.shape
    K2, N = b.shape
    assert K == K2, f"Incompatible dimensions: A[{M},{K}] x B[{K2},{N}]"

    BLOCK_SIZE_M = 64
    BLOCK_SIZE_N = 128
    # 确保M是BLOCK_SIZE_M的倍数
    assert M % BLOCK_SIZE_M == 0, f"M={M} must be multiple of BLOCK_SIZE_M={BLOCK_SIZE_M}"
    assert N % BLOCK_SIZE_N == 0, f"N={N} must be multiple of BLOCK_SIZE_N={BLOCK_SIZE_N}"

    grid = (1, )  # 单核心
    workspace = torch.empty((BLOCK_SIZE_M * BLOCK_SIZE_N, ), dtype=torch.float32, device=a.device)
    c = torch.empty((M, N), dtype=torch.float32, device=a.device)

    cv_mix_matmul_add_kernel[grid](a, b, c, workspace, stride_am=a.stride(0), stride_ak=a.stride(1),
                                   stride_bk=b.stride(0), stride_bn=b.stride(1), stride_cn=c.stride(0), M=M, N=N,
                                   BLOCK_SIZE_M=BLOCK_SIZE_M, BLOCK_SIZE_N=BLOCK_SIZE_N, K=K, multibuffer=False)
    return c


def cv_mix_matmul_add_double_buffer(a: torch.Tensor, b: torch.Tensor):
    """双缓冲包装函数: 单核心，M方向切分流水"""
    M, K = a.shape
    K2, N = b.shape
    assert K == K2, f"Incompatible dimensions: A[{M},{K}] x B[{K2},{N}]"

    BLOCK_SIZE_M = 128
    BLOCK_SIZE_N = 128
    assert M % BLOCK_SIZE_M == 0, f"M={M} must be multiple of BLOCK_SIZE_M={BLOCK_SIZE_M}"
    assert N % BLOCK_SIZE_N == 0, f"N={N} must be multiple of BLOCK_SIZE_N={BLOCK_SIZE_N}"

    grid = (1, )  # 单核心
    # 双缓冲workspace (f32)
    workspace = torch.empty((2 * BLOCK_SIZE_M * BLOCK_SIZE_N, ), dtype=torch.float32, device=a.device)
    c = torch.empty((M, N), dtype=torch.float32, device=a.device)

    cv_mix_matmul_add_double_buffer_kernel[grid](a, b, c, workspace, stride_am=a.stride(0), stride_ak=a.stride(1),
                                                 stride_bk=b.stride(0), stride_bn=b.stride(1), stride_cn=c.stride(0),
                                                 M=M, N=N, BLOCK_SIZE_M=BLOCK_SIZE_M, BLOCK_SIZE_N=BLOCK_SIZE_N, K=K,
                                                 disable_auto_inject_block_sync=True, multibuffer=False)
    return c


def test_cv_mix_matmul_add():
    """测试函数"""
    DEVICE_ID = 0
    DEVICE = f"npu:{DEVICE_ID}"
    torch_npu.npu.set_device(DEVICE_ID)

    BLOCK_SIZE_M = 64
    BLOCK_SIZE_N = 128

    # 测试配置 - 确保M是BLOCK_SIZE_M的倍数，N是BLOCK_SIZE_N的倍数
    test_cases = [
        {"M": BLOCK_SIZE_M * 128, "N": BLOCK_SIZE_N * 1, "K": 128, "name": "small"},
        {"M": BLOCK_SIZE_M * 512, "N": BLOCK_SIZE_N * 1, "K": 256, "name": "medium"},
        {"M": BLOCK_SIZE_M * 1024, "N": BLOCK_SIZE_N * 1, "K": 256, "name": "big"},
    ]

    dtype = torch.float16

    print("=" * 60)
    print("CV Mix 手排流水测试 (Cube: MatMul, Vector: Add)")
    print("=" * 60)

    for case in test_cases:
        M, N, K = case["M"], case["N"], case["K"]
        print(f"\n测试用例: {case['name']} - M={M}, N={N}, K={K}")

        # 创建输入
        a = torch.randn((M, K), dtype=dtype, device=DEVICE)
        b = torch.randn((K, N), dtype=dtype, device=DEVICE)

        # PyTorch参考: (A @ B) + 1 + 100 * 0.001
        c_ref = (a @ b).to(torch.float32) + 1.1

        # ===== 单缓冲版本 =====
        print("  [单缓冲版本]")
        c_triton = None
        try:
            c_triton = cv_mix_matmul_add(a, b)
            torch.testing.assert_close(c_triton, c_ref, rtol=1e-2, atol=1e-2)
            print("    ✓ 精度验证通过")

            triton_time = do_bench_npu(lambda: cv_mix_matmul_add(a, b), clear_l2_cache=True, collect_prof=False)
            print(f"    Time: {triton_time:.4f} us")
        except Exception as e:
            print(f"    ✗ 失败: {e}")

        # ===== 双缓冲版本 =====
        print("  [双缓冲版本]")
        c_db = None
        try:
            c_db = cv_mix_matmul_add_double_buffer(a, b)
            torch.testing.assert_close(c_db, c_ref, rtol=1e-2, atol=1e-2)
            print("    ✓ 精度验证通过")

            db_time = do_bench_npu(lambda: cv_mix_matmul_add_double_buffer(a, b), clear_l2_cache=True,
                                   collect_prof=False)
            print(f"    Time: {db_time:.4f} us")
            if 'triton_time' in locals():
                speedup = triton_time / db_time
                print(f"    加速比: {speedup:.2f}x")
        except Exception as e:
            print(f"    ✗ 失败: {e}")

        # ===== 单缓冲 vs 双缓冲对比 =====
        if c_triton is not None and c_db is not None:
            print("  [单缓冲 vs 双缓冲对比]")
            try:
                torch.testing.assert_close(c_triton, c_db, rtol=1e-5, atol=1e-5)
                print("    ✓ 单缓冲和双缓冲结果一致")
            except Exception as e:
                print(f"    ✗ 单缓冲和双缓冲结果不一致: {e}")

    print("\n" + "=" * 60)
    print("测试完成!")
    print("=" * 60)


if __name__ == "__main__":
    test_cv_mix_matmul_add()
