import torch
import torch_npu
import triton
import triton.language as tl
import triton.runtime.driver as driver


# get device properties of npu
def get_npu_properties():
    device = torch.npu.current_device()
    return driver.active.utils.get_device_properties(device)


@triton.autotune(
configs=[
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 256, "BLOCK_K": 256, "BLOCK_TRESHHOLD": 4}),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 256, "BLOCK_K": 256, "BLOCK_TRESHHOLD": 5}),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 256, "BLOCK_K": 256, "BLOCK_TRESHHOLD": 6}),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 256, "BLOCK_K": 256, "BLOCK_TRESHHOLD": 7}),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 256, "BLOCK_K": 256, "BLOCK_TRESHHOLD": 8}),
        triton.Config({"BLOCK_M": 256, "BLOCK_N": 128, "BLOCK_K": 256, "BLOCK_TRESHHOLD": 4}),
        triton.Config({"BLOCK_M": 256, "BLOCK_N": 128, "BLOCK_K": 256, "BLOCK_TRESHHOLD": 5}),
        triton.Config({"BLOCK_M": 256, "BLOCK_N": 128, "BLOCK_K": 256, "BLOCK_TRESHHOLD": 6}),
        triton.Config({"BLOCK_M": 256, "BLOCK_N": 128, "BLOCK_K": 256, "BLOCK_TRESHHOLD": 7}),
        triton.Config({"BLOCK_M": 256, "BLOCK_N": 128, "BLOCK_K": 256, "BLOCK_TRESHHOLD": 8}),
    ],
    key=["M", "N", "K"]
)
@triton.jit
def matmul_kernel(
        mat_a, mat_b, mat_c,
        M: tl.constexpr,
        N: tl.constexpr,
        K: tl.constexpr,
        num_cores: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_K: tl.constexpr,
        BLOCK_TRESHHOLD: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    task_m_idx = 0
    task_n_idx = 0

    '''
    水平分核方式每个任务块编号如下
    [0,  1,  2,  3,  4,  5,  6,  7]
    [8,  9,  10, 11, 12, 13, 14, 15]
    [16, 17, 18, 19, 20, 21, 22, 23]
    [24, 25, 26, 27, 28, 29, 30, 31]
    [32, 33, 34, 35, 36, 37, 38, 39]
    [40, 41, 42, 43, 44, 45, 46, 47]
    [48, 49, 50, 51, 52, 53, 54, 55]
    [56, 57, 58, 59, 60, 61, 62, 63]
    0核处理 0 20 40 60 4块任务
    1核处理 1 21 41 61 4块任务
    2核处理 2 22 42 62 4块任务
    ...
    19核处理 19 39 59 3块任务
    
    大shape下如果使用传统水平分核方式,会有如下问题
    1:同一时间大量核心需要访问同一块左矩阵内存,产生Bank冲突,导致硬件访问效率降低
    2:当完成一整行mat_c运算时,已经将所有右矩阵数据全部使用上,右矩阵较大时会超过L2Cache的容量上限,
      从而导致L2Cache的搬入及换出,此后每行运算都会或多或少产生CacheMiss,导致L2Cche命中率较低,影响
      算子执行效率
    此处使用8 * 8对角线分核方式可以按8 * 8的方块沿对角线方向分核计算,可以很大程度优化上面两点。

    此处以8*8对角线分核为例,实际以BLOCK_TRESHHOLD为tune参数选择最优的阈值
    8 * 8 对角线分核方式中,每8 * 8分格内任务块编号如下
    [0,  8,  16, 24, 32, 40, 48, 56]
    [57, 1,  9,  17, 25, 33, 41, 49]
    [50, 58, 2,  10, 18, 26, 34, 42]
    [43, 51, 59, 3,  11, 19, 27, 35]
    [36, 44, 52, 60, 4,  12, 20, 28]
    [29, 37, 45, 53, 61, 5,  13, 21]
    [22, 30, 38, 46, 54, 62, 6,  14]
    [15, 23, 31, 39, 47, 55, 63, 7]
    
    M轴方向超过8个基本块时,使用对角线分核可以明显减小Bank冲突 
    当右矩阵大小超过L2Cache大小时,采取对角线分核可以提升L2Cache利用率
    所以当矩阵在M和N方向均超过8块时使能对角线分核即可有优化,当右矩阵大小超过L2Cache大小时优化效果尤为明显
    '''
    NUM_BLOCKS_M = triton.cdiv(M, BLOCK_M)
    NUM_BLOCKS_N = triton.cdiv(N, BLOCK_N)
    NUM_BLOCKS = NUM_BLOCKS_M * NUM_BLOCKS_N
    #当任务量较多时，可以使能对角线分核策略进行优化
    if NUM_BLOCKS_M >= BLOCK_TRESHHOLD and NUM_BLOCKS_N >= BLOCK_TRESHHOLD:
        for block_idx in range (
            pid, NUM_BLOCKS, num_cores
        ):
            #8 * 8 对角线分核代码实现 
            curThresholdM = BLOCK_TRESHHOLD if block_idx < (NUM_BLOCKS_M // BLOCK_TRESHHOLD * BLOCK_TRESHHOLD) * NUM_BLOCKS_N else NUM_BLOCKS_M % BLOCK_TRESHHOLD
            curThresholdM_thresholdN = curThresholdM * BLOCK_TRESHHOLD
            curThresholdN = BLOCK_TRESHHOLD if block_idx % (NUM_BLOCKS_N * BLOCK_TRESHHOLD) < (curThresholdM * NUM_BLOCKS_N) // curThresholdM_thresholdN * curThresholdM_thresholdN else NUM_BLOCKS_N % BLOCK_TRESHHOLD
            localRelativeBlock = block_idx % (BLOCK_TRESHHOLD * NUM_BLOCKS_N) % (BLOCK_TRESHHOLD * curThresholdM)
            task_m_idx = localRelativeBlock % curThresholdM + block_idx // (BLOCK_TRESHHOLD * NUM_BLOCKS_N) * BLOCK_TRESHHOLD
            #求最小公倍数，方便求基本块的坐标
            x, y = curThresholdM, curThresholdN if curThresholdM > curThresholdN else curThresholdN, curThresholdM
            while y != 0:
                x, y = y, x % y
            lcm = curThresholdM * curThresholdN // x
            task_n_idx = (localRelativeBlock + (localRelativeBlock // lcm)) % curThresholdN + block_idx % (BLOCK_TRESHHOLD * NUM_BLOCKS_N) // curThresholdM_thresholdN * BLOCK_TRESHHOLD
            
            m_start = task_m_idx * BLOCK_M
            n_start = task_n_idx * BLOCK_N
            
            mat_c_block = tl.zeros((BLOCK_M, BLOCK_N),dtype = tl.float32)
            for k_start in range(0, K, BLOCK_K):
                mat_a_offset = ((m_start + tl.arange(0, BLOCK_M)) * K)[:, None] + (
                    k_start + tl.arange(0, BLOCK_K)
                )[None, :]
                mat_a_mask = ((m_start + tl.arange(0, BLOCK_M)) < M)[:, None] & (
                    (k_start + tl.arange(0, BLOCK_K)) < K
                )[None, :]
                mat_a_block = tl.load(mat_a + mat_a_offset, mask = mat_a_mask, other = 0.0)  # @hint: dot_pad_only_k
                mat_b_offset = ((k_start + tl.arange(0, BLOCK_K)) * N)[:, None] + ( 
                    n_start + tl.arange(0, BLOCK_N)
                )[None, :]
                mat_b_mask = ((k_start + tl.arange(0, BLOCK_K)) < K)[:, None] & (
                    (n_start + tl.arange(0, BLOCK_N)) < N
                )[None, :]
                mat_b_block = tl.load(mat_b + mat_b_offset, mask = mat_b_mask, other = 0.0)  # @hint: dot_pad_only_k
                mat_c_block = tl.dot(mat_a_block, mat_b_block, mat_c_block)
            mat_c_offset = ((m_start + tl.arange(0, BLOCK_M)) * N)[:, None] + (
                n_start + tl.arange(0, BLOCK_N)
            )[None, :]
            mat_c_mask = ((m_start + tl.arange(0, BLOCK_M)) < M)[:, None] & (
                (n_start + tl.arange(0, BLOCK_N)) < N
            )[None, :]
            tl.store(mat_c + mat_c_offset, mat_c_block.to(tl.bfloat16), mask = mat_c_mask)
    else:
        #传统顺序分核
        for block_idx in range (
            pid, NUM_BLOCKS, num_cores
        ):
            task_m_idx = block_idx // NUM_BLOCKS_N
            task_n_idx = block_idx % NUM_BLOCKS_N
            m_start = task_m_idx * BLOCK_M
            n_start = task_n_idx * BLOCK_N
            
            mat_c_block = tl.zeros((BLOCK_M, BLOCK_N),dtype = tl.float32)
            for k_start in range(0, K, BLOCK_K):
                mat_a_offset = ((m_start + tl.arange(0, BLOCK_M)) * K)[:, None] + (
                    k_start + tl.arange(0, BLOCK_K)
                )[None, :]
                mat_a_mask = ((m_start + tl.arange(0, BLOCK_M)) < M)[:, None] & (
                    (k_start + tl.arange(0, BLOCK_K)) < K
                )[None, :]
                mat_a_block = tl.load(mat_a + mat_a_offset, mask = mat_a_mask, other = 0.0)  # @hint: dot_pad_only_k
                mat_b_offset = ((k_start + tl.arange(0, BLOCK_K)) * N)[:, None] + ( 
                    n_start + tl.arange(0, BLOCK_N)
                )[None, :]
                mat_b_mask = ((k_start + tl.arange(0, BLOCK_K)) < K)[:, None] & (
                    (n_start + tl.arange(0, BLOCK_N)) < N
                )[None, :]
                mat_b_block = tl.load(mat_b + mat_b_offset, mask = mat_b_mask, other = 0.0)  # @hint: dot_pad_only_k
                mat_c_block = tl.dot(mat_a_block, mat_b_block, mat_c_block)
            mat_c_offset = ((m_start + tl.arange(0, BLOCK_M)) * N)[:, None] + (
                n_start + tl.arange(0, BLOCK_N)
            )[None, :]
            mat_c_mask = ((m_start + tl.arange(0, BLOCK_M)) < M)[:, None] & (
                (n_start + tl.arange(0, BLOCK_N)) < N
            )[None, :]
            tl.store(mat_c + mat_c_offset, mat_c_block.to(tl.bfloat16), mask = mat_c_mask)

def triton_matmul(
    mat_a,
    mat_b,
):
    m = mat_a.shape[0]
    k = mat_a.shape[1]
    n = mat_b.shape[1]
    mat_c = torch.empty(m, n, dtype=mat_a.dtype, device=mat_a.device)
    
    '''
    NPU芯片更加亲和512B对齐场景,如下分块通用性能较好,可以使用autotune选取最优
    BLOCK_M = 128
    BLOCK_N = 256
    BLOCK_K = 256    
    '''

    num_cores = get_npu_properties()["num_aicore"]

    matmul_kernel[(num_cores,)] (
        mat_a,
        mat_b,
        mat_c,
        m,
        n,
        k,
        num_cores
    )
    return mat_c


if __name__ == "__main__":
    M = 2048
    K = 7168
    N = 16384
    
    mat_a = torch.randn([M, K], dtype = torch.bfloat16, device = "npu")
    mat_b = torch.randn([K, N], dtype = torch.bfloat16, device = "npu")
    
    result = triton_matmul(mat_a, mat_b)
    golden = torch.matmul(mat_a, mat_b)
    
    mask = golden.abs() < 1.0
    tmpatol = tmprtol = 2 ** -6
    try:
        torch.testing.assert_close(result[mask], golden[mask], atol = tmpatol, rtol = 0)
        torch.testing.assert_close(result[~mask], golden[~mask], atol = 0, rtol = tmprtol)
        print("run matmul success")
    except:
        print(f"[ERROR] M={M} ,K={K}, N={N}存在精度问题")