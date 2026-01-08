import torch
import triton
import triton.language as tl

# active driver
driver = triton.runtime.driver.active
# torch.cuda, torch.aipu, torch.npu
torch_device_fn = triton.runtime.driver.active.get_device_interface()
# device
if hasattr(driver, "get_active_torch_device"):
    device = triton.runtime.driver.active.get_active_torch_device()
else:
    device = triton.runtime.driver.active.get_current_device()


@triton.jit
def bmm_kernel(
    A,
    B,
    O,
    M,
    N,
    K,
    TILE_M: tl.constexpr = 64,
    TILE_N: tl.constexpr = 64,
    TILE_K: tl.constexpr = 64,
    GROUP_M: tl.constexpr = 1,
    DIVISIBLE_M: tl.constexpr = True,
    DIVISIBLE_N: tl.constexpr = True,
    DIVISIBLE_K: tl.constexpr = True,
):
    # batch offsets
    pid_b = tl.program_id(2)
    A += pid_b * M * K
    B += pid_b * K * N
    O += pid_b * M * N

    pidx = tl.program_id(0)
    pidy = tl.program_id(1)
    if GROUP_M == 1:
        pid_m, pid_n = pidx, pidy
    else:
        # reorder CTAs
        gridx = tl.num_programs(0)
        gridy = tl.num_programs(1)
        pid = pidx + pidy * gridx

        num_CTA_per_group = gridy * GROUP_M

        group_id = pid // num_CTA_per_group
        inner_group_id = pid % num_CTA_per_group
        GROUP_SIZE = tl.where((group_id * GROUP_M + GROUP_M) > gridx, gridx % GROUP_M, GROUP_M)
        pid_m = group_id * GROUP_M + inner_group_id % GROUP_SIZE
        pid_n = inner_group_id // GROUP_SIZE

    offs_m = pid_m * TILE_M + tl.arange(0, TILE_M)
    offs_n = pid_n * TILE_N + tl.arange(0, TILE_N)
    offs_k = tl.arange(0, TILE_K)

    a_ptrs = A + offs_m[:, None] * K + offs_k[None, :]
    b_ptrs = B + offs_k[:, None] * N + offs_n[None, :]
    o_ptrs = O + offs_m[:, None] * N + offs_n[None, :]

    num_iters = tl.cdiv(K, TILE_K)
    o = tl.zeros((TILE_M, TILE_N), dtype=tl.float32)
    for i in range(num_iters):
        mask_a = offs_k[None, :] < K - i * TILE_K
        mask_b = offs_k[:, None] < K - i * TILE_K
        a = tl.load(a_ptrs, mask=mask_a)
        b = tl.load(b_ptrs, mask=mask_b)

        a_ptrs += TILE_K
        b_ptrs += TILE_K * N

        o += tl.dot(a, b, allow_tf32=False)

    mask_m = (pid_m * TILE_M + tl.arange(0, TILE_M)) < M
    mask_n = (pid_n * TILE_N + tl.arange(0, TILE_N)) < N
    mask_c = mask_m[:, None] & mask_n[None, :]
    tl.store(o_ptrs, o, mask_c)


def bmm(A, B):
    batch, M, K = A.shape
    _, _, N = B.shape
    A = A.contiguous()
    B = B.contiguous()
    out = torch.empty((batch, M, N), dtype=A.dtype, device=A.device)

    grid_fn = lambda meta: (
        triton.cdiv(meta["M"], meta["TILE_M"]),
        triton.cdiv(meta["N"], meta["TILE_N"]),
        batch,
    )
    with torch_device_fn.device(A.device):
        bmm_kernel[grid_fn](A, B, out, M, N, K)
    return out


def check(name, ref, res, equal_nan=False, reduce_dim=1, atol=1e-4):
    RESOLUTION = {
        torch.bool: 0,
        torch.uint8: 0,
        torch.int8: 0,
        torch.int16: 0,
        torch.int32: 0,
        torch.int64: 0,
        torch.float8_e4m3fn: 1e-3,
        torch.float8_e5m2: 1e-3,
        torch.float8_e4m3fnuz: 1e-3,
        torch.float8_e5m2fnuz: 1e-3,
        torch.float16: 1e-3,
        torch.float32: 1.3e-6,
        torch.bfloat16: 0.016,
        torch.float64: 1e-7,
        torch.complex32: 1e-3,
        torch.complex64: 1.3e-6,
    }
    res = res.cpu()
    print(f"The maximum difference out {name} between torch and triton is "
          f"{torch.max(torch.abs(ref - res))}")
    rtol = RESOLUTION[ref.dtype]
    assert torch.allclose(res, ref, atol=atol * reduce_dim, rtol=rtol), (res, ref)


if __name__ == "__main__":
    # param
    M = 4
    N = 2
    K = 8
    dtype = torch.float32
    batch = 4

    # inp
    mat1 = torch.randn((batch, M, K), dtype=dtype, device=device)
    mat2 = torch.randn((batch, K, N), dtype=dtype, device=device)
    ref_mat1 = mat1.cpu()
    ref_mat2 = mat2.cpu()

    # op
    ref_out = torch.bmm(ref_mat1, ref_mat2)
    res_out = bmm(mat1, mat2)
    check("value", ref_out, res_out, reduce_dim=K)
