import imp
import torch
import triton
import triton.language as tl
from triton.experimental import tle

TILE_NUM = 16
M = 4096
K = 1024
N = 4096
BLOCK_M = M // TILE_NUM
BLOCK_K = K
SUB_N = N // TILE_NUM

TILE_PHYSICAL_RELATION = [0, 1, 2, 3, 7, 11, 15, 14, 13, 12, 8, 9, 10, 6, 5, 4]

MESH = tle.device_mesh(
    None,
    _shape=(TILE_NUM, ),
    _dim_names=("tile", ),
    _physical_ids=tuple(TILE_PHYSICAL_RELATION),
)


@triton.jit
def dsa_shift_n_gemm_kernel(
    A_ptr,
    B_ptr,
    C_ptr,
    send_next_tile_lut_ptr,
    ring_index_lut_ptr,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_K: tl.constexpr,
    SUB_N: tl.constexpr,
    TILE_NUM: tl.constexpr,
):
    pid = tl.program_id(0)
    send_next_tile = tl.load(send_next_tile_lut_ptr + pid)
    ring_index = tl.load(ring_index_lut_ptr + pid)

    offs_m = pid * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_k = tl.arange(0, BLOCK_K)

    a_ptrs = A_ptr + offs_m[:, None] * K + offs_k[None, :]
    a = tl.load(a_ptrs)

    shard_idx = ring_index
    offs_sub_n = shard_idx * SUB_N + tl.arange(0, SUB_N)
    b_ptrs = B_ptr + offs_k[:, None] * N + offs_sub_n[None, :]
    b_init = tl.load(b_ptrs)

    send_buf = tle.language.dsa.alloc((BLOCK_K, SUB_N), tl.float16)
    recv_buf = tle.language.dsa.alloc((BLOCK_K, SUB_N), tl.float16)

    offs_buf_k = tl.arange(0, BLOCK_K)[:, None] + tl.zeros((1, SUB_N), dtype=tl.int32)
    offs_buf_n = tl.arange(0, SUB_N)[None, :] + tl.zeros((BLOCK_K, 1), dtype=tl.int32)

    send_ptr = tle.language.dsa.local_ptr(send_buf, [offs_buf_k, offs_buf_n])
    recv_ptr = tle.language.dsa.local_ptr(recv_buf, [offs_buf_k, offs_buf_n])

    remote_recv_buf = tle.remote(recv_buf, send_next_tile)
    remote_recv_ptr = tle.language.dsa.local_ptr(remote_recv_buf, [offs_buf_k, offs_buf_n])

    tl.store(send_ptr, b_init)

    for step in range(TILE_NUM):
        b_cur = tl.load(send_ptr)
        c_part = tl.dot(a, b_cur, out_dtype=tl.float32)

        offs_n = shard_idx * SUB_N + tl.arange(0, SUB_N)
        c_ptrs = C_ptr + offs_m[:, None] * N + offs_n[None, :]
        tl.store(c_ptrs, c_part.to(tl.float16))

        if step < TILE_NUM - 1:
            tl.store(remote_recv_ptr, tl.load(send_ptr))
            # tle.distributed_barrier(MESH)
            tl.store(send_ptr, tl.load(recv_ptr))
            # tle.distributed_barrier(MESH)

            shard_idx = tl.where(shard_idx == 0, TILE_NUM - 1, shard_idx - 1)


def build_ring_luts(mesh, device):
    phys = mesh.physical_ids
    n = mesh.size
    send_next = torch.empty(n, dtype=torch.int32)
    ring_index = torch.empty(n, dtype=torch.int32)
    for i in range(n):
        cur = phys[i]
        nxt = phys[(i + 1) % n]
        send_next[cur] = nxt
        ring_index[cur] = i
    return send_next.to(device), ring_index.to(device)


def run():
    device = triton.runtime.driver.active.get_active_torch_device()
    a = torch.randn((M, K), device=device, dtype=torch.float16)
    b = torch.randn((K, N), device=device, dtype=torch.float16)
    c = torch.empty((M, N), device=device, dtype=torch.float16)

    send_next_lut, ring_index_lut = build_ring_luts(MESH, device)

    grid = (TILE_NUM, )
    dsa_shift_n_gemm_kernel[grid](
        a,
        b,
        c,
        send_next_lut,
        ring_index_lut,
        M=M,
        N=N,
        K=K,
        BLOCK_M=BLOCK_M,
        BLOCK_K=BLOCK_K,
        SUB_N=SUB_N,
        TILE_NUM=TILE_NUM,
    )
    a_f32 = a.cpu().float()
    b_f32 = b.cpu().float()
    c_f32 = c.cpu().float()
    ref = torch.matmul(a_f32, b_f32)

    max_diff = (c_f32 - ref).abs().max().item()
    passed = torch.allclose(c_f32, ref, atol=1e-1, rtol=1e-1)

    print(f"Shift-N Ring-GEMM: M={M}, N={N}, K={K}, TILE_NUM={TILE_NUM}")
    print(f"BLOCK_M={BLOCK_M}, BLOCK_K={BLOCK_K}, SUB_N={SUB_N}")
    print(f"Physical ring: {TILE_PHYSICAL_RELATION}")
    print(f"max_abs_diff = {max_diff:.6f}")

    if passed:
        print("PASS")
    else:
        print("FAIL")
        diff = (c_f32 - ref).abs()
        idx = diff.argmax().item()
        r, col = idx // N, idx % N
        print(f"  worst @ ({r},{col}): got={c_f32[r,col]:.4f}  ref={ref[r,col]:.4f}")

    # import flag_gems
    # with flag_gems.use_gems():
    #     ref_out = torch.mm(a, b)
    # # Compare on CPU to avoid unsupported torch.testing ops on TXDA backend.
    # res_out = c.detach().cpu().to(torch.float32)
    # golden_cpu = ref_out.detach().cpu().to(torch.float32)
    # # ref = torch.matmul(a_cpu, b_cpu)
    # max_abs = (res_out - golden_cpu).abs().max().item()

    # # diff = (c_cpu - ref).abs()
    # # flat_idx = diff.argmax().item()
    # # row = flat_idx // diff.shape[1]
    # # col = flat_idx % diff.shape[1]
    # # print(f"[DEBUG] split-k max_abs_diff={max_abs}")
    # # print(f"[DEBUG] split-k worst_idx=({row}, {col})")
    # # print(f"[DEBUG] c_cpu[{row},{col}]={c_cpu[row, col].item()}")
    # # print(f"[DEBUG] ref  [{row},{col}]={ref[row, col].item()}")
    # # print("[DEBUG] c_cpu[0:4, 0:8]=")
    # # print(c_cpu[0:4, 0:8])
    # # print("[DEBUG] ref[0:4, 0:8]=")
    # # print(ref[0:4, 0:8])

    # if not torch.allclose(res_out, golden_cpu, atol=1e-3, rtol=1e-2):
    #     raise AssertionError(f"Mismatch: max_abs_diff={max_abs}")
    # print(
    #     f"PASS: M={M}, N={N}, K={K}, BLOCK_M={BLOCK_M}, "
    #     f"BLOCK_K={BLOCK_K}, TILE_NUM={TILE_NUM}, "
    #     f"mode=ring, max_abs_diff={max_abs}"
    # )


if __name__ == "__main__":
    run()
