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
    _shape=(TILE_NUM,),
    _dim_names=("tile",),
    _physical_ids=tuple(TILE_PHYSICAL_RELATION),
)


@triton.jit
def dsa_shift_n_gemm_kernel(
    A_ptr, B_ptr, C_ptr,
    physical_ids_ptr, ring_index_lut_ptr,
    M: tl.constexpr, N: tl.constexpr, K: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_K: tl.constexpr,
    SUB_N: tl.constexpr, TILE_NUM: tl.constexpr,
):
    # Use tle.shard_id() to obtain the current tile's physical id.
    pid = tle.shard_id(MESH, axis=0)

    # ring_index = logical ring position of this physical tile.
    ring_index = tl.load(ring_index_lut_ptr + pid)

    # Compute send target from the mesh topology.
    # send_next_tile = physical_ids[(ring_index + 1) % TILE_NUM]
    next_ring_pos = tl.where(ring_index == TILE_NUM - 1, 0, ring_index + 1)
    send_next_tile = tl.load(physical_ids_ptr + next_ring_pos)

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

    # Mark recv_buf for remote access.  send_next_tile becomes the DTE
    # target (__Send's tileId).  The recv source is resolved at runtime
    # by __Send from the topology passed via scope=MESH.
    remote_recv_buf = tle.remote(recv_buf, send_next_tile, scope=MESH)
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
            tle.distributed_barrier(MESH)
            tl.store(send_ptr, tl.load(recv_ptr))
            tle.distributed_barrier(MESH)

            shard_idx = tl.where(shard_idx == 0, TILE_NUM - 1, shard_idx - 1)


def build_mesh_luts(mesh, device):
    """Build topology LUTs from device_mesh.physical_ids.

    Returns two tensors:
      physical_ids  — the ring order as a device tensor.
      ring_index    — inverse mapping: physical_id → ring position.
    """
    phys = mesh.physical_ids
    n = mesh.size
    physical_ids = torch.tensor(phys, dtype=torch.int32)
    ring_index = torch.empty(n, dtype=torch.int32)
    for i, p in enumerate(phys):
        ring_index[p] = i
    return physical_ids.to(device), ring_index.to(device)


def run():
    device = triton.runtime.driver.active.get_active_torch_device()
    a = torch.randn((M, K), device=device, dtype=torch.float16)
    b = torch.randn((K, N), device=device, dtype=torch.float16)
    c = torch.empty((M, N), device=device, dtype=torch.float16)

    physical_ids, ring_index_lut = build_mesh_luts(MESH, device)

    grid = (TILE_NUM,)
    dsa_shift_n_gemm_kernel[grid](
        a, b, c,
        physical_ids, ring_index_lut,
        M=M, N=N, K=K,
        BLOCK_M=BLOCK_M, BLOCK_K=BLOCK_K,
        SUB_N=SUB_N, TILE_NUM=TILE_NUM,
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


if __name__ == "__main__":
    run()
