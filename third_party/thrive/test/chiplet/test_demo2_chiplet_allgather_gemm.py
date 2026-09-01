"""All-gather of A's rows across chiplet dies followed by GEMM.

A is sharded on the M dimension; each die first pushes its local A-shard to
its peer via putmem, then each block gathers the A-shard of die src_die and
computes C[src_die] = A[src_die] @ B^T.
"""
import pytest
import torch
import triton
import triton.experimental.tle.language as tle
import triton.language as tl

N_DIES = 4
M, N, K = 256, 256, 128
M_per_rank = M // N_DIES
BLOCK_M = M_per_rank
BLOCK_N = N
BLOCK_K = K


@triton.jit
def ag_gemm_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    buffer_ptr,
    signal_ptr,
    mesh: tl.constexpr,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    M_per_rank: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid = tl.program_id(0)
    pe = tle.shard_id(mesh, "chiplet")
    npes = tle.n_pes(None)

    offs_m = tl.arange(0, BLOCK_M)
    offs_k = tl.arange(0, BLOCK_K)
    offs_n = tl.arange(0, BLOCK_N)

    src_die = (pe + pid) % npes
    if src_die != pe:
        tle.chiplet.putmem(
            buffer_ptr + pe * M_per_rank * K,
            a_ptr,
            M_per_rank * K,
            src_die,
            blocking=False,
        )
        tle.chiplet.fence()
        tle.chiplet.notify(
            signal_ptr + pe,
            src_die,
            signal=1,
            sig_op="set",
        )

    if src_die != pe:
        tle.chiplet.wait(signal_ptr + src_die, wait_value=1)
        a_shard = tl.load(buffer_ptr + src_die * M_per_rank * K + offs_m[:, None] * K + offs_k[None, :])
    else:
        a_shard = tl.load(a_ptr + offs_m[:, None] * K + offs_k[None, :])

    b = tl.load(b_ptr + offs_n[:, None] * K + offs_k[None, :])
    c_shard = tl.dot(a_shard, b.T)

    tl.store(
        c_ptr + src_die * M_per_rank * N + offs_m[:, None] * N + offs_n[None, :],
        c_shard.to(tl.float16),
    )


def test_allgather_gemm():
    mesh = tle.device_mesh({"chiplet": N_DIES})
    spec_shard = tle.sharding(mesh, split=(("chiplet", ), None))
    spec_repl = tle.sharding(mesh)

    a_host = torch.randn(M, K, dtype=torch.float16)
    b_host = torch.randn(N, K, dtype=torch.float16)
    golden = (a_host @ b_host.T).to(torch.float16)

    a_dev = tle.make_sharded_tensor(a_host, spec_shard)
    b_dev = tle.make_sharded_tensor(b_host, spec_repl)
    c_dev = tle.make_sharded_tensor(torch.zeros(M, N, dtype=torch.float16), spec_repl)
    buffer_dev = tle.make_sharded_tensor(torch.zeros(M, K, dtype=torch.float16), spec_repl)
    signal_dev = tle.make_sharded_tensor(torch.zeros(N_DIES, dtype=torch.int64), spec_repl)

    ag_gemm_kernel[(N_DIES, )](
        a_dev,
        b_dev,
        c_dev,
        buffer_dev,
        signal_dev,
        mesh,
        M=M,
        N=N,
        K=K,
        M_per_rank=M_per_rank,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_K=BLOCK_K,
    )

    c_host = c_dev.cpu()
    assert torch.allclose(c_host, golden, atol=1e-3, rtol=1e-3), \
        "C mismatch vs golden (a_host @ b_host.T)"


if __name__ == "__main__":
    test_allgather_gemm()
