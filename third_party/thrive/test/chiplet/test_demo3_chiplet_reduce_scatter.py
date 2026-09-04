"""GEMM with reduce-scatter across chiplet dies along the K dimension.

A and B are sharded on K, so C = A @ B is a sum over each die's K-segment
partial. Each block computes its own partial, scatters it to the peer dies via
putmem, then waits for and reduces all peer partials to form its C shard.
"""
import torch
import triton
import triton.experimental.tle.language as tle
import triton.language as tl

NUM_DIES = 4
M, K, N = 512, 256, 512
BLOCK_M = 128
BLOCK_N = 128
BLOCK_K = 64
N_TILES_PER_BLOCK = 1
N_TILES_VAL = N // BLOCK_N
NUM_BLOCKS = N_TILES_VAL


@triton.jit
def _partial_gemm(
    a_ptr,
    b_ptr,
    m_off,
    n_off,
    M: tl.constexpr,
    K_local: tl.constexpr,
    N: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    K_TILES: tl.constexpr,
):
    """Compute A[m_off:m_off+BM, :] @ B[:, n_off:n_off+BN] using local K_local."""
    a_bp = tl.make_block_ptr(
        base=a_ptr,
        shape=(M, K_local),
        strides=(K_local, 1),
        offsets=(m_off, 0),
        block_shape=(BLOCK_M, BLOCK_K),
        order=(1, 0),
    )
    b_bp = tl.make_block_ptr(
        base=b_ptr,
        shape=(K_local, N),
        strides=(N, 1),
        offsets=(0, n_off),
        block_shape=(BLOCK_K, BLOCK_N),
        order=(1, 0),
    )
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for _ in range(K_TILES):
        acc += tl.dot(tl.load(a_bp), tl.load(b_bp))
        a_bp = tl.advance(a_bp, (0, BLOCK_K))
        b_bp = tl.advance(b_bp, (BLOCK_K, 0))
    return acc


@triton.jit
def gemm_rs_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    staging_ptr,
    recv_buf_ptr,
    signal_ptr,
    mesh: tl.constexpr,
    M: tl.constexpr,
    N: tl.constexpr,
    K_local: tl.constexpr,
    M_out: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    K_TILES: tl.constexpr,
    NUM_BLOCKS_CST: tl.constexpr,
    SEGMENT_STRIDE: tl.constexpr,
    TILE_STRIDE: tl.constexpr,
):
    pid = tl.program_id(0)
    pe = tle.shard_id(mesh, "chiplet")
    die_id = pe
    npes = tle.n_pes(None)

    n_off = pid * BLOCK_N
    my_sig_off = die_id * NUM_BLOCKS_CST + pid

    for s in tl.range(npes):
        if s != die_id:
            m_global = s * M_out
            acc = _partial_gemm(
                a_ptr,
                b_ptr,
                m_global,
                n_off,
                M,
                K_local,
                N,
                BLOCK_M,
                BLOCK_N,
                BLOCK_K,
                K_TILES,
            )
            stg_bp = tl.make_block_ptr(
                base=staging_ptr,
                shape=(NUM_BLOCKS_CST * BLOCK_M, BLOCK_N),
                strides=(BLOCK_N, 1),
                offsets=(pid * BLOCK_M, 0),
                block_shape=(BLOCK_M, BLOCK_N),
                order=(1, 0),
            )
            tl.store(stg_bp, acc)

            tle.chiplet.putmem(
                recv_buf_ptr + die_id * SEGMENT_STRIDE + pid * TILE_STRIDE,
                staging_ptr + pid * TILE_STRIDE,
                TILE_STRIDE,
                s,
                blocking=False,
            )
            tle.chiplet.fence()
            tle.chiplet.notify(
                signal_ptr + my_sig_off,
                s,
                signal=1,
                sig_op="set",
            )

    own_m = die_id * M_out
    acc = _partial_gemm(
        a_ptr,
        b_ptr,
        own_m,
        n_off,
        M,
        K_local,
        N,
        BLOCK_M,
        BLOCK_N,
        BLOCK_K,
        K_TILES,
    )

    for j in tl.range(npes):
        if j != die_id:
            tle.chiplet.wait(signal_ptr + j * NUM_BLOCKS_CST + pid, wait_value=1)
            recv_bp = tl.make_block_ptr(
                base=recv_buf_ptr + j * SEGMENT_STRIDE + pid * TILE_STRIDE,
                shape=(BLOCK_M, BLOCK_N),
                strides=(BLOCK_N, 1),
                offsets=(0, 0),
                block_shape=(BLOCK_M, BLOCK_N),
                order=(1, 0),
            )
            acc += tl.load(recv_bp)

    c_bp = tl.make_block_ptr(
        base=c_ptr,
        shape=(M_out, N),
        strides=(N, 1),
        offsets=(0, n_off),
        block_shape=(BLOCK_M, BLOCK_N),
        order=(1, 0),
    )
    tl.store(c_bp, acc.to(tl.float16))


def test_gemm_rs():
    mesh = tle.device_mesh({"chiplet": NUM_DIES})
    spec_a = tle.sharding(mesh, split=(None, ("chiplet", )))
    spec_b = tle.sharding(mesh, split=(("chiplet", ), None))
    spec_c = tle.sharding(mesh, split=(("chiplet", ), None))
    spec_repl = tle.sharding(mesh)

    M_total, K_total, N_total = M, K, N
    K_local = K_total // NUM_DIES
    M_out = M_total // NUM_DIES
    K_TILES_val = int(K_local // BLOCK_K)

    a_host = torch.randn(M_total, K_total, dtype=torch.float16)
    b_host = torch.randn(K_total, N_total, dtype=torch.float16)
    golden = (a_host @ b_host).to(torch.float16)

    a_dev = tle.make_sharded_tensor(a_host, spec_a)
    b_dev = tle.make_sharded_tensor(b_host, spec_b)
    c_dev = tle.make_sharded_tensor(torch.zeros(M_total, N_total, dtype=torch.float16), spec_c)
    staging_dev = tle.make_sharded_tensor(torch.zeros(NUM_BLOCKS * BLOCK_M, BLOCK_N, dtype=torch.float32), spec_repl)
    recv_buf_dev = tle.make_sharded_tensor(torch.zeros(NUM_DIES * NUM_BLOCKS * BLOCK_M, BLOCK_N, dtype=torch.float32),
                                           spec_repl)
    signal_dev = tle.make_sharded_tensor(torch.zeros(NUM_DIES * NUM_BLOCKS, dtype=torch.int64), spec_repl)

    gemm_rs_kernel[(NUM_BLOCKS, )](
        a_dev,
        b_dev,
        c_dev,
        staging_dev,
        recv_buf_dev,
        signal_dev,
        mesh,
        M=M_total,
        N=N_total,
        K_local=K_local,
        M_out=M_out,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_K=BLOCK_K,
        K_TILES=K_TILES_val,
        NUM_BLOCKS_CST=NUM_BLOCKS,
        SEGMENT_STRIDE=M_out * N,
        TILE_STRIDE=BLOCK_M * BLOCK_N,
    )

    c_host = c_dev.cpu()
    for pe in range(NUM_DIES):
        actual = c_host[pe * M_out:(pe + 1) * M_out, :]
        expected = golden[pe * M_out:(pe + 1) * M_out, :]
        assert torch.allclose(actual, expected, atol=1e-2, rtol=1e-2), \
            f"die {pe}: C mismatch vs golden"


if __name__ == "__main__":
    test_gemm_rs()
