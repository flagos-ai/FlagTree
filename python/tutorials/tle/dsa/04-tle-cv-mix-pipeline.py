"""
CV Mix manual pipeline example
====================
Demonstrates pipelined parallelism between the Cube and Vector cores:
- Cube: matrix multiplication (tl.dot)
- Vector: an add + a 100-iteration loop (to emulate longer compute time)

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
    CV Mix pipeline example (single core, split along M):
    1. Cube: compute C = A @ B (via tl.dot), output in f32
    2. Vector: compute D = C + 1 + 100*0.001 (to emulate longer compute time)
    """
    num_blocks = M // BLOCK_SIZE_M

    for block_idx in range(num_blocks):
        start_m = block_idx * BLOCK_SIZE_M

        # Offsets
        offs_am = start_m + tl.arange(0, BLOCK_SIZE_M)
        offs_an = tl.arange(0, BLOCK_SIZE_N)
        offs_k = tl.arange(0, K)

        # ===== Cube compute: matrix multiplication =====
        a = tl.load(a_ptr + offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
        b = tl.load(b_ptr + offs_k[:, None] * stride_bk + offs_an[None, :] * stride_bn)

        # Cube matrix multiplication
        accumulator = tl.dot(a, b)

        d = accumulator + 1.0
        # Loop 100 times to emulate longer Vector compute time
        for _ in range(100):
            d = d + 0.001

        # Store the final result (using global offsets)
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
    Double-buffered pipeline: single core, split along M, Cube and Vector run in parallel
    - num_blocks: number of blocks along M
    - Two buffers alternate to form the pipeline
    """
    num_blocks = M // BLOCK_SIZE_M

    # Initialization: mark both buffers as available
    tle.dsa.ascend.sync_block_set('vector', 'cube', 0, pipe.PIPE_MTE2, pipe.PIPE_FIX)
    tle.dsa.ascend.sync_block_set('vector', 'cube', 1, pipe.PIPE_MTE2, pipe.PIPE_FIX)

    for block_idx in range(num_blocks):
        start_m = block_idx * BLOCK_SIZE_M
        buffer_id = block_idx % 2
        ws_offset = buffer_id * BLOCK_SIZE_M * BLOCK_SIZE_N

        # Offsets
        offs_am = start_m + tl.arange(0, BLOCK_SIZE_M)
        offs_an = tl.arange(0, BLOCK_SIZE_N)
        offs_k = tl.arange(0, K)

        # Local offsets within the workspace (always starting from 0)
        offs_m_local = tl.arange(0, BLOCK_SIZE_M)

        # ===== Cube compute: matrix multiplication =====
        a = tl.load(a_ptr + offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
        b = tl.load(b_ptr + offs_k[:, None] * stride_bk + offs_an[None, :] * stride_bn)

        accumulator = tl.dot(a, b)

        # Wait for the corresponding buffer to be released by Vector
        tle.dsa.ascend.sync_block_wait("vector", "cube", buffer_id, pipe.PIPE_MTE2, pipe.PIPE_FIX)

        # Store to the workspace (f32), using local offsets + the buffer offset
        tl.store(workspace_ptr + ws_offset + offs_m_local[:, None] * BLOCK_SIZE_N + offs_an[None, :], accumulator)

        # Notify Vector that the data is ready
        tle.dsa.ascend.sync_block_set("cube", "vector", buffer_id, pipe.PIPE_FIX, pipe.PIPE_MTE2)

        # Wait for Vector to finish processing
        tle.dsa.ascend.sync_block_wait("cube", "vector", buffer_id, pipe.PIPE_FIX, pipe.PIPE_MTE2)

        # ===== Vector compute: load f32 and add (loop 100 times to emulate longer compute time) =====
        c_reload = tl.load(workspace_ptr + ws_offset + offs_m_local[:, None] * BLOCK_SIZE_N + offs_an[None, :])
        d = c_reload + 1.0
        # Loop 100 times to emulate longer Vector compute time
        for _ in range(100):
            d = d + 0.001

        # Store the final result
        tl.store(c_ptr + offs_am[:, None] * stride_cn + offs_an[None, :], d)

        # Release the buffer
        tle.dsa.ascend.sync_block_set('vector', 'cube', buffer_id, pipe.PIPE_MTE2, pipe.PIPE_FIX)


@triton.jit
def _cv_mix_cube_producer(
    c_writer,
    block_idx,
    a_ptr,
    b_ptr,
    stride_am,
    stride_ak,
    stride_bk,
    stride_bn,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    K: tl.constexpr,
):
    start_m = block_idx * BLOCK_SIZE_M

    # Offsets
    offs_am = start_m + tl.arange(0, BLOCK_SIZE_M)
    offs_an = tl.arange(0, BLOCK_SIZE_N)
    offs_k = tl.arange(0, K)

    # ===== Cube compute: matrix multiplication =====
    a = tl.load(a_ptr + offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b = tl.load(b_ptr + offs_k[:, None] * stride_bk + offs_an[None, :] * stride_bn)
    accumulator = tl.dot(a, b)

    # ===== Write pipe slot: wait free -> write workspace -> set ready =====
    write_slot = c_writer.acquire(block_idx)
    tl.store(write_slot.c, accumulator)
    c_writer.commit(block_idx)


@triton.jit
def _cv_mix_vector_consumer(
    c_reader,
    block_idx,
    c_ptr,
    stride_cn,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
):
    start_m = block_idx * BLOCK_SIZE_M

    # Offsets
    offs_am = start_m + tl.arange(0, BLOCK_SIZE_M)
    offs_an = tl.arange(0, BLOCK_SIZE_N)

    # ===== Vector compute: wait ready -> read slot -> compute -> set free =====
    result = c_reader.wait(block_idx)
    c_reload = tl.load(result.slot.c)
    d = c_reload + 1.0
    # Loop 100 times to emulate longer Vector compute time
    for _ in range(100):
        d = d + 0.001

    # Store the final result
    tl.store(c_ptr + offs_am[:, None] * stride_cn + offs_an[None, :], d)

    # Release the buffer
    c_reader.release(block_idx)


@triton.jit
def cv_mix_matmul_add_pipe_kernel(
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
    Double-buffered pipeline (tle.pipe version): semantically equivalent to the handwritten sync_block kernel above
    - Workspace ring stage selection, event id allocation and the initial free
      markers are all handled by tle.pipe; only the producer/consumer four-step handshake remains in the kernel
    """
    num_blocks = M // BLOCK_SIZE_M

    # The GM workspace is split into ring stages by capacity; the pipe handles synchronization only, not data
    c_workspace = tle.dsa.workspace(
        workspace_ptr,
        capacity=2,
        shape=[BLOCK_SIZE_M, BLOCK_SIZE_N],
        dtype=tl.float32,
    )

    scheduler = tle.dsa.ascend.pipe_scheduler(
        ready_sync=tle.dsa.ascend.SyncSpec(
            sender="cube",
            receiver="vector",
            sender_pipe=pipe.PIPE_FIX,
            receiver_pipe=pipe.PIPE_MTE2,
        ),
        free_sync=tle.dsa.ascend.SyncSpec(
            sender="vector",
            receiver="cube",
            sender_pipe=pipe.PIPE_MTE2,
            receiver_pipe=pipe.PIPE_FIX,
        ),
        event_base=0,
    )

    c_pipe = tle.pipe(
        capacity=2,
        scope="cta",
        name="cv_mix_c_pipe",
        c=c_workspace,
    )
    c_writer = c_pipe.writer()
    c_reader = c_pipe.reader()

    for block_idx in range(num_blocks):
        # run_pipe inlines the producer and consumer functions in order
        # The scheduler is written out explicitly here, mirroring the handwritten sync_block version above
        # In fact, for the default directions and default event_id, the scheduler can be omitted, i.e.:
        # tle.dsa.ascend.run_pipe([(_cv_mix_cube_producer, ...), (_cv_mix_vector_consumer, ...)])
        # It is spelled out here to exercise the scheduler API
        tle.dsa.ascend.run_pipe([
            (
                _cv_mix_cube_producer,
                (
                    c_writer,
                    block_idx,
                    a_ptr,
                    b_ptr,
                    stride_am,
                    stride_ak,
                    stride_bk,
                    stride_bn,
                    BLOCK_SIZE_M,
                    BLOCK_SIZE_N,
                    K,
                ),
            ),
            (
                _cv_mix_vector_consumer,
                (
                    c_reader,
                    block_idx,
                    c_ptr,
                    stride_cn,
                    BLOCK_SIZE_M,
                    BLOCK_SIZE_N,
                ),
            ),
        ], scheduler=scheduler)


def cv_mix_matmul_add(a: torch.Tensor, b: torch.Tensor):
    """Single-buffer wrapper (single core): D = (A @ B) + 1.1"""
    M, K = a.shape
    K2, N = b.shape
    assert K == K2, f"Incompatible dimensions: A[{M},{K}] x B[{K2},{N}]"

    BLOCK_SIZE_M = 64
    BLOCK_SIZE_N = 128
    # Make sure M is a multiple of BLOCK_SIZE_M
    assert M % BLOCK_SIZE_M == 0, f"M={M} must be multiple of BLOCK_SIZE_M={BLOCK_SIZE_M}"
    assert N % BLOCK_SIZE_N == 0, f"N={N} must be multiple of BLOCK_SIZE_N={BLOCK_SIZE_N}"

    grid = (1, )  # single core
    workspace = torch.empty((BLOCK_SIZE_M * BLOCK_SIZE_N, ), dtype=torch.float32, device=a.device)
    c = torch.empty((M, N), dtype=torch.float32, device=a.device)

    cv_mix_matmul_add_kernel[grid](a, b, c, workspace, stride_am=a.stride(0), stride_ak=a.stride(1),
                                   stride_bk=b.stride(0), stride_bn=b.stride(1), stride_cn=c.stride(0), M=M, N=N,
                                   BLOCK_SIZE_M=BLOCK_SIZE_M, BLOCK_SIZE_N=BLOCK_SIZE_N, K=K, multibuffer=False)
    return c


def cv_mix_matmul_add_double_buffer(a: torch.Tensor, b: torch.Tensor):
    """Double-buffered wrapper: single core, pipelined along M"""
    M, K = a.shape
    K2, N = b.shape
    assert K == K2, f"Incompatible dimensions: A[{M},{K}] x B[{K2},{N}]"

    BLOCK_SIZE_M = 128
    BLOCK_SIZE_N = 128
    assert M % BLOCK_SIZE_M == 0, f"M={M} must be multiple of BLOCK_SIZE_M={BLOCK_SIZE_M}"
    assert N % BLOCK_SIZE_N == 0, f"N={N} must be multiple of BLOCK_SIZE_N={BLOCK_SIZE_N}"

    grid = (1, )  # single core
    # Double-buffered workspace (f32)
    workspace = torch.empty((2 * BLOCK_SIZE_M * BLOCK_SIZE_N, ), dtype=torch.float32, device=a.device)
    c = torch.empty((M, N), dtype=torch.float32, device=a.device)

    cv_mix_matmul_add_double_buffer_kernel[grid](a, b, c, workspace, stride_am=a.stride(0), stride_ak=a.stride(1),
                                                 stride_bk=b.stride(0), stride_bn=b.stride(1), stride_cn=c.stride(0),
                                                 M=M, N=N, BLOCK_SIZE_M=BLOCK_SIZE_M, BLOCK_SIZE_N=BLOCK_SIZE_N, K=K,
                                                 disable_auto_inject_block_sync=True, multibuffer=False)
    return c


def cv_mix_matmul_add_pipe(a: torch.Tensor, b: torch.Tensor):
    """Double-buffered wrapper (tle.pipe version): single core, pipelined along M"""
    M, K = a.shape
    K2, N = b.shape
    assert K == K2, f"Incompatible dimensions: A[{M},{K}] x B[{K2},{N}]"

    BLOCK_SIZE_M = 128
    BLOCK_SIZE_N = 128
    assert M % BLOCK_SIZE_M == 0, f"M={M} must be multiple of BLOCK_SIZE_M={BLOCK_SIZE_M}"
    assert N % BLOCK_SIZE_N == 0, f"N={N} must be multiple of BLOCK_SIZE_N={BLOCK_SIZE_N}"

    grid = (1, )
    workspace = torch.empty((2 * BLOCK_SIZE_M * BLOCK_SIZE_N, ), dtype=torch.float32, device=a.device)
    c = torch.empty((M, N), dtype=torch.float32, device=a.device)

    cv_mix_matmul_add_pipe_kernel[grid](a, b, c, workspace, stride_am=a.stride(0), stride_ak=a.stride(1),
                                        stride_bk=b.stride(0), stride_bn=b.stride(1), stride_cn=c.stride(0), M=M, N=N,
                                        BLOCK_SIZE_M=BLOCK_SIZE_M, BLOCK_SIZE_N=BLOCK_SIZE_N, K=K,
                                        disable_auto_inject_block_sync=True)
    return c


def test_cv_mix_matmul_add():
    """Test function"""
    DEVICE_ID = 0
    DEVICE = f"npu:{DEVICE_ID}"
    torch_npu.npu.set_device(DEVICE_ID)

    BLOCK_SIZE_M = 64
    BLOCK_SIZE_N = 128

    # Test configs - make sure M is a multiple of BLOCK_SIZE_M and N is a multiple of BLOCK_SIZE_N
    test_cases = [
        {"M": BLOCK_SIZE_M * 128, "N": BLOCK_SIZE_N * 1, "K": 128, "name": "small"},
        {"M": BLOCK_SIZE_M * 512, "N": BLOCK_SIZE_N * 1, "K": 256, "name": "medium"},
        {"M": BLOCK_SIZE_M * 1024, "N": BLOCK_SIZE_N * 1, "K": 256, "name": "big"},
    ]

    dtype = torch.float16

    print("=" * 60)
    print("CV Mix manual pipeline test (Cube: MatMul, Vector: Add)")
    print("=" * 60)

    for case in test_cases:
        M, N, K = case["M"], case["N"], case["K"]
        print(f"\nTest case: {case['name']} - M={M}, N={N}, K={K}")

        # Create inputs
        a = torch.randn((M, K), dtype=dtype, device=DEVICE)
        b = torch.randn((K, N), dtype=dtype, device=DEVICE)

        # PyTorch reference: (A @ B) + 1 + 100 * 0.001
        c_ref = (a @ b).to(torch.float32) + 1.1

        # ===== Single-buffer version =====
        print("  [Single-buffer version]")
        c_triton = None
        try:
            c_triton = cv_mix_matmul_add(a, b)
            torch.testing.assert_close(c_triton, c_ref, rtol=1e-2, atol=1e-2)
            print("    ✓ Accuracy check passed")

            triton_time = do_bench_npu(lambda: cv_mix_matmul_add(a, b), clear_l2_cache=True)
            print(f"    Time: {triton_time:.4f} us")
        except Exception as e:
            print(f"    ✗ Failed: {e}")

        # ===== Double-buffer version =====
        print("  [Double-buffer version]")
        c_db = None
        try:
            c_db = cv_mix_matmul_add_double_buffer(a, b)
            torch.testing.assert_close(c_db, c_ref, rtol=1e-2, atol=1e-2)
            print("    ✓ Accuracy check passed")

            db_time = do_bench_npu(lambda: cv_mix_matmul_add_double_buffer(a, b), clear_l2_cache=True)
            print(f"    Time: {db_time:.4f} us")
            if 'triton_time' in locals():
                speedup = triton_time / db_time
                print(f"    Speedup: {speedup:.2f}x")
        except Exception as e:
            print(f"    ✗ Failed: {e}")

        # ===== Double-buffer tle.pipe version =====
        print("  [Double-buffer tle.pipe version]")
        c_pipe_out = None
        try:
            c_pipe_out = cv_mix_matmul_add_pipe(a, b)
            torch.testing.assert_close(c_pipe_out, c_ref, rtol=1e-2, atol=1e-2)
            print("    ✓ Accuracy check passed")

            pipe_time = do_bench_npu(lambda: cv_mix_matmul_add_pipe(a, b), clear_l2_cache=True)
            print(f"    Time: {pipe_time:.4f} us")
            if 'db_time' in locals():
                print(f"    pipe/double-buffer time ratio: {pipe_time / db_time:.2f}x")
        except Exception as e:
            print(f"    ✗ Failed: {e}")

        # ===== Single-buffer vs double-buffer comparison =====
        if c_triton is not None and c_db is not None:
            print("  [Single-buffer vs double-buffer comparison]")
            try:
                torch.testing.assert_close(c_triton, c_db, rtol=1e-5, atol=1e-5)
                print("    ✓ Single-buffer and double-buffer results match")
            except Exception as e:
                print(f"    ✗ Single-buffer and double-buffer results differ: {e}")

        # ===== Handwritten sync_block vs tle.pipe comparison =====
        if c_db is not None and c_pipe_out is not None:
            print("  [sync_block vs tle.pipe comparison]")
            try:
                torch.testing.assert_close(c_db, c_pipe_out, rtol=1e-5, atol=1e-5)
                print("    ✓ Handwritten sync_block and tle.pipe results match")
            except Exception as e:
                print(f"    ✗ Handwritten sync_block and tle.pipe results differ: {e}")

    print("\n" + "=" * 60)
    print("All tests done!")
    print("=" * 60)


if __name__ == "__main__":
    test_cv_mix_matmul_add()
