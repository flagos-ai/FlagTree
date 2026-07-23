"""
Matrix Multiplication with Warp Specialization (Enflame GCU)
=============================================================

This tutorial demonstrates matrix multiplication with warp specialization
on Enflame GCU400, comparing four approaches:

1. **Auto Warp Specialize** (via `tl.range(warp_specialize=True)`):
   The GCU compiler automatically partitions the loop body:
   - Default partition (1 warp): loads A and B tiles via DTE -> shared memory
   - Worker partition (4 warps): computes dot + stores result
   Uses `tl.make_block_ptr` for structured memory access.

2. **TLE WS (consumer-default)** (via `tle.gpu.warp_specialize` + `tle.pipe`):
   Manually defined partitions with consumer as default:
     default partition (4 warps) = consumer (dot + store)
     worker  partition (1 warp)  = producer (load via DTE)

3. **TLE WS (producer-default)** - matches Auto WS structure:
   Manually defined partitions with producer as default:
     default partition (1 warp)  = producer (load via DTE)
     worker  partition (4 warps) = consumer (dot + store)
   This directly mirrors the Auto WS IR layout.

4. **Torch AOT** - matches topsaten.

Migrated from:
  - kurama/triton_gcu/test/python/triton/benchmark/gcu_tutorials/gcu400/openai-03-matrix-multiplication.py
  - kurama/triton_gcu/test/python/triton/gcu/matmul_warp_specialize.py
"""

import sys
import pytest
import torch
import triton
import triton.language as tl
import importlib.util
if importlib.util.find_spec("triton.backends.enflame") is None:
    import triton_gcu.triton
from torch_gcu import transfer_to_gcu
import triton.experimental.tle.language as tle

DEVICE = triton.runtime.driver.active.get_active_torch_device()

# ===========================================================================
# Autotune configs (matching kurama openai-03-matrix-multiplication.py)
# ===========================================================================


def get_autotune_config():
    configs = []
    for max_grid in [24, 48]:
        configs.append(
            triton.Config({'BLOCK_SIZE_M': 256, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 256, 'MAX_GRID_DIM': max_grid},
                          num_stages=2, num_warps=4))
        configs.append(
            triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 256, 'MAX_GRID_DIM': max_grid},
                          num_stages=2, num_warps=4))
        configs.append(
            triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 256, 'MAX_GRID_DIM': max_grid},
                          num_stages=2, num_warps=4))
        configs.append(
            triton.Config({'BLOCK_SIZE_M': 256, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 256, 'MAX_GRID_DIM': max_grid},
                          num_stages=2, num_warps=4))
    configs.append(
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 256, 'MAX_GRID_DIM': 24}, num_stages=2,
                      num_warps=2))
    configs.append(
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 256, 'MAX_GRID_DIM': 48}, num_stages=2,
                      num_warps=2))
    return configs


# ===========================================================================
# Approach 1: Auto Warp Specialize (with autotune, matching kurama)
# ===========================================================================


@triton.autotune(configs=get_autotune_config(), key=['M', 'N', 'K'])
@triton.jit
def matmul_ws_auto_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    M,
    N,
    K,
    stride_am,
    stride_ak,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    ACTIVATION: tl.constexpr,
    MAX_GRID_DIM: tl.constexpr,
    WARP_SPECIALIZE: tl.constexpr,
):
    """Persistent-thread matmul with auto warp specialization.
    The outer tl.range loop is warp-specialized: the compiler splits load
    (producer) and dot+store (consumer) across warp groups automatically.
    """
    pid_mn = tl.program_id(0)
    grid_m = tl.cdiv(M, BLOCK_SIZE_M)
    grid_n = tl.cdiv(N, BLOCK_SIZE_N)
    for pid in tl.range(pid_mn, grid_m * grid_n, MAX_GRID_DIM, warp_specialize=WARP_SPECIALIZE):
        x = pid // grid_n
        y = pid % grid_n
        L_block_ptr = tl.make_block_ptr(base=a_ptr, shape=(M, K), strides=(stride_am, stride_ak),
                                        offsets=(x * BLOCK_SIZE_M, 0), block_shape=(BLOCK_SIZE_M, BLOCK_SIZE_K),
                                        order=(1, 0))
        R_block_ptr = tl.make_block_ptr(base=b_ptr, shape=(K, N), strides=(stride_bk, stride_bn),
                                        offsets=(0, y * BLOCK_SIZE_N), block_shape=(BLOCK_SIZE_K, BLOCK_SIZE_N),
                                        order=(1, 0))
        O_block_ptr = tl.make_block_ptr(base=c_ptr, shape=(M, N), strides=(stride_cm, stride_cn),
                                        offsets=(x * BLOCK_SIZE_M, y * BLOCK_SIZE_N),
                                        block_shape=(BLOCK_SIZE_M, BLOCK_SIZE_N), order=(1, 0))

        accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
        for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
            a = tl.load(L_block_ptr, boundary_check=(
                0,
                1,
            ), padding_option="zero")
            b = tl.load(R_block_ptr, boundary_check=(
                0,
                1,
            ), padding_option="zero")
            accumulator += tl.dot(a, b, out_dtype=tl.float32)
            L_block_ptr = tl.advance(L_block_ptr, (0, BLOCK_SIZE_K))
            R_block_ptr = tl.advance(R_block_ptr, (BLOCK_SIZE_K, 0))
        if ACTIVATION == "leaky_relu":
            accumulator = leaky_relu(accumulator)
        c = accumulator.to(tl.float16)
        tl.store(O_block_ptr, c, boundary_check=(
            0,
            1,
        ))


@triton.jit
def leaky_relu(x):
    x = x + 1
    return tl.where(x >= 0, x, 0.01 * x)


# ===========================================================================
# Approach 2: TLE Explicit Warp Specialize (tle.gpu.warp_specialize + pipe)
# Manual producer/consumer partition matching the auto WS IR structure:
#   default partition (4 warps) = consumer (dot + store)
#   worker  partition (1 warp)  = producer (load via DTE)
# ===========================================================================


@triton.jit
def _tle_consumer_default(
    c_ptr,
    reader,
    M,
    N,
    K,
    stride_cm,
    stride_cn,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    MAX_GRID_DIM: tl.constexpr,
    USE_LEAKY_RELU: tl.constexpr,
):
    """Consumer as default (4 warps): dot + store."""
    pid = tl.program_id(0)
    grid_m = tl.cdiv(M, BLOCK_SIZE_M)
    grid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_k_tiles = tl.cdiv(K, BLOCK_SIZE_K)

    for tile_id in tl.range(pid, grid_m * grid_n, MAX_GRID_DIM):
        x = tile_id // grid_n
        y = tile_id % grid_n
        accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
        for k in range(0, num_k_tiles):
            result = reader.wait(k)
            a = tl.load(tle.gpu.local_ptr(result.slot.a_buf))
            b = tl.load(tle.gpu.local_ptr(result.slot.b_buf))
            accumulator += tl.dot(a, b, out_dtype=tl.float32)
            reader.release(k)
        if USE_LEAKY_RELU:
            accumulator = leaky_relu(accumulator)
        c = accumulator.to(tl.float16)

        O_block_ptr = tl.make_block_ptr(base=c_ptr, shape=(M, N), strides=(stride_cm, stride_cn),
                                        offsets=(x * BLOCK_SIZE_M, y * BLOCK_SIZE_N),
                                        block_shape=(BLOCK_SIZE_M, BLOCK_SIZE_N), order=(1, 0))
        tl.store(O_block_ptr, c, boundary_check=(0, 1))


@triton.jit
def _tle_producer_worker(
    a_ptr,
    b_ptr,
    writer,
    M,
    N,
    K,
    stride_am,
    stride_ak,
    stride_bk,
    stride_bn,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    MAX_GRID_DIM: tl.constexpr,
):
    """Producer as worker (1 warp): loads A and B tiles into pipe."""
    pid = tl.program_id(0)
    grid_m = tl.cdiv(M, BLOCK_SIZE_M)
    grid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_k_tiles = tl.cdiv(K, BLOCK_SIZE_K)
    offs_m = tl.arange(0, BLOCK_SIZE_M)
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    offs_n = tl.arange(0, BLOCK_SIZE_N)

    for tile_id in tl.range(pid, grid_m * grid_n, MAX_GRID_DIM):
        x = tile_id // grid_n
        y = tile_id % grid_n

        for k in range(0, num_k_tiles):
            slot = writer.acquire(k)

            a_row = x * BLOCK_SIZE_M + offs_m
            a_col = k * BLOCK_SIZE_K + offs_k
            a_mask = (a_row[:, None] < M) & (a_col[None, :] < K)
            a = tl.load(a_ptr + a_row[:, None] * stride_am + a_col[None, :] * stride_ak, mask=a_mask, other=0.0)
            b_row = k * BLOCK_SIZE_K + offs_k
            b_col = y * BLOCK_SIZE_N + offs_n
            b_mask = (b_row[:, None] < K) & (b_col[None, :] < N)
            b = tl.load(b_ptr + b_row[:, None] * stride_bk + b_col[None, :] * stride_bn, mask=b_mask, other=0.0)
            tl.store(tle.gpu.local_ptr(slot.a_buf), a)
            tl.store(tle.gpu.local_ptr(slot.b_buf), b)
            writer.commit(k)


@triton.jit
def matmul_ws_tle_consumer_default_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    M,
    N,
    K,
    stride_am,
    stride_ak,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    MAX_GRID_DIM: tl.constexpr,
    NUM_STAGES: tl.constexpr,
    USE_LEAKY_RELU: tl.constexpr,
):
    """TLE WS: default=consumer(4w dot+store), worker=producer(1w load)."""
    a_buf = tle.gpu.alloc(
        [NUM_STAGES, BLOCK_SIZE_M, BLOCK_SIZE_K],  # type: ignore[arg-type]
        dtype=tl.float16,
        layout=None,
        scope=tle.gpu.smem,
        nv_mma_shared_layout=False,
    )
    b_buf = tle.gpu.alloc(
        [NUM_STAGES, BLOCK_SIZE_K, BLOCK_SIZE_N],  # type: ignore[arg-type]
        dtype=tl.float16,
        layout=None,
        scope=tle.gpu.smem,
        nv_mma_shared_layout=False,
    )
    p = tle.pipe(capacity=NUM_STAGES, a_buf=a_buf, b_buf=b_buf)
    writer = p.writer()
    reader = p.reader()

    tle.gpu.warp_specialize(
        [
            (_tle_consumer_default, (
                c_ptr,
                reader,
                M,
                N,
                K,
                stride_cm,
                stride_cn,
                tl.constexpr(BLOCK_SIZE_M),
                tl.constexpr(BLOCK_SIZE_N),
                tl.constexpr(BLOCK_SIZE_K),
                tl.constexpr(MAX_GRID_DIM),
                tl.constexpr(USE_LEAKY_RELU),
            )),
            (_tle_producer_worker, (
                a_ptr,
                b_ptr,
                writer,
                M,
                N,
                K,
                stride_am,
                stride_ak,
                stride_bk,
                stride_bn,
                tl.constexpr(BLOCK_SIZE_M),
                tl.constexpr(BLOCK_SIZE_N),
                tl.constexpr(BLOCK_SIZE_K),
                tl.constexpr(MAX_GRID_DIM),
            )),
        ],
        [1],
        [8],
    )


# ===========================================================================
# Approach 3: TLE WS (producer-default) - mirrors Auto WS structure
#   default partition (1 warp)  = producer (load)
#   worker  partition (4 warps) = consumer (dot + store)
# ===========================================================================


@triton.jit
def _tle_producer_default(
    a_ptr,
    b_ptr,
    writer,
    M,
    N,
    K,
    stride_am,
    stride_ak,
    stride_bk,
    stride_bn,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    MAX_GRID_DIM: tl.constexpr,
):
    """Producer as default (1 warp): loads A and B tiles into pipe."""
    pid = tl.program_id(0)
    grid_m = tl.cdiv(M, BLOCK_SIZE_M)
    grid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_k_tiles = tl.cdiv(K, BLOCK_SIZE_K)
    offs_m = tl.arange(0, BLOCK_SIZE_M)
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    offs_n = tl.arange(0, BLOCK_SIZE_N)

    for tile_id in tl.range(pid, grid_m * grid_n, MAX_GRID_DIM):
        x = tile_id // grid_n
        y = tile_id % grid_n
        for k in range(0, num_k_tiles):
            slot = writer.acquire(k)
            a_row = x * BLOCK_SIZE_M + offs_m
            a_col = k * BLOCK_SIZE_K + offs_k
            a_mask = (a_row[:, None] < M) & (a_col[None, :] < K)
            a = tl.load(a_ptr + a_row[:, None] * stride_am + a_col[None, :] * stride_ak, mask=a_mask, other=0.0)
            b_row = k * BLOCK_SIZE_K + offs_k
            b_col = y * BLOCK_SIZE_N + offs_n
            b_mask = (b_row[:, None] < K) & (b_col[None, :] < N)
            b = tl.load(b_ptr + b_row[:, None] * stride_bk + b_col[None, :] * stride_bn, mask=b_mask, other=0.0)
            tl.store(tle.gpu.local_ptr(slot.a_buf), a)
            tl.store(tle.gpu.local_ptr(slot.b_buf), b)
            writer.commit(k)


@triton.jit
def _tle_consumer_worker(
    c_ptr,
    reader,
    M,
    N,
    K,
    stride_cm,
    stride_cn,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    MAX_GRID_DIM: tl.constexpr,
    USE_LEAKY_RELU: tl.constexpr,
):
    """Consumer as worker (4 warps): dot + store."""
    pid = tl.program_id(0)
    grid_m = tl.cdiv(M, BLOCK_SIZE_M)
    grid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_k_tiles = tl.cdiv(K, BLOCK_SIZE_K)

    for tile_id in tl.range(pid, grid_m * grid_n, MAX_GRID_DIM):
        x = tile_id // grid_n
        y = tile_id % grid_n
        accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
        for k in range(0, num_k_tiles):
            result = reader.wait(k)
            a = tl.load(tle.gpu.local_ptr(result.slot.a_buf))
            b = tl.load(tle.gpu.local_ptr(result.slot.b_buf))
            accumulator += tl.dot(a, b, out_dtype=tl.float32)
            reader.release(k)
        if USE_LEAKY_RELU:
            accumulator = leaky_relu(accumulator)
        c = accumulator.to(tl.float16)

        O_block_ptr = tl.make_block_ptr(base=c_ptr, shape=(M, N), strides=(stride_cm, stride_cn),
                                        offsets=(x * BLOCK_SIZE_M, y * BLOCK_SIZE_N),
                                        block_shape=(BLOCK_SIZE_M, BLOCK_SIZE_N), order=(1, 0))
        tl.store(O_block_ptr, c, boundary_check=(0, 1))


@triton.jit
def matmul_ws_tle_producer_default_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    M,
    N,
    K,
    stride_am,
    stride_ak,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    MAX_GRID_DIM: tl.constexpr,
    NUM_STAGES: tl.constexpr,
    USE_LEAKY_RELU: tl.constexpr,
):
    """TLE WS: default=producer(1w load), worker=consumer(4w dot+store).
    Mirrors the Auto WS partition structure exactly."""
    a_buf = tle.gpu.alloc([NUM_STAGES, BLOCK_SIZE_M, BLOCK_SIZE_K], dtype=tl.float16,  # type: ignore[arg-type]
                          layout=None, scope=tle.gpu.smem, nv_mma_shared_layout=True)
    b_buf = tle.gpu.alloc([NUM_STAGES, BLOCK_SIZE_K, BLOCK_SIZE_N], dtype=tl.float16,  # type: ignore[arg-type]
                          layout=None, scope=tle.gpu.smem, nv_mma_shared_layout=True)
    p = tle.pipe(capacity=NUM_STAGES, a_buf=a_buf, b_buf=b_buf)
    writer = p.writer()
    reader = p.reader()

    tle.gpu.warp_specialize(
        [
            (_tle_producer_default, (
                a_ptr,
                b_ptr,
                writer,
                M,
                N,
                K,
                stride_am,
                stride_ak,
                stride_bk,
                stride_bn,
                tl.constexpr(BLOCK_SIZE_M),
                tl.constexpr(BLOCK_SIZE_N),
                tl.constexpr(BLOCK_SIZE_K),
                tl.constexpr(MAX_GRID_DIM),
            )),
            (_tle_consumer_worker, (
                c_ptr,
                reader,
                M,
                N,
                K,
                stride_cm,
                stride_cn,
                tl.constexpr(BLOCK_SIZE_M),
                tl.constexpr(BLOCK_SIZE_N),
                tl.constexpr(BLOCK_SIZE_K),
                tl.constexpr(MAX_GRID_DIM),
                tl.constexpr(USE_LEAKY_RELU),
            )),
        ],
        [4],
        [8],
    )


# ===========================================================================
# Python wrappers
# ===========================================================================


def matmul_auto_ws(a, b, activation=""):
    """Auto warp specialize matmul (with autotune)."""
    assert a.shape[1] == b.shape[0], "Incompatible dimensions"
    assert a.is_contiguous(), "Matrix A must be contiguous"
    assert b.is_contiguous(), "Matrix B must be contiguous"
    M, K = a.shape
    K, N = b.shape
    c = torch.empty((M, N), device=a.device, dtype=a.dtype)
    grid = lambda META: (META['MAX_GRID_DIM'], )
    matmul_ws_auto_kernel[grid](
        a,
        b,
        c,
        M,
        N,
        K,
        a.stride(0),
        a.stride(1),
        b.stride(0),
        b.stride(1),
        c.stride(0),
        c.stride(1),
        ACTIVATION=activation,
        WARP_SPECIALIZE=True,
    )
    return c


def matmul_tle_ws_consumer_default(a, b, max_grid_dim=24, activation=""):
    """TLE WS: default=consumer(4w), worker=producer(1w)."""
    assert a.shape[1] == b.shape[0]
    assert a.is_contiguous() and b.is_contiguous()
    M, K = a.shape
    K, N = b.shape
    c = torch.empty((M, N), device=a.device, dtype=a.dtype)
    BLOCK_M, BLOCK_N, BLOCK_K = 256, 256, 256
    use_leaky_relu = 1 if activation == "leaky_relu" else 0
    matmul_ws_tle_consumer_default_kernel[(max_grid_dim, )](
        a,
        b,
        c,
        M,
        N,
        K,
        a.stride(0),
        a.stride(1),
        b.stride(0),
        b.stride(1),
        c.stride(0),
        c.stride(1),
        BLOCK_SIZE_M=BLOCK_M,
        BLOCK_SIZE_N=BLOCK_N,
        BLOCK_SIZE_K=BLOCK_K,
        MAX_GRID_DIM=max_grid_dim,
        NUM_STAGES=2,
        USE_LEAKY_RELU=use_leaky_relu,
        num_warps=4,
        num_stages=2,
    )
    return c


def matmul_tle_ws_producer_default(a, b, max_grid_dim=24, activation=""):
    """TLE WS: default=producer(1w), worker=consumer(4w).
    Mirrors Auto WS partition layout: kernel num_warps=1 for the default
    region (load), worker_num_warps=[4] for the consumer partition (dot+store)."""
    assert a.shape[1] == b.shape[0]
    assert a.is_contiguous() and b.is_contiguous()
    M, K = a.shape
    K, N = b.shape
    c = torch.empty((M, N), device=a.device, dtype=a.dtype)
    BLOCK_M, BLOCK_N, BLOCK_K = 256, 256, 256
    use_leaky_relu = 1 if activation == "leaky_relu" else 0
    matmul_ws_tle_producer_default_kernel[(max_grid_dim, )](
        a,
        b,
        c,
        M,
        N,
        K,
        a.stride(0),
        a.stride(1),
        b.stride(0),
        b.stride(1),
        c.stride(0),
        c.stride(1),
        BLOCK_SIZE_M=BLOCK_M,
        BLOCK_SIZE_N=BLOCK_N,
        BLOCK_SIZE_K=BLOCK_K,
        MAX_GRID_DIM=max_grid_dim,
        NUM_STAGES=2,
        USE_LEAKY_RELU=use_leaky_relu,
        num_warps=1,
        num_stages=2,
    )
    return c


# ===========================================================================
# Correctness tests
# ===========================================================================


class TestMatmulWarpSpecialize:
    """End-to-end tests for warp-specialized matmul on GCU."""

    @pytest.mark.parametrize("M,N,K", [(512, 512, 512), (1286, 1286, 1286)])
    def test_auto_ws(self, M, N, K):
        """Auto warp specialize via tl.range(warp_specialize=True) + block_ptr."""
        torch.manual_seed(0)
        a = torch.randn((M, K), dtype=torch.float16).gcu()
        b = torch.randn((K, N), dtype=torch.float16).gcu()
        triton_out = matmul_auto_ws(a, b)
        torch_out = torch.matmul(a, b)
        torch.testing.assert_close(triton_out, torch_out, atol=1e-2, rtol=1e-2)

    @pytest.mark.parametrize("M,N,K", [(512, 512, 512), (1286, 1286, 1286)])
    def test_tle_ws_consumer_default(self, M, N, K):
        """TLE WS: default=consumer(4w dot+store), worker=producer(1w load)."""
        torch.manual_seed(0)
        a = torch.randn((M, K), dtype=torch.float16).gcu()
        b = torch.randn((K, N), dtype=torch.float16).gcu()
        triton_out = matmul_tle_ws_consumer_default(a, b)
        torch_out = torch.matmul(a, b)
        torch.testing.assert_close(triton_out, torch_out, atol=1e-2, rtol=1e-2)

    @pytest.mark.parametrize("M,N,K", [(512, 512, 512), (1286, 1286, 1286)])
    def test_tle_ws_producer_default(self, M, N, K):
        """TLE WS: default=producer(1w load), worker=consumer(4w dot+store).
        Mirrors Auto WS partition layout: kernel num_warps=1, worker_num_warps=[4]."""
        torch.manual_seed(0)
        a = torch.randn((M, K), dtype=torch.float16).gcu()
        b = torch.randn((K, N), dtype=torch.float16).gcu()
        triton_out = matmul_tle_ws_producer_default(a, b)
        torch_out = torch.matmul(a, b)
        torch.testing.assert_close(triton_out, torch_out, atol=1e-2, rtol=1e-2)


# ===========================================================================
# Benchmark (matching kurama x_vals range: 256*1 to 256*19)
# ===========================================================================


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=['M', 'N', 'K'],
        x_vals=[256 * i for i in range(1, 20)],
        line_arg='provider',
        line_vals=['torch', 'triton', 'triton_tle_ws_cd', 'triton_tle_ws_pd'],
        line_names=["Torch", "Auto-WS", "TLE WS (consumer-default)", "TLE WS (producer-default)"],
        styles=[('green', '-'), ('blue', '-'), ("orange", "-"), ("red", "-")],
        ylabel="TFLOPS",
        plot_name="matmul-warp-specialize-comparison",
        args={},
    ))
def benchmark(M, N, K, provider):
    a = torch.randn((M, K), dtype=torch.float16).gcu()
    b = torch.randn((K, N), dtype=torch.float16).gcu()
    quantiles = [0.5, 0.2, 0.8]
    ms, min_ms, max_ms = 1.0, 1.0, 1.0
    if provider == 'torch':
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: torch.matmul(a, b), quantiles=quantiles)
    if provider == 'triton':
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: matmul_auto_ws(a, b), quantiles=quantiles)
    if provider == 'triton_tle_ws_cd':
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: matmul_tle_ws_consumer_default(a, b), quantiles=quantiles)
    if provider == 'triton_tle_ws_pd':
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: matmul_tle_ws_producer_default(a, b), quantiles=quantiles)
    torch_output = torch.matmul(a, b)
    if provider == 'triton':
        triton_output = matmul_auto_ws(a, b)
    elif provider == 'triton_tle_ws_cd':
        triton_output = matmul_tle_ws_consumer_default(a, b)
    elif provider == 'triton_tle_ws_pd':
        triton_output = matmul_tle_ws_producer_default(a, b)
    else:
        triton_output = torch_output
    if torch.allclose(triton_output, torch_output, atol=1e-2, rtol=1e-2):
        print(f"✅ Triton({provider}) and Torch match, M = {M}, N = {N}, K = {K}")
    else:
        max_diff = (triton_output - torch_output).abs().max().item()
        print(f"❌ Triton({provider}) and Torch differ (max_diff={max_diff:.2f}), M = {M}, N = {N}, K = {K}")
    perf = lambda ms: 2 * M * N * K * 1e-12 / (ms * 1e-3)
    return perf(ms), perf(max_ms), perf(min_ms)


if __name__ == "__main__":
    benchmark.run(show_plots=True, print_data=True, save_path='./')
