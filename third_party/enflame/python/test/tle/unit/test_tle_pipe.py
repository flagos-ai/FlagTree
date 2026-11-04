"""E2E tests for TLE pipe mechanics on the GCU backend.

Covers three pipe-focused scenarios (each driven by warp_specialize):
  1. is_closed rejection: GCU400 fast_pipeline rejects is_closed usage
     because it breaks strict producer/consumer pairing.
  2. Pipe key collision & multi-pipe fixPipeSlotIndex: two anonymous pipes
     with the same capacity/field_names but different memdescs must not be
     merged; both pipes in the same outer tile loop must be fixed.
  3. SPMC (Single-Producer Multi-Consumer): named readers sharing one pipe,
     including partial field subscription.
"""
import importlib.util

import pytest
import torch
import triton
import triton.language as tl

if importlib.util.find_spec("triton.backends.enflame") is None:
    import triton_gcu.triton
from torch_gcu import transfer_to_gcu  # noqa: F401  (enables .gcu() on tensors)
import triton.experimental.tle.language as tle

DEVICE = triton.runtime.driver.active.get_active_torch_device()

# ===========================================================================
# 1. is_closed rejection
# ===========================================================================


@triton.jit
def _bad_producer(x_ptr, writer, numel, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < numel
    x = tl.load(x_ptr + offs, mask=mask, other=0.0)
    slot = writer.acquire(0)
    smem_ptrs = tle.gpu.local_ptr(slot.data, (tl.arange(0, BLOCK), ))
    tl.store(smem_ptrs, x, mask=mask)
    writer.commit(0)
    writer.close(0)


@triton.jit
def _bad_consumer(out_ptr, reader, numel, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    result = reader.wait(0)
    is_closed = result.is_closed  # This must trigger error
    # Use is_closed in a Triton operation to ensure it has IR uses
    x = tl.where(is_closed, 0.0, 1.0)
    smem_ptrs = tle.gpu.local_ptr(result.slot.data, (tl.arange(0, BLOCK), ))
    data = tl.load(smem_ptrs)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < numel
    out = data + x
    tl.store(out_ptr + offs, out, mask=mask)
    reader.release(0)


@triton.jit
def _bad_kernel(x_ptr, out_ptr, numel, BLOCK: tl.constexpr, NUM_STAGES: tl.constexpr):
    pid = tl.program_id(0)
    smem = tle.gpu.alloc((NUM_STAGES, BLOCK), dtype=tl.float32, layout=None, scope=tle.gpu.smem,
                         nv_mma_shared_layout=False)
    p = tle.pipe(capacity=NUM_STAGES, data=smem)
    writer = p.writer()
    reader = p.reader()
    tle.gpu.warp_specialize(
        [
            (_bad_consumer, (out_ptr, reader, numel, tl.constexpr(BLOCK))),
            (_bad_producer, (x_ptr, writer, numel, tl.constexpr(BLOCK))),
        ],
        [1],
        [8],
    )


def test_is_closed_rejected():
    """Verify that using is_closed triggers a compilation error."""
    x = torch.randn(64, device=DEVICE, dtype=torch.float32)
    out = torch.zeros_like(x)
    rejected = False
    try:
        _bad_kernel[(1, )](x, out, 64, BLOCK=64, NUM_STAGES=2, num_warps=4)
        print("FAIL: is_closed should have been rejected")
    except Exception as e:
        err = str(e)
        # The MLIR pass error message may or may not propagate to Python
        # exception. Check both the exception string and any compilation
        # failure as indication of rejection.
        if "is_closed" in err:
            print("PASS: is_closed correctly rejected with error")
            rejected = True
        elif "PassManager" in err or "Pipeline run failed" in err:
            # PassManager failure is expected — the error message from
            # emitError is in MLIR diagnostics, not always in Python exception.
            print("PASS: is_closed correctly rejected (compilation failed)")
            rejected = True
        else:
            print(f"UNEXPECTED error: {err[:200]}")
    assert rejected, "is_closed should have been rejected by the compiler"


# ===========================================================================
# 2. Pipe key collision & multi-pipe fixPipeSlotIndex
# ===========================================================================


@triton.jit
def _dual_pipe_consumer(
    c_ptr,
    f_ptr,
    reader1,
    reader2,
    M,
    N,
    K,
    stride_cm,
    stride_cn,
    stride_fm,
    stride_fn,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    MAX_GRID_DIM: tl.constexpr,
):
    pid = tl.program_id(0)
    grid_m = tl.cdiv(M, BLOCK_SIZE_M)
    grid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_k_tiles = tl.cdiv(K, BLOCK_SIZE_K)

    for tile_id in tl.range(pid, grid_m * grid_n, MAX_GRID_DIM):
        x = tile_id // grid_n
        y = tile_id % grid_n

        acc1 = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
        for k in range(0, num_k_tiles):
            result = reader1.wait(k)
            a = tl.load(tle.gpu.local_ptr(result.slot.a_buf))
            b = tl.load(tle.gpu.local_ptr(result.slot.b_buf))
            acc1 += tl.dot(a, b, out_dtype=tl.float32)
            reader1.release(k)
        c = acc1.to(tl.float16)
        O1 = tl.make_block_ptr(base=c_ptr, shape=(M, N), strides=(stride_cm, stride_cn),
                               offsets=(x * BLOCK_SIZE_M, y * BLOCK_SIZE_N), block_shape=(BLOCK_SIZE_M, BLOCK_SIZE_N),
                               order=(1, 0))
        tl.store(O1, c, boundary_check=(0, 1))

        acc2 = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
        for k in range(0, num_k_tiles):
            result = reader2.wait(k)
            d = tl.load(tle.gpu.local_ptr(result.slot.a_buf))
            e = tl.load(tle.gpu.local_ptr(result.slot.b_buf))
            acc2 += tl.dot(d, e, out_dtype=tl.float32)
            reader2.release(k)
        f = acc2.to(tl.float16)
        O2 = tl.make_block_ptr(base=f_ptr, shape=(M, N), strides=(stride_fm, stride_fn),
                               offsets=(x * BLOCK_SIZE_M, y * BLOCK_SIZE_N), block_shape=(BLOCK_SIZE_M, BLOCK_SIZE_N),
                               order=(1, 0))
        tl.store(O2, f, boundary_check=(0, 1))


@triton.jit
def _dual_pipe_producer(
    a_ptr,
    b_ptr,
    d_ptr,
    e_ptr,
    writer1,
    writer2,
    M,
    N,
    K,
    stride_am,
    stride_ak,
    stride_bk,
    stride_bn,
    stride_dm,
    stride_dk,
    stride_ek,
    stride_en,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    MAX_GRID_DIM: tl.constexpr,
):
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
            slot1 = writer1.acquire(k)
            a_row = x * BLOCK_SIZE_M + offs_m
            a_col = k * BLOCK_SIZE_K + offs_k
            a_mask = (a_row[:, None] < M) & (a_col[None, :] < K)
            a = tl.load(a_ptr + a_row[:, None] * stride_am + a_col[None, :] * stride_ak, mask=a_mask, other=0.0)
            b_row = k * BLOCK_SIZE_K + offs_k
            b_col = y * BLOCK_SIZE_N + offs_n
            b_mask = (b_row[:, None] < K) & (b_col[None, :] < N)
            b = tl.load(b_ptr + b_row[:, None] * stride_bk + b_col[None, :] * stride_bn, mask=b_mask, other=0.0)
            tl.store(tle.gpu.local_ptr(slot1.a_buf), a)
            tl.store(tle.gpu.local_ptr(slot1.b_buf), b)
            writer1.commit(k)

        for k in range(0, num_k_tiles):
            slot2 = writer2.acquire(k)
            d_row = x * BLOCK_SIZE_M + offs_m
            d_col = k * BLOCK_SIZE_K + offs_k
            d_mask = (d_row[:, None] < M) & (d_col[None, :] < K)
            d = tl.load(d_ptr + d_row[:, None] * stride_dm + d_col[None, :] * stride_dk, mask=d_mask, other=0.0)
            e_row = k * BLOCK_SIZE_K + offs_k
            e_col = y * BLOCK_SIZE_N + offs_n
            e_mask = (e_row[:, None] < K) & (e_col[None, :] < N)
            e = tl.load(e_ptr + e_row[:, None] * stride_ek + e_col[None, :] * stride_en, mask=e_mask, other=0.0)
            tl.store(tle.gpu.local_ptr(slot2.a_buf), d)
            tl.store(tle.gpu.local_ptr(slot2.b_buf), e)
            writer2.commit(k)


@triton.jit
def dual_pipe_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    d_ptr,
    e_ptr,
    f_ptr,
    M,
    N,
    K,
    stride_am,
    stride_ak,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    stride_dm,
    stride_dk,
    stride_ek,
    stride_en,
    stride_fm,
    stride_fn,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    MAX_GRID_DIM: tl.constexpr,
    NUM_STAGES: tl.constexpr,
):
    a_buf = tle.gpu.alloc((NUM_STAGES, BLOCK_SIZE_M, BLOCK_SIZE_K), dtype=tl.float16, layout=None, scope=tle.gpu.smem,
                          nv_mma_shared_layout=False)
    b_buf = tle.gpu.alloc((NUM_STAGES, BLOCK_SIZE_K, BLOCK_SIZE_N), dtype=tl.float16, layout=None, scope=tle.gpu.smem,
                          nv_mma_shared_layout=False)
    d_buf = tle.gpu.alloc((NUM_STAGES, BLOCK_SIZE_M, BLOCK_SIZE_K), dtype=tl.float16, layout=None, scope=tle.gpu.smem,
                          nv_mma_shared_layout=False)
    e_buf = tle.gpu.alloc((NUM_STAGES, BLOCK_SIZE_K, BLOCK_SIZE_N), dtype=tl.float16, layout=None, scope=tle.gpu.smem,
                          nv_mma_shared_layout=False)

    p1 = tle.pipe(capacity=NUM_STAGES, a_buf=a_buf, b_buf=b_buf)
    p2 = tle.pipe(capacity=NUM_STAGES, a_buf=d_buf, b_buf=e_buf)
    writer1 = p1.writer()
    reader1 = p1.reader()
    writer2 = p2.writer()
    reader2 = p2.reader()

    tle.gpu.warp_specialize(
        [
            (_dual_pipe_consumer, (
                c_ptr,
                f_ptr,
                reader1,
                reader2,
                M,
                N,
                K,
                stride_cm,
                stride_cn,
                stride_fm,
                stride_fn,
                tl.constexpr(BLOCK_SIZE_M),
                tl.constexpr(BLOCK_SIZE_N),
                tl.constexpr(BLOCK_SIZE_K),
                tl.constexpr(MAX_GRID_DIM),
            )),
            (_dual_pipe_producer, (
                a_ptr,
                b_ptr,
                d_ptr,
                e_ptr,
                writer1,
                writer2,
                M,
                N,
                K,
                stride_am,
                stride_ak,
                stride_bk,
                stride_bn,
                stride_dm,
                stride_dk,
                stride_ek,
                stride_en,
                tl.constexpr(BLOCK_SIZE_M),
                tl.constexpr(BLOCK_SIZE_N),
                tl.constexpr(BLOCK_SIZE_K),
                tl.constexpr(MAX_GRID_DIM),
            )),
        ],
        [1],
        [8],
    )


def dual_pipe_matmul(a, b, d, e, max_grid_dim=24):
    assert a.shape[1] == b.shape[0]
    assert d.shape[1] == e.shape[0]
    assert a.is_contiguous() and b.is_contiguous()
    assert d.is_contiguous() and e.is_contiguous()
    M, K = a.shape
    K2, N = b.shape
    assert K == K2
    c = torch.empty((M, N), device=a.device, dtype=a.dtype)
    f = torch.empty((M, N), device=a.device, dtype=a.dtype)
    BLOCK_M, BLOCK_N, BLOCK_K = 128, 128, 256
    dual_pipe_kernel[(max_grid_dim, )](
        a,
        b,
        c,
        d,
        e,
        f,
        M,
        N,
        K,
        a.stride(0),
        a.stride(1),
        b.stride(0),
        b.stride(1),
        c.stride(0),
        c.stride(1),
        d.stride(0),
        d.stride(1),
        e.stride(0),
        e.stride(1),
        f.stride(0),
        f.stride(1),
        BLOCK_SIZE_M=BLOCK_M,
        BLOCK_SIZE_N=BLOCK_N,
        BLOCK_SIZE_K=BLOCK_K,
        MAX_GRID_DIM=max_grid_dim,
        NUM_STAGES=2,
        num_warps=4,
        num_stages=2,
    )
    return c, f


def _run_dual_pipe_shape(M):
    torch.manual_seed(42)
    a = torch.randn((M, M), dtype=torch.float16).gcu()
    b = torch.randn((M, M), dtype=torch.float16).gcu()
    d = torch.randn((M, M), dtype=torch.float16).gcu()
    e = torch.randn((M, M), dtype=torch.float16).gcu()

    c_out, f_out = dual_pipe_matmul(a, b, d, e)
    torch_c = torch.matmul(a, b)
    torch_f = torch.matmul(d, e)

    num_k_tiles = M // 256
    total_tiles = (M // 256)**2
    multi_tile = total_tiles > 24

    c_ok = torch.allclose(c_out, torch_c, atol=1e-2, rtol=1e-2)
    f_ok = torch.allclose(f_out, torch_f, atol=1e-2, rtol=1e-2)

    tag = 'odd' if num_k_tiles % 2 else 'even'
    status = 'PASS' if (c_ok and f_ok) else 'FAIL'
    detail = ''
    if not c_ok:
        detail += f' C_diff={((c_out - torch_c).abs().max().item()):.2f}'
    if not f_ok:
        detail += f' F_diff={((f_out - torch_f).abs().max().item()):.2f}'
    print(f"{status}  M={M:>5} k_tiles={num_k_tiles:>2}({tag:>4}) "
          f"tiles={total_tiles:>3} multi={str(multi_tile):>5}{detail}")
    return c_ok and f_ok


class TestTLEPipeKeyAndMultiPipe:
    """Pipe key collision fix and multi-pipe fixPipeSlotIndex."""

    @pytest.mark.parametrize("M", [256, 1280, 2560])
    def test_pipe_key_and_multi_pipe(self, M):
        ok = _run_dual_pipe_shape(M)
        assert ok, f"dual-pipe matmul failed for M={M}"


# ===========================================================================
# 3. SPMC (Single-Producer Multi-Consumer)
# ===========================================================================

# --- Scenario 3a: 2 named consumers reading all fields (single field). -----


@triton.jit
def _spmc_producer(x_ptr, writer, numel, BLOCK: tl.constexpr):
    """Producer (default, 1 warp): loads data into pipe."""
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < numel
    x = tl.load(x_ptr + offs, mask=mask, other=0.0)
    slot = writer.acquire(0)
    smem_ptrs = tle.gpu.local_ptr(slot.data, (tl.arange(0, BLOCK), ))
    tl.store(smem_ptrs, x, mask=mask)
    writer.commit(0)


@triton.jit
def _spmc_consumer_a(out_ptr, reader, numel, BLOCK: tl.constexpr):
    """Consumer A / "compute_a" (worker 0, 2 warps): reads from pipe, computes data * 2."""
    pid = tl.program_id(0)
    result = reader.wait(0)
    smem_ptrs = tle.gpu.local_ptr(result.slot.data, (tl.arange(0, BLOCK), ))
    x = tl.load(smem_ptrs)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < numel
    out = x * 2.0
    tl.store(out_ptr + offs, out, mask=mask)
    reader.release(0)


@triton.jit
def _spmc_consumer_b(out_ptr, reader, numel, BLOCK: tl.constexpr):
    """Consumer B / "compute_b" (worker 1, 2 warps): reads from pipe, computes data + 1."""
    pid = tl.program_id(0)
    result = reader.wait(0)
    smem_ptrs = tle.gpu.local_ptr(result.slot.data, (tl.arange(0, BLOCK), ))
    x = tl.load(smem_ptrs)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < numel
    out = x + 1.0
    tl.store(out_ptr + offs, out, mask=mask)
    reader.release(0)


@triton.jit
def _spmc_kernel(x_ptr, out_a_ptr, out_b_ptr, numel, BLOCK: tl.constexpr, NUM_STAGES: tl.constexpr):
    pid = tl.program_id(0)
    smem = tle.gpu.alloc((NUM_STAGES, BLOCK), dtype=tl.float32, layout=None, scope=tle.gpu.smem,
                         nv_mma_shared_layout=False)
    p = tle.pipe(capacity=NUM_STAGES, readers=("compute_a", "compute_b"), data=smem)
    writer = p.writer()
    reader_a = p.reader("compute_a")
    reader_b = p.reader("compute_b")
    tle.gpu.warp_specialize(
        [
            (_spmc_producer, (x_ptr, writer, numel, tl.constexpr(BLOCK))),
            (_spmc_consumer_a, (out_a_ptr, reader_a, numel, tl.constexpr(BLOCK))),
            (_spmc_consumer_b, (out_b_ptr, reader_b, numel, tl.constexpr(BLOCK))),
        ],
        [2, 2],
        [8, 8],
    )


def test_spmc():
    """Test SPMC with 2 named consumers reading from same pipe."""
    numel = 64
    x = torch.randn(numel, device=DEVICE, dtype=torch.float32)
    out_a = torch.zeros_like(x)
    out_b = torch.zeros_like(x)

    _spmc_kernel[(1, )](x, out_a, out_b, numel, BLOCK=64, NUM_STAGES=2, num_warps=1)

    expected_a = x * 2.0
    expected_b = x + 1.0

    a_ok = torch.allclose(out_a, expected_a, atol=1e-5, rtol=1e-5)
    b_ok = torch.allclose(out_b, expected_b, atol=1e-5, rtol=1e-5)

    if a_ok and b_ok:
        print(f"PASS spmc test (numel={numel})")
    else:
        if not a_ok:
            print(f"FAIL spmc: consumer_a mismatch, "
                  f"max_diff={abs(out_a - expected_a).max():.4f}")
        if not b_ok:
            print(f"FAIL spmc: consumer_b mismatch, "
                  f"max_diff={abs(out_b - expected_b).max():.4f}")
    assert a_ok and b_ok, "SPMC multi-consumer mismatch"


# --- Scenario 3b: partial field subscription (two fields). ----------------


@triton.jit
def _spmc_field_producer(x_ptr, m_ptr, writer, numel, BLOCK: tl.constexpr):
    """Producer (default, 1 warp): loads data and meta into pipe."""
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < numel
    x = tl.load(x_ptr + offs, mask=mask, other=0.0)
    m = tl.load(m_ptr + offs, mask=mask, other=0.0)
    slot = writer.acquire(0)
    data_ptrs = tle.gpu.local_ptr(slot.data, (tl.arange(0, BLOCK), ))
    meta_ptrs = tle.gpu.local_ptr(slot.meta, (tl.arange(0, BLOCK), ))
    tl.store(data_ptrs, x, mask=mask)
    tl.store(meta_ptrs, m, mask=mask)
    writer.commit(0)


@triton.jit
def _spmc_full_consumer(out_ptr, reader, numel, BLOCK: tl.constexpr):
    """Consumer A / "full_reader" (worker 0, 2 warps): reads both fields, computes data * meta."""
    pid = tl.program_id(0)
    result = reader.wait(0)
    data_ptrs = tle.gpu.local_ptr(result.slot.data, (tl.arange(0, BLOCK), ))
    meta_ptrs = tle.gpu.local_ptr(result.slot.meta, (tl.arange(0, BLOCK), ))
    x = tl.load(data_ptrs)
    m = tl.load(meta_ptrs)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < numel
    out = x * m
    tl.store(out_ptr + offs, out, mask=mask)
    reader.release(0)


@triton.jit
def _spmc_data_consumer(out_ptr, reader, numel, BLOCK: tl.constexpr):
    """Consumer B / "data_reader" (worker 1, 2 warps): reads only "data" field."""
    pid = tl.program_id(0)
    result = reader.wait(0)
    data_ptrs = tle.gpu.local_ptr(result.slot.data, (tl.arange(0, BLOCK), ))
    x = tl.load(data_ptrs)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < numel
    out = x + 1.0
    tl.store(out_ptr + offs, out, mask=mask)
    reader.release(0)


@triton.jit
def _spmc_field_kernel(x_ptr, m_ptr, out_a_ptr, out_b_ptr, numel, BLOCK: tl.constexpr, NUM_STAGES: tl.constexpr):
    pid = tl.program_id(0)
    data_buf = tle.gpu.alloc((NUM_STAGES, BLOCK), dtype=tl.float32, layout=None, scope=tle.gpu.smem,
                             nv_mma_shared_layout=False)
    meta_buf = tle.gpu.alloc((NUM_STAGES, BLOCK), dtype=tl.float32, layout=None, scope=tle.gpu.smem,
                             nv_mma_shared_layout=False)
    p = tle.pipe(
        capacity=NUM_STAGES,
        readers=("full_reader", "data_reader"),
        data=data_buf,
        meta=meta_buf,
    )
    writer = p.writer()
    full_reader = p.reader("full_reader")
    data_reader = p.reader("data_reader", fields=("data", ))
    tle.gpu.warp_specialize(
        [
            (_spmc_field_producer, (x_ptr, m_ptr, writer, numel, tl.constexpr(BLOCK))),
            (_spmc_full_consumer, (out_a_ptr, full_reader, numel, tl.constexpr(BLOCK))),
            (_spmc_data_consumer, (out_b_ptr, data_reader, numel, tl.constexpr(BLOCK))),
        ],
        [2, 2],
        [8, 8],
    )


def test_spmc_fields():
    """Test SPMC with partial field subscription."""
    numel = 64
    x = torch.randn(numel, device=DEVICE, dtype=torch.float32)
    m = torch.randn(numel, device=DEVICE, dtype=torch.float32)
    out_a = torch.zeros_like(x)
    out_b = torch.zeros_like(x)

    _spmc_field_kernel[(1, )](x, m, out_a, out_b, numel, BLOCK=64, NUM_STAGES=2, num_warps=1)

    expected_a = x * m
    expected_b = x + 1.0

    a_ok = torch.allclose(out_a, expected_a, atol=1e-5, rtol=1e-5)
    b_ok = torch.allclose(out_b, expected_b, atol=1e-5, rtol=1e-5)

    if a_ok and b_ok:
        print(f"PASS spmc_fields test (numel={numel})")
    else:
        if not a_ok:
            print(f"FAIL spmc_fields: full_reader mismatch, "
                  f"max_diff={abs(out_a - expected_a).max():.4f}")
        if not b_ok:
            print(f"FAIL spmc_fields: data_reader mismatch, "
                  f"max_diff={abs(out_b - expected_b).max():.4f}")
    assert a_ok and b_ok, "SPMC field subscription mismatch"


if __name__ == "__main__":
    test_is_closed_rejected()
    for M in [256, 1280, 2560]:
        _run_dual_pipe_shape(M)
    test_spmc()
    test_spmc_fields()
