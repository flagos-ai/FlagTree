"""
Functional tests for GatherScatterView lowering (tile_load / tile_store).

Coverage:

  * gather dim arrangements: S D, S S, D S D, and S D D;
  * transposed payload layout;
  * sparse out-of-bounds gather/scatter, payload upper bounds, and zero-length
    payload loads;
  * S D scatter with prefix/suffix canaries, and S S D scatter;
  * leading dense partial/full OOB and negative payload blocks, for both
    gather and scatter with physically allocated guard regions.

Every test compares against a CPU reference with zero tolerance. Run with:

    python -m pytest -v test/CommonIR/Ascend/dsl_extension/test_gather_scatter_view.py

Remaining gaps are documented in `documents/commonir/tensorview_known_issues.md`.
"""

import pytest

import torch
import torch_npu  # noqa: F401  (registers the npu device)
import triton
import triton.language as tl
import triton.experimental.tle as tle


@pytest.fixture(autouse=True)
def seed_random():
    torch.manual_seed(0)


# ---------------------------------------------------------------- arrangements


@triton.jit
def k_gather_sd(src, out, idx, n_rows: tl.constexpr, M: tl.constexpr, BL: tl.constexpr):
    """S D -- one sparse dim, one payload dim. The baseline shape."""
    v = tle.make_gather_scatter_view(src, shape=[n_rows, M], strides=[M, 1], tile=[BL, M], sparse_dim=[0])
    r = tl.load(idx + tl.arange(0, BL))
    t = tle.tile_load(v, index=(r, 0), dst_space=tle.language.dsa.ascend.UB)
    t2 = t * 1.0
    rows = tl.arange(0, BL)[:, None]
    cols = tl.arange(0, M)[None, :]
    tl.store(out + rows * M + cols, t2)


@triton.jit
def k_gather_ssd(src, out, ridx, cidx, D0: tl.constexpr, D1: tl.constexpr, T0: tl.constexpr, T1: tl.constexpr):
    """S S -- two sparse dims; the access is an outer product of the indices."""
    v = tle.make_gather_scatter_view(src, shape=[D0, D1], strides=[D1, 1], tile=[T0, T1], sparse_dim=[0, 1])
    r = tl.load(ridx + tl.arange(0, T0))
    c = tl.load(cidx + tl.arange(0, T1))
    t = tle.tile_load(v, index=(r, c), dst_space=tle.language.dsa.ascend.UB)
    t2 = t * 1.0
    rows = tl.arange(0, T0)[:, None]
    cols = tl.arange(0, T1)[None, :]
    tl.store(out + rows * T1 + cols, t2)


@triton.jit
def k_gather_dsd(src, out, i1, D0: tl.constexpr, D1: tl.constexpr, D2: tl.constexpr, T0: tl.constexpr, T1: tl.constexpr,
                 T2: tl.constexpr):
    """D S D -- the sparse dim is not a prefix; dim 0 is a plain loop level."""
    v = tle.make_gather_scatter_view(src, shape=[D0, D1, D2], strides=[D1 * D2, D2, 1], tile=[T0, T1, T2],
                                     sparse_dim=[1])
    c = tl.load(i1 + tl.arange(0, T1))
    t = tle.tile_load(v, index=(0, c, 0), dst_space=tle.language.dsa.ascend.UB)
    t2 = t * 1.0
    off = (tl.arange(0, T0)[:, None, None] * (T1 * T2) + tl.arange(0, T1)[None, :, None] * T2 +
           tl.arange(0, T2)[None, None, :])
    tl.store(out + off, t2)


@triton.jit
def k_gather_sdd(src, out, i0, D0: tl.constexpr, D1: tl.constexpr, D2: tl.constexpr, T0: tl.constexpr):
    """S D D -- the payload spans two dense dims and moves as one block."""
    v = tle.make_gather_scatter_view(src, shape=[D0, D1, D2], strides=[D1 * D2, D2, 1], tile=[T0, D1, D2],
                                     sparse_dim=[0])
    a = tl.load(i0 + tl.arange(0, T0))
    t = tle.tile_load(v, index=(a, 0, 0), dst_space=tle.language.dsa.ascend.UB)
    t2 = t * 1.0
    off = (tl.arange(0, T0)[:, None, None] * (D1 * D2) + tl.arange(0, D1)[None, :, None] * D2 +
           tl.arange(0, D2)[None, None, :])
    tl.store(out + off, t2)


@triton.jit
def k_gather_transposed(src, out, i0, D0: tl.constexpr, D1: tl.constexpr, D2: tl.constexpr, T0: tl.constexpr,
                        T1: tl.constexpr, T2: tl.constexpr):
    """Transposed dim_map: innermost payload stride != 1.

    This one must *not* take the block DMA -- the logical block is a transpose
    of the memory layout, not a skip -- so it exercises the scalar fallback.
    """
    v = tle.make_gather_scatter_view(src, shape=[D0, D1, D2], strides=[D1 * D2, 1, D1], tile=[T0, T1, T2],
                                     sparse_dim=[0])
    a = tl.load(i0 + tl.arange(0, T0))
    t = tle.tile_load(v, index=(a, 0, 0), dst_space=tle.language.dsa.ascend.UB)
    t2 = t * 1.0
    off = (tl.arange(0, T0)[:, None, None] * (T1 * T2) + tl.arange(0, T1)[None, :, None] * T2 +
           tl.arange(0, T2)[None, None, :])
    tl.store(out + off, t2)


# --------------------------------------------------------- payload clamp / OOB


@triton.jit
def k_gather_payload_oob(src, out, idx, n_rows: tl.int64, m_cols: tl.constexpr, BL: tl.constexpr, TC: tl.constexpr):
    """The tile is TC wide but the payload dim's extent is only m_cols < TC.

    Columns past the extent must be padded, not read from the next row.
    """
    v = tle.make_gather_scatter_view(src, shape=[n_rows, m_cols], strides=[m_cols, 1], tile=[BL, TC], sparse_dim=[0])
    r = tl.load(idx + tl.arange(0, BL))
    t = tle.tile_load(v, index=(r, 0), dst_space=tle.language.dsa.ascend.UB)
    t2 = t * 1.0
    rows = tl.arange(0, BL)[:, None]
    cols = tl.arange(0, TC)[None, :]
    tl.store(out + rows * TC + cols, t2)


@triton.jit
def k_gather_zero_len(src, out, idx, n_rows: tl.int64, m_cols: tl.int64, BL: tl.constexpr, TC: tl.constexpr):
    """Payload block 1 starts at TC, past an extent of m_cols < TC.

    The valid length clamps to 0, so the whole block must take the pad.
    """
    v = tle.make_gather_scatter_view(src, shape=[n_rows, m_cols], strides=[m_cols, 1], tile=[BL, TC], sparse_dim=[0])
    r = tl.load(idx + tl.arange(0, BL))
    t = tle.tile_load(v, index=(r, 1), dst_space=tle.language.dsa.ascend.UB)
    t2 = t * 1.0
    rows = tl.arange(0, BL)[:, None]
    cols = tl.arange(0, TC)[None, :]
    tl.store(out + rows * TC + cols, t2)


# --------------------------------------------------------------------- scatter


@triton.jit
def k_scatter_sd(src, dst, idx, n_rows: tl.constexpr, M: tl.constexpr, BL: tl.constexpr):
    """S D scatter. An out-of-range coordinate must be dropped, not clamped."""
    rows = tl.arange(0, BL)[:, None]
    cols = tl.arange(0, M)[None, :]
    val = tl.load(src + rows * M + cols)
    r = tl.load(idx + tl.arange(0, BL))
    v = tle.make_gather_scatter_view(dst, shape=[n_rows, M], strides=[M, 1], tile=[BL, M], sparse_dim=[0])
    tle.tile_store(v, value=val, index=(r, 0), src_space=tle.language.dsa.ascend.UB)


@triton.jit
def k_scatter_ssd(src, dst, i0, i1, D0: tl.constexpr, D1: tl.constexpr, D2: tl.constexpr, T0: tl.constexpr,
                  T1: tl.constexpr, T2: tl.constexpr):
    """Two sparse dims on the store side, payload on dim 2."""
    off = (tl.arange(0, T0)[:, None, None] * (T1 * T2) + tl.arange(0, T1)[None, :, None] * T2 +
           tl.arange(0, T2)[None, None, :])
    val = tl.load(src + off)
    a = tl.load(i0 + tl.arange(0, T0))
    c = tl.load(i1 + tl.arange(0, T1))
    v = tle.make_gather_scatter_view(dst, shape=[D0, D1, D2], strides=[D1 * D2, D2, 1], tile=[T0, T1, T2],
                                     sparse_dim=[0, 1])
    tle.tile_store(v, value=val, index=(a, c, 0), src_space=tle.language.dsa.ascend.UB)


# ------------------------------------------------------- dense OOB regressions


@triton.jit
def k_gather_leading_dense_oob(src, out, idx, D0: tl.constexpr, block: tl.int32):
    v = tle.make_gather_scatter_view(src, shape=[D0, 8, 32], strides=[256, 32, 1], tile=[4, 4, 32], sparse_dim=[1])
    r = tl.load(idx + tl.arange(0, 4))
    t = tle.tile_load(v, index=(block, r, 0), dst_space=tle.language.dsa.ascend.UB)
    off = (tl.arange(0, 4)[:, None, None] * 128 + tl.arange(0, 4)[None, :, None] * 32 + tl.arange(0, 32)[None, None, :])
    tl.store(out + off, t * 1.0)


@triton.jit
def k_scatter_leading_dense_oob(src, dst, idx, D0: tl.constexpr, block: tl.int32):
    off = (tl.arange(0, 4)[:, None, None] * 128 + tl.arange(0, 4)[None, :, None] * 32 + tl.arange(0, 32)[None, None, :])
    val = tl.load(src + off)
    r = tl.load(idx + tl.arange(0, 4))
    v = tle.make_gather_scatter_view(dst, shape=[D0, 8, 32], strides=[256, 32, 1], tile=[4, 4, 32], sparse_dim=[1])
    tle.tile_store(v, value=val, index=(block, r, 0), src_space=tle.language.dsa.ascend.UB)


@triton.jit
def k_gather_negative_payload_block(src, out, idx):
    v = tle.make_gather_scatter_view(src, shape=[4, 32], strides=[32, 1], tile=[4, 32], sparse_dim=[0])
    r = tl.load(idx + tl.arange(0, 4))
    t = tle.tile_load(v, index=(r, -1), dst_space=tle.language.dsa.ascend.UB)
    off = tl.arange(0, 4)[:, None] * 32 + tl.arange(0, 32)[None, :]
    tl.store(out + off, t * 1.0)


@triton.jit
def k_scatter_negative_payload_block(src, dst, idx):
    off = tl.arange(0, 4)[:, None] * 32 + tl.arange(0, 32)[None, :]
    val = tl.load(src + off)
    r = tl.load(idx + tl.arange(0, 4))
    v = tle.make_gather_scatter_view(dst, shape=[4, 32], strides=[32, 1], tile=[4, 32], sparse_dim=[0])
    tle.tile_store(v, value=val, index=(r, -1), src_space=tle.language.dsa.ascend.UB)


# ------------------------------------------------------------------ test cases


def test_gather_sd_sparse_oob():
    # S D, with a negative and a past-extent index
    N, M, BL = 64, 128, 32
    src = torch.arange(N * M, dtype=torch.float32, device="npu").reshape(N, M)
    idx = torch.randint(0, N, (BL, ), dtype=torch.int32, device="npu")
    idx[0], idx[1] = -1, N + 7
    out = torch.full((BL, M), 99.0, dtype=torch.float32, device="npu")
    k_gather_sd[(1, )](src, out, idx, N, M, BL)
    ic = idx.cpu()
    exp = torch.zeros((BL, M))
    for i in range(BL):
        j = int(ic[i])
        if 0 <= j < N:
            exp[i] = src[j].cpu()
    torch.testing.assert_close(out.cpu(), exp, rtol=0, atol=0)


def test_gather_ss_outer_product():
    # S S -- outer product, one OOB per sparse dim
    D0, D1, T0, T1 = 64, 64, 16, 16
    src = torch.arange(D0 * D1, dtype=torch.float32, device="npu").reshape(D0, D1)
    r = torch.randint(0, D0, (T0, ), dtype=torch.int32, device="npu")
    c = torch.randint(0, D1, (T1, ), dtype=torch.int32, device="npu")
    r[0], c[1] = -1, D1 + 3
    out = torch.full((T0, T1), 99.0, dtype=torch.float32, device="npu")
    k_gather_ssd[(1, )](src, out, r, c, D0, D1, T0, T1)
    rc, cc = r.cpu(), c.cpu()
    exp = torch.zeros((T0, T1))
    for i in range(T0):
        for j in range(T1):
            a, b = int(rc[i]), int(cc[j])
            if 0 <= a < D0 and 0 <= b < D1:
                exp[i, j] = src[a, b].cpu()
    torch.testing.assert_close(out.cpu(), exp, rtol=0, atol=0)


def test_gather_dsd_non_prefix_sparse():
    # D S D -- sparse dim is not a prefix
    D0, D1, D2, T0, T1, T2 = 16, 32, 128, 4, 8, 128
    src = torch.arange(D0 * D1 * D2, dtype=torch.float32, device="npu").reshape(D0, D1, D2)
    i1 = torch.randint(0, D1, (T1, ), dtype=torch.int32, device="npu")
    i1[1], i1[2] = -1, D1 + 3
    out = torch.full((T0, T1, T2), 99.0, dtype=torch.float32, device="npu")
    k_gather_dsd[(1, )](src, out, i1, D0, D1, D2, T0, T1, T2)
    cc = i1.cpu()
    exp = torch.zeros((T0, T1, T2))
    for i in range(T0):
        for j in range(T1):
            b = int(cc[j])
            if 0 <= b < D1:
                exp[i, j] = src[i, b].cpu()
    torch.testing.assert_close(out.cpu(), exp, rtol=0, atol=0)


def test_gather_sdd_multi_dim_payload():
    # S D D -- payload spans two dims
    D0, D1, D2, T0 = 32, 8, 64, 8
    src = torch.arange(D0 * D1 * D2, dtype=torch.float32, device="npu").reshape(D0, D1, D2)
    i0 = torch.randint(0, D0, (T0, ), dtype=torch.int32, device="npu")
    i0[0], i0[1] = -1, D0 + 4
    out = torch.full((T0, D1, D2), 99.0, dtype=torch.float32, device="npu")
    k_gather_sdd[(1, )](src, out, i0, D0, D1, D2, T0)
    ac = i0.cpu()
    exp = torch.zeros((T0, D1, D2))
    for i in range(T0):
        a = int(ac[i])
        if 0 <= a < D0:
            exp[i] = src[a].cpu()
    torch.testing.assert_close(out.cpu(), exp, rtol=0, atol=0)


def test_gather_transposed_layout():
    # transposed dim_map -- must stay correct via the scalar fallback
    D0, D1, D2, T0, T1, T2 = 16, 8, 32, 4, 8, 32
    src = torch.arange(D0 * D1 * D2, dtype=torch.float32, device="npu").reshape(D0, D1, D2)
    i0 = torch.randint(0, D0, (T0, ), dtype=torch.int32, device="npu")
    out = torch.full((T0, T1, T2), 99.0, dtype=torch.float32, device="npu")
    k_gather_transposed[(1, )](src, out, i0, D0, D1, D2, T0, T1, T2)
    flat, ac = src.reshape(-1).cpu(), i0.cpu()
    exp = torch.zeros((T0, T1, T2))
    for i in range(T0):
        a = int(ac[i])
        if 0 <= a < D0:
            for j in range(T1):
                for q in range(T2):
                    exp[i, j, q] = flat[a * (D1 * D2) + j * 1 + q * D1]
    torch.testing.assert_close(out.cpu(), exp, rtol=0, atol=0)


def test_gather_payload_tile_exceeds_extent():
    # tile wider than the payload extent: the tail must be pad, and must NOT
    # be the next row's data (which is what an unclamped DMA would read).
    N, M, BL, TC = 8, 10, 4, 16
    src = torch.zeros((N, M), dtype=torch.float32, device="npu")
    for i in range(N):
        for j in range(M):
            src[i, j] = 100 * i + j
    idx = torch.arange(BL, dtype=torch.int32, device="npu")
    out = torch.full((BL, TC), -1.0, dtype=torch.float32, device="npu")
    k_gather_payload_oob[(1, )](src, out, idx, N, M, BL, TC)
    exp = torch.zeros((BL, TC))
    for i in range(BL):
        exp[i, :M] = src[int(idx.cpu()[i]), :].cpu()
    torch.testing.assert_close(out.cpu(), exp, rtol=0, atol=0)


def test_gather_payload_oob_at_buffer_end():
    # last valid row: the tail would run past the whole buffer
    N, M, BL, TC = 8, 10, 4, 16
    big = torch.full((N * M + 64, ), -7.0, dtype=torch.float32, device="npu")
    base = big[:N * M].reshape(N, M)
    for i in range(N):
        for j in range(M):
            base[i, j] = 100 * i + j
    idx = torch.full((BL, ), N - 1, dtype=torch.int32, device="npu")
    out = torch.full((BL, TC), -1.0, dtype=torch.float32, device="npu")
    k_gather_payload_oob[(1, )](big, out, idx, N, M, BL, TC)
    exp = torch.zeros((BL, TC))
    exp[:, :M] = base[N - 1].cpu()
    torch.testing.assert_close(out.cpu(), exp, rtol=0, atol=0)


def test_gather_zero_length_payload():
    # a payload block entirely past the extent -> zero-length transfer
    N, M, BL, TC = 8, 10, 4, 16
    src = torch.zeros((N, M), dtype=torch.float32, device="npu")
    for i in range(N):
        for j in range(M):
            src[i, j] = 100 * i + j
    idx = torch.full((BL, ), N - 1, dtype=torch.int32, device="npu")
    out = torch.full((BL, TC), -1.0, dtype=torch.float32, device="npu")
    k_gather_zero_len[(1, )](src, out, idx, N, M, BL, TC)
    torch.testing.assert_close(out.cpu(), torch.zeros((BL, TC)), rtol=0, atol=0)


def test_scatter_sd_sparse_oob():
    # Canaries on BOTH sides: a -1 coordinate addresses offset -M, which lands
    # *before* the buffer, so a trailing-only canary would miss it.
    N, M, BL = 64, 128, 32
    PRE = 32
    big = torch.full((PRE + N + 32, M), -7.0, dtype=torch.float32, device="npu")
    dst = big[PRE:PRE + N]
    pre0, post0 = big[:PRE].clone(), big[PRE + N:].clone()
    src = torch.arange(BL * M, dtype=torch.float32, device="npu").reshape(BL, M)
    idx = torch.randperm(N, device="npu")[:BL].to(torch.int32)
    idx[0], idx[1] = -1, N + 5
    exp = dst.clone().cpu()
    ic = idx.cpu()
    for i in range(BL):
        j = int(ic[i])
        if 0 <= j < N:
            exp[j] = src[i].cpu()
    k_scatter_sd[(1, )](src, dst, idx, N, M, BL)
    torch.testing.assert_close(dst.cpu(), exp, rtol=0, atol=0)
    torch.testing.assert_close(big[:PRE].cpu(), pre0.cpu(), rtol=0, atol=0)
    torch.testing.assert_close(big[PRE + N:].cpu(), post0.cpu(), rtol=0, atol=0)


def test_scatter_ssd_sparse_oob():
    # two sparse dims on the store side
    D0, D1, D2, T0, T1, T2 = 16, 32, 64, 4, 8, 64
    dst = torch.full((D0, D1, D2), -7.0, dtype=torch.float32, device="npu")
    src = torch.arange(T0 * T1 * T2, dtype=torch.float32, device="npu").reshape(T0, T1, T2)
    i0 = torch.randperm(D0, device="npu")[:T0].to(torch.int32)
    i1 = torch.randperm(D1, device="npu")[:T1].to(torch.int32)
    i0[0], i1[1] = -1, D1 + 2
    exp = dst.clone().cpu()
    ac, cc = i0.cpu(), i1.cpu()
    for i in range(T0):
        for j in range(T1):
            a, b = int(ac[i]), int(cc[j])
            if 0 <= a < D0 and 0 <= b < D1:
                exp[a, b] = src[i, j].cpu()
    k_scatter_ssd[(1, )](src, dst, i0, i1, D0, D1, D2, T0, T1, T2)
    torch.testing.assert_close(dst.cpu(), exp, rtol=0, atol=0)


@pytest.mark.parametrize("D0,block", [(2, 0), (3, 0), (2, 1)], ids=["partial-two-planes", "partial-tail", "fully-oob"])
def test_gather_leading_dense_oob(D0, block):
    # Eight physical planes contain even an erroneous access to planes 4..7.
    PRE, PLANE, PHYSICAL_PLANES = 256, 8 * 32, 8
    host = torch.full((PRE + PHYSICAL_PLANES * PLANE + PRE, ), -7.0)
    logical = (torch.arange(D0 * PLANE, dtype=torch.float32) + 1).reshape(D0, 8, 32)
    host[PRE:PRE + D0 * PLANE] = logical.flatten()
    backing = host.to("npu")
    src = backing[PRE:PRE + D0 * PLANE].reshape(D0, 8, 32)
    idx = torch.arange(4, dtype=torch.int32, device="npu")
    out = torch.full((4, 4, 32), 99.0, device="npu")
    exp = torch.zeros((4, 4, 32))
    for i in range(4):
        dense = block * 4 + i
        if 0 <= dense < D0:
            exp[i] = logical[dense, :4]
    k_gather_leading_dense_oob[(1, )](src, out, idx, D0, block)
    torch.testing.assert_close(out.cpu(), exp, rtol=0, atol=0)


@pytest.mark.parametrize("D0,block", [(2, 0), (3, 0), (2, 1)], ids=["partial-two-planes", "partial-tail", "fully-oob"])
def test_scatter_leading_dense_oob(D0, block):
    # Keep both canaries allocated while testing logical, not physical, OOB.
    PRE, PLANE, PHYSICAL_PLANES = 256, 8 * 32, 8
    backing = torch.full((PRE + PHYSICAL_PLANES * PLANE + PRE, ), -7.0, device="npu")
    dst = backing[PRE:PRE + D0 * PLANE].reshape(D0, 8, 32)
    values = (torch.arange(4 * 4 * 32, dtype=torch.float32) + 1).reshape(4, 4, 32)
    src = values.to("npu")
    idx = torch.arange(4, dtype=torch.int32, device="npu")
    exp = backing.cpu().clone()
    logical = exp[PRE:PRE + D0 * PLANE].reshape(D0, 8, 32)
    for i in range(4):
        dense = block * 4 + i
        if 0 <= dense < D0:
            logical[dense, :4] = values[i]
    k_scatter_leading_dense_oob[(1, )](src, dst, idx, D0, block)
    # Include untouched sparse rows and both canaries in the same assertion.
    torch.testing.assert_close(backing.cpu(), exp, rtol=0, atol=0)


def test_gather_negative_payload_block():
    # The erroneous row-0 access starts at -32; leave a full row before src.
    PRE, N, M = 32, 4, 32
    host = torch.full((PRE + N * M + M, ), -7.0)
    host[PRE:PRE + N * M] = torch.arange(N * M, dtype=torch.float32) + 1
    backing = host.to("npu")
    src = backing[PRE:PRE + N * M].reshape(N, M)
    idx = torch.arange(N, dtype=torch.int32, device="npu")
    out = torch.full((N, M), 99.0, device="npu")
    k_gather_negative_payload_block[(1, )](src, out, idx)
    # Block -1 covers columns [-32, 0), invalid for every sparse row.
    torch.testing.assert_close(out.cpu(), torch.zeros((N, M)), rtol=0, atol=0)


def test_scatter_negative_payload_block():
    PRE, N, M = 32, 4, 32
    backing = torch.full((PRE + N * M + M, ), -7.0, device="npu")
    dst = backing[PRE:PRE + N * M].reshape(N, M)
    src = (torch.arange(N * M, dtype=torch.float32, device="npu") + 1).reshape(N, M)
    idx = torch.arange(N, dtype=torch.int32, device="npu")
    exp = backing.cpu().clone()
    k_scatter_negative_payload_block[(1, )](src, dst, idx)
    # A bad transfer can hit either the prefix or a preceding logical row.
    torch.testing.assert_close(backing.cpu(), exp, rtol=0, atol=0)
