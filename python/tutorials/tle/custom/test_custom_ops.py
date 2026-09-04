# Copyright 2026- Xcoresigma Technology Co., Ltd
"""
Ascend custom op per-op correctness tests
===================================================

Minimal standalone correctness checks for every op registered in
custom_ops.bc, one group per op:

  - gather_gm_to_l1      (fp16 / bf16): verified through a following tl.dot
  - gather_gm_to_ub      (fp16 / bf16): verified by storing the result to GM
  - sort_1d_pack         (all three sort paths BASE / S4096_K129_512 /
                          S4096_K1_128_K2048, plus an index_offset case)
  - merge_exhaust_sort4  (4-way / 2-way / 3-way with a hole way)
  - unpack_sort          (split proposals into value / index)

Correctness only, no benchmarking.
"""

import numpy as np
import torch
import torch_npu
import triton
import triton.experimental.tle as tle
import triton.language as tl
from triton.experimental.tle.language.dsa.ascend.custom_ops import (
    SORT_IMPL_BASE,
    SORT_IMPL_S4096_K129_512,
    SORT_IMPL_S4096_K1_128_K2048,
)

DEVICE = "npu"

SMALLK_SORT_CHUNK = 1024
SMALLK_SORT_WAYS = 4

# ══════════════════════════════════════════════════════════════════════════
# Kernels
# ══════════════════════════════════════════════════════════════════════════


@triton.jit
def gather_gm_to_l1_dot_kernel(
    src,
    src_index,
    query,
    output,
    NUM_ROWS: tl.constexpr,
    TILE_SIZE: tl.constexpr,
    D: tl.constexpr,
    DTYPE: tl.constexpr,
):
    src_2d = tl.make_block_ptr(
        base=src,
        shape=(NUM_ROWS, D),
        strides=(D, 1),
        offsets=(0, 0),
        block_shape=(TILE_SIZE, D),
        order=(1, 0),
    )
    src_index_2d = tl.make_block_ptr(
        base=src_index,
        shape=(TILE_SIZE, 1),
        strides=(1, 1),
        offsets=(0, 0),
        block_shape=(TILE_SIZE, 1),
        order=(1, 0),
    )
    tile_k = tl.full((TILE_SIZE, D), 0, DTYPE)
    tile_k = tle.dsa.ascend.raw(
        "gather_gm_to_l1",
        src_2d,
        src_index_2d,
        TILE_SIZE,
        D,
        out=tile_k,
    )

    query_2d = tl.make_block_ptr(
        base=query,
        shape=(TILE_SIZE, D),
        strides=(D, 1),
        offsets=(0, 0),
        block_shape=(TILE_SIZE, D),
        order=(1, 0),
    )
    tile_q = tl.load(query_2d)
    result = tl.dot(tile_q, tl.trans(tile_k))

    output_2d = tl.make_block_ptr(
        base=output,
        shape=(TILE_SIZE, TILE_SIZE),
        strides=(TILE_SIZE, 1),
        offsets=(0, 0),
        block_shape=(TILE_SIZE, TILE_SIZE),
        order=(1, 0),
    )
    tl.store(output_2d, result)


@triton.jit
def gather_gm_to_ub_store_kernel(
    src,
    src_index,
    output,
    NUM_ROWS: tl.constexpr,
    TILE_SIZE: tl.constexpr,
    D: tl.constexpr,
    DTYPE: tl.constexpr,
):
    src_2d = tl.make_block_ptr(
        base=src,
        shape=(NUM_ROWS, D),
        strides=(D, 1),
        offsets=(0, 0),
        block_shape=(TILE_SIZE, D),
        order=(1, 0),
    )
    src_index_2d = tl.make_block_ptr(
        base=src_index,
        shape=(TILE_SIZE, 1),
        strides=(1, 1),
        offsets=(0, 0),
        block_shape=(TILE_SIZE, 1),
        order=(1, 0),
    )
    tile_v = tl.full((TILE_SIZE, D), 0, DTYPE)
    tile_v = tle.dsa.ascend.raw(
        "gather_gm_to_ub",
        src_2d,
        src_index_2d,
        TILE_SIZE,
        D,
        out=tile_v,
    )

    output_2d = tl.make_block_ptr(
        base=output,
        shape=(TILE_SIZE, D),
        strides=(D, 1),
        offsets=(0, 0),
        block_shape=(TILE_SIZE, D),
        order=(1, 0),
    )
    tl.store(output_2d, tile_v)


@triton.jit
def sort_pack_kernel(X, OutGM, N: tl.constexpr, K: tl.constexpr, INDEX_OFFSET: tl.constexpr, SORT_IMPL: tl.constexpr,
                     TMP_SIZE: tl.constexpr):
    """Single-segment sort_1d_pack: load N f32 values from GM into UB, sort,
    then write the K proposals (2K f32) back to OutGM."""
    NP2: tl.constexpr = triton.next_power_of_2(N)
    KP2: tl.constexpr = triton.next_power_of_2(K * 2)
    tmp = tl.zeros([TMP_SIZE], dtype=tl.float32)
    src_ub = tle.dsa.alloc([N], dtype=tl.float32, mem_addr_space=tle.dsa.ascend.UB)
    tle.dsa.copy(X + tl.arange(0, NP2), src_ub, [N])
    props = tl.zeros([KP2], dtype=tl.float32)
    props = tle.dsa.ascend.raw("sort_1d_pack", tle.dsa.to_tensor(src_ub), tmp, True, K, INDEX_OFFSET, SORT_IMPL,
                               out=props)
    tl.store(OutGM + tl.arange(0, KP2), props)


@triton.jit
def merge_exhaust_kernel(SrcGM, OutGM, ConsGM, WAY_CAP: tl.constexpr, WAYS: tl.constexpr, L0: tl.constexpr,
                         L1: tl.constexpr, L2: tl.constexpr, L3: tl.constexpr, OUT_CAP: tl.constexpr):
    """One merge_exhaust_sort4 call: the four proposal ways sit in a single
    UB buffer at fixed offsets 0/1/2/3 * WAY_CAP; merge once, then write the
    safe prefix and the per-way consumed counts back to GM."""
    IN_LEN: tl.constexpr = 4 * WAY_CAP * 2
    IN_P2: tl.constexpr = triton.next_power_of_2(IN_LEN)
    OUT_P2: tl.constexpr = triton.next_power_of_2(OUT_CAP * 2)
    in_ub = tle.dsa.alloc([IN_LEN], dtype=tl.float32, mem_addr_space=tle.dsa.ascend.UB)
    out_t = tl.zeros([OUT_P2], dtype=tl.float32)
    cons = tl.zeros([4], dtype=tl.int32)

    tle.dsa.copy(SrcGM + tl.arange(0, IN_P2), in_ub, [IN_LEN])
    out_t, cons = tle.dsa.ascend.raw("merge_exhaust_sort4", tle.dsa.to_tensor(in_ub), WAYS, 0 * WAY_CAP, 1 * WAY_CAP,
                                     2 * WAY_CAP, 3 * WAY_CAP, L0, L1, L2, L3, out=[out_t, cons])

    tl.store(OutGM + tl.arange(0, OUT_P2), out_t)
    tl.store(ConsGM + tl.arange(0, 4), cons)


@triton.jit
def unpack_sort_kernel(SrcGM, Yv, Yi, K: tl.constexpr):
    """unpack_sort: split K proposals into two GM outputs, value (f32) and
    index (i32)."""
    KP2: tl.constexpr = triton.next_power_of_2(K * 2)
    s_ub = tle.dsa.alloc([K * 2], dtype=tl.float32, mem_addr_space=tle.dsa.ascend.UB)
    dval = tl.zeros([K], dtype=tl.float32)
    didx = tl.zeros([K], dtype=tl.int32)
    offs = tl.arange(0, K)
    tle.dsa.copy(SrcGM + tl.arange(0, KP2), s_ub, [K * 2])
    dval, didx = tle.dsa.ascend.raw("unpack_sort", tle.dsa.to_tensor(s_ub), K, out=[dval, didx])
    tl.store(Yv + offs, dval)
    tl.store(Yi + offs, didx)


# ══════════════════════════════════════════════════════════════════════════
# Host-side helpers
# ══════════════════════════════════════════════════════════════════════════


def _encode_props(values, indices):
    """(value f32, index i32) lists -> interleaved f32 numpy array
    [v0, idx0_as_f32, v1, idx1_as_f32, ...]."""
    raw = np.empty(2 * len(values), dtype=np.int32)
    raw[0::2] = np.asarray(values, dtype=np.float32).view(np.int32)
    raw[1::2] = np.asarray(indices, dtype=np.int32)
    return raw.view(np.float32)


def _decode_props(props):
    """Interleaved proposal f32 array -> (value f32, index i32)."""
    raw = np.ascontiguousarray(props, dtype=np.float32).view(np.int32)
    return raw[0::2].view(np.float32), raw[1::2]


def _sort_tmp_size(seg_len: int, sort_run_len: int, sort_impl: int) -> int:
    if sort_impl == SORT_IMPL_BASE:
        return seg_len * 4
    if sort_impl == SORT_IMPL_S4096_K1_128_K2048:
        props_ab = seg_len * 2
        group_buf = 4 * 512 * 2
        return 2 * props_ab + 2 * group_buf + (2 * group_buf + 8)
    candidates = SMALLK_SORT_WAYS * sort_run_len * 2
    chunk_tmp = SMALLK_SORT_CHUNK * 4
    merge_out = SMALLK_SORT_WAYS * sort_run_len * 2
    return candidates + chunk_tmp + merge_out


# ══════════════════════════════════════════════════════════════════════════
# gather_gm_to_l1 / gather_gm_to_ub
# ══════════════════════════════════════════════════════════════════════════

NUM_ROWS = 32
TILE_SIZE = 16
D = 16
L1_INDEX = [3, 4, 9, 10, 2, 7, 0, 1, 12, 13, 5, 8, 11, 6, 15, 14]
UB_INDEX = [20, 21, 19, 30, 31, 25, 24, 18, 27, 28, 22, 29, 26, 23, 17, 16]

_TL_DTYPE = {torch.float16: tl.float16, torch.bfloat16: tl.bfloat16}


def test_gather_gm_to_l1(torch_dtype, tol):
    torch.manual_seed(0)
    src = torch.randn((NUM_ROWS, D), dtype=torch_dtype, device=DEVICE)
    query = torch.randn((TILE_SIZE, D), dtype=torch_dtype, device=DEVICE)
    output = torch.empty((TILE_SIZE, TILE_SIZE), dtype=torch.float32, device=DEVICE)
    src_index = torch.tensor(L1_INDEX, dtype=torch.int32, device=DEVICE)

    gather_gm_to_l1_dot_kernel[(1, )](
        src,
        src_index,
        query,
        output,
        NUM_ROWS=NUM_ROWS,
        TILE_SIZE=TILE_SIZE,
        D=D,
        DTYPE=_TL_DTYPE[torch_dtype],
        disable_auto_cv_work_space_manage=True,
    )
    torch_npu.npu.synchronize()

    gathered_src = src[src_index.long(), :]
    expected = torch.matmul(query.float(), gathered_src.float().transpose(0, 1))
    torch.testing.assert_close(output.cpu(), expected.cpu(), rtol=tol, atol=tol)
    print(f"[PASS] gather_gm_to_l1 dot correctness ({torch_dtype})")


def test_gather_gm_to_ub(torch_dtype):
    torch.manual_seed(0)
    src = torch.randn((NUM_ROWS, D), dtype=torch_dtype, device=DEVICE)
    output = torch.empty((TILE_SIZE, D), dtype=torch_dtype, device=DEVICE)
    src_index = torch.tensor(UB_INDEX, dtype=torch.int32, device=DEVICE)

    gather_gm_to_ub_store_kernel[(1, )](
        src,
        src_index,
        output,
        NUM_ROWS=NUM_ROWS,
        TILE_SIZE=TILE_SIZE,
        D=D,
        DTYPE=_TL_DTYPE[torch_dtype],
        disable_auto_cv_work_space_manage=True,
    )
    torch_npu.npu.synchronize()

    expected = src[src_index.long(), :]
    torch.testing.assert_close(output.cpu(), expected.cpu(), rtol=0, atol=0)
    print(f"[PASS] gather_gm_to_ub store correctness ({torch_dtype})")


# ══════════════════════════════════════════════════════════════════════════
# sort_1d_pack
# ══════════════════════════════════════════════════════════════════════════


def _run_sort_case(n, k, index_offset, sort_impl, label):
    torch.manual_seed(0)
    x = torch.rand((n, ), dtype=torch.float32, device=DEVICE)
    out = torch.zeros((triton.next_power_of_2(k * 2), ), dtype=torch.float32, device=DEVICE)

    sort_pack_kernel[(1, )](
        x,
        out,
        N=n,
        K=k,
        INDEX_OFFSET=index_offset,
        SORT_IMPL=sort_impl,
        TMP_SIZE=_sort_tmp_size(n, k, sort_impl),
    )
    torch_npu.npu.synchronize()

    values, indices = _decode_props(out.cpu().numpy()[:k * 2])
    x_np = x.cpu().numpy()
    ref = np.sort(x_np)[::-1][:k]

    assert indices.min() >= index_offset and indices.max() < index_offset + n, (
        f"{label}: index out of range [{indices.min()}, {indices.max()}]")
    np.testing.assert_allclose(values, ref, rtol=1e-5, atol=1e-5,
                               err_msg=f"{label}: values are not the descending top-{k}")
    np.testing.assert_allclose(x_np[indices.astype(np.int64) - index_offset], values, rtol=1e-5, atol=1e-5,
                               err_msg=f"{label}: index does not match value")
    print(f"[PASS] sort_1d_pack {label} (N={n}, K={k}, offset={index_offset})")


def test_sort_1d_pack():
    # BASE generic path + a non-zero index_offset case
    _run_sort_case(2048, 100, 0, SORT_IMPL_BASE, "BASE")
    _run_sort_case(2048, 100, 500, SORT_IMPL_BASE, "BASE index_offset")
    # 4x1024 small-K path: 4096 inputs, 128 < K <= 512
    _run_sort_case(4096, 256, 0, SORT_IMPL_S4096_K129_512, "S4096_K129_512")
    # Layered path: 4096 inputs; K <= 128 can early-stop, K == 2048 uses the
    # fixed merge tree
    _run_sort_case(4096, 64, 0, SORT_IMPL_S4096_K1_128_K2048, "S4096_K1_128_K2048 early-stop")
    _run_sort_case(4096, 2048, 0, SORT_IMPL_S4096_K1_128_K2048, "S4096_K1_128_K2048 fixed-tree")


# ══════════════════════════════════════════════════════════════════════════
# merge_exhaust_sort4
# ══════════════════════════════════════════════════════════════════════════


def _make_way(rng, length, tag):
    """Generate one descending proposal way: globally unique values with a
    one-to-one value/index mapping."""
    values = np.sort(rng.random(length))[::-1].astype(np.float32)
    indices = np.arange(length, dtype=np.int32) + tag * 100000
    return list(zip(values.tolist(), indices.astype(np.int64).tolist()))


def _run_merge_case(lengths, label, seed=0):
    rng = np.random.default_rng(seed)
    way_cap = max(lengths)
    src = np.zeros(4 * way_cap * 2, dtype=np.float32)
    ways = []
    for tag, ln in enumerate(lengths):
        if ln > 0:
            way = _make_way(rng, ln, tag)
            ways.append((tag, way))
            src[tag * way_cap * 2:tag * way_cap * 2 + ln * 2] = _encode_props(*zip(*way))

    src_gm = torch.from_numpy(src).to(DEVICE)
    out_gm = torch.zeros((4 * way_cap * 2, ), dtype=torch.float32, device=DEVICE)
    cons_gm = torch.zeros((4, ), dtype=torch.int32, device=DEVICE)

    merge_exhaust_kernel[(1, )](
        src_gm,
        out_gm,
        cons_gm,
        WAY_CAP=way_cap,
        WAYS=sum(1 for ln in lengths if ln > 0),
        L0=lengths[0],
        L1=lengths[1],
        L2=lengths[2],
        L3=lengths[3],
        OUT_CAP=4 * way_cap,
    )
    torch_npu.npu.synchronize()

    cons = cons_gm.cpu().numpy()
    total = int(cons[:len(lengths)].sum())
    assert total > 0, f"{label}: merge produced no output"
    for tag, ln in enumerate(lengths):
        assert 0 <= cons[tag] <= ln, (f"{label}: way{tag} consumed count {cons[tag]} exceeds length {ln}")

    out_v, out_i = _decode_props(out_gm.cpu().numpy()[:total * 2])
    assert np.all(out_v[:-1] >= out_v[1:]), f"{label}: output prefix is not descending"

    out_pairs = list(zip(out_v.tolist(), out_i.tolist()))
    full_merge = sorted((p for _, way in ways for p in way), key=lambda p: -p[0])
    assert out_pairs == full_merge[:total], (f"{label}: output is not a safe prefix of the full merge")

    value_to_way = {p[0]: tag for tag, way in ways for p in way}
    for tag, way in ways:
        used = [p for p in out_pairs if value_to_way[p[0]] == tag]
        assert used == way[:cons[tag]], (
            f"{label}: way{tag} proposals actually consumed do not match the reported consumed count")
    print(f"[PASS] merge_exhaust_sort4 {label} "
          f"(lens={lengths}, consumed={cons[:len(lengths)].tolist()}, total={total})")


def test_merge_exhaust_sort4():
    _run_merge_case((64, 64, 64, 64), "4-way")
    _run_merge_case((64, 64, 0, 0), "2-way")
    _run_merge_case((64, 0, 64, 32), "3-way with hole")


# ══════════════════════════════════════════════════════════════════════════
# unpack_sort
# ══════════════════════════════════════════════════════════════════════════


def test_unpack_sort():
    rng = np.random.default_rng(0)
    k = 128
    values = np.sort(rng.random(k))[::-1].astype(np.float32)
    indices = rng.integers(0, 10000, size=k).astype(np.int32)
    src = torch.from_numpy(_encode_props(values, indices)).to(DEVICE)
    yv = torch.zeros((k, ), dtype=torch.float32, device=DEVICE)
    yi = torch.zeros((k, ), dtype=torch.int32, device=DEVICE)

    unpack_sort_kernel[(1, )](src, yv, yi, K=k)
    torch_npu.npu.synchronize()

    np.testing.assert_array_equal(yv.cpu().numpy(), values, err_msg="unpack_sort values mismatch")
    np.testing.assert_array_equal(yi.cpu().numpy(), indices, err_msg="unpack_sort indices mismatch")
    print("[PASS] unpack_sort correctness")


def main():
    for torch_dtype, tol in ((torch.float16, 1e-3), (torch.bfloat16, 1e-2)):
        test_gather_gm_to_l1(torch_dtype, tol)
        test_gather_gm_to_ub(torch_dtype)
    test_sort_1d_pack()
    test_merge_exhaust_sort4()
    test_unpack_sort()
    print("\nAll custom op correctness tests passed.")


if __name__ == "__main__":
    main()
