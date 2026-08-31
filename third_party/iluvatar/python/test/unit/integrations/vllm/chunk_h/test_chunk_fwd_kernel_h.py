# SPDX-License-Identifier: Apache-2.0
"""vLLM / FLA `chunk_gated_delta_rule_fwd_h` integration.

A ragged last chunk (T < BT) uses `make_block_ptr` + `boundary_check` without
`padding_option="zero"`. After pipelining, that load becomes
`ttg.async_copy_global_to_local` with a mask and no `other`; masked-off shared
memory must still be zero-filled or `tl.dot` along T spreads NaN into
`final_state`.
"""

from __future__ import annotations

import pytest
import torch
import triton
import triton.language as tl

BT = 64
FIXED_BV = 32
FIXED_NUM_WARPS = 2
FIXED_NUM_STAGES = 2

exp = tl.exp
exp2 = tl.exp2


def prepare_chunk_indices(cu_seqlens: torch.Tensor, chunk_size: int) -> torch.Tensor:
    lens = cu_seqlens[1:] - cu_seqlens[:-1]
    indices = torch.cat([torch.arange(n) for n in triton.cdiv(lens, chunk_size).tolist()])
    return torch.stack([indices.eq(0).cumsum(0) - 1, indices], 1).to(cu_seqlens)


def prepare_chunk_offsets(cu_seqlens: torch.Tensor, chunk_size: int) -> torch.Tensor:
    lens = cu_seqlens[1:] - cu_seqlens[:-1]
    return torch.cat([cu_seqlens.new_tensor([0]), triton.cdiv(lens, chunk_size)]).cumsum(-1)


@triton.heuristics({
    "USE_G": lambda args: args["g"] is not None,
    "USE_GK": lambda args: args["gk"] is not None,
    "USE_INITIAL_STATE": lambda args: args["h0"] is not None,
    "STORE_FINAL_STATE": lambda args: args["ht"] is not None,
    "SAVE_NEW_VALUE": lambda args: args["v_new"] is not None,
    "IS_VARLEN": lambda args: args["cu_seqlens"] is not None,
})
@triton.jit(do_not_specialize=["T"])
def chunk_gated_delta_rule_fwd_kernel_h_blockdim64(
    k,
    v,
    w,
    v_new,
    g,
    gk,
    h,
    h0,
    ht,
    cu_seqlens,
    chunk_offsets,
    T,
    H: tl.constexpr,
    Hg: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BV: tl.constexpr,
    USE_G: tl.constexpr,
    USE_GK: tl.constexpr,
    USE_INITIAL_STATE: tl.constexpr,
    STORE_FINAL_STATE: tl.constexpr,
    SAVE_NEW_VALUE: tl.constexpr,
    IS_VARLEN: tl.constexpr,
    USE_EXP2: tl.constexpr,
):
    i_v, i_nh = tl.program_id(0), tl.program_id(1)
    i_n, i_h = i_nh // H, i_nh % H
    if IS_VARLEN:
        bos, eos = tl.load(cu_seqlens + i_n).to(tl.int32), tl.load(cu_seqlens + i_n + 1).to(tl.int32)
        T = eos - bos
        NT = tl.cdiv(T, BT)
        boh = tl.load(chunk_offsets + i_n).to(tl.int32)
    else:
        bos, eos = i_n * T, i_n * T + T
        NT = tl.cdiv(T, BT)
        boh = i_n * NT

    b_h1 = tl.zeros([BV, 64], dtype=tl.float32)
    if K > 64:
        b_h2 = tl.zeros([BV, 64], dtype=tl.float32)
    if K > 128:
        b_h3 = tl.zeros([BV, 64], dtype=tl.float32)
    if K > 192:
        b_h4 = tl.zeros([BV, 64], dtype=tl.float32)

    h += ((boh * H + i_h) * V * K).to(tl.int64)
    v += ((bos * H + i_h) * V).to(tl.int64)
    k += ((bos * Hg + i_h // (H // Hg)) * K).to(tl.int64)
    w += ((bos * H + i_h) * K).to(tl.int64)
    if SAVE_NEW_VALUE:
        v_new += ((bos * H + i_h) * V).to(tl.int64)
    stride_v = H * V
    stride_h = H * V * K
    stride_k = Hg * K
    stride_w = H * K
    if USE_INITIAL_STATE:
        h0 = h0 + i_nh * V * K
    if STORE_FINAL_STATE:
        ht = ht + i_nh * V * K

    if USE_INITIAL_STATE:
        p_h0_1 = tl.make_block_ptr(h0, (V, K), (K, 1), (i_v * BV, 0), (BV, 64), (1, 0))
        b_h1 += tl.load(p_h0_1, boundary_check=(0, 1)).to(tl.float32)
        if K > 64:
            p_h0_2 = tl.make_block_ptr(h0, (V, K), (K, 1), (i_v * BV, 64), (BV, 64), (1, 0))
            b_h2 += tl.load(p_h0_2, boundary_check=(0, 1)).to(tl.float32)
        if K > 128:
            p_h0_3 = tl.make_block_ptr(h0, (V, K), (K, 1), (i_v * BV, 128), (BV, 64), (1, 0))
            b_h3 += tl.load(p_h0_3, boundary_check=(0, 1)).to(tl.float32)
        if K > 192:
            p_h0_4 = tl.make_block_ptr(h0, (V, K), (K, 1), (i_v * BV, 192), (BV, 64), (1, 0))
            b_h4 += tl.load(p_h0_4, boundary_check=(0, 1)).to(tl.float32)

    for i_t in range(NT):
        p_h1 = tl.make_block_ptr(h + i_t.to(tl.int64) * stride_h, (V, K), (K, 1), (i_v * BV, 0), (BV, 64), (1, 0))
        tl.store(p_h1, b_h1.to(p_h1.dtype.element_ty), boundary_check=(0, 1))
        if K > 64:
            p_h2 = tl.make_block_ptr(h + i_t.to(tl.int64) * stride_h, (V, K), (K, 1), (i_v * BV, 64), (BV, 64), (1, 0))
            tl.store(p_h2, b_h2.to(p_h2.dtype.element_ty), boundary_check=(0, 1))
        if K > 128:
            p_h3 = tl.make_block_ptr(h + i_t.to(tl.int64) * stride_h, (V, K), (K, 1), (i_v * BV, 128), (BV, 64), (1, 0))
            tl.store(p_h3, b_h3.to(p_h3.dtype.element_ty), boundary_check=(0, 1))
        if K > 192:
            p_h4 = tl.make_block_ptr(h + i_t.to(tl.int64) * stride_h, (V, K), (K, 1), (i_v * BV, 192), (BV, 64), (1, 0))
            tl.store(p_h4, b_h4.to(p_h4.dtype.element_ty), boundary_check=(0, 1))

        p_w = tl.make_block_ptr(w, (T, K), (stride_w, 1), (i_t * BT, 0), (BT, 64), (1, 0))
        b_w = tl.load(p_w, boundary_check=(0, 1))
        b_v = tl.dot(b_w, tl.trans(b_h1).to(b_w.dtype))
        if K > 64:
            p_w = tl.make_block_ptr(w, (T, K), (stride_w, 1), (i_t * BT, 64), (BT, 64), (1, 0))
            b_w = tl.load(p_w, boundary_check=(0, 1))
            b_v += tl.dot(b_w, tl.trans(b_h2).to(b_w.dtype))
        if K > 128:
            p_w = tl.make_block_ptr(w, (T, K), (stride_w, 1), (i_t * BT, 128), (BT, 64), (1, 0))
            b_w = tl.load(p_w, boundary_check=(0, 1))
            b_v += tl.dot(b_w, tl.trans(b_h3).to(b_w.dtype))
        if K > 192:
            p_w = tl.make_block_ptr(w, (T, K), (stride_w, 1), (i_t * BT, 192), (BT, 64), (1, 0))
            b_w = tl.load(p_w, boundary_check=(0, 1))
            b_v += tl.dot(b_w, tl.trans(b_h4).to(b_w.dtype))
        p_v = tl.make_block_ptr(v, (T, V), (stride_v, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
        b_v = tl.load(p_v, boundary_check=(0, 1)) - b_v

        if SAVE_NEW_VALUE:
            p_v = tl.make_block_ptr(v_new, (T, V), (stride_v, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
            tl.store(p_v, b_v.to(p_v.dtype.element_ty), boundary_check=(0, 1))

        last_idx = min((i_t.to(tl.int64) + 1) * BT, T) - 1
        if USE_G:
            m_t = (i_t.to(tl.int64) * BT + tl.arange(0, BT)) < T
            b_g_last = tl.load(g + bos * H + last_idx * H + i_h)
            p_g = tl.make_block_ptr(g + bos * H + i_h, (T, ), (H, ), (i_t * BT, ), (BT, ), (0, ))
            b_g = tl.load(p_g, boundary_check=(0, ))
            if USE_EXP2:
                b_v = b_v * tl.where(m_t, exp2(b_g_last - b_g), 0)[:, None]
                b_g_last = exp2(b_g_last)
            else:
                b_v = b_v * tl.where(m_t, exp(b_g_last - b_g), 0)[:, None]
                b_g_last = exp(b_g_last)
            b_h1 *= b_g_last
            if K > 64:
                b_h2 *= b_g_last
            if K > 128:
                b_h3 *= b_g_last
            if K > 192:
                b_h4 *= b_g_last

        if USE_GK:
            o_k1 = tl.arange(0, 64)
            b_gk_last1 = tl.load(gk + (bos + last_idx) * H * K + i_h * K + o_k1, mask=(o_k1 < K), other=0.0)
            if USE_EXP2:
                b_h1 *= exp2(b_gk_last1)[None, :]
            else:
                b_h1 *= exp(b_gk_last1)[None, :]
            if K > 64:
                o_k2 = 64 + o_k1
                b_gk_last2 = tl.load(gk + (bos + last_idx) * H * K + i_h * K + o_k2, mask=(o_k2 < K), other=0.0)
                if USE_EXP2:
                    b_h2 *= exp2(b_gk_last2)[None, :]
                else:
                    b_h2 *= exp(b_gk_last2)[None, :]
            if K > 128:
                o_k3 = 128 + o_k1
                b_gk_last3 = tl.load(gk + (bos + last_idx) * H * K + i_h * K + o_k3, mask=(o_k3 < K), other=0.0)
                if USE_EXP2:
                    b_h3 *= exp2(b_gk_last3)[None, :]
                else:
                    b_h3 *= exp(b_gk_last3)[None, :]
            if K > 192:
                o_k4 = 192 + o_k1
                b_gk_last4 = tl.load(gk + (bos + last_idx) * H * K + i_h * K + o_k4, mask=(o_k4 < K), other=0.0)
                if USE_EXP2:
                    b_h4 *= exp2(b_gk_last4)[None, :]
                else:
                    b_h4 *= exp(b_gk_last4)[None, :]
        b_v = b_v.to(k.dtype.element_ty)

        p_k = tl.make_block_ptr(k, (K, T), (1, stride_k), (0, i_t * BT), (64, BT), (0, 1))
        b_k = tl.load(p_k, boundary_check=(0, 1))
        b_h1 += tl.trans(tl.dot(b_k, b_v))
        if K > 64:
            p_k = tl.make_block_ptr(k, (K, T), (1, stride_k), (64, i_t * BT), (64, BT), (0, 1))
            b_k = tl.load(p_k, boundary_check=(0, 1))
            b_h2 += tl.trans(tl.dot(b_k, b_v))
        if K > 128:
            p_k = tl.make_block_ptr(k, (K, T), (1, stride_k), (128, i_t * BT), (64, BT), (0, 1))
            b_k = tl.load(p_k, boundary_check=(0, 1))
            b_h3 += tl.trans(tl.dot(b_k, b_v))
        if K > 192:
            p_k = tl.make_block_ptr(k, (K, T), (1, stride_k), (192, i_t * BT), (64, BT), (0, 1))
            b_k = tl.load(p_k, boundary_check=(0, 1))
            b_h4 += tl.trans(tl.dot(b_k, b_v))

    if STORE_FINAL_STATE:
        p_ht = tl.make_block_ptr(ht, (V, K), (K, 1), (i_v * BV, 0), (BV, 64), (1, 0))
        tl.store(p_ht, b_h1.to(p_ht.dtype.element_ty), boundary_check=(0, 1))
        if K > 64:
            p_ht = tl.make_block_ptr(ht, (V, K), (K, 1), (i_v * BV, 64), (BV, 64), (1, 0))
            tl.store(p_ht, b_h2.to(p_ht.dtype.element_ty), boundary_check=(0, 1))
        if K > 128:
            p_ht = tl.make_block_ptr(ht, (V, K), (K, 1), (i_v * BV, 128), (BV, 64), (1, 0))
            tl.store(p_ht, b_h3.to(p_ht.dtype.element_ty), boundary_check=(0, 1))
        if K > 192:
            p_ht = tl.make_block_ptr(ht, (V, K), (K, 1), (i_v * BV, 192), (BV, 64), (1, 0))
            tl.store(p_ht, b_h4.to(p_ht.dtype.element_ty), boundary_check=(0, 1))


def chunk_gated_delta_rule_fwd_h(k, w, u, g=None, initial_state=None, output_final_state=False, cu_seqlens=None):
    B, T, Hg, K, V = *k.shape, u.shape[-1]
    H = u.shape[-2]
    if cu_seqlens is None:
        N, NT, chunk_offsets = B, triton.cdiv(T, BT), None
    else:
        chunk_indices = prepare_chunk_indices(cu_seqlens, BT)
        N, NT = len(cu_seqlens) - 1, len(chunk_indices)
        chunk_offsets = prepare_chunk_offsets(cu_seqlens, BT)

    h = k.new_empty(B, NT, H, V, K)
    final_state = k.new_empty(N, H, V, K, dtype=torch.float32) if output_final_state else None
    v_new = torch.empty_like(u)

    grid = (triton.cdiv(V, FIXED_BV), N * H)
    chunk_gated_delta_rule_fwd_kernel_h_blockdim64[grid](k=k, v=u, w=w, v_new=v_new, g=g, gk=None, h=h,
                                                         h0=initial_state, ht=final_state, cu_seqlens=cu_seqlens,
                                                         chunk_offsets=chunk_offsets, T=T, H=H, Hg=Hg, K=K, V=V, BT=BT,
                                                         BV=FIXED_BV, USE_EXP2=False, num_warps=FIXED_NUM_WARPS,
                                                         num_stages=FIXED_NUM_STAGES)
    return h, v_new, final_state


def chunk_gated_delta_rule_fwd_h_ref(k, w, u, g, initial_state, chunk_size=BT):
    B, T, Hg, K = k.shape
    H, V = u.shape[-2], u.shape[-1]
    NT = (T + chunk_size - 1) // chunk_size
    group = H // Hg
    h_out = torch.empty(B, NT, H, V, K, device=k.device, dtype=k.dtype)
    v_new = torch.empty_like(u)
    final_state = torch.empty(B, H, V, K, device=k.device, dtype=torch.float32)

    for b in range(B):
        for i_h in range(H):
            hg = i_h // group
            if initial_state is None:
                h_state = torch.zeros(V, K, device=k.device, dtype=torch.float32)
            else:
                # Kernel reads a V*K block as (V, K), strides (K, 1).
                h_state = initial_state[b, i_h].reshape(V, K).float()
            for i_t in range(NT):
                t0 = i_t * chunk_size
                t1 = min(t0 + chunk_size, T)
                n_t = t1 - t0
                h_out[b, i_t, i_h] = h_state.to(k.dtype)

                w_chunk = torch.zeros(chunk_size, K, device=k.device, dtype=w.dtype)
                u_chunk = torch.zeros(chunk_size, V, device=k.device, dtype=u.dtype)
                k_chunk = torch.zeros(chunk_size, K, device=k.device, dtype=k.dtype)
                w_chunk[:n_t] = w[b, t0:t1, i_h]
                u_chunk[:n_t] = u[b, t0:t1, i_h]
                k_chunk[:n_t] = k[b, t0:t1, hg]

                b_v = u_chunk.float() - (w_chunk @ h_state.to(w.dtype).T).float()
                v_new[b, t0:t1, i_h] = b_v[:n_t].to(u.dtype)

                if g is not None:
                    g_last = g[b, t1 - 1, i_h].float()
                    g_chunk = torch.zeros(chunk_size, device=k.device, dtype=torch.float32)
                    g_chunk[:n_t] = g[b, t0:t1, i_h]
                    in_bound = (t0 + torch.arange(chunk_size, device=k.device)) < T
                    b_v = b_v * torch.where(in_bound, torch.exp(g_last - g_chunk), 0)[:, None]
                    h_state = h_state * torch.exp(g_last)

                b_v = b_v.to(k.dtype)
                h_state = h_state + (k_chunk.T @ b_v).T.float()
            final_state[b, i_h] = h_state
    return h_out, v_new, final_state


def _poison_shared_memory(device):
    # Pre-fix async_copy leaves masked-off smem untouched. If the buffer happens
    # to be zeros the kernel looks correct; fill it with NaN so the T < BT case
    # fails before the backend zero-fill lands.
    nan_mat = torch.full((512, 512), float("nan"), device=device, dtype=torch.bfloat16)
    _ = nan_mat @ nan_mat
    torch.cuda.synchronize()


@pytest.mark.parametrize("T", [7, 63])
def test_chunk_gated_delta_rule_fwd_h_ragged_t(T):
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for this integration test.")

    device = torch.device("cuda")
    torch.manual_seed(0)
    H, K, V = 4, 128, 128

    k = torch.randn(1, T, H, K, device=device, dtype=torch.bfloat16) * 0.1
    w = torch.randn(1, T, H, K, device=device, dtype=torch.bfloat16) * 0.1
    u = torch.randn(1, T, H, V, device=device, dtype=torch.bfloat16) * 0.1
    g = torch.randn(1, T, H, device=device, dtype=torch.float32) * 0.5
    initial_state = torch.zeros(1, H, K, V, device=device, dtype=torch.bfloat16)
    cu_seqlens = torch.tensor([0, T], dtype=torch.int32, device=device)

    _poison_shared_memory(device)
    h, v_new, final_state = chunk_gated_delta_rule_fwd_h(k=k, w=w, u=u, g=g, initial_state=initial_state,
                                                         output_final_state=True, cu_seqlens=cu_seqlens)
    torch.cuda.synchronize()

    h_ref, v_new_ref, final_ref = chunk_gated_delta_rule_fwd_h_ref(k, w, u, g, initial_state)

    assert not torch.isnan(h).any().item()
    assert not torch.isnan(v_new).any().item()
    assert not torch.isnan(final_state).any().item()
    assert torch.isfinite(final_state).all().item()
    torch.testing.assert_close(h.float(), h_ref.float(), rtol=5e-2, atol=5e-2)
    torch.testing.assert_close(v_new.float(), v_new_ref.float(), rtol=5e-2, atol=5e-2)
    torch.testing.assert_close(final_state, final_ref, rtol=5e-2, atol=5e-2)
