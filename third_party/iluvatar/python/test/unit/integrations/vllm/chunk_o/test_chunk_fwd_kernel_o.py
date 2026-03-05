# SPDX-License-Identifier: Apache-2.0
import os

import pytest
import torch
import triton
import triton.language as tl


def _get_device() -> str:
    return os.getenv("CHUNK_O_DEVICE", "cuda:0")


def _dtype_from_str(name: str) -> torch.dtype:
    name = name.lower()
    if name in ("bf16", "bfloat16"):
        return torch.bfloat16
    if name in ("fp16", "float16"):
        return torch.float16
    raise ValueError(f"Unsupported dtype={name}")


def _prepare_chunk_indices(cu_seqlens: torch.Tensor, chunk_size: int) -> torch.Tensor:
    lens = cu_seqlens[1:] - cu_seqlens[:-1]
    counts = (lens + chunk_size - 1) // chunk_size
    indices = torch.cat([torch.arange(int(n), device=cu_seqlens.device) for n in counts.tolist()])
    return torch.stack([indices.eq(0).cumsum(0) - 1, indices], 1).to(cu_seqlens)


BKV_LIST = [32, 64]
NUM_WARPS = [2, 4, 8]


@triton.heuristics({
    "USE_G": lambda args: args["g"] is not None,
    "IS_VARLEN": lambda args: args["cu_seqlens"] is not None,
})
@triton.autotune(
    configs=[
        triton.Config({"BK": BK, "BV": BV}, num_warps=num_warps, num_stages=num_stages)
        for BK in BKV_LIST
        for BV in BKV_LIST
        for num_warps in NUM_WARPS
        for num_stages in [2, 3, 4]
    ],
    key=["H", "K", "V", "BT"],
)
@triton.jit(do_not_specialize=["T"])
def chunk_fwd_kernel_o(
    q,
    k,
    v,
    h,
    g,
    o,
    cu_seqlens,
    chunk_indices,
    scale,
    T,
    H: tl.constexpr,
    Hg: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    USE_G: tl.constexpr,
    IS_VARLEN: tl.constexpr,
):
    i_v, i_t, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    i_b, i_h = i_bh // H, i_bh % H

    if IS_VARLEN:
        i_tg = i_t
        i_n = tl.load(chunk_indices + i_t * 2).to(tl.int32)
        i_t = tl.load(chunk_indices + i_t * 2 + 1).to(tl.int32)
        bos = tl.load(cu_seqlens + i_n).to(tl.int32)
        eos = tl.load(cu_seqlens + i_n + 1).to(tl.int32)
        T = eos - bos
        NT = tl.cdiv(T, BT)
    else:
        NT = tl.cdiv(T, BT)
        i_tg = i_b * NT + i_t
        bos, eos = i_b * T, i_b * T + T

    q += (bos * Hg + i_h // (H // Hg)) * K
    k += (bos * Hg + i_h // (H // Hg)) * K
    v += (bos * H + i_h) * V
    o += (bos * H + i_h) * V
    h += (i_tg * H + i_h).to(tl.int64) * K * V

    b_o = tl.zeros([BT, BV], dtype=tl.float32)
    b_A = tl.zeros([BT, BT], dtype=tl.float32)

    for i_k in range(tl.cdiv(K, BK)):
        p_q = tl.make_block_ptr(q, (T, K), (Hg * K, 1), (i_t * BT, i_k * BK), (BT, BK), (1, 0))
        p_k = tl.make_block_ptr(k, (K, T), (1, Hg * K), (i_k * BK, i_t * BT), (BK, BT), (0, 1))
        p_h = tl.make_block_ptr(h, (K, V), (V, 1), (i_k * BK, i_v * BV), (BK, BV), (1, 0))
        b_q = tl.load(p_q, boundary_check=(0, 1))
        b_k = tl.load(p_k, boundary_check=(0, 1))
        b_h = tl.load(p_h, boundary_check=(0, 1))
        b_o += tl.dot(b_q, b_h)
        b_A += tl.dot(b_q, b_k)

    if USE_G:
        g += bos * H + i_h
        p_g = tl.make_block_ptr(g, (T, ), (H, ), (i_t * BT, ), (BT, ), (0, ))
        b_g = tl.load(p_g, boundary_check=(0, ))
        b_o = b_o * tl.exp(b_g)[:, None]
        b_A = b_A * tl.exp(b_g[:, None] - b_g[None, :])

    o_t = i_t * BT + tl.arange(0, BT)
    m_t = o_t < T
    m_A = (o_t[:, None] >= o_t[None, :]) & (m_t[:, None] & m_t)
    b_A = tl.where(m_A, b_A, 0)

    p_v = tl.make_block_ptr(v, (T, V), (H * V, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
    p_o = tl.make_block_ptr(o, (T, V), (H * V, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
    b_v = tl.load(p_v, boundary_check=(0, 1))

    b_o = b_o * scale + tl.dot(b_A.to(b_v.dtype), b_v) * scale
    tl.store(p_o, b_o.to(p_o.dtype.element_ty), boundary_check=(0, 1))


def _chunk_fwd_o_reference(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    h: torch.Tensor,
    g: torch.Tensor | None,
    cu_seqlens: torch.Tensor,
    chunk_size: int,
    scale: float,
) -> torch.Tensor:
    bsz, t_len, hg, k_dim = q.shape
    h_dim = v.shape[-2]
    v_dim = v.shape[-1]
    assert bsz == 1, "Reference only supports bsz=1."
    assert cu_seqlens.numel() == 2, "Reference expects one sequence."

    bt = min(chunk_size, max(16, triton.next_power_of_2(t_len)))
    nt = (t_len + bt - 1) // bt
    out = torch.empty_like(v)
    heads_per_group = h_dim // hg

    for i_t in range(nt):
        t_start = i_t * bt
        t_end = min(t_start + bt, t_len)
        q_chunk = q[0, t_start:t_end]
        k_chunk = k[0, t_start:t_end]
        v_chunk = v[0, t_start:t_end]
        g_chunk_all = None if g is None else g[0, t_start:t_end]
        for i_h in range(h_dim):
            group = i_h // heads_per_group
            q_h = q_chunk[:, group, :]
            k_h = k_chunk[:, group, :]
            v_h = v_chunk[:, i_h, :]
            h_h = h[i_t, i_h]
            q_h_f = q_h.float()
            k_h_f = k_h.float()
            h_h_f = h_h.float()
            b_o = q_h_f @ h_h_f
            b_A = q_h_f @ k_h_f.transpose(0, 1)
            if g is not None:
                g_h = g_chunk_all[:, i_h].float()
                exp_g = torch.exp(g_h)
                b_o = b_o * exp_g[:, None]
                b_A = b_A * (exp_g[:, None] / exp_g[None, :])
            mask = torch.tril(torch.ones((t_end - t_start, t_end - t_start), device=q.device, dtype=torch.bool))
            b_A = torch.where(mask, b_A, torch.zeros_like(b_A))
            b_o = b_o * scale + (b_A.to(v_h.dtype) @ v_h).float() * scale
            out[0, t_start:t_end, i_h, :] = b_o.to(out.dtype)
    return out


CASES = [
    (128, 8, 4, 64, 64, 32, "bf16", True),
    (128, 8, 4, 64, 64, 32, "fp16", False),
    (192, 16, 8, 64, 128, 64, "bf16", True),
    (256, 16, 8, 128, 64, 64, "fp16", True),
    (256, 16, 8, 128, 128, 64, "bf16", False),
    (384, 16, 8, 64, 64, 64, "fp16", False),
    (512, 16, 8, 128, 128, 64, "bf16", True),
    (768, 16, 8, 64, 128, 64, "fp16", False),
    (1024, 8, 4, 128, 64, 64, "bf16", True),
    (1024, 16, 8, 128, 128, 64, "fp16", True),
]


@pytest.mark.parametrize(
    "t_len,h_dim,hg,k_dim,v_dim,chunk_size,dtype_name,use_g",
    CASES,
    ids=[
        "t128-h8-hg4-k64-v64-c32-bf16-g",
        "t128-h8-hg4-k64-v64-c32-fp16-nog",
        "t192-h16-hg8-k64-v128-c64-bf16-g",
        "t256-h16-hg8-k128-v64-c64-fp16-g",
        "t256-h16-hg8-k128-v128-c64-bf16-nog",
        "t384-h16-hg8-k64-v64-c64-fp16-nog",
        "t512-h16-hg8-k128-v128-c64-bf16-g",
        "t768-h16-hg8-k64-v128-c64-fp16-nog",
        "t1024-h8-hg4-k128-v64-c64-bf16-g",
        "t1024-h16-hg8-k128-v128-c64-fp16-g",
    ],
)
def test_chunk_fwd_kernel_o_correctness(
    t_len: int,
    h_dim: int,
    hg: int,
    k_dim: int,
    v_dim: int,
    chunk_size: int,
    dtype_name: str,
    use_g: bool,
) -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for this integration test.")

    device = _get_device()
    dtype = _dtype_from_str(dtype_name)
    bsz = 1
    nt = (t_len + chunk_size - 1) // chunk_size
    scale = k_dim**-0.5

    torch.manual_seed(0)
    q = torch.randn((bsz, t_len, hg, k_dim), device=device, dtype=dtype)
    k = torch.randn_like(q)
    v = torch.randn((bsz, t_len, h_dim, v_dim), device=device, dtype=dtype)
    g = (torch.randn((bsz, t_len, h_dim), device=device, dtype=torch.float32) if use_g else None)
    h_state = torch.randn((nt, h_dim, k_dim, v_dim), device=device, dtype=dtype)
    cu_seqlens = torch.tensor([0, t_len], device=device, dtype=torch.int64)

    bt = min(chunk_size, max(16, triton.next_power_of_2(t_len)))
    chunk_indices = _prepare_chunk_indices(cu_seqlens, bt)
    out = torch.empty_like(v)

    def grid(meta):
        return (triton.cdiv(v_dim, meta["BV"]), len(chunk_indices), bsz * h_dim)

    chunk_fwd_kernel_o[grid](
        q,
        k,
        v,
        h_state,
        g,
        out,
        cu_seqlens,
        chunk_indices,
        scale,
        T=t_len,
        H=h_dim,
        Hg=hg,
        K=k_dim,
        V=v_dim,
        BT=bt,
    )
    ref = _chunk_fwd_o_reference(
        q=q,
        k=k,
        v=v,
        h=h_state,
        g=g,
        cu_seqlens=cu_seqlens,
        chunk_size=chunk_size,
        scale=scale,
    )
    torch.testing.assert_close(out, ref, rtol=2e-1, atol=2e-1)
