import math
from typing import Iterable

import pytest
import torch
import triton
import triton.language as tl


def _iter_sequences(B: int, T: int, cu_seqlens: torch.Tensor | None):
    if cu_seqlens is None:
        for b in range(B):
            yield b, 0, T
    else:
        assert B == 1, "cu_seqlens expects flattened batch with B=1."
        for n in range(len(cu_seqlens) - 1):
            start = int(cu_seqlens[n].item())
            end = int(cu_seqlens[n + 1].item())
            yield 0, start, end


def prepare_chunk_indices(cu_seqlens: torch.Tensor, chunk_size: int) -> torch.Tensor:
    pairs: list[tuple[int, int]] = []
    for n in range(len(cu_seqlens) - 1):
        start = int(cu_seqlens[n].item())
        end = int(cu_seqlens[n + 1].item())
        for t0 in range(0, end - start, chunk_size):
            pairs.append((n, t0 // chunk_size))
    if not pairs:
        return torch.empty((0, 2), device=cu_seqlens.device, dtype=torch.int32)
    return torch.tensor(pairs, device=cu_seqlens.device, dtype=torch.int32)


def chunk_local_cumsum_ref(g: torch.Tensor, chunk_size: int, cu_seqlens: torch.Tensor | None) -> torch.Tensor:
    B, T, H = g.shape
    out = torch.zeros_like(g, dtype=torch.float32)
    for b, start, end in _iter_sequences(B, T, cu_seqlens):
        for h in range(H):
            for t0 in range(start, end, chunk_size):
                t1 = min(t0 + chunk_size, end)
                out[b, t0:t1, h] = torch.cumsum(g[b, t0:t1, h].float(), dim=0)
    return out


def build_A_inv_ref(
    k: torch.Tensor,
    beta: torch.Tensor,
    g_cumsum: torch.Tensor,
    chunk_size: int,
    cu_seqlens: torch.Tensor | None,
) -> torch.Tensor:
    B, T, Hg, K = k.shape
    H = beta.shape[-1]
    group = H // Hg
    A_inv = torch.zeros((B, T, H, chunk_size), device=k.device, dtype=torch.float32)
    for b, start, end in _iter_sequences(B, T, cu_seqlens):
        for h in range(H):
            hg = h // group
            for t0 in range(start, end, chunk_size):
                t1 = min(t0 + chunk_size, end)
                k_chunk = k[b, t0:t1, hg, :].float()
                beta_chunk = beta[b, t0:t1, h].float()
                g_chunk = g_cumsum[b, t0:t1, h].float()
                A = (k_chunk * beta_chunk[:, None]) @ k_chunk.T
                A = A * torch.exp(g_chunk[:, None] - g_chunk[None, :])
                A = torch.tril(A, diagonal=-1)
                BT = t1 - t0
                block = torch.zeros((chunk_size, chunk_size), device=k.device)
                block[:BT, :BT] = A
                inv = torch.linalg.inv(torch.eye(chunk_size, device=k.device) + block)
                A_inv[b, t0:t1, h, :BT] = inv[:BT, :BT]
    return A_inv


def recompute_w_u_ref(
    k: torch.Tensor,
    v: torch.Tensor,
    beta: torch.Tensor,
    g_cumsum: torch.Tensor,
    A_inv: torch.Tensor,
    chunk_size: int,
    cu_seqlens: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    B, T, Hg, K = k.shape
    H = v.shape[-2]
    V = v.shape[-1]
    group = H // Hg
    w = k.new_zeros(B, T, H, K, dtype=torch.float32)
    u = v.new_zeros(B, T, H, V, dtype=torch.float32)
    for b, start, end in _iter_sequences(B, T, cu_seqlens):
        for h in range(H):
            hg = h // group
            for t0 in range(start, end, chunk_size):
                t1 = min(t0 + chunk_size, end)
                Ai = A_inv[b, t0:t1, h, :t1 - t0].float()
                beta_chunk = beta[b, t0:t1, h].float()
                g_chunk = g_cumsum[b, t0:t1, h].float()
                v_chunk = v[b, t0:t1, h, :].float()
                u_chunk = Ai @ (v_chunk * beta_chunk[:, None])
                u[b, t0:t1, h, :] = u_chunk
                k_chunk = k[b, t0:t1, hg, :].float()
                w_chunk = Ai @ (k_chunk * beta_chunk[:, None] * torch.exp(g_chunk)[:, None])
                w[b, t0:t1, h, :] = w_chunk
    return w, u


@triton.heuristics({"IS_VARLEN": lambda args: args["cu_seqlens"] is not None})
@triton.jit(do_not_specialize=["T"])
def recompute_w_u_fwd_kernel(
    k,
    v,
    beta,
    w,
    u,
    A,
    g,
    cu_seqlens,
    chunk_indices,
    T,
    H: tl.constexpr,
    Hg: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    IS_VARLEN: tl.constexpr,
):
    i_t, i_bh = tl.program_id(0), tl.program_id(1)
    i_b, i_h = i_bh // H, i_bh % H
    if IS_VARLEN:
        i_n = tl.load(chunk_indices + i_t * 2).to(tl.int32)
        i_t = tl.load(chunk_indices + i_t * 2 + 1).to(tl.int32)
        bos = tl.load(cu_seqlens + i_n).to(tl.int32)
        eos = tl.load(cu_seqlens + i_n + 1).to(tl.int32)
        T = eos - bos
    else:
        bos, eos = i_b * T, i_b * T + T
    p_beta = tl.make_block_ptr(beta + bos * H + i_h, (T, ), (H, ), (i_t * BT, ), (BT, ), (0, ))
    p_g = tl.make_block_ptr(g + (bos * H + i_h), (T, ), (H, ), (i_t * BT, ), (BT, ), (0, ))
    p_A = tl.make_block_ptr(A + (bos * H + i_h) * BT, (T, BT), (H * BT, 1), (i_t * BT, 0), (BT, BT), (1, 0))
    b_beta = tl.load(p_beta, boundary_check=(0, ))
    b_A = tl.load(p_A, boundary_check=(0, 1))
    b_g = tl.exp(tl.load(p_g, boundary_check=(0, )))

    for i_v in range(tl.cdiv(V, BV)):
        p_v = tl.make_block_ptr(
            v + (bos * H + i_h) * V,
            (T, V),
            (H * V, 1),
            (i_t * BT, i_v * BV),
            (BT, BV),
            (1, 0),
        )
        p_u = tl.make_block_ptr(
            u + (bos * H + i_h) * V,
            (T, V),
            (H * V, 1),
            (i_t * BT, i_v * BV),
            (BT, BV),
            (1, 0),
        )
        b_v = tl.load(p_v, boundary_check=(0, 1))
        b_vb = (b_v * b_beta[:, None]).to(b_v.dtype)
        b_u = tl.dot(b_A, b_vb, allow_tf32=False)
        tl.store(p_u, b_u.to(p_u.dtype.element_ty), boundary_check=(0, 1))

    for i_k in range(tl.cdiv(K, BK)):
        p_k = tl.make_block_ptr(
            k + (bos * Hg + i_h // (H // Hg)) * K,
            (T, K),
            (Hg * K, 1),
            (i_t * BT, i_k * BK),
            (BT, BK),
            (1, 0),
        )
        p_w = tl.make_block_ptr(
            w + (bos * H + i_h) * K,
            (T, K),
            (H * K, 1),
            (i_t * BT, i_k * BK),
            (BT, BK),
            (1, 0),
        )
        b_k = tl.load(p_k, boundary_check=(0, 1))
        b_kb = (b_k * b_beta[:, None] * b_g[:, None]).to(b_k.dtype)
        b_w = tl.dot(b_A, b_kb)
        tl.store(p_w, b_w.to(p_w.dtype.element_ty), boundary_check=(0, 1))


def _make_cu_seqlens(lengths: Iterable[int], device: torch.device) -> torch.Tensor:
    cu = [0]
    acc = 0
    for l in lengths:
        acc += l
        cu.append(acc)
    return torch.tensor(cu, device=device, dtype=torch.int32)


CASES = [(B, T, H, Hg, K, V, is_varlen, num_warps, num_stages, dtype)
         for B in [1]
         for T in [17, 64, 128]
         for H in [4, 8]
         for Hg in [2, 4]
         for K in [128]
         for V in [128]
         for is_varlen in [False, True]
         for num_warps in [4, 8]
         for num_stages in [2, 3]
         for dtype in [torch.float16, torch.bfloat16]]


@pytest.mark.parametrize("B,T,H,Hg,K,V,is_varlen,num_warps,num_stages,dtype", CASES)
def test_recompute_w_u_fwd_kernel(B, T, H, Hg, K, V, is_varlen, num_warps, num_stages, dtype):
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required")

    device = torch.device("cuda")
    torch.manual_seed(0)
    if is_varlen:
        lengths = [T // 2, T // 4, T - (T // 2 + T // 4)]
        cu_seqlens = _make_cu_seqlens(lengths, device)
        B = 1
    else:
        cu_seqlens = None

    k = torch.randn((B, T, Hg, K), device=device, dtype=dtype) * 0.2
    v = torch.randn((B, T, H, V), device=device, dtype=dtype) * 0.2
    g = torch.randn((B, T, H), device=device, dtype=torch.float32) * 0.1
    beta = torch.randn((B, T, H), device=device, dtype=dtype) * 0.1

    BT = 64
    BK = 64
    BV = 64

    g_cumsum = chunk_local_cumsum_ref(g, chunk_size=BT, cu_seqlens=cu_seqlens)
    A_inv = build_A_inv_ref(
        k=k,
        beta=beta,
        g_cumsum=g_cumsum,
        chunk_size=BT,
        cu_seqlens=cu_seqlens,
    ).to(dtype)
    w_ref, u_ref = recompute_w_u_ref(
        k=k,
        v=v,
        beta=beta,
        g_cumsum=g_cumsum,
        A_inv=A_inv,
        chunk_size=BT,
        cu_seqlens=cu_seqlens,
    )

    chunk_indices = prepare_chunk_indices(cu_seqlens, BT) if is_varlen else None
    NT = math.ceil(T / BT) if not is_varlen else len(chunk_indices)
    w = torch.empty((B, T, H, K), device=device, dtype=dtype)
    u = torch.empty((B, T, H, V), device=device, dtype=dtype)
    grid = (NT, B * H)

    recompute_w_u_fwd_kernel[grid](
        k,
        v,
        beta,
        w,
        u,
        A_inv,
        g_cumsum,
        cu_seqlens,
        chunk_indices,
        T,
        H=H,
        Hg=Hg,
        K=K,
        V=V,
        BT=BT,
        BK=BK,
        BV=BV,
        IS_VARLEN=is_varlen,
        num_warps=num_warps,
        num_stages=num_stages,
    )

    assert not torch.isnan(w).any().item()
    assert not torch.isnan(u).any().item()

    torch.testing.assert_close(w.float(), w_ref, rtol=5e-2, atol=5e-2, check_dtype=False)
    torch.testing.assert_close(u.float(), u_ref, rtol=5e-2, atol=5e-2, check_dtype=False)
