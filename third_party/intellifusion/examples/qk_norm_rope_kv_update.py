"""E02: head RMSNorm -> split-half RoPE -> unique-position KV cache write."""
import math
import sys

import pytest
import torch
import triton
import triton.language as tl


@triton.jit
def _qk_norm_rope(Q, K, V, QW, KW, COS, SIN, POS, QO, KO, KC, VC, NQ: tl.constexpr, NK: tl.constexpr, D: tl.constexpr,
                  EPS: tl.constexpr, BLOCK: tl.constexpr):
    token = tl.program_id(0)
    head = tl.program_id(1)
    kh = head - NQ
    col = tl.arange(0, BLOCK)
    mask = col < D
    partner = (col + D // 2) % D
    pos = tl.load(POS + token)
    if head < NQ:
        offset = (token * NQ + head) * D
        x = tl.load(Q + offset + col, mask, other=0).to(tl.float32)
        xp = tl.load(Q + offset + partner, mask, other=0).to(tl.float32)
        w = tl.load(QW + col, mask, other=0).to(tl.float32)
        wp = tl.load(QW + partner, mask, other=0).to(tl.float32)
    else:
        offset = (token * NK + kh) * D
        x = tl.load(K + offset + col, mask, other=0).to(tl.float32)
        xp = tl.load(K + offset + partner, mask, other=0).to(tl.float32)
        w = tl.load(KW + col, mask, other=0).to(tl.float32)
        wp = tl.load(KW + partner, mask, other=0).to(tl.float32)
    inv = tl.rsqrt(tl.sum(x * x, 0) / D + EPS)
    # Materialize RMSNorm rounding before applying RoPE, matching the CPU oracle.
    x = ((x * inv) * w).to(Q.dtype.element_ty).to(tl.float32)
    xp = ((xp * inv) * wp).to(Q.dtype.element_ty).to(tl.float32)
    cs_offset = pos * (D // 2) + col % (D // 2)
    c = tl.load(COS + cs_offset, mask, other=0)
    s = tl.load(SIN + cs_offset, mask, other=0)
    rotated = x * c + tl.where(col < D // 2, -xp, xp) * s
    if head < NQ:
        tl.store(QO + offset + col, rotated, mask)
    else:
        tl.store(KO + offset + col, rotated, mask)
        cache_offset = (pos * NK + kh) * D + col
        tl.store(KC + cache_offset, rotated, mask)
        value = tl.load(V + offset + col, mask, other=0)
        tl.store(VC + cache_offset, value, mask)


def qk_norm_rope_kv_update(q, k, v, q_weight, k_weight, cos, sin, positions, k_cache, v_cache, eps=1e-6):
    """Normalize Q/K, rotate both, write rotated K and original V into caches.

    Q [T,Hq,D], K/V [T,Hkv,D], caches [capacity,Hkv,D], weights [D].
    Hq must be a multiple of Hkv. All payloads share FP32/BF16 dtype/device;
    cos/sin [capacity,D/2] use FP32. positions is a CPU int32/int64 [T]
    tensor, used both as RoPE position and physical cache slot. Slots must be
    distinct and in range. Q/K norms round to payload dtype before FP32 RoPE;
    final rotation rounds on store. Inputs are unchanged except the two caches.
    This inference-only API returns (rotated_q, rotated_k), with no backward.
    """
    if not math.isfinite(eps) or eps <= 0:
        raise ValueError("eps must be finite and positive")
    if q.ndim != 3 or k.ndim != 3 or k.shape != v.shape:
        raise ValueError("Q/K/V must have compatible rank-three shapes")
    tokens, nq, dim = q.shape
    if k.shape[0] != tokens or k.shape[2] != dim or nq <= 0 or k.shape[1] <= 0 or nq % k.shape[1]:
        raise ValueError("Q/K token and head dimensions must match, with Hq divisible by Hkv")
    if dim < 2 or dim > 1024 or dim % 2:
        raise ValueError("head dimension must be even and between 2 and 1024")
    if k_cache.ndim != 3 or k_cache.shape != v_cache.shape or k_cache.shape[1:] != k.shape[1:]:
        raise ValueError("KV caches must have shape [capacity,Hkv,D]")
    capacity = k_cache.shape[0]
    if q_weight.shape != (dim, ) or k_weight.shape != (dim, ):
        raise ValueError("Q/K weights must have shape [D]")
    payloads = (q, k, v, q_weight, k_weight, k_cache, v_cache)
    if q.device.type != "ifwa" or q.dtype not in (torch.float32, torch.bfloat16):
        raise ValueError("payloads must be FP32 or BF16 on IFWA")
    if any(t.device != q.device or t.dtype != q.dtype or not t.is_contiguous() or t.requires_grad for t in payloads):
        raise ValueError("payloads must be contiguous with the same device/dtype and no gradients")
    if any(t.device != q.device or t.dtype != torch.float32 or t.shape != (
            capacity, dim // 2) or not t.is_contiguous() or t.requires_grad for t in (cos, sin)):
        raise ValueError("cos/sin must be contiguous FP32 [capacity,D/2] tables on the payload device")
    if positions.device.type != "cpu" or positions.dtype not in (torch.int32,
                                                                 torch.int64) or positions.shape != (tokens, ):
        raise ValueError("positions must be a CPU int32/int64 [T] tensor")
    if tokens and (positions.min().item() < 0 or positions.max().item() >= capacity):
        raise ValueError("cache positions are out of range")
    if positions.unique().numel() != tokens:
        raise ValueError("cache positions must be unique to prevent write races")
    # For contiguous tensors, overlapping byte intervals would permit cache writes
    # to clobber data read by another program, or make K/V cache writes race.
    def overlaps(a, b):
        return a.numel() and b.numel() and max(a.data_ptr(), b.data_ptr()) < min(
            a.data_ptr() + a.numel() * a.element_size(),
            b.data_ptr() + b.numel() * b.element_size())

    if overlaps(k_cache, v_cache) or any(
            overlaps(c, t) for c in (k_cache, v_cache) for t in (q, k, v, q_weight, k_weight, cos, sin)):
        raise ValueError("cache storage must not overlap inputs or each other")
    qo, ko = torch.empty_like(q), torch.empty_like(k)
    if tokens:
        pos_dev = positions.contiguous().to(q.device)
        _qk_norm_rope[(tokens, nq + k.shape[1])](q, k, v, q_weight, k_weight, cos, sin, pos_dev, qo,
                                                 ko, k_cache, v_cache, nq, k.shape[1], dim, eps,
                                                 triton.next_power_of_2(dim), num_warps=4, enable_fp_fusion=False)
    return qo, ko


def _reference(x, weight, cos, sin, positions, eps):
    xf = x.float()
    norm = ((xf * torch.rsqrt(xf.square().mean(-1, keepdim=True) + eps)) * weight.float()).to(x.dtype).float()
    half = x.shape[-1] // 2
    partner = torch.cat((-norm[..., half:], norm[..., :half]), dim=-1)
    c = cos[positions].repeat(1, 2)[:, None, :]
    s = sin[positions].repeat(1, 2)[:, None, :]
    return (norm * c + partner * s).to(x.dtype)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("tokens,nq,nk,dim", [(1, 2, 2, 32), (3, 4, 2, 48), (0, 8, 1, 64)])
@pytest.mark.parametrize("eps", [1e-6, 1e-3])
def test_qk_norm_rope_kv_update(dtype, tokens, nq, nk, dim, eps):
    generator = torch.Generator().manual_seed(23)
    q = torch.randn((tokens, nq, dim), generator=generator).to(dtype)
    k = torch.randn((tokens, nk, dim), generator=generator).to(dtype)
    v = torch.randn((tokens, nk, dim), generator=generator).to(dtype)
    qw, kw = (torch.randn(dim, generator=generator).to(dtype) for _ in range(2))
    capacity = 9
    positions = torch.tensor([5, 1, 7][:tokens], dtype=torch.int64)
    angle = torch.arange(capacity)[:, None] * torch.pow(10000., -torch.arange(dim // 2).float()[None, :] * 2 / dim)
    cos, sin = angle.cos(), angle.sin()
    kc = torch.full((capacity, nk, dim), -3.25, dtype=dtype)
    vc = torch.full_like(kc, 2.5)
    expected_q = _reference(q, qw, cos, sin, positions, eps)
    expected_k = _reference(k, kw, cos, sin, positions, eps)
    expected_kc, expected_vc = kc.clone(), vc.clone()
    expected_kc[positions], expected_vc[positions] = expected_k, v
    cpu = (q, k, v, qw, kw, cos, sin, kc, vc)
    dev = [t.to("ifwa:0") for t in cpu]
    qo, ko = qk_norm_rope_kv_update(*dev[:7], positions, *dev[7:], eps=eps)
    tolerance = 1e-5 if dtype == torch.float32 else 1e-2
    for actual, expected in ((qo.cpu(), expected_q), (ko.cpu(), expected_k), (dev[7].cpu(), expected_kc)):
        torch.testing.assert_close(actual, expected, rtol=tolerance, atol=tolerance)
    torch.testing.assert_close(dev[8].cpu(), expected_vc, rtol=0, atol=0)
    untouched = torch.ones(capacity, dtype=torch.bool)
    untouched[positions] = False
    torch.testing.assert_close(dev[7].cpu()[untouched], kc[untouched], rtol=0, atol=0)
    for actual, original in zip(dev[:7], cpu[:7]):
        torch.testing.assert_close(actual.cpu(), original, rtol=0, atol=0)


if __name__ == "__main__":
    target = triton.runtime.driver.active.get_current_target()
    print(f"E02 target={target}", flush=True)
    if target.backend != "ifwa":
        raise RuntimeError("this example requires IFWA")
    raise SystemExit(pytest.main([__file__, *sys.argv[1:]]))
