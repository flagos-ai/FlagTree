"""E13: paged MLA-style decode from FP8 latent KV and positional keys."""
import math
import sys

import pytest
import torch
import triton
import triton.language as tl


@triton.jit
def _mla_decode(QC, QP, LATENT, KPE, SCALES, PAGES, LENGTHS, WV, OUT, HQ: tl.constexpr, DC: tl.constexpr,
                DR: tl.constexpr, DV: tl.constexpr, PS: tl.constexpr, MAXP: tl.constexpr, SCALE: tl.constexpr,
                BN: tl.constexpr, BC: tl.constexpr, BR: tl.constexpr, BV: tl.constexpr):
    item = tl.program_id(0)
    batch = item // HQ
    head = item % HQ
    c = tl.arange(0, BC)
    r = tl.arange(0, BR)
    v = tl.arange(0, BV)
    tokens0 = tl.arange(0, BN)
    qc = tl.load(QC + item * DC + c, c < DC, other=0).to(tl.float32)
    qp = tl.load(QP + item * DR + r, r < DR, other=0).to(tl.float32)
    length = tl.load(LENGTHS + batch)
    running_max = tl.full((), -float("inf"), tl.float32)
    running_sum = tl.full((), 0, tl.float32)
    latent_sum = tl.zeros((BC, ), tl.float32)
    fp8_zero = tl.full((BN, BC), 0, tl.uint8).to(tl.float8e4nv, bitcast=True)
    for tile in range(tl.cdiv(length, BN)):
        tokens = tile * BN + tokens0
        page = tl.load(PAGES + batch * MAXP + tokens // PS, tokens < length, other=0)
        physical = page * PS + tokens % PS
        raw = tl.load(LATENT + physical[:, None] * DC + c[None, :], (tokens[:, None] < length) & (c[None, :] < DC),
                      other=fp8_zero).to(tl.float32)
        scale = tl.load(SCALES + physical, tokens < length, other=0)
        latent = raw * scale[:, None]
        kpe = tl.load(KPE + physical[:, None] * DR + r[None, :], (tokens[:, None] < length) & (r[None, :] < DR),
                      other=0).to(tl.float32)
        score = (tl.sum(latent * qc[None, :], 1) + tl.sum(kpe * qp[None, :], 1)) * SCALE
        score = tl.where(tokens < length, score, -float("inf"))
        new_max = tl.maximum(running_max, tl.max(score, 0))
        alpha = tl.exp(running_max - new_max)
        probabilities = tl.exp(score - new_max)
        latent_sum = latent_sum * alpha + tl.sum(probabilities[:, None] * latent, 0)
        running_sum = running_sum * alpha + tl.sum(probabilities, 0)
        running_max = new_max
    latent_mean = latent_sum / running_sum
    projection = tl.load(WV + head * DC * DV + c[:, None] * DV + v[None, :], (c[:, None] < DC) & (v[None, :] < DV),
                         other=0).to(tl.float32)
    result = tl.sum(latent_mean[:, None] * projection, 0)
    tl.store(OUT + item * DV + v, result, v < DV)


def mla_decode_fp8_kv(q_content, q_pos, latent_cache, k_pos_cache, latent_scales, value_projection, page_table, lengths,
                      *, scale=None):
    """Return BF16 [B,Hq,Dv] from a shared compressed latent cache.

    q_content is already absorbed into latent space [B,Hq,Dc]; q_pos and
    k_pos_cache are already RoPE-rotated [B,Hq,Dr] and [P,PS,Dr]. Latent cache
    is E4M3FN [P,PS,Dc] with FP32 per-token dequant scales [P,PS]. Wv is
    BF16 [Hq,Dc,Dv]. Page table and lengths are validated CPU integer metadata.
    Scores use both content and position; softmax accumulates dequantized latent
    values before per-head Wv projection. No separate dense KV expansion.
    """
    if q_content.ndim != 3 or q_pos.ndim != 3 or latent_cache.ndim != 3 or k_pos_cache.ndim != 3:
        raise ValueError("expected rank-three query and cache tensors")
    b, hq, dc = q_content.shape
    p, ps, cached_dc = latent_cache.shape
    dr = q_pos.shape[2]
    if value_projection.ndim != 3:
        raise ValueError("Wv must be [Hq,Dc,Dv]")
    dv = value_projection.shape[2]
    if min(b, hq, p, ps) <= 0 or not 1 <= dc <= 128 or not 1 <= dr <= 128 or not 1 <= dv <= 128:
        raise ValueError("invalid MLA dimensions")
    if q_pos.shape != (b, hq, dr) or cached_dc != dc or k_pos_cache.shape != (p, ps, dr):
        raise ValueError("query and cache dimensions disagree")
    if latent_scales.shape != (p, ps) or value_projection.shape != (hq, dc, dv):
        raise ValueError("scales or value projection have wrong shape")
    tensors = (q_content, q_pos, latent_cache, k_pos_cache, latent_scales, value_projection)
    dtypes = (torch.bfloat16, torch.bfloat16, torch.float8_e4m3fn, torch.bfloat16, torch.float32, torch.bfloat16)
    if q_content.device.type != "ifwa" or any(
            t.device != q_content.device or t.dtype != dtype or not t.is_contiguous() or t.requires_grad
            for t, dtype in zip(tensors, dtypes)):
        raise ValueError("requires contiguous IFWA BF16 queries/weights, FP8 latent, FP32 scales")
    pages = torch.as_tensor(page_table)
    seq_lengths = torch.as_tensor(lengths)
    if pages.device.type != "cpu" or seq_lengths.device.type != "cpu" or pages.ndim != 2 or pages.shape[0] != b:
        raise ValueError("page table must be CPU [B,max_pages]")
    if pages.shape[1] < 1 or seq_lengths.shape != (b, ) or pages.dtype not in (
            torch.int32, torch.int64) or seq_lengths.dtype not in (torch.int32, torch.int64):
        raise ValueError("page table and lengths must have integer shapes")
    if not bool(((seq_lengths >= 1) & (seq_lengths <= ps * pages.shape[1])).all()):
        raise ValueError("lengths outside page-table capacity")
    for sequence in range(b):
        used = pages[sequence, :triton.cdiv(int(seq_lengths[sequence]), ps)]
        if not bool(((used >= 0) & (used < p)).all()):
            raise ValueError("used physical page is outside cache")
    scale = (dc + dr)**-0.5 if scale is None else float(scale)
    if not math.isfinite(scale) or scale <= 0:
        raise ValueError("scale must be finite and positive")
    pages_dev = pages.to(q_content.device, dtype=torch.int32).contiguous()
    lengths_dev = seq_lengths.to(q_content.device, dtype=torch.int32).contiguous()
    out = torch.empty((b, hq, dv), device=q_content.device, dtype=torch.bfloat16)
    _mla_decode[(b * hq, )](q_content, q_pos, latent_cache, k_pos_cache, latent_scales, pages_dev, lengths_dev,
                            value_projection, out, hq, dc, dr, dv, ps, pages.shape[1], scale, 32,
                            triton.next_power_of_2(dc), triton.next_power_of_2(dr), triton.next_power_of_2(dv),
                            num_warps=4, enable_fp_fusion=False)
    return out


def reference(qc, qp, latent, kp, scales, wv, pages, lengths, scale):
    out = torch.empty((*qc.shape[:2], wv.shape[-1]), dtype=torch.bfloat16)
    ps = latent.shape[1]
    for batch, length in enumerate(lengths):
        physical = [(pages[batch, token // ps], token % ps) for token in range(length)]
        c = torch.stack([latent[p, slot].float() * scales[p, slot] for p, slot in physical])
        positional = torch.stack([kp[p, slot].float() for p, slot in physical])
        for head in range(qc.shape[1]):
            scores = (c @ qc[batch, head].float() + positional @ qp[batch, head].float()) * scale
            weighted_latent = torch.softmax(scores, 0) @ c
            out[batch, head] = weighted_latent @ wv[head].float()
    return out


@pytest.mark.parametrize("shape,lengths", [((1, 2, 32, 16, 32, 7), [9]), ((2, 4, 32, 16, 32, 8), [35, 17])])
@pytest.mark.parametrize("scale_factor", [0.5, 1.5])
def test_mla_decode(shape, lengths, scale_factor):
    b, hq, dc, dr, dv, ps = shape
    generator = torch.Generator().manual_seed(113)
    max_pages = triton.cdiv(max(lengths), ps) + 1
    physical_pages = b * max_pages + 3
    qc = torch.randn((b, hq, dc), generator=generator).to(torch.bfloat16)
    qp = torch.randn((b, hq, dr), generator=generator).to(torch.bfloat16)
    latent = torch.randn((physical_pages, ps, dc), generator=generator).to(torch.float8_e4m3fn)
    kp = torch.randn((physical_pages, ps, dr), generator=generator).to(torch.bfloat16)
    scales = torch.rand((physical_pages, ps), generator=generator) * scale_factor + 0.1
    wv = (torch.randn((hq, dc, dv), generator=generator) / 8).to(torch.bfloat16)
    permutation = torch.randperm(physical_pages, generator=generator)
    pages = torch.full((b, max_pages), -1, dtype=torch.int32)
    for batch, length in enumerate(lengths):
        count = triton.cdiv(length, ps)
        pages[batch, :count] = permutation[batch * max_pages:batch * max_pages + count].int()
    scale = 0.7 / math.sqrt(dc + dr)
    expected = reference(qc, qp, latent, kp, scales, wv, pages, lengths, scale)
    originals = (qc, qp, latent, kp, scales, wv)
    device_inputs = [tensor.to("ifwa:0") for tensor in originals]
    actual = mla_decode_fp8_kv(*device_inputs, pages, lengths, scale=scale).cpu()
    assert torch.isfinite(actual).all()
    torch.testing.assert_close(actual, expected, rtol=2e-2, atol=2e-2)
    for original, device in zip(originals, device_inputs):
        torch.testing.assert_close(device.cpu().view(torch.uint8), original.view(torch.uint8), rtol=0, atol=0)


if __name__ == "__main__":
    if triton.runtime.driver.active.get_current_target().backend != "ifwa":
        raise RuntimeError("E13 requires IFWA")
    raise SystemExit(pytest.main([__file__, *sys.argv[1:]]))
