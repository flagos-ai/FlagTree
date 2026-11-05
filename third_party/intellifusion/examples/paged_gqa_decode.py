"""E09: paged GQA decode with physical page gathers and online softmax."""
import math
import sys

import pytest
import torch
import triton
import triton.language as tl


@triton.jit
def _decode(Q, K, V, Pages, Lengths, O, HQ: tl.constexpr, HKV: tl.constexpr, D: tl.constexpr, PS: tl.constexpr,
            MAXP: tl.constexpr, SCALE: tl.constexpr, BN: tl.constexpr, BD: tl.constexpr):
    item = tl.program_id(0)
    batch = item // HQ
    head = item % HQ
    kv_head = head // (HQ // HKV)
    dims = tl.arange(0, BD)
    q = tl.load(Q + item * D + dims, dims < D, other=0).to(tl.float32)
    length = tl.load(Lengths + batch)
    running_max = tl.full((), -float("inf"), tl.float32)
    running_sum = tl.full((), 0, tl.float32)
    acc = tl.zeros((BD, ), tl.float32)
    offsets = tl.arange(0, BN)
    for start in range(0, tl.cdiv(length, BN)):
        tokens = start * BN + offsets
        physical = tl.load(Pages + batch * MAXP + tokens // PS, tokens < length, other=0)
        addresses = ((physical * PS + tokens % PS) * HKV + kv_head) * D
        k = tl.load(K + addresses[:, None] + dims[None, :], (tokens[:, None] < length) & (dims[None, :] < D),
                    other=0).to(tl.float32)
        scores = tl.sum(k * q[None, :], 1) * SCALE
        scores = tl.where(tokens < length, scores, -float("inf"))
        new_max = tl.maximum(running_max, tl.max(scores, 0))
        alpha = tl.exp(running_max - new_max)
        probs = tl.exp(scores - new_max)
        v = tl.load(V + addresses[:, None] + dims[None, :], (tokens[:, None] < length) & (dims[None, :] < D),
                    other=0).to(tl.float32)
        acc = acc * alpha + tl.sum(probs[:, None] * v, 0)
        running_sum = running_sum * alpha + tl.sum(probs, 0)
        running_max = new_max
    tl.store(O + item * D + dims, acc / running_sum, dims < D)


def paged_gqa_decode(q, key_cache, value_cache, page_table, lengths, *, scale=None):
    """Decode Q[B,Hq,D] against cache[P,page_size,Hkv,D], producing Q-shaped output.

    Q/cache are contiguous IFWA FP16/BF16 tensors with equal dtype/device.
    Hq is a multiple of Hkv. D is 1..128. Lengths and page_table[B,max_pages]
    must be CPU integer metadata. Only pages needed by each sequence must have
    valid physical indices; unused entries may be -1. Shared pages are allowed
    because this operator is read-only. No cache update is performed.
    """
    if q.ndim != 3 or key_cache.ndim != 4 or value_cache.shape != key_cache.shape:
        raise ValueError("expected Q[B,Hq,D] and matching K/V[P,page_size,Hkv,D]")
    b, hq, d = q.shape
    p, ps, hkv, cache_d = key_cache.shape
    if min(b, hq, p, ps, hkv) <= 0 or not 1 <= d <= 128 or cache_d != d or hq % hkv:
        raise ValueError("nonempty matching dimensions and Hq divisible by Hkv are required")
    if q.device.type != "ifwa" or q.dtype not in (torch.float16, torch.bfloat16):
        raise ValueError("Q must be an IFWA FP16/BF16 tensor")
    tensors = (q, key_cache, value_cache)
    if any(t.device != q.device or t.dtype != q.dtype or not t.is_contiguous() or t.requires_grad for t in tensors):
        raise ValueError("inputs must be contiguous inference tensors of equal dtype/device")
    pages = torch.as_tensor(page_table)
    seq_lengths = torch.as_tensor(lengths)
    if pages.device.type != "cpu" or seq_lengths.device.type != "cpu":
        raise ValueError("page_table and lengths must be CPU metadata")
    if pages.ndim != 2 or pages.shape[0] != b or pages.shape[1] <= 0 or seq_lengths.shape != (b, ):
        raise ValueError("expected page_table[B,max_pages] and lengths[B]")
    if pages.dtype not in (torch.int32, torch.int64) or seq_lengths.dtype not in (torch.int32, torch.int64):
        raise ValueError("page metadata must use integer dtype")
    if not bool(((seq_lengths >= 1) & (seq_lengths <= ps * pages.shape[1])).all()):
        raise ValueError("lengths must fit the logical page table and be positive")
    for sequence in range(b):
        used = pages[sequence, :triton.cdiv(int(seq_lengths[sequence]), ps)]
        if not bool(((used >= 0) & (used < p)).all()):
            raise ValueError("used physical page indices must be within the cache")
    scale = d**-0.5 if scale is None else float(scale)
    if not math.isfinite(scale) or scale <= 0:
        raise ValueError("scale must be finite and positive")
    device_pages = pages.to(device=q.device, dtype=torch.int32).contiguous()
    device_lengths = seq_lengths.to(device=q.device, dtype=torch.int32).contiguous()
    out = torch.empty_like(q)
    _decode[(b * hq, )](q, key_cache, value_cache, device_pages, device_lengths, out, hq, hkv, d, ps, pages.shape[1],
                        scale, 32, triton.next_power_of_2(d), num_warps=4)
    return out


def reference(q, key_cache, value_cache, pages, lengths, scale):
    out = torch.empty_like(q)
    ps = key_cache.shape[1]
    groups = q.shape[1] // key_cache.shape[2]
    for sequence, length in enumerate(lengths):
        logical_k = torch.stack([key_cache[pages[sequence, token // ps], token % ps] for token in range(length)])
        logical_v = torch.stack([value_cache[pages[sequence, token // ps], token % ps] for token in range(length)])
        for head in range(q.shape[1]):
            kv_head = head // groups
            scores = logical_k[:, kv_head].float() @ q[sequence, head].float() * scale
            out[sequence, head] = torch.softmax(scores, 0) @ logical_v[:, kv_head].float()
    return out


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("shape,lengths", [((1, 2, 2, 16, 8), [1]), ((2, 4, 2, 32, 8), [35, 7]),
                                           ((2, 4, 1, 64, 16), [67, 31]), ((1, 2, 1, 17, 7), [37])])
def test_decode(dtype, shape, lengths):
    b, hq, hkv, d, ps = shape
    max_pages = triton.cdiv(max(lengths), ps) + 1
    total_pages = b * max_pages + 3
    generator = torch.Generator().manual_seed(909)
    q = torch.randn(b, hq, d, generator=generator).to(dtype)
    key = torch.randn(total_pages, ps, hkv, d, generator=generator).to(dtype)
    value = torch.randn(total_pages, ps, hkv, d, generator=generator).to(dtype)
    # Physical pages deliberately have no logical adjacency; unused slots stay -1.
    permutation = torch.randperm(total_pages, generator=generator)
    pages = torch.full((b, max_pages), -1, dtype=torch.int32)
    for sequence, length in enumerate(lengths):
        count = triton.cdiv(length, ps)
        pages[sequence, :count] = permutation[sequence * max_pages:sequence * max_pages + count].int()
    scale = 0.8 / math.sqrt(d)
    expected = reference(q, key, value, pages, lengths, scale)
    device_inputs = [t.to("ifwa:0") for t in (q, key, value)]
    actual = paged_gqa_decode(*device_inputs, pages, lengths, scale=scale).cpu()
    assert torch.isfinite(actual).all()
    tolerance = 2e-3 if dtype == torch.float16 else 1e-2
    torch.testing.assert_close(actual, expected, rtol=tolerance, atol=tolerance)
    for original, device in zip((q, key, value), device_inputs):
        torch.testing.assert_close(device.cpu(), original, rtol=0, atol=0)


if __name__ == "__main__":
    target = triton.runtime.driver.active.get_current_target()
    print(f"target={target}", flush=True)
    if target.backend != "ifwa":
        raise RuntimeError("this example requires IFWA")
    raise SystemExit(pytest.main([__file__, *sys.argv[1:]]))
