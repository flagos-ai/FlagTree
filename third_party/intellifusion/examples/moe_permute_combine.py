"""E11: stable token-to-expert permutation and weighted combine."""
import sys

import pytest
import torch
import triton
import triton.language as tl


@triton.jit
def _permute(X, IDS, PACKED, INVERSE, T: tl.constexpr, K: tl.constexpr, H: tl.constexpr, E: tl.constexpr,
             BA: tl.constexpr, BH: tl.constexpr):
    assignment = tl.program_id(0)
    expert = tl.load(IDS + assignment)
    all_assignments = tl.arange(0, BA)
    all_experts = tl.load(IDS + all_assignments, all_assignments < T * K, other=E)
    before_expert = tl.sum(((all_experts < expert) & (all_assignments < T * K)).to(tl.int32), 0)
    rank = tl.sum(((all_experts == expert) & (all_assignments < assignment)).to(tl.int32), 0)
    sorted_index = before_expert + rank
    h = tl.arange(0, BH)
    value = tl.load(X + (assignment // K) * H + h, h < H, other=0)
    tl.store(PACKED + sorted_index * H + h, value, h < H)
    tl.store(INVERSE + assignment, sorted_index)


@triton.jit
def _count_only(IDS, COUNTS, T: tl.constexpr, K: tl.constexpr, E: tl.constexpr, BA: tl.constexpr):
    expert = tl.program_id(0)
    positions = tl.arange(0, BA)
    ids = tl.load(IDS + positions, positions < T * K, other=E)
    count = tl.sum(((ids == expert) & (positions < T * K)).to(tl.int32), 0)
    tl.store(COUNTS + expert, count)


@triton.jit
def _combine(Y, INVERSE, WEIGHTS, OUT, T: tl.constexpr, K: tl.constexpr, H: tl.constexpr, BH: tl.constexpr):
    token = tl.program_id(0)
    h = tl.arange(0, BH)
    result = tl.zeros((BH, ), tl.float32)
    for slot in range(K):
        position = tl.load(INVERSE + token * K + slot)
        weight = tl.load(WEIGHTS + token * K + slot)
        value = tl.load(Y + position * H + h, h < H, other=0).to(tl.float32)
        result += value * weight
    tl.store(OUT + token * H + h, result, h < H)


def stable_permute(x, expert_ids, expert_count):
    """Return (packed, inverse, counts) for row-major [token,slot] assignments.

    packed[sum(counts[:e]) + rank_e(token,slot)] == x[token]. Duplicate
    expert IDs are valid. IDs and inverse map are int32 IFWA tensors.
    """
    if x.ndim != 2 or expert_ids.ndim != 2 or x.shape[0] != expert_ids.shape[0]:
        raise ValueError("expected x[T,H] and expert_ids[T,K]")
    t, h = x.shape
    k = expert_ids.shape[1]
    if h < 1 or h > 1024 or k < 1 or t * k > 1024 or expert_count < 1:
        raise ValueError("requires H in 1..1024, K>0, T*K<=1024 and expert_count>0")
    if x.device.type != "ifwa" or x.dtype not in (torch.float32, torch.bfloat16):
        raise ValueError("x must be IFWA FP32/BF16")
    if expert_ids.device != x.device or expert_ids.dtype != torch.int32:
        raise ValueError("expert IDs must be IFWA int32")
    if not x.is_contiguous() or not expert_ids.is_contiguous() or x.requires_grad:
        raise ValueError("inputs must be contiguous inference tensors")
    if t:
        ids_cpu = expert_ids.cpu()
        if not bool(((ids_cpu >= 0) & (ids_cpu < expert_count)).all()):
            raise ValueError("expert IDs out of range")
    packed = torch.empty((t * k, h), device=x.device, dtype=x.dtype)
    inverse = torch.empty((t, k), device=x.device, dtype=torch.int32)
    counts = torch.empty((expert_count, ), device=x.device, dtype=torch.int32)
    if t:
        _permute[(t * k, )](x, expert_ids, packed, inverse, t, k, h, expert_count, triton.next_power_of_2(t * k),
                            triton.next_power_of_2(h), num_warps=4)
        _count_only[(expert_count, )](expert_ids, counts, t, k, expert_count, triton.next_power_of_2(t * k),
                                      num_warps=4)
    else:
        counts.zero_()
    return packed, inverse, counts


def weighted_combine(expert_output, inverse, weights):
    """Gather sorted expert results and sum slots with FP32 weights."""
    if inverse.ndim != 2 or weights.shape != inverse.shape or expert_output.ndim != 2:
        raise ValueError("expected output[A,H] and inverse/weights[T,K]")
    t, k = inverse.shape
    a, h = expert_output.shape
    if a != t * k or h < 1 or h > 1024 or expert_output.device.type != "ifwa":
        raise ValueError("output rows must equal T*K on IFWA")
    if expert_output.dtype not in (torch.float32,
                                   torch.bfloat16) or inverse.dtype != torch.int32 or weights.dtype != torch.float32:
        raise ValueError("requires FP32/BF16 output, int32 inverse and FP32 weights")
    if any(v.device != expert_output.device or not v.is_contiguous() or v.requires_grad
           for v in (expert_output, inverse, weights)):
        raise ValueError("all inputs must be contiguous inference tensors on one device")
    if t:
        inverse_cpu = inverse.cpu()
        if not bool(((inverse_cpu >= 0) & (inverse_cpu < a)).all()):
            raise ValueError("inverse index out of range")
    result = torch.empty((t, h), device=expert_output.device, dtype=expert_output.dtype)
    if t:
        _combine[(t, )](expert_output, inverse, weights, result, t, k, h, triton.next_power_of_2(h), num_warps=4)
    return result


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("ids,e,h", [(torch.tensor([[2, 0], [2, 1], [0, 2]], dtype=torch.int32), 5, 17),
                                     (torch.tensor([[3, 3], [3, 0], [3, 1], [1, 1]], dtype=torch.int32), 4, 32),
                                     (torch.empty((0, 2), dtype=torch.int32), 3, 16)])
def test_permute_combine(dtype, ids, e, h):
    t, k = ids.shape
    generator = torch.Generator().manual_seed(111)
    x = torch.randn((t, h), generator=generator).to(dtype)
    w = torch.rand((t, k), generator=generator)
    order = sorted(range(t * k), key=lambda a: (ids.flatten()[a].item(), a))
    expected_packed = torch.stack([x[a // k] for a in order]) if order else x.new_empty((0, h))
    expected_inverse = torch.empty((t * k, ), dtype=torch.int32)
    for sorted_index, original in enumerate(order):
        expected_inverse[original] = sorted_index
    expected_inverse = expected_inverse.reshape(t, k)
    expected_counts = torch.bincount(ids.flatten().long(), minlength=e).int()
    dev_x, dev_ids, dev_w = x.to("ifwa:0"), ids.to("ifwa:0"), w.to("ifwa:0")
    packed, inverse, counts = stable_permute(dev_x, dev_ids, e)
    torch.testing.assert_close(packed.cpu(), expected_packed, rtol=0, atol=0)
    torch.testing.assert_close(inverse.cpu(), expected_inverse, rtol=0, atol=0)
    torch.testing.assert_close(counts.cpu(), expected_counts, rtol=0, atol=0)
    # External expert computation is represented by independently generated
    # results in packed order, so combine is tested without reusing permute math.
    y = torch.randn((t * k, h), generator=generator).to(dtype)
    expected = torch.zeros((t, h), dtype=torch.float32)
    for original in range(t * k):
        expected[original // k] += y[expected_inverse.flatten()[original]].float() * w.flatten()[original]
    out = weighted_combine(y.to("ifwa:0"), inverse, dev_w)
    torch.testing.assert_close(out.cpu(), expected.to(dtype), rtol=2e-2 if dtype == torch.bfloat16 else 1e-5,
                               atol=2e-2 if dtype == torch.bfloat16 else 1e-5)
    torch.testing.assert_close(dev_x.cpu(), x, rtol=0, atol=0)
    torch.testing.assert_close(dev_ids.cpu(), ids, rtol=0, atol=0)


if __name__ == "__main__":
    if triton.runtime.driver.active.get_current_target().backend != "ifwa":
        raise RuntimeError("E11 requires IFWA")
    raise SystemExit(pytest.main([__file__, *sys.argv[1:]]))
