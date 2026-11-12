"""E12: ragged expert gate/up -> SiLU -> down GEMM on IFWA."""
import sys

import pytest
import torch
import triton
import triton.language as tl


@triton.jit
def _expert_mlp(X, WG, WU, WD, OFFSETS, COUNTS, Y, H: tl.constexpr, F: tl.constexpr, BM: tl.constexpr,
                BF: tl.constexpr):
    expert = tl.program_id(0)
    tile = tl.program_id(1)
    start = tl.load(OFFSETS + expert)
    count = tl.load(COUNTS + expert)
    row = tile * BM + tl.arange(0, BM)
    hi = tl.arange(0, 32)
    fi = tl.arange(0, BF)
    x = tl.load(X + (start + row[:, None]) * H + hi[None, :], row[:, None] < count, other=0)
    gate_weight = tl.load(WG + expert * F * H + fi[None, :] * H + hi[:, None], fi[None, :] < F, other=0)
    up_weight = tl.load(WU + expert * F * H + fi[None, :] * H + hi[:, None], fi[None, :] < F, other=0)
    gate = tl.dot(x, gate_weight)
    up = tl.dot(x, up_weight)
    activated = (gate / (1.0 + tl.exp(-gate)) * up).to(x.dtype)
    down_weight = tl.load(WD + expert * H * F + hi[None, :] * F + fi[:, None], fi[:, None] < F, other=0)
    result = tl.dot(activated, down_weight)
    tl.store(Y + (start + row[:, None]) * H + hi[None, :], result, row[:, None] < count)


def grouped_expert_gemm(packed_x, gate_weight, up_weight, down_weight, counts):
    """Return FP32 expert outputs in packed row order.

    packed_x [sum(counts),32], gate/up weights [E,F,32], down [E,32,F],
    BF16 on IFWA; F=32 or64. Counts are CPU nonnegative integer metadata.
    Gate and up GEMMs accumulate FP32, activation is rounded to BF16 before
    down GEMM. The input is already grouped by expert, with empty experts valid.
    """
    if packed_x.ndim != 2 or packed_x.shape[1] != 32 or gate_weight.ndim != 3:
        raise ValueError("expected packed_x[T,32], gate_weight[E,F,32]")
    e, f, h = gate_weight.shape
    if e < 1 or h != 32 or f not in (32, 64) or up_weight.shape != gate_weight.shape or down_weight.shape != (e, 32, f):
        raise ValueError("weights must have shapes [E,F,32] and [E,32,F], F=32/64")
    if packed_x.device.type != "ifwa" or packed_x.dtype != torch.bfloat16:
        raise ValueError("inputs must be IFWA BF16")
    if any(t.device != packed_x.device or t.dtype != torch.bfloat16 or not t.is_contiguous() or t.requires_grad
           for t in (packed_x, gate_weight, up_weight, down_weight)):
        raise ValueError("all inputs must be contiguous BF16 inference tensors on the same device")
    counts_cpu = torch.as_tensor(counts)
    if counts_cpu.device.type != "cpu" or counts_cpu.shape != (e, ) or counts_cpu.dtype not in (torch.int32,
                                                                                                torch.int64):
        raise ValueError("counts must be CPU int32/int64 [E]")
    if bool((counts_cpu < 0).any()) or int(counts_cpu.sum()) != packed_x.shape[0]:
        raise ValueError("counts must be nonnegative and sum to packed rows")
    offsets = torch.cat((counts_cpu.new_zeros(1), counts_cpu.cumsum(0)[:-1])).to(torch.int32)
    count_dev = counts_cpu.to(device=packed_x.device, dtype=torch.int32)
    offset_dev = offsets.to(device=packed_x.device)
    out = torch.empty((packed_x.shape[0], 32), dtype=torch.float32, device=packed_x.device)
    max_count = int(counts_cpu.max()) if e else 0
    if max_count:
        _expert_mlp[(e, triton.cdiv(max_count,
                                    16))](packed_x, gate_weight, up_weight, down_weight, offset_dev, count_dev, out, 32,
                                          f, 16, f, num_warps=4, num_stages=1, enable_fp_fusion=False)
    return out


def reference(x, wg, wu, wd, counts):
    out = torch.empty((x.shape[0], 32), dtype=torch.float32)
    offset = 0
    for expert, count in enumerate(counts):
        if count:
            rows = x[offset:offset + count].float()
            gate = rows @ wg[expert].float().T
            up = rows @ wu[expert].float().T
            act = (torch.nn.functional.silu(gate) * up).to(torch.bfloat16)
            out[offset:offset + count] = act.float() @ wd[expert].float().T
        offset += count
    return out


@pytest.mark.parametrize("counts,f", [([3, 0, 5], 32), ([0, 17, 1, 0], 64), ([0, 0], 32), ([2, 1, 0], 64)])
def test_grouped_expert(counts, f):
    e = len(counts)
    t = sum(counts)
    generator = torch.Generator().manual_seed(112)
    x = (torch.randn((t, 32), generator=generator) / 8).to(torch.bfloat16)
    wg = (torch.randn((e, f, 32), generator=generator) / 8).to(torch.bfloat16)
    wu = (torch.randn((e, f, 32), generator=generator) / 8).to(torch.bfloat16)
    wd = (torch.randn((e, 32, f), generator=generator) / 8).to(torch.bfloat16)
    expected = reference(x, wg, wu, wd, counts)
    inputs = [v.to("ifwa:0") for v in (x, wg, wu, wd)]
    actual = grouped_expert_gemm(*inputs, counts).cpu()
    torch.testing.assert_close(actual, expected, rtol=3e-2, atol=3e-2)
    for original, device in zip((x, wg, wu, wd), inputs):
        torch.testing.assert_close(device.cpu(), original, rtol=0, atol=0)


if __name__ == "__main__":
    if triton.runtime.driver.active.get_current_target().backend != "ifwa":
        raise RuntimeError("E12 requires IFWA")
    raise SystemExit(pytest.main([__file__, *sys.argv[1:]]))
