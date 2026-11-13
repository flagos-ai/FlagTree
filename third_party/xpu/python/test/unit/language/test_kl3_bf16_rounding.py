"""KL3 BF16 conversions must round finite ties and preserve observable outputs."""
import pytest
import torch
import triton
import triton.language as tl


@pytest.fixture(scope="module", autouse=True)
def kl3():
    if not torch.cuda.is_available():
        pytest.skip("requires P800")
    target = triton.runtime.driver.active.get_current_target()
    if target.arch != 3 or target.warp_size != 1:
        pytest.skip("requires KL3")


@triton.jit
def _broadcast_add(src, bias, rounded, stored, WIDTH: tl.constexpr, BLOCK: tl.constexpr):
    idx = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    a = tl.load(src + idx).to(tl.float32)
    b = tl.load(bias + idx % WIDTH).to(tl.float32)
    total = a + b
    tl.store(rounded + idx, total.to(tl.bfloat16).to(tl.float32))
    tl.store(stored + idx, total)


@triton.jit
def _cast_only(src, rounded, stored, BLOCK: tl.constexpr):
    idx = tl.arange(0, BLOCK)
    value = tl.load(src + idx)
    tl.store(rounded + idx, value.to(tl.bfloat16).to(tl.float32))
    tl.store(stored + idx, value)


def _mismatches(actual, expected):
    # Check the sign of zero and every finite bit; NaN payload selection may vary.
    word = torch.int32 if actual.dtype == torch.float32 else torch.int16
    same_bits = actual.view(word) == expected.view(word)
    return int((~(same_bits | (actual.isnan() & expected.isnan()))).sum())


def _observe(rounded, stored, expected):
    actual32, actual16 = rounded.cpu(), stored.cpu()
    wanted32 = expected.float()
    counts = (_mismatches(actual32, wanted32), _mismatches(actual16, expected))
    return counts


@pytest.mark.parametrize("block", [1024, 2048, 4096, 8192])
def test_broadcast_rounding(block):
    rng = torch.Generator().manual_seed(61)
    a = torch.randn((16, 1024), generator=rng).bfloat16()
    bias = torch.randn(1024, generator=rng).bfloat16()
    a[:, :4] = torch.tensor([1., 1.0078125, -1., -1.0078125])
    bias[:4] = torch.tensor([1/256, 1/256, -1/256, -1/256])
    bias[4:7] = torch.tensor([float("nan"), float("inf"), -float("inf")])
    expected = (a.float() + bias.float()).bfloat16()
    # A finite sentinel also catches unwritten elements at NaN positions.
    rounded = torch.full(a.shape, -999., dtype=torch.float32, device="cuda")
    stored = torch.full_like(rounded, -999., dtype=torch.bfloat16)
    _broadcast_add[(a.numel() // block,)](a.cuda(), bias.cuda(), rounded, stored, 1024, block,
                                         num_warps=4, num_stages=1, is_sdnn=False)
    counts = _observe(rounded, stored, expected)
    assert counts == (0, 0), f"block={block}: roundtrip={counts[0]}, store={counts[1]}"


def test_cast_boundaries():
    # Encode inputs by bits to retain signed zero and ties with both retained parities.
    bits = [0x00000000, 0x80000000, 0x3f807fff, 0x3f808000, 0x3f808001,
            0x3f817fff, 0x3f818000, 0x3f818001, 0xbf808000, 0xbf818000,
            0x7f800000, 0xff800000, 0x7fc00001, 0xffc10000, 0x3e800000, 0x40800000]
    x = torch.tensor(bits, dtype=torch.int64).int().view(torch.float32).repeat(64)
    expected = x.bfloat16()
    rounded = torch.full_like(x, -999., device="cuda")
    stored = torch.full_like(rounded, -999., dtype=torch.bfloat16)
    _cast_only[(1,)](x.cuda(), rounded, stored, x.numel(), num_warps=4, num_stages=1, is_sdnn=False)
    counts = _observe(rounded, stored, expected)
    assert counts == (0, 0), f"cast-only: roundtrip={counts[0]}, store={counts[1]}"
