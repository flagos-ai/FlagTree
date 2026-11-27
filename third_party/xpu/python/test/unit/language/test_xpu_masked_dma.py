import pytest
import torch
import triton
import triton.language as tl


@triton.jit
def _masked_copy(src, predicates, dst, LIMIT: tl.constexpr, SIZE: tl.constexpr, OTHER: tl.constexpr,
                 UNIFORM: tl.constexpr):
    offsets = tl.arange(0, SIZE)
    if UNIFORM:
        active = tl.load(predicates)
    else:
        active = tl.load(predicates + offsets)
    value = tl.load(src + offsets, (offsets < LIMIT) & active, other=OTHER)
    tl.store(dst + offsets, value)


@pytest.mark.parametrize("keep", ["all", "none", "alternating"])
@pytest.mark.parametrize("other", [0.0, -7.0])
@pytest.mark.parametrize("size", [32, 1024])
def test_masked_load_with_runtime_predicate(device, keep, other, size):
    limit = size - 3
    offsets = torch.arange(size)
    src = offsets.float() + 1
    active = offsets % 2 == 0 if keep == "alternating" else torch.tensor([keep == "all"])
    expected = torch.where((offsets < limit) & active, src, other)
    dst = torch.empty(size, dtype=src.dtype, device=device)
    _masked_copy[(1, )](src.to(device), active.to(device), dst, limit, size, other, keep != "alternating")
    torch.testing.assert_close(dst.cpu(), expected, rtol=0, atol=0)
