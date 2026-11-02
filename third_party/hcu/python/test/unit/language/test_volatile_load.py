import re

import pytest
import torch
import triton
import triton.language as tl


@triton.jit
def volatile_load_kernel(src, dst, n, VOLATILE: tl.constexpr, VECTOR: tl.constexpr, MASKED: tl.constexpr):
    offsets = tl.arange(0, 256) if VECTOR else 0
    if MASKED:
        value = tl.load(src + offsets, offsets < n, other=0, volatile=VOLATILE)
    else:
        value = tl.load(src + offsets, volatile=VOLATILE)
    tl.store(dst + offsets, value)


@pytest.mark.parametrize("volatile", [False, True])
@pytest.mark.parametrize("vector", [False, True])
@pytest.mark.parametrize("masked", [False, True])
@pytest.mark.parametrize("buffer_ops", [False, True])
def test_volatile_load(volatile, vector, masked, buffer_ops, monkeypatch, device):
    monkeypatch.setenv("AMDGCN_USE_BUFFER_OPS", str(int(buffer_ops)))
    size = 256 if vector else 1
    src = torch.arange(1, size + 1, dtype=torch.int32, device=device)
    dst = torch.empty_like(src)
    kernel = volatile_load_kernel[(1, )](src, dst, 129, volatile, vector, masked)

    for stage in ("ttir", "ttgir"):
        assert ("isVolatile = true" in kernel.asm[stage]) == volatile
    assert bool(re.search(r"\bload volatile\b", kernel.asm["llir"])) == volatile
    if vector and buffer_ops and not volatile:
        assert "llvm.amdgcn.raw.ptr.buffer.load" in kernel.asm["llir"]

    expected = src.clone()
    if masked:
        expected[129:] = 0
    torch.testing.assert_close(dst, expected)
