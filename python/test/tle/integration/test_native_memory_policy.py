"""Native policies preserve ordinary memory operations and their visibility."""
import pytest
import torch
import triton
import triton.language as tl
from triton.experimental.tle import language as tle
from triton._flagtree_backend import FLAGTREE_BACKEND

BACKEND = FLAGTREE_BACKEND or "nvidia"
DEVICE = "musa" if BACKEND == "mthreads" else "cuda"
POLICY_ATTR = {
    "nvidia": "evictionPolicy = l2_evict_first",
    "mthreads": "cacheModifier = musa_i3_o3_",
    "hcu": "cacheModifier = cs",
    "metax": "cacheModifier = cs",
    "ppu": "cacheModifier = cs",
}[BACKEND]
if BACKEND == "mthreads":
    import torch_musa  # noqa: F401


def policy():
    return {
        "nvidia": tle.gpu.CachePolicy(l2=tle.gpu.FractionalEvictionPolicy("evict_first")),
        "mthreads": tle.gpu.mthreads.LSUCachePolicy(inner_persistence=3, outer_persistence=3),
        "hcu": tle.gpu.hcu.CachePolicy(".cs"),
        "metax": tle.gpu.metax.CachePolicy("cs"),
        "ppu": tle.gpu.ppu.CachePolicy(4),
    }[BACKEND]


@triton.jit
def memory_kernel(X, Y, N: tl.constexpr, POLICY: tl.constexpr, OMIT: tl.constexpr = False, MODIFIER: tl.constexpr = ""):
    i = tl.program_id(0) * 256 + tl.arange(0, 256)
    if OMIT:
        x = tl.load(X + i, i < N, other=0)
        tl.store(Y + i, x + 1, i < N)
    else:
        x = tl.load(X + i, i < N, other=0, cache_policy=POLICY, cache_modifier=MODIFIER)
        tl.store(Y + i, x + 1, i < N, cache_policy=POLICY)


@pytest.mark.parametrize("n", [1, 509, 65533])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16, torch.int32])
def test_mask_update_and_output_overwrite(n, dtype):
    x = torch.zeros(n + 3, dtype=dtype, device=DEVICE)
    y = torch.full_like(x, -11)
    for value in (7, 19):
        x.fill_(value)
        y.fill_(-11)
        kernel = memory_kernel[(triton.cdiv(n, 256), )](x, y, n, policy())
        torch.testing.assert_close(y[:n], x[:n] + 1, rtol=0, atol=0)
        assert torch.all(y[n:] == -11)
    assert POLICY_ATTR in kernel.asm["ttir"]
    assert kernel.metadata.shared == 0


def test_none_preserves_default_and_equal_policies_specialize_by_value():
    x = torch.zeros(509, device=DEVICE)
    y = torch.empty_like(x)
    omitted = memory_kernel[(2, )](x, y, 509, None, True)
    explicit = memory_kernel[(2, )](x, y, 509, None)
    assert POLICY_ATTR not in explicit.asm["ttir"]
    assert omitted.metadata.shared == explicit.metadata.shared == 0
    first = memory_kernel[(2, )](x, y, 509, policy())
    equal = memory_kernel[(2, )](x, y, 509, policy())
    assert first.hash == equal.hash


@pytest.mark.parametrize("invalid,match", [
    ("legacy", "cannot be combined"),
    ("aiu", "requires a global-to-shared copy"),
    ("backend", "not supported by this backend/path"),
])
def test_invalid_policy_is_rejected(invalid, match):
    x = torch.zeros(256, device=DEVICE)
    p = policy()
    modifier = ""
    if invalid == "legacy":
        modifier = ".cg"
    elif invalid == "aiu":
        p = tle.gpu.ppu.AIUCachePolicy(kp=5)
    else:
        p = tle.gpu.hcu.CachePolicy(".cs") if BACKEND != "hcu" else tle.gpu.ppu.CachePolicy(4)
    with pytest.raises(Exception, match=match):
        memory_kernel[(1, )](x, x, 256, p, False, modifier)


@triton.jit
def boolean_memory_kernel(X, Y, N: tl.constexpr, POLICY: tl.constexpr):
    offsets = tl.program_id(0) * 256 + tl.arange(0, 256)
    value = tl.load(X + offsets, offsets < N, other=False, cache_policy=POLICY)
    tl.store(Y + offsets, ~value, offsets < N, cache_policy=POLICY)


def test_boolean_memory_policy_reaches_the_underlying_byte_access():
    x = torch.arange(512, device=DEVICE) % 3 == 0
    y = torch.zeros_like(x)
    kernel = boolean_memory_kernel[(2, )](x, y, 509, policy())
    torch.testing.assert_close(y[:509], ~x[:509], rtol=0, atol=0)
    assert not torch.any(y[509:])
    assert kernel.asm["ttir"].count(POLICY_ATTR) == 2
