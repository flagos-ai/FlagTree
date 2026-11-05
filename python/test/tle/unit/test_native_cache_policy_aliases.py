"""Native MThreads and PPU numeric aliases and unsafe-mode rejection."""
import pytest
import triton.language as tl
from triton.experimental.tle.language import gpu


@pytest.mark.parametrize("inner", range(6))
@pytest.mark.parametrize("outer", range(4))
def test_tme_integer_aliases(inner, outer):
    names = ("none", "once", "normal", "persist")
    numeric = gpu.mthreads.TMECachePolicy(tl.constexpr(inner), tl.constexpr(outer))
    named = gpu.mthreads.TMECachePolicy(names[inner] if inner < 4 else inner, names[outer])
    assert numeric == named and hash(numeric) == hash(named)
    assert numeric.to_attributes(True)["inner"] == inner
    assert numeric.to_attributes(True)["outer"] == outer


@pytest.mark.parametrize("inner,outer", [(-1, 0), (6, 0), (0, 4), (0, 5), (True, 0), (0, True)])
def test_invalid_tme_integer_aliases(inner, outer):
    with pytest.raises(ValueError):
        gpu.mthreads.TMECachePolicy(inner, outer)


@pytest.mark.parametrize("kp,modifier", [(1, ".nc"), (2, ".cg"), (4, ".cs"), (5, ".lu"), (6, ".bl")])
def test_ppu_lsu_aliases(kp, modifier):
    a, b = gpu.ppu.CachePolicy(tl.constexpr(kp)), gpu.ppu.CachePolicy(modifier)
    assert a == b and hash(a) == hash(b) and a.type.mangle() == b.type.mangle()
    a.validate_copy(False, True)
    if kp in (1, 5):
        with pytest.raises(ValueError, match="store"):
            a.validate_copy(False, False)
    else:
        a.validate_copy(False, False)


@pytest.mark.parametrize("kp", [0, 3, 7, 8, True, 1.0])
def test_ppu_lsu_rejects_unverified_aliases(kp):
    with pytest.raises(ValueError):
        gpu.ppu.CachePolicy(kp)


@pytest.mark.parametrize("load", [False, True])
def test_ppu_ba_rejected(load):
    with pytest.raises(ValueError, match="visibility"):
        gpu.ppu.CachePolicy(".ba").validate_copy(False, load)


@pytest.mark.parametrize("load", [False, True])
def test_musa_cache_enum_encodes_every_lsu_tuple(load):
    from triton._C.libtriton import ir
    modes = set()
    for inner in range(6):
        for outer in range(6):
            for l2 in range(2):
                policy = gpu.mthreads.LSUCachePolicy(inner, outer, l2)
                cache, evict = policy.to_ir(load)
                assert evict == ir.EVICTION_POLICY.NORMAL
                value = int(cache) - 128
                assert (value // 12, value // 2 % 6, value % 2) == (inner, outer, l2)
                modes.add(cache)
    assert len(modes) == 72


def test_ppu_aiu_modes_are_distinct_from_lsu_modifiers():
    aiu = {gpu.ppu.AIUCachePolicy(kp).to_ir(True)[0] for kp in range(1, 7)}
    lsu = {gpu.ppu.CachePolicy(kp).to_ir(True)[0] for kp in (1, 2, 4, 5, 6)}
    assert len(aiu) == 6 and not aiu.intersection(lsu)
    with pytest.raises(ValueError, match="kp7 is unsupported"):
        gpu.ppu.AIUCachePolicy(7).to_ir(True)
