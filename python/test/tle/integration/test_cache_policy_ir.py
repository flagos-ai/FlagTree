"""Native policy objects reuse cache enums without merging independent controls."""
import re

import pytest
import torch
import triton
import triton.language as tl
from triton._flagtree_backend import FLAGTREE_BACKEND
from triton.experimental.tle.language import gpu

BACKEND = FLAGTREE_BACKEND or "nvidia"
DEVICE = "musa" if BACKEND == "mthreads" else "cuda"
if BACKEND == "mthreads":
    import torch_musa  # noqa: F401


@triton.jit
def _memory(X, Y, LOAD: tl.constexpr, STORE: tl.constexpr, READ: tl.constexpr = "", WRITE: tl.constexpr = ""):
    i = tl.arange(0, 256)
    x = tl.load(X + i, i < 253, other=0, cache_modifier=READ, cache_policy=LOAD)
    tl.store(Y + i, x + 1, i < 253, cache_modifier=WRITE, cache_policy=STORE)


MODIFIERS = {
    "nvidia": (["ca", "cg", "cs", "lu", "cv"], ["wb", "cg", "cs", "wt"]),
    "hcu": (["", "ca", "cg", "cs", "cv"], ["", "wb", "cg", "cs"]),
    "metax": (["ca", "cg", "cs", "lu", "cv"], ["wb", "cg", "cs", "wt"]),
    "ppu": (["", "ca", "cg", "cs", "cv", "lu", "nc", "bl"], ["", "wb", "cg", "cs", "wt", "bl"]),
    "mthreads": ([], []),
}


@pytest.mark.skipif(BACKEND == "mthreads", reason="MUSA LSU uses a persistence tuple")
@pytest.mark.parametrize("read,write", [(".ca", ".wb"), (".cg", ".cg"), (".cv", ".cs")])
def test_string_and_object_share_memory_controls(read, write):
    cls = gpu.CachePolicy if BACKEND == "nvidia" else getattr(gpu, BACKEND).CachePolicy
    x = torch.arange(256, dtype=torch.float32, device=DEVICE)
    for args in ((None, None, read, write), (cls(read), cls(write), "", "")):
        y = torch.full_like(x, -17)
        kernel = _memory[(1, )](x, y, *args)
        torch.testing.assert_close(y[:253], x[:253] + 1, rtol=0, atol=0)
        assert torch.all(y[253:] == -17)
        for stage in ("ttir", "ttgir"):
            controls = re.findall(r"cacheModifier = (\w+)", kernel.asm[stage])
            assert controls == [read[1:], write[1:]]


@pytest.mark.skipif(BACKEND != "hcu", reason="gfx936 write-through buffer store")
def test_hcu_legacy_write_through_wait():
    x = torch.arange(256, dtype=torch.float32, device=DEVICE)
    y = torch.full_like(x, -17)
    kernel = _memory[(1, )](x, y, None, None, "", ".wt")
    torch.testing.assert_close(y[:253], x[:253] + 1, rtol=0, atol=0)
    assert torch.all(y[253:] == -17)
    assert re.search(r"buffer_store .*cacheModifier = wt", kernel.asm["ttgir"])
    assert "call void @llvm.amdgcn.s.waitcnt(" in kernel.asm["llir"]


@pytest.mark.parametrize("is_load,modifier",
                         [(True, m) for m in MODIFIERS[BACKEND][0]] + [(False, m) for m in MODIFIERS[BACKEND][1]])
def test_simple_modifier_uses_existing_cache_enum(is_load, modifier):
    cls = gpu.CachePolicy if BACKEND == "nvidia" else getattr(gpu, BACKEND).CachePolicy
    policy = cls("." + modifier if modifier else "")
    # Give both accesses explicit policies to exercise direction normalization.
    default = cls("ca" if BACKEND in ("nvidia", "metax") else ".ca") if not is_load else cls("wb" if BACKEND in (
        "nvidia", "metax") else ".wb")
    x = torch.full((256, ), 17., device=DEVICE)
    y = torch.full_like(x, -7)
    kernel = _memory[(1, )](x, y, policy if is_load else default, default if is_load else policy)
    torch.testing.assert_close(y[:253], x[:253] + 1, rtol=0, atol=0)
    assert torch.all(y[253:] == -7)
    for stage in ("ttir", "ttgir"):
        ir = kernel.asm[stage]
        kinds = ("tt.load ", ".buffer_load ") if is_load else ("tt.store ", ".buffer_store ")
        access = next(line for line in ir.splitlines() if any(kind in line for kind in kinds))
        if modifier:
            assert f"cacheModifier = {modifier}" in access
        assert "tt.cache_policy" not in ir
        assert "cache_modifier =" not in ir


@pytest.mark.skipif(BACKEND != "hcu", reason="HCU gfx936 buffer controls")
@pytest.mark.parametrize("modifier", [".ca", ".cg", ".cs", ".cv"])
def test_hcu_hint_keeps_the_default_buffer_path(modifier):
    x = torch.full((256, ), 17., device=DEVICE)
    y = torch.full_like(x, -7)
    default = _memory[(1, )](x, y, None, None)
    hinted = _memory[(1, )](x, y, gpu.hcu.CachePolicy(modifier), gpu.hcu.CachePolicy(".cs"))
    for kernel in (default, hinted):
        assert ".buffer_load " in kernel.asm["ttgir"]
        assert ".buffer_store " in kernel.asm["ttgir"]
    # The gfx936 target maps these buffer modifiers to default control bits.
    # Compare the memory intrinsic operands, excluding debug metadata.
    def memory_instructions(kernel):
        return [
            re.sub(r", !dbg !\d+", "", line.strip())
            for line in kernel.asm["llir"].splitlines()
            if "call " in line and any(name in line for name in ("buffer.load", "buffer.store"))
        ]

    assert memory_instructions(default)
    assert memory_instructions(default) == memory_instructions(hinted)
    torch.testing.assert_close(y[:253], x[:253] + 1, rtol=0, atol=0)
    assert torch.all(y[253:] == -7)


@pytest.mark.skipif(BACKEND != "nvidia", reason="NVIDIA independent L1/L2 controls")
@pytest.mark.parametrize("level", ["l1", "l2"])
def test_nvidia_levels_remain_independent(level):
    policy = gpu.CachePolicy(
        l1="evict_first" if level == "l1" else None,
        l2=gpu.FractionalEvictionPolicy("evict_last") if level == "l2" else None,
    )
    x = torch.full((256, ), 23., device=DEVICE)
    y = torch.zeros_like(x)
    kernel = _memory[(1, )](x, y, policy, None)
    torch.testing.assert_close(y[:253], x[:253] + 1, rtol=0, atol=0)
    assert "tt.cache_policy" not in kernel.asm["ttir"]
    assert f"evictionPolicy = {level}_evict_" in kernel.asm["ttir"]
    loads = [line for line in kernel.asm["ptx"].splitlines() if re.search(r"\bld\.global", line)]
    assert loads
    assert any("L1::evict_first" in line for line in loads) == (level == "l1")
    assert any("L2::cache_hint" in line for line in loads) == (level == "l2")


_FOREIGN_FIELDS = {
    "hcu": ["cacheModifier = ppu_aiu_kp5", "evictionPolicy = l2_evict_last"],
    "ppu": ["cacheModifier = musa_i3_o3_l1"],
}


@pytest.mark.parametrize("foreign", _FOREIGN_FIELDS.get(BACKEND, ["cacheModifier = ppu_aiu_kp5"]))
def test_ir_rejects_foreign_native_policy(tmp_path, capfd, foreign):
    policy = {
        "nvidia": gpu.CachePolicy(l2=gpu.FractionalEvictionPolicy("evict_last")),
        "mthreads": gpu.mthreads.LSUCachePolicy(3, 3, 1),
        "hcu": gpu.hcu.CachePolicy(".cs"),
        "metax": gpu.metax.CachePolicy("cs"),
        "ppu": gpu.ppu.CachePolicy(".cs"),
    }[BACKEND]
    x = torch.ones(256, device=DEVICE)
    y = torch.zeros_like(x)
    kernel = _memory[(1, )](x, y, policy, policy)
    # The same formal cache field carries backend-specific modes. A foreign
    # native mode must fail lowering, never silently become a default modifier.
    ir = kernel.asm["ttir"]
    ir = re.sub(r"(?:cacheModifier|evictionPolicy) = \w+", foreign, ir)
    path = tmp_path / "foreign.ttir"
    path.write_text(ir)
    with pytest.raises(RuntimeError):
        triton.compile(str(path), target=triton.runtime.driver.active.get_current_target())
    diagnostic = capfd.readouterr().err.lower()
    assert ("eviction scope" if foreign.startswith("evictionPolicy") else "cache") in diagnostic


@pytest.mark.skipif(BACKEND != "nvidia", reason="PTX comparison")
def test_scoped_eviction_survives_ir_location_replacement(monkeypatch, tmp_path):
    monkeypatch.setenv("USE_IR_LOC", "ttgir")
    monkeypatch.setenv("TRITON_CACHE_DIR", str(tmp_path))
    x = torch.ones(256, device=DEVICE)
    y = torch.empty_like(x)
    policy = gpu.CachePolicy(l2=gpu.FractionalEvictionPolicy("evict_last"))
    kernel = _memory[(1, )](x, y, policy, policy)
    torch.testing.assert_close(y[:253], x[:253] + 1, rtol=0, atol=0)
    assert "tt.cache_policy" not in kernel.asm["ttgir"]
    assert "evictionPolicy = l2_evict_last" in kernel.asm["ttgir"]
    assert "createpolicy.fractional.L2::evict_last" in kernel.asm["ptx"]
    accesses = [line for line in kernel.asm["ptx"].splitlines() if re.search(r"\b(?:ld|st)\.global", line)]
    assert accesses and all(".L2::cache_hint" in line for line in accesses)


@triton.jit
def _legacy_eviction(X, Y, EVICT: tl.constexpr):
    i = tl.arange(0, 256)
    value = tl.load(X + i, eviction_policy=EVICT)
    tl.store(Y + i, value, eviction_policy=EVICT)


@pytest.mark.skipif(BACKEND != "nvidia", reason="NVIDIA legacy eviction semantics")
@pytest.mark.parametrize("priority", ["evict_first", "evict_last"])
def test_legacy_eviction_preserves_both_levels(priority):
    x = torch.arange(256, dtype=torch.float32, device=DEVICE)
    y = torch.empty_like(x)
    kernel = _legacy_eviction[(1, )](x, y, priority)
    torch.testing.assert_close(y, x, rtol=0, atol=0)
    assert f"evictionPolicy = {priority}" in kernel.asm["ttir"]
    assert f"createpolicy.fractional.L2::{priority}" in kernel.asm["ptx"]
    accesses = [line for line in kernel.asm["ptx"].splitlines() if re.search(r"\b(?:ld|st)\.global", line)]
    assert accesses and all(f"L1::{priority}" in line and "L2::cache_hint" in line for line in accesses)


@pytest.mark.skipif(BACKEND != "ppu", reason="PPU explicit LSU default")
def test_ppu_explicit_lsu_default_roundtrips(tmp_path):
    x = torch.ones(256, device=DEVICE)
    y = torch.empty_like(x)
    policy = gpu.ppu.CachePolicy()
    kernel = _memory[(1, )](x, y, policy, policy)
    torch.testing.assert_close(y[:253], x[:253] + 1, rtol=0, atol=0)
    source = tmp_path / "lsu-default.ttir"
    source.write_text(kernel.asm["ttir"])
    rebuilt = triton.compile(str(source), target=triton.runtime.driver.active.get_current_target())
    for ir in (kernel.asm["ttir"], rebuilt.asm["ttgir"]):
        assert "cacheModifier = ppu_lsu_default" in ir
        assert "async_aiu_copy" not in ir
