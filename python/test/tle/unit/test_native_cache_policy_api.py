"""Public native policy exports and target/path validation without a GPU."""
import importlib
from types import SimpleNamespace

import pytest
from triton.experimental.tle.language import gpu
from triton.experimental.tle.language.gpu._cache_policy import resolve_policy


def resolve(policy, backend="nvidia", descriptor=False, load=True, common_ir=False, arch=None):
    options = SimpleNamespace(backend_name=backend, arch=arch or ("gfx936" if backend in ("hip", "hcu") else 90))
    semantic = SimpleNamespace(builder=SimpleNamespace(options=options))
    return resolve_policy(policy, descriptor, load, semantic, backend == "musa", False, common_ir)


def test_nvidia_public_exports_and_removed_namespace():
    for name in ("CachePolicy", "FractionalEvictionPolicy"):
        assert name in gpu.__all__
        assert getattr(gpu, name) is not None
    assert "nvidia" not in gpu.__all__
    assert not hasattr(gpu, "nvidia")
    with pytest.raises(ModuleNotFoundError) as error:
        importlib.import_module("triton.experimental.tle.language.gpu.nvidia")
    assert error.value.name == "triton.experimental.tle.language.gpu.nvidia"
    dispatcher = importlib.import_module("triton.experimental.tle.language.gpu._cache_policy")
    assert not hasattr(dispatcher, "CachePolicy")


@pytest.mark.parametrize("name",
                         ["RuntimeEvictionPolicy", "make_fractional_eviction_policy", "make_range_eviction_policy"])
def test_dynamic_policy_api_is_not_exported(name):
    assert name not in gpu.__all__
    assert not hasattr(gpu, name)
    implementation = importlib.import_module("triton.experimental.tle.language.gpu._nvidia_cache_policy")
    assert not hasattr(implementation, name)


@pytest.mark.parametrize("value", [0, 1, object()])
def test_raw_policy_values_are_rejected(value):
    with pytest.raises(TypeError, match="compile-time FractionalEvictionPolicy"):
        gpu.CachePolicy(l2=value)


def test_empty_nvidia_policy_requires_none():
    with pytest.raises(ValueError, match="use None for the default"):
        gpu.CachePolicy()


@pytest.mark.parametrize("level", ["l1", "l2"])
@pytest.mark.parametrize("intent", ["reuse", "streaming", "bypass", "default"])
def test_removed_generic_intents_are_rejected(level, intent):
    with pytest.raises(ValueError if level == "l1" else TypeError):
        gpu.CachePolicy(**{level: intent})


@pytest.mark.parametrize("backend", [None, "nvidia", "mthreads", "ppu", "hcu", "metax"])
def test_removed_backend_argument(backend):
    with pytest.raises(TypeError, match="backend"):
        gpu.CachePolicy(l1="evict_normal", backend=backend)


@pytest.mark.parametrize("backend", ["nvidia", "musa", "ppu", "hcu", "maca", "amd", "unknown"])
@pytest.mark.parametrize("common_ir", [False, True])
def test_none_preserves_backend_default(backend, common_ir):
    assert resolve(None, backend, common_ir=common_ir) is None


NATIVE_POLICIES = [
    ("nvidia", gpu.CachePolicy(l1="evict_normal")),
    ("musa", gpu.mthreads.LSUCachePolicy()),
    ("ppu", gpu.ppu.CachePolicy()),
    ("hcu", gpu.hcu.CachePolicy()),
    ("maca", gpu.metax.CachePolicy("ca")),
]


@pytest.mark.parametrize("owner,policy", NATIVE_POLICIES)
@pytest.mark.parametrize("backend", ["nvidia", "musa", "ppu", "hcu", "maca"])
def test_native_objects_are_target_bound_and_keep_identity(owner, policy, backend):
    if owner == backend:
        assert resolve(policy, backend) is policy
    else:
        with pytest.raises(ValueError, match="backend/path"):
            resolve(policy, backend)


@pytest.mark.parametrize("backend,policy", NATIVE_POLICIES)
def test_common_ir_rejects_native_objects(backend, policy):
    with pytest.raises(ValueError, match="not supported"):
        resolve(policy, backend, common_ir=True)


@pytest.mark.parametrize("priority", ["evict_first", "evict_last", "evict_normal", "evict_unchanged"])
@pytest.mark.parametrize("load", [False, True])
def test_nvidia_tma_accepts_native_l2(priority, load):
    policy = gpu.CachePolicy(l2=gpu.FractionalEvictionPolicy(priority))
    assert resolve(policy, descriptor=True, load=load) is policy


@pytest.mark.parametrize("policy", [gpu.CachePolicy(l1="evict_last"), gpu.CachePolicy(cache_modifier="cg")])
def test_nvidia_tma_rejects_non_l2_controls(policy):
    with pytest.raises(ValueError, match="only L2"):
        resolve(policy, descriptor=True)


def test_mthreads_path_types_remain_distinct():
    tme, lsu = gpu.mthreads.TMECachePolicy(), gpu.mthreads.LSUCachePolicy()
    assert resolve(tme, "musa", descriptor=True) is tme
    assert resolve(lsu, "musa") is lsu
    for policy, descriptor in ((tme, False), (lsu, True)):
        with pytest.raises(ValueError, match="backend/path"):
            resolve(policy, "musa", descriptor=descriptor)


def test_hip_target_uses_selected_backend(monkeypatch):
    import triton._flagtree_backend as selected
    policy = gpu.hcu.CachePolicy()
    monkeypatch.setattr(selected, "FLAGTREE_BACKEND", "hcu")
    assert resolve(policy, "hip") is policy
    with pytest.raises(ValueError, match="gfx936"):
        resolve(policy, "hip", arch="gfx90a")
    assert resolve(None, "hip", arch="gfx90a") is None
    monkeypatch.setattr(selected, "FLAGTREE_BACKEND", "")
    with pytest.raises(ValueError, match="backend amd"):
        resolve(policy, "hip")
