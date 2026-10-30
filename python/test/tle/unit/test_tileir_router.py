from types import SimpleNamespace

import pytest

pytest.importorskip("triton.backends.tileir.router")

from triton.backends.compiler import GPUTarget
from triton.backends.tileir import router

TILEIR_PRIMS = frozenset({
    "create_mem_token",
    "join_mem_tokens",
    "load_view_tko",
    "store_view_tko",
    "dim",
    "make_tensor_view",
    "make_partition_view",
    "make_view",
})


@pytest.fixture(autouse=True)
def _stub_tileir_primitives(monkeypatch):
    monkeypatch.setattr(router, "_tileir_supported_tle_primitives", lambda: TILEIR_PRIMS)


def _cuda_target():
    return GPUTarget(backend="cuda", arch=90, warp_size=32)


def test_scan_uses_tle_supported_whitelist():
    assert router._scan_source_for_tle("def k():\n    tle.make_view()\n") is router.TLEUsage.TILEIR_SUBSET
    assert router._scan_source_for_tle("def k():\n    tle.gpu.alloc()\n") is router.TLEUsage.OTHER
    # Mixed TileIR + NVIDIA TLE is treated as OTHER (nvidia fallback).
    assert router._scan_source_for_tle("def k():\n    tle.make_view()\n    tle.gpu.alloc()\n") is router.TLEUsage.OTHER


def test_routing_requires_env(monkeypatch):
    jit_fn = SimpleNamespace(fn=None, _flagtree_tle_usage=None)
    monkeypatch.setattr(router, "kernel_tle_usage", lambda _fn: router.TLEUsage.TILEIR_SUBSET)

    monkeypatch.delenv("FLAGTREE_USE_TILEIR", raising=False)
    assert router.pick_target_for_kernel(_cuda_target(), jit_fn).backend == "cuda"

    monkeypatch.setenv("FLAGTREE_USE_TILEIR", "1")
    assert router.pick_target_for_kernel(_cuda_target(), jit_fn).backend == "tileir"
