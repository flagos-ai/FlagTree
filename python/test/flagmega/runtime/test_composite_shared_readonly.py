# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: MIT
"""The composite adapter preserves the public content-verified pool contract."""

from types import SimpleNamespace

import pytest

from triton.flagmega.runtime import loader
from triton.flagmega.runtime.module import (
    GeneratedTirCallGraphModule,
    GeneratedTirCompositeModelModule,
)
from triton.flagmega.runtime.shared_readonly import SharedReadonlyPool


@pytest.mark.parametrize("share", [False, True])
def test_public_composite_loading_preserves_shared_pool(monkeypatch, tmp_path, share):
    # Isolate public adapter dispatch from device allocation. The subclass's
    # real load override must still run its ownership initialization.
    runtime = GeneratedTirCompositeModelModule.__new__(GeneratedTirCompositeModelModule)
    runtime._kda_config = SimpleNamespace(logical_types=None)
    received = []

    def allocate(self, device="cuda:0", *, shared_rdata=None):
        received.append((self, device, shared_rdata))
        return self

    monkeypatch.setattr(GeneratedTirCallGraphModule, "load", allocate)
    manifest = {
        "status": "executable", "target": "nvidia-sm90", "codegen_backend": "cpp",
        "codegen": {"kind": "composite-adapter-test"},
    }
    monkeypatch.setattr(loader, "load_artifact", lambda *a, **k: (manifest, None))
    monkeypatch.setattr(loader.package_registry, "resolve",
                        lambda *a: SimpleNamespace(factory=lambda *a: runtime))
    pool = SharedReadonlyPool(16, device="cpu") if share else None
    result = loader.load(tmp_path, device="cpu", shared_rdata=pool)
    assert result is runtime
    assert received == [(runtime, "cpu", pool)]
