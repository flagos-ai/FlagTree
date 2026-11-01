# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Static descriptor bindings live with preparation; external ones follow calls."""

from types import SimpleNamespace

import pytest
from triton.flagmega.runtime import module as runtime_module
from triton.flagmega.runtime import tensor_descriptor
from triton.flagmega.runtime.module import GeneratedTirCallGraphModule

from .helpers import single_spec


@pytest.mark.parametrize("pool_count", [1, 218])
@pytest.mark.parametrize("has_external_descriptor", [False, True])
def test_prepared_pool_descriptors_do_not_rematerialize_per_launch(
    monkeypatch, pool_count, has_external_descriptor
):
    torch = pytest.importorskip("torch")
    launches, prepared_arguments = [], []

    def prepare(kernel, arguments, dynamic_indices, **options):
        prepared_arguments.append(arguments)
        return SimpleNamespace(launch=lambda *args, stream=None: launches.append((args, stream)))

    monkeypatch.setattr(runtime_module, "verify_buffer_plan", lambda module: object())
    monkeypatch.setattr(runtime_module, "prepare_jit_kernel", prepare)
    specs = [single_spec(name=f"pool_{i}", source="weights") for i in range(pool_count)]
    if has_external_descriptor:
        specs.insert(1, single_spec(name="external", source="source"))
    dynamic_descriptor_indices = (1,) if has_external_descriptor else ()
    codegen = {
        "runtime_binding": {
            "arguments": ({"name": "source", "buffer": "source_buffer"},),
            "pools": ({"name": "weights", "storage": "rdata", "nbytes": 48},),
            "signature": ("source", "weights"),
        },
        "host_tensor_descriptor_specs": tuple(specs),
        "signature_arguments": ("source", "weights", *(spec["name"] for spec in specs)),
        "dynamic_argument_indices": (0, *(2 + i for i in dynamic_descriptor_indices)),
        "symbol": "entry", "num_warps": 1, "grid": (1, 1, 1),
    }
    runtime = GeneratedTirCallGraphModule(
        artifact=None, manifest={"codegen": codegen},
        ir_module=SimpleNamespace(metadata={}), kernel=object(),
    )
    pool = torch.empty((3, 4), dtype=torch.float32)
    runtime._pool_values = {"weights": pool}
    runtime._mark_loaded("cpu")
    views = []
    original_view = tensor_descriptor._view_typed_buffer

    def track_view(storage, *args, **kwargs):
        views.append(storage.data_ptr())
        return original_view(storage, *args, **kwargs)

    monkeypatch.setattr(tensor_descriptor, "_view_typed_buffer", track_view)
    source = torch.empty((3, 4), dtype=torch.float32)
    replacement = torch.empty((5, 4), dtype=torch.float32)
    runtime.prepare(source)
    assert views.count(pool.data_ptr()) == pool_count
    views.clear()

    runtime.run_into(source, stream="stream")
    runtime.run_into(replacement, stream="stream")

    assert pool.data_ptr() not in views, "Prepared static descriptors must not rebuild byte views at launch."
    assert launches[0][0][0] is source and launches[1][0][0] is replacement
    assert all(stream == "stream" for _, stream in launches)
    if has_external_descriptor:
        assert launches[0][0][1].base.data_ptr() == source.data_ptr()
        assert launches[1][0][1].base.data_ptr() == replacement.data_ptr()
        assert launches[1][0][1].shape == [5, 4]
    else:
        assert all(len(args) == 1 for args, _ in launches)

    # A new preparation validates and binds all static descriptors again.
    new_pool = pool.clone()
    runtime._pool_values = {"weights": new_pool}
    views.clear()
    runtime.prepare(replacement)
    assert views.count(new_pool.data_ptr()) == pool_count
    for index, spec in enumerate(specs):
        expected = replacement if spec["source"] == "source" else new_pool
        assert prepared_arguments[-1][2 + index].base.data_ptr() == expected.data_ptr()
