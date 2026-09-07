# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT

from types import SimpleNamespace

import pytest

import triton.flagmega.runtime.module as runtime_module
from triton.flagmega.runtime.module import GeneratedTirCallGraphModule

from .helpers import single_spec


class _Prepared:
    resource_report = {}

    def __init__(self):
        self.launches = []

    def launch(self, *arguments, stream=None):
        self.launches.append((arguments, stream))


def _runtime(monkeypatch, descriptor_spec):
    prepared = _Prepared()
    captured = {}

    def prepare(kernel, arguments, dynamic_indices, **options):
        captured.update(
            arguments=arguments,
            dynamic_indices=dynamic_indices,
            options=options,
        )
        return prepared

    monkeypatch.setattr(runtime_module, "verify_buffer_plan", lambda module: object())
    monkeypatch.setattr(runtime_module, "prepare_jit_kernel", prepare)
    binding = {
        "arguments": ({
            "name": "source",
            "buffer": "source_buffer",
        },),
        "pools": (),
        "signature": ("source",),
    }
    codegen = {
        "runtime_binding": binding,
        "host_tensor_descriptor_specs": (descriptor_spec,),
        "signature_arguments": ("source", "weight_descriptor"),
        "dynamic_argument_indices": (0, 1),
        "symbol": "entry",
        "num_warps": 1,
        "grid": (1, 1, 1),
    }
    result = GeneratedTirCallGraphModule(
        artifact=None,
        manifest={"codegen": codegen},
        ir_module=SimpleNamespace(),
        kernel=object(),
    )
    result._mark_loaded("cpu")
    return result, prepared, captured


def test_external_backed_descriptor_is_a_dynamic_prepared_argument(monkeypatch):
    torch = pytest.importorskip("torch")
    runtime, prepared, captured = _runtime(
        monkeypatch,
        single_spec(source="source"),
    )
    first = torch.empty((3, 4), dtype=torch.float32)
    second = torch.empty((5, 4), dtype=torch.float32)

    runtime.prepare(first)
    runtime.run_into(second, stream="stream")

    assert captured["dynamic_indices"] == (0, 1)
    assert captured["arguments"][0] is first
    assert captured["arguments"][1].base.data_ptr() == first.data_ptr()
    assert prepared.launches[0][0][0] is second
    assert prepared.launches[0][0][1].base.data_ptr() == second.data_ptr()
    assert prepared.launches[0][0][1].shape == [5, 4]
    assert prepared.launches[0][1] == "stream"
