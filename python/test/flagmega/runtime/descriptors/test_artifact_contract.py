# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT

from types import SimpleNamespace

import pytest

import triton.flagmega.runtime.module as runtime_module
from triton.flagmega.errors import ArtifactError
from triton.flagmega.runtime.module import GeneratedTirCallGraphModule

from .helpers import single_spec


def _construct(monkeypatch, specs, *, signature, dynamic):
    monkeypatch.setattr(runtime_module, "verify_buffer_plan", lambda module: object())
    binding = {
        "arguments": ({"name": "source", "buffer": "buffer"},),
        "pools": (),
        "signature": ("source",),
    }
    return GeneratedTirCallGraphModule(
        artifact=None,
        manifest={"codegen": {
            "runtime_binding": binding,
            "host_tensor_descriptor_specs": specs,
            "signature_arguments": signature,
            "dynamic_argument_indices": dynamic,
        }},
        ir_module=SimpleNamespace(),
        kernel=object(),
    )


def test_artifact_rejects_descriptor_source_outside_runtime_roots(monkeypatch):
    with pytest.raises(ArtifactError, match="unbound source"):
        _construct(
            monkeypatch,
            (single_spec(source="missing"),),
            signature=("source", "weight_descriptor"),
            dynamic=(0,),
        )


def test_artifact_rejects_descriptor_signature_or_dynamic_index_drift(monkeypatch):
    spec = single_spec(source="source")
    with pytest.raises(ArtifactError, match="generated signature"):
        _construct(
            monkeypatch,
            (spec,),
            signature=("source",),
            dynamic=(0, 1),
        )
    with pytest.raises(ArtifactError, match="dynamic arguments"):
        _construct(
            monkeypatch,
            (spec,),
            signature=("source", "weight_descriptor"),
            dynamic=(0,),
        )
