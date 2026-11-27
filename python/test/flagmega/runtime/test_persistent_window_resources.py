# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Persistent communication storage must survive scratch reuse in one window."""

from types import SimpleNamespace

import torch
import pytest

import triton.experimental.tle.language as tle
import triton.flagmega.runtime.module as runtime_module
from triton.flagmega.runtime.module import GeneratedTirCallGraphModule
from triton.flagmega.ir.persistent_window import plan_window_resources, validate_window_resources
from triton.flagmega.errors import IRVerificationError


def _request():
    return {"storage": "workspace", "resources": [
        {"name": "channel", "nbytes": 8448, "alignment": 256,
         "initialization": "zero", "programs": ["first", "second"]},
    ]}


def test_channel_range_follows_whole_scratch_and_reuses_only_ordered_actors():
    pools = [{"storage": "workspace", "nbytes": 4097}]
    programs = [{"symbol": "first", "depends_on": []}, {"symbol": "second", "depends_on": ["first"]}]
    plan = plan_window_resources(_request(), pools, programs, "workspace")
    assert plan["resources"][0]["offset"] == 4352
    assert plan["allocation_nbytes"] == 16384
    assert validate_window_resources(plan, pools) == plan
    with pytest.raises(IRVerificationError, match="ordered by dependencies"):
        plan_window_resources(_request(), pools, [{**p, "depends_on": []} for p in programs], "workspace")


@pytest.mark.parametrize("field,value", [("offset", 4096), ("nbytes", 20000), ("alignment", 3),
                                         ("initialization", "uninitialized")])
def test_invalid_or_unprotected_persistent_range_is_rejected(field, value):
    pools = [{"storage": "workspace", "nbytes": 4097}]
    programs = [{"symbol": "first", "depends_on": []}, {"symbol": "second", "depends_on": ["first"]}]
    plan = plan_window_resources(_request(), pools, programs, "workspace")
    plan["resources"][0][field] = value
    with pytest.raises(IRVerificationError):
        validate_window_resources(plan, pools)


def test_load_reserves_and_initializes_persistent_window_tail(monkeypatch):
    runtime = object.__new__(GeneratedTirCallGraphModule)
    runtime.codegen = {
        "device_window_storage": "workspace",
        "persistent_window_resources": {
            "schema": "flagmega.persistent-window-resources/v1",
            "storage": "workspace",
            "scratch_nbytes": 4096,
            "allocation_nbytes": 4352,
            "resources": [{"name": "channel", "offset": 4096, "nbytes": 256,
                           "alignment": 256, "initialization": "zero", "programs": ["first", "second"]}],
        },
    }
    runtime.manifest = {"target": "nvidia-sm90"}
    runtime.runtime_binding = {"pools": [{"name": "workspace", "storage": "workspace", "nbytes": 4096}]}
    runtime.buffer_plan = SimpleNamespace(memory_space_map={
        "workspace": SimpleNamespace(kind="global", allocation_scope=SimpleNamespace(value="function")),
    })
    runtime._needs_device_dptr = True
    runtime._device_window = None
    runtime._function_offset_specs = ()
    runtime._mark_loaded = lambda device: None
    monkeypatch.setattr(runtime_module, "_torch", lambda: torch)
    monkeypatch.setattr(runtime_module, "_validate_artifact_device", lambda *args: None)
    monkeypatch.setattr(tle, "allocate_dist_tensor", lambda shape, **kwargs: torch.full(shape, 165, dtype=torch.uint8))

    runtime.load("cpu")
    assert runtime._device_window.numel() == 4352
    scratch = runtime._pool_values["workspace"]
    assert scratch.numel() == 4096
    assert scratch.data_ptr() == runtime._device_window.data_ptr()
    assert (scratch == 165).all()
    assert (runtime._device_window[4096:] == 0).all()
    scratch.fill_(77)
    assert (runtime._device_window[4096:] == 0).all()
