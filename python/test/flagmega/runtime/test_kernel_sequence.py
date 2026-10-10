# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Kernel-launch-boundary sequencing (replaces in-kernel grid barriers for
targets that opt in via codegen["kernels"]) -- see
python/tutorials/flagmega/01-qwen3-1.7b-bf16/iluvator-bi-v150/.local/
perf-iteration/ITERATION.md, Trial 31-32.

These tests exercise `_kernel_launch_specs` and `loader._resolve_kernel`
directly against minimal fakes, without constructing a full
GeneratedTirCallGraphModule (which needs a real IR module/buffer plan) --
the logic under test only depends on `self.codegen`/`self.kernels`.
"""
from types import SimpleNamespace

import pytest

from triton.flagmega.errors import ArtifactError
from triton.flagmega.runtime.loader import _resolve_kernel
from triton.flagmega.runtime.module import GeneratedTirCallGraphModule


def _specs(codegen, kernels):
    fake = SimpleNamespace(codegen=codegen, kernels=kernels)
    return GeneratedTirCallGraphModule._kernel_launch_specs(fake)


def test_single_kernel_manifest_is_unchanged():
    codegen = {"dynamic_argument_indices": [0, 1], "grid": [16, 1, 1]}
    specs = _specs(codegen, kernels=("k0",))
    assert specs == ({"dynamic_argument_indices": (0, 1), "grid": (16, 1, 1)},)


def test_single_kernel_manifest_rejects_a_resolved_kernel_tuple():
    codegen = {"dynamic_argument_indices": [0], "grid": [1]}
    with pytest.raises(ArtifactError, match="no 'kernels' list"):
        _specs(codegen, kernels=("k0", "k1"))


def test_multi_kernel_manifest_returns_one_spec_per_kernel_in_order():
    codegen = {
        "kernels": [
            {"symbol": "launch1", "dynamic_argument_indices": [0], "grid": [16, 1, 1]},
            {"symbol": "launch2", "dynamic_argument_indices": [0, 2], "grid": [16, 1, 1]},
        ],
    }
    specs = _specs(codegen, kernels=("k0", "k1"))
    assert specs == (
        {"dynamic_argument_indices": (0,), "grid": (16, 1, 1)},
        {"dynamic_argument_indices": (0, 2), "grid": (16, 1, 1)},
    )


def test_multi_kernel_manifest_rejects_a_resolved_kernel_count_mismatch():
    codegen = {
        "kernels": [
            {"symbol": "launch1", "dynamic_argument_indices": [0], "grid": [1]},
            {"symbol": "launch2", "dynamic_argument_indices": [0], "grid": [1]},
        ],
    }
    with pytest.raises(ArtifactError, match="2 kernel.*1 were resolved"):
        _specs(codegen, kernels=("k0",))


class _FakeGenerated:
    def __init__(self, **symbols):
        for name, value in symbols.items():
            setattr(self, name, value)


def test_resolve_kernel_single_symbol_returns_a_bare_kernel_not_a_tuple():
    generated = _FakeGenerated(flagmega_main="the-kernel")
    codegen = {"symbol": "flagmega_main"}
    kernel = _resolve_kernel(generated, codegen, source="src.py")
    assert kernel == "the-kernel"
    assert not isinstance(kernel, tuple)


def test_resolve_kernel_missing_single_symbol_raises():
    generated = _FakeGenerated()
    codegen = {"symbol": "flagmega_main"}
    with pytest.raises(ArtifactError, match="no entry symbol 'flagmega_main'"):
        _resolve_kernel(generated, codegen, source="src.py")


def test_resolve_kernel_sequence_returns_kernels_in_manifest_order():
    generated = _FakeGenerated(launch1="k1", launch2="k2")
    codegen = {"kernels": [{"symbol": "launch1"}, {"symbol": "launch2"}]}
    kernels = _resolve_kernel(generated, codegen, source="src.py")
    assert kernels == ("k1", "k2")


def test_resolve_kernel_sequence_missing_symbol_raises():
    generated = _FakeGenerated(launch1="k1")
    codegen = {"kernels": [{"symbol": "launch1"}, {"symbol": "launch2"}]}
    with pytest.raises(ArtifactError, match="no entry symbol 'launch2'"):
        _resolve_kernel(generated, codegen, source="src.py")


def test_resource_report_stays_a_single_dict_for_one_kernel():
    module = SimpleNamespace(_prepared=(SimpleNamespace(resource_report={"spill_bytes": 0}),))
    report = GeneratedTirCallGraphModule.resource_report.fget(module)
    assert report == {"spill_bytes": 0}


def test_resource_report_is_a_tuple_of_reports_for_a_kernel_sequence():
    module = SimpleNamespace(_prepared=(
        SimpleNamespace(resource_report={"spill_bytes": 0}),
        SimpleNamespace(resource_report={"spill_bytes": 4}),
    ))
    report = GeneratedTirCallGraphModule.resource_report.fget(module)
    assert report == ({"spill_bytes": 0}, {"spill_bytes": 4})


def test_resource_report_is_none_before_prepare():
    module = SimpleNamespace(_prepared=None)
    assert GeneratedTirCallGraphModule.resource_report.fget(module) is None


def test_launch_external_launches_every_prepared_kernel_in_order(monkeypatch):
    calls = []

    class _Prepared:
        def __init__(self, tag):
            self.tag = tag

        def launch(self, *args, stream=None):
            calls.append((self.tag, args, stream))

    module = SimpleNamespace(
        _prepared=(_Prepared("first"), _Prepared("second")),
        _dynamic_descriptor_indices=(),
    )
    module._require_prepared = lambda: None
    module._validate_external_arguments = lambda args: None
    module._materialize_tensor_descriptors = lambda args: ()

    GeneratedTirCallGraphModule._launch_external(module, ("arg0",), stream="s")
    assert [tag for tag, _args, _stream in calls] == ["first", "second"]
    assert all(args == ("arg0",) and stream == "s" for _tag, args, stream in calls)
