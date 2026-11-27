# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT

from pathlib import Path

import pytest

from triton.flagmega.artifacts import write_artifact
from triton.flagmega.compiler import Compiler
from triton.flagmega.errors import ArtifactError
from triton.flagmega import ir as fm
from triton.flagmega.runtime import (
    GeneratedTirSingleTensorModule,
    RuntimePackageRegistry,
    load,
)


def _make_add_module():
    builder = fm.IRBuilder(dialect="high_level", stage="imported", metadata={"model": "runtime-registry"})
    value_type = fm.tensor_type("float32", [17])
    lhs = builder.var("lhs", value_type, id="lhs")
    rhs = builder.var("rhs", value_type, id="rhs")
    output = builder.call("math.add", [lhs, rhs], value_type, id="output")
    builder.function("main", [lhs, rhs], [output])
    return builder.build(entry="main")


def test_registry_resolves_exact_kind_and_target_without_model_if_chain():
    registry = RuntimePackageRegistry()
    factory = lambda *_args: "adapter"
    spec = registry.register("unit/v1", "test-target", factory)

    assert registry.resolve("unit/v1", "test-target") is spec
    assert registry.create(
        "unit/v1", "test-target", Path("artifact"), {}, object(), object()
    ) == "adapter"


def test_registry_rejects_duplicates_and_reports_available_adapters():
    registry = RuntimePackageRegistry()
    registry.register("unit/v1", "target-a", lambda *_args: object())

    with pytest.raises(ArtifactError, match="already registered"):
        registry.register("unit/v1", "target-a", lambda *_args: object())
    with pytest.raises(ArtifactError, match="unit/v1.*target-a"):
        registry.resolve("missing/v1", "target-b")


def test_artifact_loader_uses_registered_package_adapter_without_loading_device(tmp_path):
    module = Compiler().compile(_make_add_module()).module
    root = write_artifact(
        module,
        tmp_path / "artifact",
        target="nvidia-sm90",
        emit_executable=True,
    )

    runtime = load(root)
    assert isinstance(runtime, GeneratedTirSingleTensorModule)
    assert runtime.ir_module.semantic_hash == module.semantic_hash
