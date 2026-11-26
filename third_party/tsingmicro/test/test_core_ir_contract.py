from __future__ import annotations

import json

import pytest
from triton.backends.tsingmicro.core_ir import (
    ArgumentSpec,
    CompileArtifact,
    CompileOptions,
    CoreIRContractError,
    Grid,
    KernelSignature,
)
from triton.backends.tsingmicro.runtime import CoreIRRuntime, load_core_ir_artifact


def test_contract_rejects_dynamic_and_non_positive_inputs():
    with pytest.raises(CoreIRContractError, match="static shape"):
        ArgumentSpec("x", "float32")
    # Grid no longer requires a single axis (7dc172d18); zero extents stay invalid.
    with pytest.raises(CoreIRContractError, match="positive"):
        Grid(4, 0)
    with pytest.raises(CoreIRContractError, match="between 1 and"):
        CompileOptions(num_stages=0)
    with pytest.raises(CoreIRContractError, match="between 1 and"):
        CompileOptions(num_stages=99)


def test_contract_accepts_pipeline_depths_up_to_max():
    for stages in (1, 2, 3):
        assert CompileOptions(num_stages=stages).num_stages == stages


def test_artifact_manifest_is_stable_and_excludes_binary():
    signature = KernelSignature((ArgumentSpec("x", "float32", shape=(128, ), readonly=True), ))
    artifact = CompileArtifact(
        kernel_name="main",
        signature=signature,
        grid=Grid(1),
        options=CompileOptions(),
        binary=b"not-json",
        kernel_path="/tmp/kernel.so",
        shared_memory_bytes=0,
        compiler_hash="compiler",
        source_hash="source",
    )

    manifest = json.loads(artifact.manifest_json())
    assert "binary" not in manifest
    assert manifest["kernel_name"] == "main"
    assert manifest["options"]["arch"] == "tx81"

    runtime = load_core_ir_artifact(artifact)
    assert isinstance(runtime, CoreIRRuntime)
    assert runtime.artifact is artifact
    assert runtime._launcher is None
    assert runtime._loaded == {}
