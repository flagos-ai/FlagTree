# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT
"""The rdata payload is digested once per process, by whoever maps it.

``load_artifact`` and the runtime adapter both used to verify the image, so a
rank paid for two full passes over a multi-hundred-gigabyte payload before the
first kernel ran.  Hashing belongs at the point the bytes are mapped: that
caller knows which chip segment it will read, and every other layer can rely on
the index metadata it already cross-checks.
"""

from __future__ import annotations

import json

import pytest
import torch

import triton.flagmega.artifacts.rdata as rdata_module
from triton.flagmega.artifacts import load_artifact, write_artifact
from triton.flagmega.compiler import Compiler
from triton.flagmega.errors import ArtifactError
from triton.flagmega.options import CompileOptions
from triton.flagmega.runtime import load as load_runtime

from ..passes.functions.constant_parameters.helpers import module as weight_module


class _Checkpoint:
    """Two offline weights, enough to make the packer emit a real image."""

    def __init__(self) -> None:
        self._tensors = {
            f"weight{index}": (torch.arange(8, dtype=torch.float32) * 0.017 + index).bfloat16()
            for index in range(2)
        }

    def load_tensor(self, key, *, device="cpu"):
        return self._tensors[key].to(device)


@pytest.fixture(scope="module")
def artifact(tmp_path_factory):
    module = Compiler(CompileOptions(bufferize_opt_level="fast")).compile(
        weight_module(distributed=True)
    ).module
    root = write_artifact(
        module,
        tmp_path_factory.mktemp("rdata-once") / "artifact",
        checkpoint=_Checkpoint(),
        target="nvidia-sm90",
        emit_executable=True,
    )
    return root


def _count_payload_digests(monkeypatch):
    """Record every pass the process makes over the packed payload."""

    passes = []
    original = rdata_module._verify_image_hashes
    original_stream = rdata_module._stream_digest

    def observe(image, expected_image_hash, ranges):
        passes.append(getattr(image, "name", str(image)))
        return original(image, expected_image_hash, ranges)

    monkeypatch.setattr(rdata_module, "_verify_image_hashes", observe)

    def observe_stream(image):
        passes.append(image.name)
        return original_stream(image)

    monkeypatch.setattr(rdata_module, "_stream_digest", observe_stream)
    return passes


def test_loading_an_artifact_can_defer_the_payload_digest(artifact, monkeypatch):
    passes = _count_payload_digests(monkeypatch)

    load_artifact(artifact, verify_rdata_payload=False)

    assert passes == []


def test_auditing_an_artifact_still_digests_the_payload(artifact, monkeypatch):
    # ``artifact verify`` exists to check the artifact itself, so deferral has to
    # be the loading caller's explicit choice rather than a new default.
    passes = _count_payload_digests(monkeypatch)

    load_artifact(artifact)

    assert passes == ["rdata.bin"]


def test_running_an_artifact_digests_the_payload_exactly_once(artifact, monkeypatch):
    passes = _count_payload_digests(monkeypatch)

    runtime = load_runtime(artifact)
    pool = next(
        value for value in runtime.runtime_binding["pools"]
        if str(value["storage"]) == "rdata"
    )
    pooled = runtime._load_rdata(torch, "cpu", int(pool["nbytes"]))

    assert pooled.numel() == int(pool["nbytes"])
    assert len(passes) == 1


def test_a_corrupt_payload_is_still_rejected_when_the_pool_is_mapped(artifact, monkeypatch):
    image = artifact / "assets" / "rdata.bin"
    original = image.read_bytes()
    image.write_bytes(bytes([original[0] ^ 0xFF]) + original[1:])
    try:
        runtime = load_runtime(artifact)
        pool = next(
            value for value in runtime.runtime_binding["pools"]
            if str(value["storage"]) == "rdata"
        )
        with pytest.raises(ArtifactError, match="hash does not match"):
            runtime._load_rdata(torch, "cpu", int(pool["nbytes"]))
    finally:
        image.write_bytes(original)


def test_deferring_the_payload_digest_still_checks_the_index(artifact):
    assets = artifact / "assets"
    index_path = assets / "rdata.index.json"
    original = index_path.read_text(encoding="utf-8")
    index = json.loads(original)
    index["entries"][0]["offset"] = int(index["segment_nbytes"]) + 1
    index_path.write_text(json.dumps(index, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    try:
        with pytest.raises(ArtifactError, match="out of bounds"):
            rdata_module.verify_rdata(assets, verify_image_bytes=False)
    finally:
        index_path.write_text(original, encoding="utf-8")


def test_an_index_without_segment_digests_is_still_runnable(artifact, monkeypatch):
    # Images packed before per-segment digests existed cannot be narrowed, so the
    # runtime verifies them whole rather than refusing to run them.
    runtime = load_runtime(artifact)
    pool = next(
        value for value in runtime.runtime_binding["pools"]
        if str(value["storage"]) == "rdata"
    )
    # Downgrade the published index after loading, so the manifest's own section
    # digests stay intact and only the rdata scope decision is under test.
    index_path = artifact / "assets" / "rdata.index.json"
    original = index_path.read_text(encoding="utf-8")
    index = json.loads(original)
    import hashlib
    payload = (artifact / "assets" / "rdata.bin").read_bytes()
    index["sha256"] = hashlib.sha256(payload).hexdigest()
    index.pop("digest_kind")
    for entry in index["entries"]:
        start = entry["offset"]
        entry["sha256"] = hashlib.sha256(payload[start:start + entry["nbytes"]]).hexdigest()
    del index["segment_sha256"]
    index["schema"] = "flagmega.rdata-index/v1"
    index_path.write_text(json.dumps(index, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    try:
        passes = _count_payload_digests(monkeypatch)
        pooled = runtime._load_rdata(torch, "cpu", int(pool["nbytes"]))

        assert pooled.numel() == int(pool["nbytes"])
        assert len(passes) == 1
    finally:
        index_path.write_text(original, encoding="utf-8")


def test_a_deferred_digest_cannot_also_be_narrowed_to_a_segment(artifact):
    # The two modes answer different questions: one defers the payload check to
    # a later caller, the other *is* that caller's check.  Combining them would
    # silently skip the payload entirely.
    with pytest.raises(ArtifactError, match="cannot both"):
        rdata_module.verify_rdata(
            artifact / "assets", segment=0, verify_image_bytes=False
        )
