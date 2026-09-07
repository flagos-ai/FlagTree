# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT

from __future__ import annotations

import json
from pathlib import Path

import pytest

from triton.flagmega.artifacts import ARTIFACT_SCHEMA, ARTIFACT_VERSION, load_artifact, write_artifact
from triton.flagmega.compiler import Compiler
from triton.flagmega.errors import ArtifactError
from triton.flagmega.artifacts.manifest import record_section, verify_sections

from .helpers import make_add_module


def _write(tmp_path, *, executable: bool = False):
    module = Compiler().compile(make_add_module()).module
    root = write_artifact(
        module,
        tmp_path / "artifact",
        target="nvidia-sm90",
        emit_executable=executable,
    )
    return root, module


def _manifest(root):
    return json.loads((root / "artifact.json").read_text(encoding="utf-8"))


def _replace_manifest(root, manifest):
    (root / "artifact.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def test_v2_manifest_indexes_ir_sections_and_runtime_abi(tmp_path):
    root, module = _write(tmp_path)
    manifest, loaded = load_artifact(root)

    assert manifest["flagmega_artifact_version"] == ARTIFACT_VERSION == 2
    assert manifest["schema"] == ARTIFACT_SCHEMA
    assert {value["name"] for value in manifest["sections"]} == {"ir.python", "ir.text"}
    assert manifest["abi"]["entry"] == "main"
    assert manifest["abi"]["functions"][0]["parameters"][0]["type"]["kind"] == "tensor"
    assert loaded.semantic_hash == module.semantic_hash


def test_manifest_rejects_tampered_section_before_loading_python(tmp_path):
    root, _ = _write(tmp_path)
    (root / "ir" / "final.py").write_text("raise AssertionError('must not execute')\n", encoding="utf-8")

    with pytest.raises(ArtifactError, match="ir.python.*(size|hash)"):
        load_artifact(root)


def test_manifest_rejects_path_escape(tmp_path):
    root, _ = _write(tmp_path)
    manifest = _manifest(root)
    manifest["sections"][0]["path"] = "../outside.py"
    _replace_manifest(root, manifest)

    with pytest.raises(ArtifactError, match="safe relative path"):
        load_artifact(root)


def test_manifest_abi_is_verified_independently_of_section_hashes(tmp_path):
    root, _ = _write(tmp_path)
    manifest = _manifest(root)
    manifest["abi"]["functions"][0]["parameters"][0]["name"] = "forged"
    _replace_manifest(root, manifest)

    with pytest.raises(ArtifactError, match="runtime ABI"):
        load_artifact(root)


def test_executable_codegen_descriptor_must_match_source_section(tmp_path):
    root, _ = _write(tmp_path, executable=True)
    manifest = _manifest(root)
    manifest["codegen"]["source_sha256"] = "0" * 64
    _replace_manifest(root, manifest)

    with pytest.raises(ArtifactError, match="codegen descriptor"):
        load_artifact(root)


def test_existing_v1_artifact_remains_readable_but_writer_is_v2(tmp_path):
    root, module = _write(tmp_path)
    manifest = _manifest(root)
    manifest["flagmega_artifact_version"] = 1
    manifest.pop("schema")
    manifest.pop("sections")
    manifest.pop("abi")
    _replace_manifest(root, manifest)

    legacy, loaded = load_artifact(root)
    assert legacy["flagmega_artifact_version"] == 1
    assert loaded.semantic_hash == module.semantic_hash


def test_section_hashing_streams_instead_of_using_read_bytes(tmp_path, monkeypatch):
    root = tmp_path / "artifact"
    root.mkdir()
    payload = root / "payload.bin"
    payload.write_bytes(b"stream me")

    def reject_unbounded_read_bytes(_path):
        raise AssertionError("section hashing must use bounded streaming reads")

    monkeypatch.setattr(Path, "read_bytes", reject_unbounded_read_bytes)
    section = record_section(root, payload, name="payload", kind="test/v1")
    verify_sections(root, (section,))

    assert section.nbytes == len(b"stream me")


def test_record_section_accepts_a_producer_digest_without_rehashing(tmp_path, monkeypatch):
    root = tmp_path / "artifact"
    root.mkdir()
    payload = root / "payload.bin"
    payload.write_bytes(b"producer-owned digest")

    import triton.flagmega.artifacts.manifest as manifest_module

    monkeypatch.setattr(
        manifest_module,
        "hash_file",
        lambda _path: (_ for _ in ()).throw(AssertionError("must not rehash")),
    )
    section = record_section(
        root,
        payload,
        name="payload",
        kind="test/v1",
        known_nbytes=payload.stat().st_size,
        known_sha256="0" * 64,
    )

    assert section.sha256 == "0" * 64


def test_deferred_section_hash_still_checks_size(tmp_path):
    root = tmp_path / "artifact"
    root.mkdir()
    payload = root / "payload.bin"
    payload.write_bytes(b"original")
    section = record_section(root, payload, name="payload", kind="test/v1")
    payload.write_bytes(b"changed!")

    verify_sections(root, (section,), defer_hashes=frozenset({"payload"}))
    payload.write_bytes(b"wrong size")
    with pytest.raises(ArtifactError, match="size"):
        verify_sections(root, (section,), defer_hashes=frozenset({"payload"}))
