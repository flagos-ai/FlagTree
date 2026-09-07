# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
import torch

import triton.flagmega.artifacts.rdata as rdata_module
import triton.flagmega.artifacts.manifest as manifest_module
from triton.flagmega.artifacts import load_artifact, pack_rdata, verify_rdata, write_artifact
from triton.flagmega.compiler import Compiler
from triton.flagmega.errors import ArtifactError
from triton.flagmega.importer import Qwen3LayerImporter, TensorByteRange

from ..qwen3.helpers import checkpoint


def _write_rdata_artifact(tmp_path):
    source = checkpoint()
    module = Compiler().compile(
        Qwen3LayerImporter(source, block_size=4, num_blocks=2).import_module()
    ).module
    root = write_artifact(module, tmp_path / "artifact", target="nvidia-sm90", checkpoint=source)
    return root, module


def test_rdata_index_is_checked_against_buffer_plan_after_its_own_hashes_pass(tmp_path):
    root, _ = _write_rdata_artifact(tmp_path)
    index_path = root / "assets" / "rdata.index.json"
    index = json.loads(index_path.read_text(encoding="utf-8"))
    index["entries"][0]["shape"] = [999]
    index_path.write_text(json.dumps(index, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    manifest_path = root / "artifact.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    payload = index_path.read_bytes()
    section = next(value for value in manifest["sections"] if value["name"] == "rdata.index")
    section["nbytes"] = len(payload)
    section["sha256"] = hashlib.sha256(payload).hexdigest()
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    with pytest.raises(ArtifactError, match="shape.*buffer plan"):
        load_artifact(root)


def test_rdata_rejects_duplicate_buffer_identity_before_overlap_check(tmp_path):
    root, _ = _write_rdata_artifact(tmp_path)
    index_path = root / "assets" / "rdata.index.json"
    index = json.loads(index_path.read_text(encoding="utf-8"))
    duplicate = dict(index["entries"][0])
    duplicate["key"] = duplicate["key"] + ".duplicate"
    index["entries"].append(duplicate)
    index_path.write_text(json.dumps(index, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    with pytest.raises(ArtifactError, match="buffer id.*duplicated"):
        verify_rdata(root / "assets")


def test_rdata_verification_streams_the_image_instead_of_reading_it_all(tmp_path, monkeypatch):
    root, module = _write_rdata_artifact(tmp_path)

    def reject_unbounded_read_bytes(_path):
        raise AssertionError("rdata verification must use bounded streaming reads")

    monkeypatch.setattr(Path, "read_bytes", reject_unbounded_read_bytes)
    index = verify_rdata(root / "assets", module=module)

    assert index["nbytes"] > 0


def test_rdata_packing_streams_recipes_with_an_uncached_weight_resolver(
    tmp_path, monkeypatch
):
    observed = []
    original = rdata_module.iter_numpy_materialized_constant_assets

    def observe(module, source, *, outputs=None):
        observed.append((source, frozenset(outputs or ())))
        yield from original(module, source, outputs=outputs)

    monkeypatch.setattr(
        rdata_module, "iter_numpy_materialized_constant_assets", observe
    )
    root, _ = _write_rdata_artifact(tmp_path)

    assert root.is_dir()
    assert observed and observed[0][0] is not None
    assert observed[0][1]


def test_rdata_packing_streams_byte_preserving_recipe_from_checkpoint_storage(
    tmp_path,
):
    source = checkpoint()
    module = Compiler().compile(
        Qwen3LayerImporter(source, block_size=4, num_blocks=2).import_module()
    ).module
    source_key = "model.embed_tokens.weight"
    tensor = source.load_tensor(source_key)
    raw = tensor.view(torch.uint8).numpy().tobytes()
    prefix = b"checked-prefix"
    backing = tmp_path / "checkpoint.storage"
    backing.write_bytes(prefix + raw + b"checked-suffix")
    loaded = []

    class StorageCheckpoint:
        @property
        def config(self):
            return source.config

        @property
        def keys(self):
            return source.keys

        def tensor_info(self, key):
            return source.tensor_info(key)

        def tensor_byte_range(self, key):
            if key == source_key:
                return TensorByteRange(backing, len(prefix), len(raw))
            return None

        def load_tensor(self, key, *, device="cpu"):
            loaded.append(key)
            return source.load_tensor(key, device=device)

    index = pack_rdata(module, StorageCheckpoint(), tmp_path / "assets")
    entry = next(
        value for value in index["entries"]
        if value["key"] == "w_embed_tokens_weight.reshard2"
    )
    with (tmp_path / "assets" / "rdata.bin").open("rb") as stream:
        stream.seek(entry["offset"])
        packed = stream.read(entry["nbytes"])

    assert packed == raw
    assert entry["sha256"] == hashlib.sha256(raw).hexdigest()
    assert source_key not in loaded


def test_canonical_rdata_plan_hashes_once_while_writing(tmp_path, monkeypatch):
    source = checkpoint()
    module = Compiler().compile(
        Qwen3LayerImporter(source, block_size=4, num_blocks=2).import_module()
    ).module

    def reject_second_image_scan(_path):
        raise AssertionError("canonical rdata packing must hash during its write pass")

    monkeypatch.setattr(rdata_module, "hash_file", reject_second_image_scan)
    index = pack_rdata(module, source, tmp_path / "assets")

    assert index["nbytes"] > 0
    assert len(index["sha256"]) == 64


def test_stable_checkpoint_rdata_cache_packs_once_and_clones_independent_images(
    tmp_path, monkeypatch
):
    source = checkpoint()
    module = Compiler().compile(
        Qwen3LayerImporter(source, block_size=4, num_blocks=2).import_module()
    ).module

    class StableCheckpoint:
        @property
        def config(self):
            return source.config

        @property
        def keys(self):
            return source.keys

        def tensor_info(self, key):
            return source.tensor_info(key)

        def tensor_byte_range(self, key):
            return None

        def load_tensor(self, key, *, device="cpu"):
            return source.load_tensor(key, device=device)

        def cache_identity(self, keys):
            return {
                "schema": "test.checkpoint/v1",
                "keys": sorted(keys),
                "revision": "immutable-1",
            }

    original = rdata_module._pack_rdata_uncached
    misses = []

    def observe(*args, **kwargs):
        misses.append(1)
        return original(*args, **kwargs)

    monkeypatch.setattr(rdata_module, "_pack_rdata_uncached", observe)
    cache = tmp_path / "cache"
    first_root = tmp_path / "first"
    second_root = tmp_path / "second"
    first = pack_rdata(module, StableCheckpoint(), first_root, cache_dir=cache)
    second = pack_rdata(module, StableCheckpoint(), second_root, cache_dir=cache)

    assert first == second
    assert misses == [1]
    assert len(tuple(cache.iterdir())) == 1
    cache_image = next(cache.iterdir()) / "rdata.bin"
    first_image = first_root / "rdata.bin"
    second_image = second_root / "rdata.bin"
    assert first_image.stat().st_ino != cache_image.stat().st_ino
    assert second_image.stat().st_ino != cache_image.stat().st_ino
    original_prefix = cache_image.read_bytes()[:16]
    with second_image.open("r+b") as stream:
        stream.write(b"changed-artifact")
    assert cache_image.read_bytes()[:16] == original_prefix


def test_rdata_cache_key_changes_with_checkpoint_identity(tmp_path):
    source = checkpoint()
    module = Compiler().compile(
        Qwen3LayerImporter(source, block_size=4, num_blocks=2).import_module()
    ).module

    class IdentifiedCheckpoint:
        def __init__(self, revision):
            self.revision = revision

        def cache_identity(self, keys):
            return {"revision": self.revision, "keys": sorted(keys)}

    first = rdata_module._rdata_cache_key(
        module, IdentifiedCheckpoint("revision-a")
    )
    second = rdata_module._rdata_cache_key(
        module, IdentifiedCheckpoint("revision-b")
    )

    assert first is not None
    assert second is not None
    assert first != second


def test_tensor_storage_buffer_is_a_zero_copy_byte_view():
    tensor = torch.arange(12, dtype=torch.float32).reshape(3, 4)

    raw = rdata_module._tensor_buffer(tensor)

    assert isinstance(raw, memoryview)
    assert raw.nbytes == tensor.numel() * tensor.element_size()
    assert raw.tobytes() == tensor.numpy().tobytes()


def test_failed_rdata_pack_removes_its_temporary_sparse_image(tmp_path, monkeypatch):
    source = checkpoint()
    module = Compiler().compile(
        Qwen3LayerImporter(source, block_size=4, num_blocks=2).import_module()
    ).module

    def fail_materialization(*_args, **_kwargs):
        raise RuntimeError("injected recipe failure")
        yield  # pragma: no cover - preserve generator call semantics

    monkeypatch.setattr(
        rdata_module, "iter_numpy_materialized_constant_assets", fail_materialization
    )
    assets = tmp_path / "assets"

    with pytest.raises(RuntimeError, match="injected recipe failure"):
        pack_rdata(module, source, assets)

    assert not tuple(assets.glob(".rdata.bin.*.tmp"))
    assert not (assets / "rdata.bin").exists()


def test_artifact_load_hashes_rdata_image_once_in_the_rdata_verifier(
    tmp_path, monkeypatch
):
    root, module = _write_rdata_artifact(tmp_path)
    original_hash_file = manifest_module.hash_file
    original_verify_image_hashes = rdata_module._verify_image_hashes
    verified_images = []

    def observe_manifest_hash(path):
        assert Path(path).name != "rdata.bin"
        return original_hash_file(path)

    def observe_rdata_hash(image, expected_image_hash, ranges):
        verified_images.append(image)
        return original_verify_image_hashes(image, expected_image_hash, ranges)

    monkeypatch.setattr(manifest_module, "hash_file", observe_manifest_hash)
    monkeypatch.setattr(rdata_module, "_verify_image_hashes", observe_rdata_hash)

    _, loaded = load_artifact(root)

    assert loaded.semantic_hash == module.semantic_hash
    assert [path.name for path in verified_images] == ["rdata.bin"]


def test_deferred_rdata_section_digest_is_cross_checked_with_verified_index(tmp_path):
    root, _ = _write_rdata_artifact(tmp_path)
    manifest_path = root / "artifact.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    image = next(value for value in manifest["sections"] if value["name"] == "rdata.image")
    image["sha256"] = "0" * 64
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    with pytest.raises(ArtifactError, match="image section.*verified index"):
        load_artifact(root)
