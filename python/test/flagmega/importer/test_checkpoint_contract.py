# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT

import json
import struct
import hashlib
import os
import time
from pathlib import Path

import pytest

from triton.flagmega.errors import ImporterError
from triton.flagmega.importer import DirectoryCheckpoint, TensorByteRange, TensorInfo
from triton.flagmega.ir import DType


def _config(directory):
    (directory / "config.json").write_text("{}", encoding="utf-8")


def _write_shard(path, header, data_bytes: int | bytes):
    payload = json.dumps(header, separators=(",", ":")).encode("utf-8")
    data = bytes(data_bytes) if isinstance(data_bytes, int) else data_bytes
    path.write_bytes(struct.pack("<Q", len(payload)) + payload + data)


def test_verified_shard_ranges_do_not_repeat_path_canonicalization(tmp_path, monkeypatch):
    _config(tmp_path)
    source = tmp_path / "model.safetensors"
    _write_shard(source, {
        "first": {"dtype": "F32", "shape": [1], "data_offsets": [0, 4]},
        "second": {"dtype": "F32", "shape": [1], "data_offsets": [4, 8]},
    }, b"abcdefgh")
    checkpoint = DirectoryCheckpoint(tmp_path)
    checkpoint.tensor_info("first")
    calls = []
    original = type(source).resolve

    def measure(path, *args, **kwargs):
        calls.append(path)
        return original(path, *args, **kwargs)

    monkeypatch.setattr(type(source), "resolve", measure)
    first = checkpoint.tensor_byte_range("first")
    second = checkpoint.tensor_byte_range("second")
    assert first.path == second.path == source
    assert first.nbytes == second.nbytes == 4
    assert second.offset == first.offset + 4
    assert calls == []


def test_verified_source_report_reuses_digest_and_invalidates_on_ctime(tmp_path, monkeypatch):
    _config(tmp_path)
    source = tmp_path / "model.safetensors"
    _write_shard(source, {"weight": {"dtype": "F32", "shape": [1], "data_offsets": [0, 4]}}, b"abcd")
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    report = tmp_path / "audit.json"
    report.write_text(json.dumps({"checkpoint": str(tmp_path), "ok": True, "shards": [
        {"file": source.name, "size": source.stat().st_size, "sha256": digest, "ok": True}
    ]}))
    checkpoint = DirectoryCheckpoint(tmp_path, integrity_report=report)
    original = hashlib.sha256
    def forbidden(*args, **kwargs):
        pytest.fail("unchanged verified source was hashed again")
    monkeypatch.setattr(hashlib, "sha256", forbidden)
    assert checkpoint.source_file_digest(source) == digest
    monkeypatch.setattr(hashlib, "sha256", original)
    before = source.stat()
    # Linux filesystem timestamps can share one coarse clock tick.
    time.sleep(0.02)
    with source.open("r+b") as stream:
        stream.seek(-1, 2)
        stream.write(b"z")
    os.utime(source, ns=(before.st_atime_ns, before.st_mtime_ns))
    assert checkpoint.source_file_digest(source) != digest
    with pytest.raises(ImporterError, match="stale"):
        DirectoryCheckpoint(tmp_path, integrity_report=report)


def test_index_rejects_absolute_and_parent_traversal_shards(tmp_path):
    _config(tmp_path)
    index = tmp_path / "model.safetensors.index.json"
    for source in ("../outside.safetensors", "/tmp/outside.safetensors"):
        index.write_text(
            json.dumps({"weight_map": {"weight": source}}),
            encoding="utf-8",
        )
        with pytest.raises(ImporterError, match="safe relative"):
            DirectoryCheckpoint(tmp_path)


def test_generic_checkpoint_rejects_symlink_that_resolves_outside_root(tmp_path):
    root = tmp_path / "checkpoint"
    outside = tmp_path / "outside"
    root.mkdir()
    outside.mkdir()
    _config(root)
    _write_shard(
        outside / "blob.safetensors",
        {"weight": {"dtype": "F32", "shape": [1], "data_offsets": [0, 4]}},
        4,
    )
    (root / "model.safetensors").symlink_to(outside / "blob.safetensors")
    (root / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {"weight": "model.safetensors"}}),
        encoding="utf-8",
    )

    with pytest.raises(ImporterError, match="escapes root"):
        DirectoryCheckpoint(root)


def test_huggingface_snapshot_accepts_same_model_blob_symlink(tmp_path):
    model_root = tmp_path / "models--Qwen--Fixture"
    snapshot = model_root / "snapshots" / "revision"
    blobs = model_root / "blobs"
    snapshot.mkdir(parents=True)
    blobs.mkdir()
    _config(snapshot)
    blob = blobs / "0123456789abcdef"
    _write_shard(
        blob,
        {"weight": {"dtype": "F32", "shape": [1], "data_offsets": [0, 4]}},
        4,
    )
    (snapshot / "model.safetensors").symlink_to(blob)
    (snapshot / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {"weight": "model.safetensors"}}),
        encoding="utf-8",
    )

    checkpoint = DirectoryCheckpoint(snapshot)

    assert checkpoint.tensor_info("weight") == TensorInfo(
        "weight",
        DType.FLOAT32,
        (1,),
        "model.safetensors",
        (0, 4),
    )


def test_header_rejects_shape_byte_count_mismatch(tmp_path):
    _config(tmp_path)
    _write_shard(
        tmp_path / "model.safetensors",
        {"weight": {"dtype": "BF16", "shape": [4], "data_offsets": [0, 4]}},
        4,
    )

    with pytest.raises(ImporterError, match="byte range.*expected 8"):
        DirectoryCheckpoint(tmp_path)


def test_header_rejects_overlapping_tensor_ranges(tmp_path):
    _config(tmp_path)
    _write_shard(
        tmp_path / "model.safetensors",
        {
            "lhs": {"dtype": "BF16", "shape": [4], "data_offsets": [0, 8]},
            "rhs": {"dtype": "BF16", "shape": [4], "data_offsets": [4, 12]},
        },
        12,
    )

    with pytest.raises(ImporterError, match="lhs.*rhs.*overlap"):
        DirectoryCheckpoint(tmp_path)


def test_valid_header_exposes_lazy_tensor_metadata(tmp_path):
    _config(tmp_path)
    _write_shard(
        tmp_path / "model.safetensors",
        {"weight": {"dtype": "F32", "shape": [2, 3], "data_offsets": [0, 24]}},
        24,
    )

    checkpoint = DirectoryCheckpoint(tmp_path)

    assert checkpoint.tensor_info("weight") == TensorInfo(
        "weight",
        DType.FLOAT32,
        (2, 3),
        "model.safetensors",
        (0, 24),
    )


def test_directory_checkpoint_exposes_verified_tensor_byte_range(tmp_path):
    _config(tmp_path)
    raw = bytes(range(24))
    shard = tmp_path / "model.safetensors"
    _write_shard(
        shard,
        {"weight": {"dtype": "F32", "shape": [2, 3], "data_offsets": [0, 24]}},
        raw,
    )

    checkpoint = DirectoryCheckpoint(tmp_path)
    byte_range = checkpoint.tensor_byte_range("weight")

    assert byte_range == TensorByteRange(shard.resolve(), shard.stat().st_size - len(raw), len(raw))
    with byte_range.path.open("rb") as stream:
        stream.seek(byte_range.offset)
        assert stream.read(byte_range.nbytes) == raw


def test_verified_byte_ranges_reuse_resolved_shard_metadata(tmp_path, monkeypatch):
    _config(tmp_path)
    _write_shard(tmp_path / "model.safetensors",
                 {"weight": {"dtype": "F32", "shape": [2], "data_offsets": [0, 8]}}, 8)
    checkpoint = DirectoryCheckpoint(tmp_path)
    first = checkpoint.tensor_byte_range("weight")

    def forbidden(*args, **kwargs):
        pytest.fail("an already-verified tensor range resolved its shard again")

    monkeypatch.setattr(checkpoint, "_resolve_shard", forbidden)
    assert checkpoint.tensor_byte_range("weight") is first


def test_cache_identity_stats_each_backing_shard_once(tmp_path, monkeypatch):
    _config(tmp_path)
    shard = tmp_path / "model.safetensors"
    _write_shard(shard, {f"w{i}": {"dtype": "F32", "shape": [2], "data_offsets": [i * 8, (i + 1) * 8]}
                         for i in range(64)}, 64 * 8)
    checkpoint = DirectoryCheckpoint(tmp_path)
    original = Path.stat
    calls = []

    def observe(path, *args, **kwargs):
        if path == shard:
            calls.append(path)
        return original(path, *args, **kwargs)

    monkeypatch.setattr(Path, "stat", observe)
    identity = checkpoint.cache_identity(checkpoint.keys)
    assert len(identity["tensors"]) == 64
    assert len(identity["sources"]) == 1
    assert calls == [shard]


def test_tensor_info_rejects_negative_shape_and_offsets():
    with pytest.raises(ImporterError, match="invalid shape"):
        TensorInfo("weight", DType.FLOAT32, (-1,), "model.safetensors")
    with pytest.raises(ImporterError, match="invalid data offsets"):
        TensorInfo(
            "weight",
            DType.FLOAT32,
            (1,),
            "model.safetensors",
            (4, 0),
        )
