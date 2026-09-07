# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT

import json
import struct

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
