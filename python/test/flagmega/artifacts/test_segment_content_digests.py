# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: MIT

import hashlib
import json
from dataclasses import replace
from threading import Barrier, Lock, get_ident

import pytest
import torch

from triton.flagmega import ir as fm
from triton.flagmega.artifacts import load_artifact, pack_rdata, verify_rdata, write_artifact
from triton.flagmega.artifacts import rdata
from triton.flagmega.artifacts.manifest import record_section, verify_sections
from triton.flagmega.errors import ArtifactError
from triton.flagmega.importer import MemoryCheckpoint, TensorInfo
from triton.flagmega.passes.tir.bufferize.planner import BufferizationOptions
from triton.flagmega.passes.tir.bufferize.policy import NttBufferizationPolicy

from .test_rdata_segment_verification import _pack


class StableCheckpoint(MemoryCheckpoint):
    def cache_identity(self, keys):
        return {"schema": "test.immutable-checkpoint/v1", "keys": sorted(keys), "revision": "fixed"}


def _cached_fixture(*, cyclic=False):
    placement = fm.Placement((4, 2), "cx", "cb")
    policy = fm.SBP.split_block_cyclic((0,), 1) if cyclic else fm.SBP.split_contiguous((0,))
    value_type = fm.DistributedType(fm.tensor_type("float32", (16, 4)),
                                    (policy, fm.SBP.split_contiguous((1,))), placement)
    builder = fm.IRBuilder(dialect="tir", stage="selected_tir")
    weight = builder.weight("weight", value_type, source="memory", key="weight", id="weight")
    builder.function("main", (), (weight,))
    module = NttBufferizationPolicy(replace(BufferizationOptions.generic(alignment=4),
                                           rdata_chip_sharding=True)).bufferize(
        replace(builder.build(entry="main"), metadata={"launch_contract": {"grid_mesh": placement.to_data()}}))
    source = StableCheckpoint({}, {"weight": TensorInfo("weight", fm.DType.FLOAT32, (16, 4), "memory")},
                              {"weight": torch.arange(64, dtype=torch.float32).reshape(16, 4)})
    return module, source


def test_cached_repack_neither_reads_nor_hashes_payload(tmp_path, monkeypatch):
    module, source = _cached_fixture()
    first = pack_rdata(module, source, tmp_path / "first", cache_dir=tmp_path / "cache")

    def forbidden(*args, **kwargs):
        pytest.fail("immutable cache hit accessed weight payload")

    monkeypatch.setattr(source, "load_tensor", forbidden)
    monkeypatch.setattr(rdata, "_pack_rdata_uncached", forbidden)
    monkeypatch.setattr(rdata, "_segment_digests", forbidden)
    second = pack_rdata(module, source, tmp_path / "second", cache_dir=tmp_path / "cache")
    assert second == first
    assert (tmp_path / "second/rdata.bin").read_bytes() == (tmp_path / "first/rdata.bin").read_bytes()


def test_cache_key_distinguishes_equal_sized_shard_layouts(tmp_path):
    contiguous, source = _cached_fixture()
    cyclic, _ = _cached_fixture(cyclic=True)
    assert fm.verify_buffer_plan(contiguous).rdata_bytes == fm.verify_buffer_plan(cyclic).rdata_bytes
    assert rdata._rdata_cache_key(contiguous, source) != rdata._rdata_cache_key(cyclic, source)
    pack_rdata(contiguous, source, tmp_path / "first", cache_dir=tmp_path / "cache")
    pack_rdata(cyclic, source, tmp_path / "second", cache_dir=tmp_path / "cache")
    pack_rdata(cyclic, source, tmp_path / "reference")
    assert (tmp_path / "second/rdata.bin").read_bytes() == (tmp_path / "reference/rdata.bin").read_bytes()
    assert (tmp_path / "first/rdata.bin").read_bytes() != (tmp_path / "reference/rdata.bin").read_bytes()


def test_chip_writers_use_independent_offsets_concurrently(tmp_path, monkeypatch):
    module, source = _cached_fixture()
    original = rdata._MappedChunkWriter.write
    rendezvous = Barrier(4)
    writers = set()
    lock = Lock()

    def write(writer, chunk):
        with lock:
            first = get_ident() not in writers
            writers.add(get_ident())
        if first:
            rendezvous.wait(timeout=5)
        return original(writer, chunk)

    monkeypatch.setattr(rdata.os, "cpu_count", lambda: 4)
    monkeypatch.setattr(rdata._MappedChunkWriter, "write", write)
    pack_rdata(module, source, tmp_path)
    assert len(writers) == 4
    verify_rdata(tmp_path, module=module)


def test_packing_hashes_each_payload_byte_once(tmp_path, monkeypatch):
    placement = fm.Placement((4, 2), "cx", "cb")
    value_type = fm.DistributedType(fm.tensor_type("float32", (128, 128)),
                                    (fm.SBP.split_contiguous((0,)), fm.SBP.split_contiguous((1,))), placement)
    builder = fm.IRBuilder(dialect="tir", stage="selected_tir")
    weight = builder.weight("weight", value_type, source="memory", key="weight", id="weight")
    builder.function("main", (), (weight,))
    module = NttBufferizationPolicy(replace(BufferizationOptions.generic(alignment=4),
                                           rdata_chip_sharding=True)).bufferize(
        replace(builder.build(entry="main"), metadata={"launch_contract": {"grid_mesh": placement.to_data()}}))
    source = MemoryCheckpoint({}, {"weight": TensorInfo("weight", fm.DType.FLOAT32, (128, 128), "memory")},
                              {"weight": torch.arange(128 * 128, dtype=torch.float32).reshape(128, 128)})
    original = hashlib.sha256
    hashed = []

    class Hasher:
        def __init__(self, initial=b""):
            self.inner = original(initial)
            hashed.append(len(initial))

        def update(self, chunk):
            hashed.append(len(chunk))
            self.inner.update(chunk)

        def hexdigest(self):
            return self.inner.hexdigest()

    monkeypatch.setattr(rdata.hashlib, "sha256", Hasher)
    index = pack_rdata(module, source, tmp_path)
    # The small domain/geometry table is the only additional hashed input.
    assert sum(hashed) <= int(index["nbytes"]) + 3 * len(json.dumps(index)) + 1024
    assert all("sha256" not in entry for entry in index["entries"])


@pytest.mark.parametrize("chunk_bytes", [1, 7, 16, 35, 64])
def test_segment_digests_cover_unaligned_windows_once(tmp_path, monkeypatch, chunk_bytes):
    payload = bytes(range(35))
    image = tmp_path / "rdata.bin"
    image.write_bytes(payload)
    monkeypatch.setattr(rdata, "_HASH_CHUNK_BYTES", chunk_bytes)
    assert rdata._segment_digests(image, 7, 5) == [hashlib.sha256(payload[i:i + 7]).hexdigest()
                                               for i in range(0, 35, 7)]


@pytest.mark.parametrize("field", ["digest_kind", "segment_sha256"])
def test_segment_content_identity_is_checked_even_when_payload_check_is_deferred(tmp_path, field):
    module, index = _pack(tmp_path)
    if field == "digest_kind":
        index[field] = "sha256"
    else:
        index[field][0] = "0" * 64
    (tmp_path / "rdata.index.json").write_text(json.dumps(index))
    with pytest.raises(ArtifactError, match="digest"):
        verify_rdata(tmp_path, module=module, verify_image_bytes=False)


@pytest.mark.parametrize("schema", ["flagmega.rdata-index/v1", "flagmega.rdata-index/v2"])
def test_legacy_payload_and_entry_digests_are_still_supported(tmp_path, schema):
    module, index = _pack(tmp_path)
    payload = (tmp_path / "rdata.bin").read_bytes()
    index["schema"] = schema
    index.pop("digest_kind")
    index["sha256"] = hashlib.sha256(payload).hexdigest()
    for entry in index["entries"]:
        start = entry["offset"]
        entry["sha256"] = hashlib.sha256(payload[start:start + entry["nbytes"]]).hexdigest()
    index.pop("segment_chunks", None)
    index["segment_sha256"] = [hashlib.sha256(payload[i:i + index["segment_nbytes"]]).hexdigest()
                               for i in range(0, len(payload), index["segment_nbytes"])]
    if schema.endswith("v1"):
        index.pop("segment_sha256")
    (tmp_path / "rdata.index.json").write_text(json.dumps(index))
    verify_rdata(tmp_path, module=module)
    if schema.endswith("v2"):
        verify_rdata(tmp_path, module=module, segment=0)


def test_segment_digest_worker_failure_propagates(tmp_path, monkeypatch):
    image = tmp_path / "rdata.bin"
    image.write_bytes(b"test")

    def broken(window):
        raise RuntimeError("digest failed")

    monkeypatch.setattr(rdata, "_stream_digest", broken)
    with pytest.raises(RuntimeError, match="digest failed"):
        rdata._segment_digests(image, 2, 2)


def test_generic_manifest_verification_understands_segment_content_identity(tmp_path):
    _, index = _pack(tmp_path)
    image = tmp_path / "rdata.bin"
    section = record_section(tmp_path, image, name="rdata.image", kind="rdata.image/v2",
                             known_sha256=index["sha256"], known_nbytes=index["nbytes"])
    verify_sections(tmp_path, (section,))
    with image.open("r+b") as stream:
        stream.seek(index["nbytes"] - 1)
        stream.write(b"\xff")
    with pytest.raises(ArtifactError, match="hash"):
        verify_sections(tmp_path, (section,))


def test_failed_chip_write_removes_unpublished_image(tmp_path, monkeypatch):
    module, source = _cached_fixture()

    def broken(writer, raw):
        raise OSError("injected write failure")

    monkeypatch.setattr(rdata._MappedChunkWriter, "write", broken)
    with pytest.raises(OSError, match="injected write failure"):
        pack_rdata(module, source, tmp_path)
    assert not tuple(tmp_path.glob(".rdata.bin.*.tmp"))
    assert not (tmp_path / "rdata.bin").exists()


def test_multi_chip_artifact_loader_checks_digest_kind(tmp_path):
    module, source = _cached_fixture()
    module = replace(module, stage="bufferized_tir")
    root = write_artifact(module, tmp_path / "artifact", target="nvidia-sm90", checkpoint=source)
    manifest, loaded = load_artifact(root)
    assert loaded.semantic_hash == module.semantic_hash
    assert manifest["rdata"]["digest_kind"] == rdata.RDATA_CHUNK_DIGEST_KIND
    for section in manifest["sections"]:
        if section["name"] == "rdata.image":
            assert section["kind"] == "rdata.image/v2"
            section["kind"] = "rdata.image/v1"
    (root / "artifact.json").write_text(json.dumps(manifest))
    with pytest.raises(ArtifactError, match="kind"):
        load_artifact(root, verify_rdata_payload=False)
