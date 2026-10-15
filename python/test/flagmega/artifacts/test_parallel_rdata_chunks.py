# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: MIT

import json
from dataclasses import replace
from threading import Barrier

import pytest

from triton.flagmega.artifacts import pack_rdata, verify_rdata
from triton.flagmega.artifacts import rdata
from triton.flagmega.errors import ArtifactError
from triton.flagmega import ir as fm
from triton.flagmega.importer import MemoryCheckpoint, TensorInfo
from triton.flagmega.passes.constants import freeze_constant_islands
from triton.flagmega.passes.tir.bufferize.planner import BufferizationOptions
from triton.flagmega.passes.tir.bufferize.policy import NttBufferizationPolicy

from .test_segment_content_digests import _cached_fixture
from .helpers import make_add_module


def test_chip_pack_hashes_written_chunks_without_rescanning(tmp_path, monkeypatch):
    module, checkpoint = _cached_fixture()

    def forbidden(*args, **kwargs):
        pytest.fail("packing reread the completed payload to calculate its digest")

    monkeypatch.setattr(rdata, "_segment_digests", forbidden)
    index = pack_rdata(module, checkpoint, tmp_path)
    assert index["digest_kind"] == "sha256-chunks/v1"
    assert len(index["segment_chunks"]) == 4
    verify_rdata(tmp_path, module=module)


@pytest.mark.parametrize("tamper", ["gap", "digest", "order"])
def test_chunk_table_is_validated_before_deferred_payload_check(tmp_path, tamper):
    module, checkpoint = _cached_fixture()
    index = pack_rdata(module, checkpoint, tmp_path)
    chunks = index["segment_chunks"][0]
    if tamper == "gap":
        chunks[0]["offset"] += 1
    elif tamper == "digest":
        chunks[0]["sha256"] = "0" * 64
    else:
        chunks.reverse()
    (tmp_path / "rdata.index.json").write_text(json.dumps(index))
    with pytest.raises(ArtifactError, match="chunk|digest"):
        verify_rdata(tmp_path, module=module, verify_image_bytes=False)


def test_independent_recipes_materialize_concurrently(tmp_path, monkeypatch):
    import numpy as np
    placement = fm.Placement((2, 2), "cx", "cb")
    value_type = fm.DistributedType(fm.tensor_type("float32", (8, 4)),
                                    (fm.SBP.split_contiguous((0,)), fm.SBP.split_contiguous((1,))), placement)
    builder = fm.IRBuilder(dialect="tir", stage="selected_tir",
                           metadata={"launch_contract": {"grid_mesh": placement.to_data()}})
    weights = tuple(builder.weight(key, value_type, source="memory", key=key, id=key)
                    for key in ("w0", "w1"))
    builder.function("main", (), weights)
    frozen = freeze_constant_islands(builder.build(entry="main"))
    module = NttBufferizationPolicy(replace(BufferizationOptions.generic(alignment=4),
                                           rdata_chip_sharding=True)).bufferize(frozen)
    assert len(module.constant_recipes) == 2
    rendezvous = Barrier(2)

    class ConcurrentCheckpoint(MemoryCheckpoint):
        def load_tensor(self, key, *, device="cpu"):
            rendezvous.wait(timeout=5)
            return super().load_tensor(key, device=device)

    values = {key:np.arange(32, dtype=np.float32).reshape(8, 4) + i * 100
              for i, key in enumerate(("w0", "w1"))}
    checkpoint = ConcurrentCheckpoint({}, {key:TensorInfo(key, fm.DType.FLOAT32, (8, 4), "memory")
                                            for key in values}, values)
    monkeypatch.setattr(rdata.os, "cpu_count", lambda: 4)
    index = pack_rdata(module, checkpoint, tmp_path)
    image = (tmp_path / "rdata.bin").read_bytes()
    for buffer in fm.verify_buffer_plan(module).buffers:
        if buffer.storage != "rdata":
            continue
        for chip in range(2):
            for block in range(2):
                offset = chip * index["segment_nbytes"] + buffer.offset + block * buffer.component_stride_bytes
                actual = np.frombuffer(image, dtype=np.float32, count=8, offset=offset).reshape(4, 2)
                np.testing.assert_array_equal(actual, values[buffer.weight_key][chip * 4:(chip + 1) * 4,
                                                                              block * 2:(block + 1) * 2])
    verify_rdata(tmp_path, module=module)


def test_manifest_metadata_does_not_serialize_the_entire_graph(tmp_path, monkeypatch):
    from triton.flagmega.artifacts import write_artifact
    module = make_add_module()

    def forbidden(*args, **kwargs):
        pytest.fail("manifest serialized the whole graph just to read module metadata")

    monkeypatch.setattr(fm.IRModule, "to_data", forbidden)
    root = write_artifact(module, tmp_path, target="test")
    assert json.loads((root / "artifact.json").read_text())["metadata"] == dict(module.metadata)


@pytest.mark.parametrize("shared_extents", [True, False])
def test_extent_assembly_preserves_cross_extent_writes_and_sparse_padding(tmp_path, monkeypatch, shared_extents):
    if not shared_extents:
        import errno
        def unsupported(*args, **kwargs):
            raise OSError(errno.EOPNOTSUPP, "filesystem has no shared extents")
        monkeypatch.setattr(rdata.fcntl, "ioctl", unsupported)
    path = tmp_path / "image.bin"
    expected = bytearray(10003)
    expected[4090:4103] = b"cross-boundary"
    expected[8999:9007] = b"last-one"
    with path.open("w+b") as stream:
        stream.truncate(len(expected))
        with rdata._StripedImage(stream, len(expected)) as image:
            image.part_nbytes = 4096
            image.write_at(4090, memoryview(b"cross-boundary"))
            image.write_at(8999, memoryview(b"last-one"))
            image.assemble()
    assert path.read_bytes() == expected
    assert not tuple(tmp_path.glob(".rdata-extents-*"))


def test_v3_segment_digest_artifacts_remain_readable(tmp_path):
    import hashlib
    module, checkpoint = _cached_fixture()
    index = pack_rdata(module, checkpoint, tmp_path)
    payload = (tmp_path / "rdata.bin").read_bytes()
    size = index["segment_nbytes"]
    index["schema"] = rdata.RDATA_INDEX_SCHEMA_V3
    index["digest_kind"] = rdata.RDATA_SEGMENT_DIGEST_KIND
    index.pop("segment_chunks")
    index["segment_sha256"] = [hashlib.sha256(payload[i:i + size]).hexdigest()
                               for i in range(0, len(payload), size)]
    index["sha256"] = rdata._segment_content_digest(index)
    (tmp_path / "rdata.index.json").write_text(json.dumps(index))
    verify_rdata(tmp_path, module=module)
    verify_rdata(tmp_path, module=module, segment=3)


def test_recipe_budget_counts_tuple_intermediates():
    value_type = fm.tensor_type("float32", (4,))
    value = fm.Node("value", "builtin.splat_const", (), value_type, attrs={"value": 1.0})
    pair = fm.Node("pair", "builtin.tuple", ("value", "value"), fm.TupleType((value_type, value_type)))
    recipe = fm.ConstantRecipe("tuple-budget", (value, pair), ("pair",))
    assert rdata._recipe_storage_budget(recipe, [], 1) == 2 * (16 + 32)
