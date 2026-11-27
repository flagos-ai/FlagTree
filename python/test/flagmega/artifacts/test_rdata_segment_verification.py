# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Per-segment rdata verification for chip-sharded images.

A chip-sharded image concatenates one segment per launch chip, and a rank only
ever maps its own segment.  Verification must therefore be scopable to that
segment: hashing the whole image on every rank makes startup cost scale with
``chip_count**2`` while proving nothing about bytes the rank never reads.
"""

from dataclasses import replace

import pytest
import torch

from triton.flagmega import ir as fm
from triton.flagmega.artifacts import pack_rdata, verify_rdata
from triton.flagmega.artifacts import rdata as rdata_module
from triton.flagmega.errors import ArtifactError
from triton.flagmega.importer import MemoryCheckpoint, TensorInfo
from triton.flagmega.passes.tir.bufferize.planner import BufferizationOptions
from triton.flagmega.passes.tir.bufferize.policy import NttBufferizationPolicy


CHIP_COUNT = 4


def _chip_sharded_module():
    """Build a 4-chip image whose weight is split along the chip axis."""

    placement = fm.Placement((CHIP_COUNT, 2), "cx", "cb")
    distributed = fm.DistributedType(
        fm.tensor_type("float32", (16, 4)),
        (fm.SBP.split_contiguous((0,)), fm.SBP.split_contiguous((1,))),
        placement,
    )
    builder = fm.IRBuilder(dialect="tir", stage="selected_tir")
    weight = builder.weight("weight", distributed, source="memory", key="weight", id="weight")
    builder.function("main", (), (weight,))
    module = replace(
        builder.build(entry="main"),
        metadata={"launch_contract": {"grid_mesh": placement.to_data()}},
    )
    return NttBufferizationPolicy(
        replace(BufferizationOptions.generic(alignment=4), rdata_chip_sharding=True)
    ).bufferize(module)


def _pack(tmp_path):
    module = _chip_sharded_module()
    source = torch.arange(64, dtype=torch.float32).reshape(16, 4)
    checkpoint = MemoryCheckpoint(
        {},
        {"weight": TensorInfo("weight", fm.DType.FLOAT32, (16, 4), "memory")},
        {"weight": source},
    )
    index = pack_rdata(module, checkpoint, tmp_path)
    assert index["chip_count"] == CHIP_COUNT
    return module, index


def _read_bytes_during(monkeypatch, tmp_path, module, **kwargs):
    """Verify while counting bytes actually pulled off the image."""

    total = {"value": 0}
    real_open = rdata_module.Path.open

    def counting_open(self, mode="r", *args, **rest):
        stream = real_open(self, mode, *args, **rest)
        if self.name != "rdata.bin":
            return stream
        real_read = stream.read

        def read(size=-1):
            chunk = real_read(size)
            total["value"] += len(chunk)
            return chunk

        stream.read = read
        return stream

    monkeypatch.setattr(rdata_module.Path, "open", counting_open)
    verify_rdata(tmp_path, module=module, **kwargs)
    return total["value"]


def test_verifying_one_segment_reads_only_that_segment(monkeypatch, tmp_path):
    module, index = _pack(tmp_path)
    segment_nbytes = int(index["segment_nbytes"])

    scoped = _read_bytes_during(monkeypatch, tmp_path, module, segment=2)

    # Reading a second segment's bytes would make every rank pay for the
    # whole model instead of its own shard.
    assert scoped <= segment_nbytes


def test_verifying_the_whole_image_still_reads_every_segment(monkeypatch, tmp_path):
    module, index = _pack(tmp_path)

    whole = _read_bytes_during(monkeypatch, tmp_path, module)

    assert whole == int(index["nbytes"]) == int(index["segment_nbytes"]) * CHIP_COUNT


@pytest.mark.parametrize("corrupted", range(CHIP_COUNT))
def test_a_corrupt_segment_is_caught_when_that_segment_is_verified(tmp_path, corrupted):
    module, index = _pack(tmp_path)
    segment_nbytes = int(index["segment_nbytes"])
    image = tmp_path / "rdata.bin"
    payload = bytearray(image.read_bytes())
    payload[corrupted * segment_nbytes] ^= 0xFF
    image.write_bytes(bytes(payload))

    # Narrowing the scope must not weaken detection inside that scope.
    with pytest.raises(ArtifactError):
        verify_rdata(tmp_path, module=module, segment=corrupted)
    # Nor may it silently pass when the caller asked for everything.
    with pytest.raises(ArtifactError):
        verify_rdata(tmp_path, module=module)


def test_a_corrupt_segment_is_ignored_by_ranks_that_never_map_it(tmp_path):
    module, index = _pack(tmp_path)
    segment_nbytes = int(index["segment_nbytes"])
    image = tmp_path / "rdata.bin"
    payload = bytearray(image.read_bytes())
    payload[3 * segment_nbytes] ^= 0xFF
    image.write_bytes(bytes(payload))

    # This is the point of per-segment scoping: rank 0 does not read rank 3's
    # bytes, so rank 3's corruption cannot be rank 0's error.
    assert verify_rdata(tmp_path, module=module, segment=0)["chip_count"] == CHIP_COUNT


def test_every_segment_digest_is_published_and_matches_its_bytes(tmp_path):
    import hashlib

    _, index = _pack(tmp_path)
    digests = index["segment_sha256"]
    assert isinstance(digests, list) and len(digests) == CHIP_COUNT
    segment_nbytes = int(index["segment_nbytes"])
    payload = (tmp_path / "rdata.bin").read_bytes()
    for chip, digest in enumerate(digests):
        window = payload[chip * segment_nbytes:(chip + 1) * segment_nbytes]
        if index.get("digest_kind") == "sha256-chunks/v1":
            from triton.flagmega.artifacts.rdata import _chunk_segment_digest
            chunks = index["segment_chunks"][chip]
            assert _chunk_segment_digest(chunks, segment_nbytes) == digest
            for chunk in chunks:
                start = chunk["offset"]
                assert hashlib.sha256(window[start:start + chunk["nbytes"]]).hexdigest() == chunk["sha256"]
        else:
            assert hashlib.sha256(window).hexdigest() == digest


def test_a_single_chip_image_publishes_its_whole_image_digest(tmp_path):
    builder = fm.IRBuilder(dialect="tir", stage="selected_tir")
    value_type = fm.tensor_type("float32", (8,))
    weight = builder.weight("weight", value_type, source="memory", key="weight", id="weight")
    builder.function("main", (), (weight,))
    module = NttBufferizationPolicy(BufferizationOptions.generic(alignment=4)).bufferize(
        builder.build(entry="main")
    )
    checkpoint = MemoryCheckpoint(
        {},
        {"weight": TensorInfo("weight", fm.DType.FLOAT32, (8,), "memory")},
        {"weight": torch.arange(8, dtype=torch.float32)},
    )
    index = pack_rdata(module, checkpoint, tmp_path)

    assert index["chip_count"] == 1
    assert index["segment_sha256"] == [index["sha256"]]
    assert verify_rdata(tmp_path, module=module, segment=0)["nbytes"] == index["nbytes"]


@pytest.mark.parametrize("segment", range(CHIP_COUNT))
def test_corruption_anywhere_in_a_segment_is_detected(tmp_path, segment):
    """Every byte of a segment is covered, not just its indexed entries.

    Entry digests describe chip 0's component only, so a later segment relies on
    its segment digest for coverage.  Corrupt the last byte, which no entry of a
    later segment could be attributed to, and it must still fail.
    """

    module, index = _pack(tmp_path)
    segment_nbytes = int(index["segment_nbytes"])
    image = tmp_path / "rdata.bin"
    payload = bytearray(image.read_bytes())
    payload[(segment + 1) * segment_nbytes - 1] ^= 0xFF
    image.write_bytes(bytes(payload))

    with pytest.raises(ArtifactError):
        verify_rdata(tmp_path, module=module, segment=segment)


def test_a_segment_scoped_verify_rejects_an_index_without_segment_digests(tmp_path):
    import json

    module, _ = _pack(tmp_path)
    index_path = tmp_path / "rdata.index.json"
    index = json.loads(index_path.read_text(encoding="utf-8"))
    payload = (tmp_path / "rdata.bin").read_bytes()
    import hashlib
    index["sha256"] = hashlib.sha256(payload).hexdigest()
    index.pop("digest_kind")
    for entry in index["entries"]:
        start = entry["offset"]
        entry["sha256"] = hashlib.sha256(payload[start:start + entry["nbytes"]]).hexdigest()
    del index["segment_sha256"]
    index["schema"] = "flagmega.rdata-index/v1"
    index_path.write_text(json.dumps(index, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    # A v1 image can still be verified whole; it just cannot be narrowed.
    assert verify_rdata(tmp_path, module=module)["chip_count"] == CHIP_COUNT
    with pytest.raises(ArtifactError, match="digest per chip segment"):
        verify_rdata(tmp_path, module=module, segment=1)


def test_an_out_of_range_segment_is_rejected(tmp_path):
    module, _ = _pack(tmp_path)

    with pytest.raises(ArtifactError, match="segment"):
        verify_rdata(tmp_path, module=module, segment=CHIP_COUNT)
