# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: MIT

import pytest

from triton.flagmega.artifacts import rdata
from triton.flagmega.errors import ArtifactError
from python.test.flagmega.artifacts.rdata_hashes.helpers import MemoryImage, REAL_SHA256, expected_ranges


@pytest.mark.parametrize("chunk_bytes", [1, 7, 64])
@pytest.mark.parametrize("spans", [
    pytest.param([], id="no_entries"),
    pytest.param([(0, 0), (7, 7), (64, 64)], id="empty_entries"),
    pytest.param([(3, 14), (14, 47)], id="adjacent_entries_and_padding"),
    pytest.param([(0, 64)], id="whole_image_entry"),
    pytest.param([(0, 48), (8, 40), (16, 32)], id="nested_entries"),
])
def test_complete_image_and_each_range_digest_are_preserved(monkeypatch, chunk_bytes, spans):
    # The public index validator rejects overlaps. This streaming helper must
    # still hash every supplied intersection correctly, including nested ranges.
    data = bytes(range(64))
    image = MemoryImage(data, chunk_bytes)
    monkeypatch.setattr(rdata, "_HASH_CHUNK_BYTES", chunk_bytes)
    rdata._verify_image_hashes(image, REAL_SHA256(data).hexdigest(), expected_ranges(data, spans))
    assert b"".join(image.chunks) == data
    assert image.closed


@pytest.mark.parametrize("spans", [[], [(0, 0)]])
def test_empty_image_and_empty_ranges(monkeypatch, spans):
    monkeypatch.setattr(rdata, "_HASH_CHUNK_BYTES", 7)
    image = MemoryImage(b"", 7)
    rdata._verify_image_hashes(image, REAL_SHA256(b"").hexdigest(), expected_ranges(b"", spans))
    assert image.reads == 1 and image.closed


@pytest.mark.parametrize("invalid", ["image", "entry", "both", "padding"])
def test_corruption_is_not_hidden_by_parallel_hashing(monkeypatch, invalid):
    original = bytes(range(64))
    data = bytearray(original)
    ranges = expected_ranges(original, [(8, 40)])
    image_digest = REAL_SHA256(original).hexdigest()
    if invalid in ("image", "both"):
        image_digest = "0" * 64
    if invalid in ("entry", "both"):
        start, end, name, _ = ranges[0]
        ranges[0] = (start, end, name, "0" * 64)
    if invalid == "padding":
        data[63] ^= 1
    image = MemoryImage(bytes(data), 7)
    monkeypatch.setattr(rdata, "_HASH_CHUNK_BYTES", 7)
    expected = "entry 'weight_0'" if invalid == "entry" else "image hash"
    with pytest.raises(ArtifactError, match=expected):
        rdata._verify_image_hashes(image, image_digest, ranges)
    assert image.closed
