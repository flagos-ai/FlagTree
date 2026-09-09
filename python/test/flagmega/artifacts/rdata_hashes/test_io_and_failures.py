# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: MIT

import pytest

from triton.flagmega.artifacts import rdata
from triton.flagmega.errors import ArtifactError
from python.test.flagmega.artifacts.rdata_hashes.helpers import HashProbe, MemoryImage, REAL_SHA256, expected_ranges


@pytest.mark.parametrize("short_read", [1, 3, 17])
def test_short_reads_keep_exact_offsets_and_all_bytes(monkeypatch, short_read):
    data = bytes(range(97))
    image = MemoryImage(data, 32, short_read=short_read)
    monkeypatch.setattr(rdata, "_HASH_CHUNK_BYTES", 32)
    rdata._verify_image_hashes(image, REAL_SHA256(data).hexdigest(), expected_ranges(data, [(2, 34), (34, 93)]))
    assert b"".join(image.chunks) == data and image.closed


def test_truncated_stream_fails_image_hash(monkeypatch):
    data = bytes(range(97))
    image = MemoryImage(data[:-3], 32)
    monkeypatch.setattr(rdata, "_HASH_CHUNK_BYTES", 32)
    with pytest.raises(ArtifactError, match="image hash"):
        rdata._verify_image_hashes(image, REAL_SHA256(data).hexdigest(), expected_ranges(data, [(0, 97)]))
    assert image.closed


@pytest.mark.parametrize("failure", ["image_hasher", "entry_hasher", "read"])
def test_errors_are_propagated_and_no_hash_worker_outlives_verification(monkeypatch, failure):
    data = bytes(range(97))
    updates = [0, 0]

    def before_update(identity, payload):
        updates[identity] += 1
        selected = (identity == 0 and failure == "image_hasher") or (identity == 1 and failure == "entry_hasher")
        if selected and updates[identity] == 2:
            raise LookupError("injected hasher failure")

    image = MemoryImage(data, 32, fail_read=1 if failure == "read" else None)
    probe = HashProbe(before_update)
    monkeypatch.setattr(rdata, "_HASH_CHUNK_BYTES", 32)
    monkeypatch.setattr(rdata.hashlib, "sha256", probe.factory)
    exception = OSError if failure == "read" else LookupError
    with pytest.raises(exception, match="injected"):
        rdata._verify_image_hashes(image, REAL_SHA256(data).hexdigest(), expected_ranges(data, [(0, 97)]))
    assert image.closed
    probe.assert_workers_joined()
