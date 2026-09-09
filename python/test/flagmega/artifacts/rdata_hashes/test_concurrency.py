# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: MIT

import threading

from triton.flagmega.artifacts import rdata
from python.test.flagmega.artifacts.rdata_hashes.helpers import HashProbe, MemoryImage, REAL_SHA256, expected_ranges


def test_image_and_entry_hash_updates_overlap_on_the_same_immutable_bytes(monkeypatch):
    data = bytes(range(64))
    ranges = expected_ranges(data, [(0, len(data))])
    image_started = threading.Event()
    entry_started = threading.Event()

    def before_update(identity, payload):
        if identity == 0:
            assert isinstance(payload, bytes)
            image_started.set()
            assert entry_started.wait(5), "image and entry hash streams must overlap"
        else:
            assert isinstance(payload, memoryview) and payload.readonly
            assert image_started.wait(5), "entry must overlap a live image hash update"
            entry_started.set()

    probe = HashProbe(before_update)
    image = MemoryImage(data, 32)
    monkeypatch.setattr(rdata, "_HASH_CHUNK_BYTES", 32)
    monkeypatch.setattr(rdata.hashlib, "sha256", probe.factory)
    rdata._verify_image_hashes(image, REAL_SHA256(data).hexdigest(), ranges)
    image_updates = [payload for identity, payload in probe.updates if identity == 0]
    entry_updates = [payload for identity, payload in probe.updates if identity == 1]
    assert len(image_updates) == len(entry_updates) == 2
    assert all(entry.obj is whole for whole, entry in zip(image_updates, entry_updates, strict=True))
    assert len(probe.threads) == 2
    assert image.closed
    probe.assert_workers_joined()


def test_finishes_current_image_hash_before_reading_the_next_chunk(monkeypatch):
    data = bytes(range(256)) * 3
    completed = []

    def before_read(read_index):
        assert len(completed) == read_index, "rdata must not queue unbounded chunk payloads"

    def after_update(identity, payload):
        if identity == 0:
            completed.append(len(payload))

    image = MemoryImage(data, 32, before_read=before_read)
    probe = HashProbe(after_update=after_update)
    monkeypatch.setattr(rdata, "_HASH_CHUNK_BYTES", 32)
    monkeypatch.setattr(rdata.hashlib, "sha256", probe.factory)
    rdata._verify_image_hashes(image, REAL_SHA256(data).hexdigest(), expected_ranges(data, [(0, len(data))]))
    assert completed == [32] * 24
    assert image.closed
    probe.assert_workers_joined()
