# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: MIT

import hashlib
import io
import threading

REAL_SHA256 = hashlib.sha256


def expected_ranges(data, spans):
    return [(start, end, f"weight_{index}", REAL_SHA256(data[start:end]).hexdigest())
            for index, (start, end) in enumerate(spans)]


class MemoryImage:
    """Observe the actual streaming boundary without allowing whole-file reads."""

    def __init__(self, data, chunk_bytes, *, short_read=None, before_read=None, fail_read=None):
        self.data = data
        self.chunk_bytes = chunk_bytes
        self.short_read = short_read
        self.before_read = before_read
        self.fail_read = fail_read
        self.reads = 0
        self.closed = False
        self.chunks = []

    def open(self, mode):
        assert mode == "rb"
        owner = self

        class Stream(io.BytesIO):

            def read(self, size=-1):
                assert size == owner.chunk_bytes
                if owner.before_read:
                    owner.before_read(owner.reads)
                if owner.fail_read == owner.reads:
                    raise OSError("injected stream failure")
                owner.reads += 1
                value = super().read(size if owner.short_read is None else min(size, owner.short_read))
                if value:
                    owner.chunks.append(value)
                return value

            def close(self):
                owner.closed = True
                super().close()

        return Stream(self.data)


class HashProbe:

    def __init__(self, before_update=None, after_update=None):
        self.before_update = before_update
        self.after_update = after_update
        self.count = 0
        self.threads = set()
        self.updates = []

    def factory(self):
        identity = self.count
        self.count += 1
        owner = self

        class Hasher:

            def __init__(self):
                self.real = REAL_SHA256()

            def update(self, data):
                owner.threads.add(threading.current_thread())
                owner.updates.append((identity, data))
                if owner.before_update:
                    owner.before_update(identity, data)
                self.real.update(data)
                if owner.after_update:
                    owner.after_update(identity, data)

            def hexdigest(self):
                return self.real.hexdigest()

        return Hasher()

    def assert_workers_joined(self):
        assert all(not t.is_alive() for t in self.threads if t is not threading.current_thread())
