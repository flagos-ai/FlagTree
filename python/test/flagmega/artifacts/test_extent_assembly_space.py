# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Copy-based assembly must consume staging extents within a bounded disk budget."""

import errno
from pathlib import Path

from triton.flagmega.artifacts import rdata


def test_copy_assembly_needs_only_one_extra_extent(tmp_path, monkeypatch):
    def no_clone(*args):
        raise OSError(errno.EOPNOTSUPP, "test filesystem has no reflinks")

    monkeypatch.setattr(rdata.fcntl, "ioctl", no_clone)
    extent_bytes, total = 4096, 4 * 4096
    payload = bytes(index % 251 for index in range(total))
    destination = tmp_path / "image.bin"
    original_write = rdata._write_all
    with destination.open("w+b") as stream:
        stream.truncate(total)
        with rdata._StripedImage(stream, total) as image:
            image.part_nbytes = extent_bytes
            image.write_at(0, memoryview(payload))

            def bounded_write(target, raw):
                original_write(target, raw)
                staging = sum(path.stat().st_size for path in Path(image.directory.name).glob("*.bin"))
                assert staging + target.tell() <= total + extent_bytes

            monkeypatch.setattr(rdata, "_write_all", bounded_write)
            image.assemble()
        stream.seek(0)
        assert stream.read() == payload
