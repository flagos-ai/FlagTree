# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Keep a verified snapshot's byte proof across normal readonly cache copies."""

import os
import shutil
import time

import pytest
from triton.flagmega.artifacts import pack_rdata, rdata, source_layout, verify_rdata
from triton.flagmega.errors import ArtifactError

from python.test.flagmega.artifacts.test_source_backed_rdata import _fixture


@pytest.mark.parametrize("warm_cache", (False, True))
def test_cached_publication_keeps_the_stable_source_copy_proof(tmp_path, monkeypatch, warm_cache):
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "proofs"))
    module, checkpoint, _ = _fixture(tmp_path)
    cache = tmp_path / "cache"
    if warm_cache:
        pack_rdata(module, checkpoint, tmp_path / "first", cache_dir=cache)
    original_hash = rdata.hash_file

    def no_repeated_bytes(path):
        if path.suffix == ".bin":
            pytest.fail(f"A stable source snapshot was unnecessarily hashed: {path}")
        return original_hash(path)

    monkeypatch.setattr(rdata, "hash_file", no_repeated_bytes)
    monkeypatch.setattr(source_layout, "hash_file", no_repeated_bytes)
    assets = tmp_path / "second"
    pack_rdata(module, checkpoint, assets, cache_dir=cache)
    verify_rdata(assets, module=module, reuse_source_proofs=True)
    verify_rdata(assets, module=module, reuse_source_proofs=True)


def test_changed_cache_source_is_rejected_before_copy_proof_publication(tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "proofs"))
    module, checkpoint, _ = _fixture(tmp_path)
    cache = tmp_path / "cache"
    index = pack_rdata(module, checkpoint, tmp_path / "first", cache_dir=cache)
    entry, = [path for path in cache.iterdir() if path.is_dir()]
    source = entry / index["sources"][0]["path"]
    before = source.stat()
    # A very small fixture can be published and rewritten in the same coarse
    # filesystem timestamp tick. This test requires a changed file version.
    time.sleep(.02)
    with source.open("r+b") as stream:
        stream.seek(-1, 2)
        value = stream.read(1)[0]
        stream.seek(-1, 2)
        stream.write(bytes((value ^ 1,)))
    os.utime(source, ns=(before.st_atime_ns, before.st_mtime_ns))
    assert source.stat().st_ctime_ns != before.st_ctime_ns
    with pytest.raises(ArtifactError, match="source.*hash"):
        pack_rdata(module, checkpoint, tmp_path / "second", cache_dir=cache)


def test_unknown_cache_source_is_audited_once_then_its_copies_reuse_proofs(tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "old-proofs"))
    module, checkpoint, _ = _fixture(tmp_path)
    cache = tmp_path / "cache"
    index = pack_rdata(module, checkpoint, tmp_path / "first", cache_dir=cache)
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "new-proofs"))
    original_hash = rdata.hash_file
    audits = []

    def audited_hash(path):
        if path.suffix == ".bin":
            audits.append(path)
        return original_hash(path)

    monkeypatch.setattr(rdata, "hash_file", audited_hash)
    monkeypatch.setattr(source_layout, "hash_file", audited_hash)
    # The proof also covers ordinary byte copies on filesystems without reflink.
    monkeypatch.setattr(rdata, "_clone_file", shutil.copyfile)
    for name in ("second", "third"):
        assets = tmp_path / name
        pack_rdata(module, checkpoint, assets, cache_dir=cache)
        verify_rdata(assets, module=module, reuse_source_proofs=True)
    assert len(audits) == len(index["sources"])
    assert all(path.is_relative_to(cache) for path in audits)


def test_source_mutation_during_cached_copy_is_rejected(tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "proofs"))
    module, checkpoint, _ = _fixture(tmp_path)
    cache = tmp_path / "cache"
    index = pack_rdata(module, checkpoint, tmp_path / "first", cache_dir=cache)
    entry, = [path for path in cache.iterdir() if path.is_dir()]
    changed = entry / index["sources"][0]["path"]
    original_clone = rdata._clone_file

    def clone_then_change(source, target):
        original_clone(source, target)
        if source == changed:
            with source.open("r+b") as stream:
                stream.seek(-1, 2)
                value = stream.read(1)[0]
                stream.seek(-1, 2)
                stream.write(bytes((value ^ 1,)))

    monkeypatch.setattr(rdata, "_clone_file", clone_then_change)
    with pytest.raises(ArtifactError, match="source snapshot changed while copying"):
        pack_rdata(module, checkpoint, tmp_path / "second", cache_dir=cache)
