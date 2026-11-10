# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Synthetic payload tests for explicit clone policy across real writer threads."""
from dataclasses import replace
import errno
import os
from pathlib import Path
import threading

import pytest

from triton.flagmega.artifacts import pack_rdata, write_artifact, verify_rdata, rdata
from python.test.flagmega.artifacts.test_source_backed_rdata import _fixture as source_fixture
from python.test.flagmega.artifacts.test_segment_content_digests import _cached_fixture as chip_fixture


ERRORS = (errno.EOPNOTSUPP, errno.ENOTTY, errno.EXDEV)


def no_copy(*args, **kwargs):
    pytest.fail("payload-copy fallback was invoked")


def failed_ioctl(monkeypatch, code, *, operation=None):
    calls = []
    original = rdata.fcntl.ioctl
    def ioctl(fd, op, arg=0, *args):
        if operation is None or op == operation:
            calls.append(threading.current_thread().name)
            raise OSError(code, "synthetic clone failure")
        return original(fd, op, arg, *args)
    monkeypatch.setattr(rdata.fcntl, "ioctl", ioctl)
    # EXDEV may still succeed via a verified same-inode bind mount. Exercise
    # the terminal EXDEV path without depending on host mount configuration.
    monkeypatch.setattr(rdata, "_overlay_clone_source", lambda *a, **k: None)
    monkeypatch.setattr(rdata, "_bind_clone_destination", lambda *a, **k: None)
    return calls


@pytest.mark.parametrize("code", (*ERRORS, errno.EIO))
def test_strict_file_clone_preserves_destination_and_cleans_temporary(tmp_path, monkeypatch, code):
    source, target = tmp_path / "source", tmp_path / "target"
    source.write_bytes(b"synthetic source")
    target.write_bytes(b"existing destination")
    failed_ioctl(monkeypatch, code)
    monkeypatch.setattr(rdata.shutil, "copyfile", no_copy)
    with pytest.raises(OSError) as failure:
        rdata._clone_file(source, target, require_reflink=True)
    assert failure.value.errno == code
    assert target.read_bytes() == b"existing destination"
    assert source.read_bytes() == b"synthetic source"
    assert not list(tmp_path.glob(".target.*.tmp"))


@pytest.mark.parametrize("code", ERRORS)
def test_default_file_clone_retains_copy_fallback(tmp_path, monkeypatch, code):
    source, target = tmp_path / "source", tmp_path / "target"
    source.write_bytes(b"synthetic source")
    failed_ioctl(monkeypatch, code)
    rdata._clone_file(source, target)
    assert target.read_bytes() == source.read_bytes()
    assert target.stat().st_ino != source.stat().st_ino


@pytest.mark.parametrize("code", (*ERRORS, errno.EIO))
def test_strict_range_assembly_never_copies_and_releases_extents(tmp_path, monkeypatch, code):
    failed_ioctl(monkeypatch, code)
    with (tmp_path / "image").open("w+b") as stream:
        stream.truncate(8192)
        with pytest.raises(OSError) as failure:
            with rdata._StripedImage(stream, 8192, require_reflink=True) as image:
                image.part_nbytes = 4096
                image.write_at(4090, memoryview(b"cross-boundary"))
                monkeypatch.setattr(rdata, "_write_all", no_copy)
                image.assemble()
        assert failure.value.errno == code
    assert not list(tmp_path.glob(".rdata-extents-*"))


@pytest.mark.parametrize("code", ERRORS)
def test_default_range_assembly_retains_copy_fallback(tmp_path, monkeypatch, code):
    failed_ioctl(monkeypatch, code)
    payload = b"x" * 4093 + b"boundary" + b"y" * 23
    target = tmp_path / "image"
    with target.open("w+b") as stream:
        stream.truncate(len(payload))
        with rdata._StripedImage(stream, len(payload)) as image:
            image.part_nbytes = 4096
            image.write_at(0, memoryview(payload))
            image.assemble()
    assert target.read_bytes() == payload
    assert not list(tmp_path.glob(".rdata-extents-*"))


@pytest.mark.parametrize("cached", (False, True))
@pytest.mark.parametrize("code", (errno.EOPNOTSUPP, errno.EXDEV))
def test_writer_strict_policy_reaches_snapshot_workers(tmp_path, monkeypatch, cached, code):
    module, checkpoint, _ = source_fixture(tmp_path)
    calls = failed_ioctl(monkeypatch, code)
    monkeypatch.setattr(rdata.shutil, "copyfile", no_copy)
    output, cache = tmp_path / "artifact", tmp_path / "cache"
    with pytest.raises(OSError) as failure:
        write_artifact(module, output, target="test", checkpoint=checkpoint,
                       rdata_cache_dir=cache if cached else None, require_reflink=True)
    assert failure.value.errno == code
    assert calls and all(name.startswith("flagmega-source-snapshot") for name in calls)
    assert not (output / "artifact.json").exists()
    assert not list(tmp_path.rglob(".rdata-source-snapshots-*"))
    if cached:
        assert not list(cache.iterdir())


@pytest.mark.parametrize("cached", (False, True))
@pytest.mark.parametrize("code", (errno.EOPNOTSUPP, errno.EXDEV))
def test_writer_strict_policy_reaches_segment_assembly(tmp_path, monkeypatch, cached, code):
    module, checkpoint = chip_fixture()
    module = replace(module, stage="bufferized_tir")
    calls = failed_ioctl(monkeypatch, code, operation=rdata._FICLONERANGE)
    original = rdata._write_all
    def allow_only_materialization(target, raw):
        if hasattr(target, "name"):
            no_copy()
        return original(target, raw)
    monkeypatch.setattr(rdata, "_write_all", allow_only_materialization)
    monkeypatch.setattr(rdata.shutil, "copyfile", no_copy)
    output, cache = tmp_path / "artifact", tmp_path / "cache"
    with pytest.raises(OSError) as failure:
        write_artifact(module, output, target="test", checkpoint=checkpoint,
                       rdata_cache_dir=cache if cached else None, require_reflink=True)
    assert failure.value.errno == code
    assert calls and all(name.startswith("flagmega-assets") for name in calls)
    assert not (output / "artifact.json").exists()
    assert not list(tmp_path.rglob(".rdata-extents-*"))
    assert not list(tmp_path.rglob(".rdata.bin.*.tmp"))
    if cached:
        assert not list(cache.iterdir())


@pytest.mark.parametrize("payload", ("rdata", "source"))
@pytest.mark.parametrize("code", (errno.EOPNOTSUPP, errno.EXDEV))
def test_cache_hit_still_requires_reflink_for_each_payload(tmp_path, monkeypatch, payload, code):
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "proofs"))
    module, checkpoint, _ = source_fixture(tmp_path)
    cache = tmp_path / "cache"
    index = pack_rdata(module, checkpoint, tmp_path / "warm", cache_dir=cache)
    entry, = [p for p in cache.iterdir() if p.is_dir()]
    victim = entry / ("rdata.bin" if payload == "rdata" else index["sources"][0]["path"])
    original = rdata.fcntl.ioctl
    calls = []
    def ioctl(fd, operation, arg=0, *args):
        if operation == rdata._FICLONE and Path(os.readlink(f"/proc/self/fd/{arg}")) == victim:
            calls.append(threading.current_thread().name)
            raise OSError(code, "synthetic cached clone failure")
        return original(fd, operation, arg, *args)
    monkeypatch.setattr(rdata.fcntl, "ioctl", ioctl)
    monkeypatch.setattr(rdata, "_overlay_clone_source", lambda *a, **k: None)
    monkeypatch.setattr(rdata, "_bind_clone_destination", lambda *a, **k: None)
    monkeypatch.setattr(rdata.shutil, "copyfile", no_copy)
    monkeypatch.setattr(rdata, "_pack_rdata_uncached", lambda *a, **k: pytest.fail("cache hit repacked"))
    output = tmp_path / "artifact"
    with pytest.raises(OSError) as failure:
        write_artifact(module, output, target="test", checkpoint=checkpoint,
                       rdata_cache_dir=cache, require_reflink=True)
    assert failure.value.errno == code
    assert calls and all(name.startswith("flagmega-assets") for name in calls)
    assert not (output / "artifact.json").exists()
    assert not list(output.rglob("*.tmp"))
    assert victim.is_file()


def test_strict_success_uses_real_reflinks_and_retains_byte_identity(tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "proofs"))
    module, checkpoint, _ = source_fixture(tmp_path)
    original = rdata.fcntl.ioctl
    operations = []
    def observe(fd, operation, arg=0, *args):
        result = original(fd, operation, arg, *args)
        operations.append(operation)
        return result
    monkeypatch.setattr(rdata.fcntl, "ioctl", observe)
    monkeypatch.setattr(rdata.shutil, "copyfile", no_copy)
    cache = tmp_path / "cache"
    root = write_artifact(module, tmp_path / "artifact", target="test", checkpoint=checkpoint,
                          rdata_cache_dir=cache, require_reflink=True)
    verify_rdata(root / "assets", module=module)
    assert rdata._FICLONE in operations
    assert (root / "artifact.json").exists()
    source, target = tmp_path / "synthetic", tmp_path / "clone"
    source.write_bytes(b"independent fixture")
    rdata._clone_file(source, target, require_reflink=True)
    assert target.read_bytes() == source.read_bytes()
    assert target.stat().st_ino != source.stat().st_ino
    target.write_bytes(b"changed clone")
    assert source.read_bytes() == b"independent fixture"
    # This fixture has source-layout virtual outputs; exercise actual striped
    # materialization separately so successful FICLONERANGE is also required.
    module, checkpoint = chip_fixture()
    pack_rdata(module, checkpoint, tmp_path / "striped", require_reflink=True)
    verify_rdata(tmp_path / "striped", module=module)
    assert rdata._FICLONERANGE in operations


@pytest.mark.parametrize("value", (None, 0, "false"))
@pytest.mark.parametrize("entry", ("pack", "artifact"))
def test_public_clone_policy_rejects_non_boolean_before_output(tmp_path, value, entry):
    output = tmp_path / "output"
    with pytest.raises(TypeError, match="require_reflink"):
        if entry == "pack":
            pack_rdata(None, None, output, require_reflink=value)
        else:
            write_artifact(None, output, target="test", require_reflink=value)
    assert not output.exists()
