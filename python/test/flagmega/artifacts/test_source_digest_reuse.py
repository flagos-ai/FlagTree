# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Reuse a proven snapshot digest only while the exact file version survives."""

import json
import os
import shutil
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import pytest
from triton.flagmega.artifacts import pack_rdata, source_layout
from triton.flagmega.artifacts import verify_rdata as audit_rdata
from triton.flagmega.errors import ArtifactError

from python.test.flagmega.artifacts.test_source_backed_rdata import _fixture


def verify_rdata(*args, **kwargs):
    return audit_rdata(*args, reuse_source_proofs=True, **kwargs)


def snapshot(tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "host-cache"))
    monkeypatch.setenv("TRITON_CACHE_DIR", str(tmp_path / "cache"))
    module, checkpoint, _ = _fixture(tmp_path)
    assets = tmp_path / "assets"
    index = pack_rdata(module, checkpoint, assets)
    return module, assets, index


def test_publication_proof_avoids_hashing_the_unchanged_clone_again(tmp_path, monkeypatch):
    module, assets, _ = snapshot(tmp_path, monkeypatch)

    def repeated_hash(path):
        pytest.fail(f"An unchanged verified snapshot was read again: {path}")

    monkeypatch.setattr(source_layout, "hash_file", repeated_hash)
    verify_rdata(assets, module=module)
    verify_rdata(assets, module=module)


def test_publication_proof_survives_a_changed_jit_cache_directory(tmp_path, monkeypatch):
    module, assets, _ = snapshot(tmp_path, monkeypatch)
    monkeypatch.setenv("TRITON_CACHE_DIR", str(tmp_path / "another-jit-cache"))

    def repeated_hash(path):
        pytest.fail(f"Rotating the JIT cache must not invalidate unchanged asset bytes: {path}")

    monkeypatch.setattr(source_layout, "hash_file", repeated_hash)
    verify_rdata(assets, module=module)


def test_publication_proof_survives_a_new_process_with_its_own_jit_cache(tmp_path, monkeypatch):
    _, assets, _ = snapshot(tmp_path, monkeypatch)
    worker = '''
import sys
from pathlib import Path
from triton.flagmega.artifacts import source_layout, verify_rdata
def repeated_hash(path):
    raise AssertionError("A fresh JIT cache cannot revoke publication's unchanged-file proof: " + str(path))
source_layout.hash_file = repeated_hash
verify_rdata(Path(sys.argv[1]), reuse_source_proofs=True)
'''
    result = subprocess.run([sys.executable, "-c", worker, str(assets)],
        env={**os.environ, "TRITON_CACHE_DIR": str(tmp_path / "child-jit-cache")},
        capture_output=True, text=True, timeout=60, check=False)
    assert result.returncode == 0, result.stdout + result.stderr


def test_same_size_mutation_invalidates_proof_even_when_mtime_is_restored(tmp_path, monkeypatch):
    module, assets, index = snapshot(tmp_path, monkeypatch)
    verify_rdata(assets, module=module)
    source = assets / index["sources"][0]["path"]
    before = source.stat()
    with source.open("r+b") as out:
        out.seek(-1, 2)
        value = out.read(1)[0]
        out.seek(-1, 2)
        out.write(bytes((value ^ 1,)))
    os.utime(source, ns=(before.st_atime_ns, before.st_mtime_ns))
    with pytest.raises(ArtifactError, match="source.*hash"):
        verify_rdata(assets, module=module)


def test_replaced_inode_invalidates_a_previous_snapshot_proof(tmp_path, monkeypatch):
    module, assets, index = snapshot(tmp_path, monkeypatch)
    source = assets / index["sources"][0]["path"]
    replacement = source.with_suffix(".replacement")
    data = bytearray(source.read_bytes())
    data[-1] ^= 1
    replacement.write_bytes(data)
    os.replace(replacement, source)
    with pytest.raises(ArtifactError, match="source.*hash"):
        verify_rdata(assets, module=module)


def test_a_copied_snapshot_is_hashed_once_then_reused(tmp_path, monkeypatch):
    module, assets, index = snapshot(tmp_path, monkeypatch)
    copied = tmp_path / "copy"
    shutil.copytree(assets, copied)
    calls = []
    original = source_layout.hash_file

    def audit(path):
        calls.append(path)
        return original(path)

    monkeypatch.setattr(source_layout, "hash_file", audit)
    verify_rdata(copied, module=module)
    assert len(calls) == len(index["sources"])
    verify_rdata(copied, module=module)
    assert len(calls) == len(index["sources"])


def test_concurrent_verifiers_share_one_hash_of_an_unknown_copy(tmp_path, monkeypatch):
    _, assets, index = snapshot(tmp_path, monkeypatch)
    copied = tmp_path / "copy"
    shutil.copytree(assets, copied)
    calls = []
    original = source_layout.hash_file

    def audit(path):
        calls.append(path)
        return original(path)

    monkeypatch.setattr(source_layout, "hash_file", audit)
    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(lambda _: verify_rdata(copied), range(4)))
    assert len(calls) == len(index["sources"])


def test_a_changed_expected_digest_cannot_reuse_the_old_proof(tmp_path, monkeypatch):
    module, assets, index = snapshot(tmp_path, monkeypatch)
    index["sources"][0]["sha256"] = "0" * 64
    index["sha256"] = source_layout.content_digest(index)
    (assets / "rdata.index.json").write_text(json.dumps(index))
    with pytest.raises(ArtifactError, match="source.*hash"):
        verify_rdata(assets, module=module)


def test_explicit_audit_still_reads_bytes_after_a_successful_runtime_verification(tmp_path, monkeypatch):
    module, assets, index = snapshot(tmp_path, monkeypatch)
    verify_rdata(assets, module=module)
    calls = []
    original = source_layout.hash_file

    def audit(path):
        calls.append(path)
        return original(path)

    monkeypatch.setattr(source_layout, "hash_file", audit)
    audit_rdata(assets, module=module)
    audit_rdata(assets, module=module)
    assert len(calls) == 2 * len(index["sources"])


def test_a_source_changed_during_its_hash_is_never_cached(tmp_path, monkeypatch):
    _, assets, _ = snapshot(tmp_path, monkeypatch)
    copied = tmp_path / "copy"
    shutil.copytree(assets, copied)
    original = source_layout.hash_file

    def changed_during_hash(path):
        digest = original(path)
        with path.open("r+b") as out:
            out.seek(-1, 2)
            value = out.read(1)[0]
            out.seek(-1, 2)
            out.write(bytes((value ^ 1,)))
        return digest

    monkeypatch.setattr(source_layout, "hash_file", changed_during_hash)
    with pytest.raises(ArtifactError, match="source changed.*hash"):
        verify_rdata(copied)
    monkeypatch.setattr(source_layout, "hash_file", original)
    with pytest.raises(ArtifactError, match="source.*hash"):
        verify_rdata(copied)


def test_four_processes_share_one_unknown_snapshot_audit(tmp_path, monkeypatch):
    _, assets, index = snapshot(tmp_path, monkeypatch)
    copied = tmp_path / "copy"
    shutil.copytree(assets, copied)
    log = tmp_path / "hashes.log"
    worker = '''
import os, sys, time
from pathlib import Path
from triton.flagmega.artifacts import source_layout, verify_rdata
root, log = map(Path, sys.argv[1:3])
original = source_layout.hash_file
def audit(path):
    with log.open("ab", buffering=0) as out:
        out.write(b"hash\\n")
    time.sleep(.1)
    return original(path)
source_layout.hash_file = audit
(log.parent / ("ready-" + sys.argv[3])).touch()
deadline = time.monotonic() + 30
while len(list(log.parent.glob("ready-*"))) < 4:
    assert time.monotonic() < deadline
    time.sleep(.01)
verify_rdata(root, reuse_source_proofs=True)
'''
    children = [subprocess.Popen([sys.executable, "-c", worker, str(copied), str(log), str(index)],
                                 stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
                for index in range(4)]
    try:
        for child in children:
            stdout, stderr = child.communicate(timeout=60)
            assert child.returncode == 0, stdout + stderr
    finally:
        for child in children:
            if child.poll() is None:
                child.kill()
                child.wait()
    assert log.read_text().splitlines() == ["hash"] * len(index["sources"])


def test_runtime_maps_a_proven_source_and_still_audits_its_physical_chunks(tmp_path, monkeypatch):
    import torch
    import triton.flagmega.runtime.module as runtime
    from triton.flagmega.artifacts import rdata

    module, assets, index = snapshot(tmp_path, monkeypatch)
    chunks = []
    original = rdata._verify_chunk_hashes

    def fresh_chunks(path, data, segment):
        chunks.append(segment)
        return original(path, data, segment)

    def repeated_source_hash(_path):
        pytest.fail("Runtime hashed an unchanged published source again")

    monkeypatch.setattr(source_layout, "hash_file", repeated_source_hash)
    monkeypatch.setattr(rdata, "_verify_chunk_hashes", fresh_chunks)
    monkeypatch.setattr(runtime, "_runtime_chip_index", lambda _count: 0)
    instance = SimpleNamespace(artifact=assets.parent, manifest={"rdata": index}, ir_module=module)
    result = runtime.GeneratedTirCallGraphModule._load_rdata(instance, torch, "cpu", index["segment_nbytes"])
    assert result.numel() == index["segment_nbytes"]
    assert chunks == [0]
