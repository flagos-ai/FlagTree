# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Host-local proof reuse for unchanged, already verified source snapshots.

The identity includes ctime as well as mtime: restoring a modified file's size
and mtime cannot reuse its old proof. Copies and replaced inodes require a new
byte audit. A publisher can also register the proof supplied by its stable,
validated source and completed byte-preserving snapshot operation.
"""

import fcntl
import hashlib
import json
import os
import socket
import tempfile
from contextlib import contextmanager
from pathlib import Path

from triton.flagmega.errors import ArtifactError

_SCHEMA = "flagmega.verified-source-file/v1"


def file_identity(path: Path) -> tuple[int, ...]:
    stat = path.stat()
    return stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns


@contextmanager
def _locked_receipt(identity, digest):
    # Keep proofs host-local even when the normal compiler cache is shared
    # between machines whose filesystem device/inode numbers can coincide.
    # File-version proofs belong to the host's asset cache. A separate or
    # freshly cleared Triton JIT cache does not change the published bytes.
    cache_home = Path(os.environ.get("XDG_CACHE_HOME") or Path.home() / ".cache")
    root = cache_home / "flagtree" / "flagmega-source-verification" / socket.gethostname()
    root.mkdir(parents=True, exist_ok=True)
    key = hashlib.sha256(json.dumps([_SCHEMA, identity, digest], separators=(",", ":")).encode()).hexdigest()
    receipt = root / f"{key}.json"
    with (root / f"{key}.lock").open("ab") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        try:
            yield receipt
        finally:
            fcntl.flock(lock, fcntl.LOCK_UN)


def _proof(identity, digest):
    return {"schema": _SCHEMA, "identity": list(identity), "sha256": digest}


def _write_receipt(path, proof):
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", dir=path.parent, prefix=f".{path.name}.", delete=False) as out:
            temporary = Path(out.name)
            json.dump(proof, out, separators=(",", ":"))
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _require_unchanged(path, identity):
    if file_identity(path) != identity:
        raise ArtifactError(f"Rdata source changed while verifying its hash: {path}.")


def remember_snapshot_digest(path: Path, digest: str, identity: tuple[int, ...]) -> None:
    """Register the publisher's copy proof, bound to its observed file version.

    Call only after a byte-preserving copy from a validated, unchanged source.
    This function does not establish that premise by hashing the destination.
    """
    with _locked_receipt(identity, digest) as receipt:
        _require_unchanged(path, identity)
        _write_receipt(receipt, _proof(identity, digest))


def verified_source_digest(path: Path, expected: str, hash_file) -> str:
    """Audit an unknown version once, sharing the proof with concurrent ranks."""
    identity = file_identity(path)
    proof = _proof(identity, expected)
    with _locked_receipt(identity, expected) as receipt:
        _require_unchanged(path, identity)
        try:
            cached = json.loads(receipt.read_text())
        except (FileNotFoundError, json.JSONDecodeError, UnicodeError):
            cached = None
        if cached == proof:
            return expected
        actual = hash_file(path)
        _require_unchanged(path, identity)
        if actual == expected:
            _write_receipt(receipt, proof)
        return actual
