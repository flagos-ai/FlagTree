# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Explicit proof-based admission of immutable cache inputs after remounts."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shutil
import tempfile

from triton.flagmega.artifacts import rdata
from triton.flagmega.artifacts.manifest import resolve_artifact_path, write_json_atomic
from triton.flagmega.artifacts.source_layout import SOURCE_LAYOUT_SCHEMA, validate_source_layout
from triton.flagmega.artifacts.source_verification import (
    file_identity, remember_snapshot_digest, verified_source_digest,
)
from triton.flagmega.errors import ArtifactError


def _key(data):
    return hashlib.sha256(json.dumps(data, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def rebind_rdata_cache_entry(module, checkpoint, cache_dir, historical_key_data) -> str:
    """Publish a cache identity using verified, unchanged checkpoint content.

    The historical key preimage must match an existing immutable cache entry.
    Layout, recipes and tensor metadata must be identical. Every backing shard
    also needs a current byte-digest proof matching a source snapshot in that
    entry. Checkpoint payloads are never read and recipes are never materialized.
    Cache assets use the ordinary publisher's clone operation; regular artifact
    and runtime integrity checks continue to apply afterwards.
    """
    current = rdata._rdata_cache_key_data(module, checkpoint)
    if current is None or not isinstance(historical_key_data, dict):
        raise ArtifactError("Cache admission requires complete checkpoint key receipts.")
    historical = historical_key_data
    old_key, new_key = _key(historical), _key(current)
    root = Path(cache_dir).resolve()
    original = root / old_key
    index = rdata._load_rdata_cache_entry(original, old_key)
    if index is None or index.get("schema") != SOURCE_LAYOUT_SCHEMA:
        raise ArtifactError("Historical cache receipt has no valid source-layout entry.")
    if {k:v for k,v in current.items() if k != "checkpoint"} != {
            k:v for k,v in historical.items() if k != "checkpoint"}:
        raise ArtifactError("Readonly cache layout or constant recipes changed.")
    old_checkpoint, new_checkpoint = historical.get("checkpoint", {}), current["checkpoint"]
    if (not isinstance(old_checkpoint, dict) or not isinstance(new_checkpoint, dict)
            or {k:v for k,v in old_checkpoint.items() if k != "sources"}
            != {k:v for k,v in new_checkpoint.items() if k != "sources"}):
        raise ArtifactError("Readonly cache tensor metadata changed.")
    validate_source_layout(original, index, module, segment=None, verify_bytes=False)
    old_sources = {s["source"]:s for s in old_checkpoint.get("sources", [])}
    new_sources = {s["source"]:s for s in new_checkpoint.get("sources", [])}
    if not old_sources or old_sources.keys() != new_sources.keys():
        raise ArtifactError("Readonly cache backing shard set changed.")
    key_to_source = {t["key"]:t["source"] for t in new_checkpoint.get("tensors", [])}
    expected = {}
    for program in index["programs"]:
        for operand in program["operands"]:
            name = key_to_source.get(operand["key"])
            snapshot = index["sources"][operand["source"]]
            proof = (snapshot["sha256"], snapshot["nbytes"])
            if name is None or (name in expected and expected[name] != proof):
                raise ArtifactError("Source snapshot does not bind unambiguous checkpoint content.")
            expected[name] = proof
    getter = getattr(checkpoint, "verified_source_digest", None)
    if not callable(getter) or expected.keys() != new_sources.keys():
        raise ArtifactError("Every cache input needs an unchanged source digest proof.")
    source_identities = {}
    for name, source in new_sources.items():
        path = Path(source["path"])
        before = file_identity(path)
        digest, size = expected[name]
        if (source["size"] != size or old_sources[name]["size"] != size
                or getter(path) != digest or file_identity(path) != before):
            raise ArtifactError(f"Missing or changed verified checkpoint proof for {name}.")
        source_identities[path] = before
    paths = [original/"rdata.bin", original/"rdata.index.json"]
    paths.extend(resolve_artifact_path(original, s["path"]) for s in index["sources"])
    identities = {p:file_identity(p) for p in paths}
    if any(p.stat().st_mode & 0o222 for p in paths):
        raise ArtifactError("Cache admission requires immutable input files.")
    def missing_snapshot_proof(path):
        raise ArtifactError(f"Cache snapshot needs an independent byte audit: {path}.")
    for snapshot in index["sources"]:
        path = resolve_artifact_path(original, snapshot["path"])
        verified_source_digest(path, snapshot["sha256"], missing_snapshot_proof)

    def unchanged():
        return (all(file_identity(p) == identity for p,identity in source_identities.items())
                and all(file_identity(p) == identity for p,identity in identities.items()))

    destination = root / new_key
    if destination.exists():
        if rdata._load_rdata_cache_entry(destination, new_key) != index or not unchanged():
            raise ArtifactError("Current cache destination is invalid or input changed.")
        return new_key
    staging = Path(tempfile.mkdtemp(prefix=f".{new_key}.", dir=root))
    try:
        # Use the ordinary publisher's independent reflinks and snapshot proof
        # transfer. The original cache's inode and ctime remain unchanged.
        snapshots = {resolve_artifact_path(original, s["path"]):s["sha256"] for s in index["sources"]}
        for path in paths:
            target = staging / path.relative_to(original)
            target.parent.mkdir(parents=True, exist_ok=True)
            rdata._clone_file(path, target)
            target.chmod(0o444)
            if path in snapshots:
                if file_identity(path) != identities[path]:
                    raise ArtifactError("Source snapshot changed while cloning its verified version.")
                remember_snapshot_digest(target, snapshots[path], file_identity(target))
        manifest = json.loads((original/"cache.json").read_text())
        manifest["key"] = new_key
        write_json_atomic(staging/"cache.json", manifest)
        (staging/"cache.json").chmod(0o444)
        if not unchanged():
            raise ArtifactError("Readonly cache input changed during admission.")
        write_json_atomic(staging/"admission.json", dict(
            schema="flagmega.rdata-cache-admission/v1", historical_key=old_key, current_key=new_key,
            layout_and_recipes_equal=True, tensor_metadata_equal=True,
            verified_source_digests={name:proof[0] for name,proof in expected.items()},
            checkpoint_payload_read=False, snapshot_payload_rehashed=False,
            cache_assets_cloned_by_normal_publisher=True, recipes_materialized=False))
        try:
            staging.rename(destination)
        except FileExistsError:
            if rdata._load_rdata_cache_entry(destination, new_key) != index:
                raise ArtifactError("Concurrent cache admission published an invalid entry.")
        return new_key
    finally:
        if staging.exists():
            shutil.rmtree(staging)


__all__ = ["rebind_rdata_cache_entry"]
