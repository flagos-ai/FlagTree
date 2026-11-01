# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: MIT

import copy
import json
import os
import time

import pytest

from triton.flagmega.artifacts import pack_rdata
from triton.flagmega.artifacts import rdata
from triton.flagmega.errors import ArtifactError
from .test_source_backed_rdata import _fixture


def _cache(tmp_path, monkeypatch):
    module, checkpoint, _ = _fixture(tmp_path)
    cache = tmp_path / 'cache'
    # Capture precisely the original key preimage, as a durable input receipt.
    captured = {}
    original = json.dumps
    def observe(value, *args, **kwargs):
        if isinstance(value, dict) and value.get('schema') == rdata.RDATA_CACHE_SCHEMA and 'checkpoint' in value:
            captured['data'] = copy.deepcopy(value)
        return original(value, *args, **kwargs)
    monkeypatch.setattr(json, 'dumps', observe)
    old_key = rdata._rdata_cache_key(module, checkpoint)
    pack_rdata(module, checkpoint, tmp_path / 'first', cache_dir=cache)
    monkeypatch.setattr(json, 'dumps', original)
    identity = checkpoint.cache_identity
    def remounted(keys):
        value = identity(keys)
        for source in value['sources']:
            source['device'] += 1
        return value
    monkeypatch.setattr(checkpoint, 'cache_identity', remounted)
    assert rdata._rdata_cache_key(module, checkpoint) != old_key
    return module, checkpoint, cache, old_key, captured['data']


def test_verified_remount_reuses_cache_without_materializing_payload(tmp_path, monkeypatch):
    from triton.flagmega.artifacts import rebind_rdata_cache_entry
    module, checkpoint, cache, old_key, historical = _cache(tmp_path, monkeypatch)
    old = cache / old_key
    identity = old.joinpath('rdata.bin').stat()
    old_manifest = old.joinpath('cache.json').read_text()
    def forbidden(*args, **kwargs):
        pytest.fail('readonly cache rebind materialized or hashed checkpoint payload')
    monkeypatch.setattr(checkpoint, 'load_tensor', forbidden)
    monkeypatch.setattr(checkpoint, 'tensor_byte_range', forbidden)
    monkeypatch.setattr(checkpoint, 'source_file_digest', forbidden)
    monkeypatch.setattr(rdata, '_pack_rdata_uncached', forbidden)
    key = rebind_rdata_cache_entry(module, checkpoint, cache, historical)
    assert key == rdata._rdata_cache_key(module, checkpoint)
    assert rdata._load_rdata_cache_entry(cache / key, key) is not None
    assert old.joinpath('rdata.bin').stat().st_ino == identity.st_ino
    assert old.joinpath('cache.json').read_text() == old_manifest
    result = pack_rdata(module, checkpoint, tmp_path/'second', cache_dir=cache)
    assert result['sha256'] == json.loads((tmp_path/'first/rdata.index.json').read_text())['sha256']
    assert (tmp_path/'second/rdata.bin').stat().st_ino != identity.st_ino
    assert rebind_rdata_cache_entry(module, checkpoint, cache, historical) == key


@pytest.mark.parametrize('trigger', ['unverified', 'changed_bytes_restored_mtime', 'tampered_preimage', 'changed_recipe', 'corrupt_index', 'unknown_snapshot_version'])
def test_cache_rebind_rejects_missing_or_changed_proof(tmp_path, monkeypatch, trigger):
    from triton.flagmega.artifacts import rebind_rdata_cache_entry
    module, checkpoint, cache, old_key, historical = _cache(tmp_path, monkeypatch)
    if trigger == 'unverified':
        checkpoint._source_digests.clear()
    elif trigger == 'changed_bytes_restored_mtime':
        path = checkpoint.root/'model.safetensors'
        before = path.stat()
        time.sleep(0.02)
        with path.open('r+b') as stream:
            stream.seek(-1, 2)
            stream.write(b'\xff')
        os.utime(path, ns=(before.st_atime_ns, before.st_mtime_ns))
    elif trigger == 'tampered_preimage':
        historical['chip_count'] += 1
    elif trigger == 'changed_recipe':
        original = rdata._rdata_cache_key_data
        def changed(*args):
            value = original(*args)
            value['recipes'][0]['fingerprint'] = '0'*64
            return value
        monkeypatch.setattr(rdata, '_rdata_cache_key_data', changed)
    elif trigger == 'corrupt_index':
        index_path = cache/old_key/'rdata.index.json'
        index_path.chmod(0o644)
        index_path.write_text('{}')
    else:
        snapshot = next((cache/old_key/'sources').iterdir())
        before = snapshot.stat().st_ctime_ns
        time.sleep(0.02)
        snapshot.chmod(0o644)
        snapshot.chmod(0o444)
        assert snapshot.stat().st_ctime_ns != before
    with pytest.raises(ArtifactError):
        rebind_rdata_cache_entry(module, checkpoint, cache, historical)
    assert len(list(cache.iterdir())) == 1
