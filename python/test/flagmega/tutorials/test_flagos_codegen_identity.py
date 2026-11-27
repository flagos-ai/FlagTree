# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Mixed-source artifacts must identify the backend actually executed."""
import hashlib
import importlib.util
from pathlib import Path


TUTORIAL = (Path(__file__).resolve().parents[3] / 'tutorials' / 'flagmega' /
            '01-qwen3-1.7b-bf16' / 'iluvator-bi-v150')


def load_adapter(monkeypatch):
    monkeypatch.syspath_prepend(str(TUTORIAL))
    spec = importlib.util.spec_from_file_location('flagos_adapter_test', TUTORIAL / 'flagos_adapter.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_cpp_identity_ignores_stale_python_source(tmp_path, monkeypatch):
    adapter = load_adapter(monkeypatch)
    (tmp_path / 'generated_kernels.py').write_bytes(b'stale python source')
    (tmp_path / 'generated_kernels.cpp').write_bytes(b'actual cpp source')
    (tmp_path / 'decode.so').write_bytes(b'actual compiled library')
    manifest = {'codegen_backend': 'cpp', 'codegen': {
        'source': 'generated_kernels.cpp', 'library': 'decode.so'}}
    identity = adapter.artifact_codegen_identity(tmp_path, manifest)
    assert identity['source_sha256'] == hashlib.sha256(b'actual cpp source').hexdigest()
    assert identity['library_sha256'] == hashlib.sha256(b'actual compiled library').hexdigest()
    (tmp_path / 'generated_kernels.py').unlink()
    assert adapter.artifact_codegen_identity(tmp_path, manifest) == identity


def test_triton_identity_needs_no_cpp_library(tmp_path, monkeypatch):
    adapter = load_adapter(monkeypatch)
    (tmp_path / 'generated_kernels.py').write_bytes(b'actual triton source')
    manifest = {'codegen_backend': 'triton', 'codegen': {'source': 'generated_kernels.py'}}
    assert adapter.artifact_codegen_identity(tmp_path, manifest) == {
        'backend': 'triton', 'source': 'generated_kernels.py',
        'source_sha256': hashlib.sha256(b'actual triton source').hexdigest()}
