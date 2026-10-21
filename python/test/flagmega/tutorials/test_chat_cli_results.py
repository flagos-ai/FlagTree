# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Standalone plots must retain prefill differences and verify actual artifacts."""
import copy
import gzip
import hashlib
import importlib.util
import json

import pytest

from test_flagos_results import TUTORIAL, report, write_report


@pytest.fixture
def chat_renderer(monkeypatch):
    monkeypatch.syspath_prepend(str(TUTORIAL))
    spec = importlib.util.spec_from_file_location('chat_results_test', TUTORIAL / 'render_chat_cli_results.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def prepare(directory):
    native, cpp = report('native'), report('flagmega')
    write_report(directory, 'native', native, compressed=True)
    write_report(directory, 'flagmega', cpp, compressed=True)
    run = {'prompt_tokens': [1, 2], 'token_ids': [6, 7, 8], 'cached_tokens': 0,
           'decode_step_ms': [1., 9.], 'decode_tokens_per_second': 999.,
           'total_ms': 40., 'prefill_ms': 30., 'ttft_ms': 30.}
    chat = {key: native[key] for key in ('checkpoint_files', 'visible_devices', 'torch', 'triton',
                                        'compiler_sha256', 'batch', 'concurrency')}
    chat.update(checkpoint_revision='checkpoint', round=0, repeats=1,
                process_rounds=1,
                performance_valid=True, vllm_imported=False, prefill_mode='compiled_token_scan',
                prefix_caching=False, artifact_codegen=cpp['executor_stats'][0]['codegen'],
                sampling={'temperature': 0., 'top_k': 0, 'top_p': 1., 'seed': 0},
                native_token_agreement=True, native_reference_prefill='native_token_scan',
                boundary='standalone generation, sampling and text decoding',
                backend={'artifact_source_sha256': 'cpp-source', 'cuda_graph': True,
                         'serving_source_sha256': 'serving-source', 'gpu': 'same'},
                scenarios=[{'prompt_length': 2, 'runs': [run]}],
                warmup_scenarios=[{'prompt_length': 2, 'runs': [copy.deepcopy(run)]}])
    path = directory / 'round00.chat_cli.json.gz'
    path.write_bytes(gzip.compress(json.dumps(chat).encode(), mtime=0))
    reference = {key: native[key] for key in ('checkpoint_files', 'visible_devices', 'torch', 'triton',
                                              'compiler_sha256', 'vllm', 'vllm_source',
                                              'flaggems_source', 'plugin_source')}
    reference.update(schema='flagmega.native-token-scan-reference/v1', accepted=True,
                     max_num_batched_tokens=1, prefill_mode='native_token_scan',
                     scenarios=[{'prompt_length': 2, 'runs': [{'prompt_tokens': [1, 2],
                                                             'token_ids': [6, 7, 8]}]}])
    reference_path = directory / 'reference.json'
    reference_path.write_text(json.dumps(reference))
    chat['native_reference_sha256'] = hashlib.sha256(reference_path.read_bytes()).hexdigest()
    path.write_bytes(gzip.compress(json.dumps(chat).encode(), mtime=0))
    return path, reference_path


def test_native_scan_verification_retains_batched_difference(tmp_path, chat_renderer):
    path, reference = prepare(tmp_path)
    rows, groups, reports = chat_renderer.aggregate_chat([path])
    assert rows[0]['decode_tokens_per_second'] == 200.  # Entire interval sum, not stored rate.
    assert rows[0]['request_tokens_per_second'] == 75.
    summary = chat_renderer.add_flagos_comparison(rows, groups, reports, tmp_path,
                                                 native_reference=reference)
    assert set(summary['scenarios'][0]['variants']) == {'native', 'flagmega', 'chat_cli'}
    assert summary['chat_cli']['native_token_agreement']
    assert not summary['chat_cli']['native_batched_tokens_exact']


@pytest.mark.parametrize('corruption', ['library', 'token', 'warmup', 'reference', 'reference_identity',
                                        'missing_reference'])
def test_chat_comparison_rejects_unverified_identity_and_tokens(tmp_path, chat_renderer, corruption):
    path, reference = prepare(tmp_path)
    chat = chat_renderer.read_report(path)
    if corruption == 'library':
        chat['artifact_codegen']['library_sha256'] = 'other-library'
    elif corruption == 'token':
        chat['scenarios'][0]['runs'][0]['token_ids'][0] = 99
        chat['warmup_scenarios'] = []
    elif corruption == 'warmup':
        chat['warmup_scenarios'][0]['runs'][0]['token_ids'][0] = 99
    elif corruption == 'reference':
        raw = json.loads(reference.read_text())
        raw['accepted'] = False
        reference.write_text(json.dumps(raw))
    elif corruption == 'reference_identity':
        chat['native_reference_sha256'] = 'different-reference'
    elif corruption == 'missing_reference':
        reference = None
    path.write_bytes(gzip.compress(json.dumps(chat).encode(), mtime=0))
    with pytest.raises((ValueError, AssertionError)):
        rows, groups, reports = chat_renderer.aggregate_chat([path])
        chat_renderer.add_flagos_comparison(rows, groups, reports, tmp_path, native_reference=reference)


def test_torch_module_version_requires_same_release_and_rejects_other_corex(chat_renderer):
    chat_renderer.require_environment({'torch': '2.10.0'}, {'torch': '2.10.0+corex.4.5'}, ('torch',))
    with pytest.raises(ValueError):
        chat_renderer.require_environment({'torch': '2.9.0'}, {'torch': '2.10.0+corex.4.5'}, ('torch',))
    with pytest.raises(ValueError):
        chat_renderer.require_environment({'torch': '2.10.0+corex.4.4'},
                                          {'torch': '2.10.0+corex.4.5'}, ('torch',))


def test_chat_comparison_rejects_shared_process_rounds(tmp_path, chat_renderer):
    path, reference = prepare(tmp_path)
    chat = chat_renderer.read_report(path)
    chat['process_rounds'] = 3
    path.write_bytes(gzip.compress(json.dumps(chat).encode(), mtime=0))
    rows, groups, reports = chat_renderer.aggregate_chat([path])
    with pytest.raises(ValueError, match='process'):
        chat_renderer.add_flagos_comparison(rows, groups, reports, tmp_path,
                                           native_reference=reference)


def test_chat_comparison_rejects_different_start_conditions(tmp_path, chat_renderer):
    path, reference = prepare(tmp_path)
    for variant in ('native', 'flagmega'):
        p = tmp_path / f'round00.{variant}.json.gz'
        raw = chat_renderer.read_report(p)
        raw.update(max_start_temperature_c=70, conditions_sha256='conditions')
        raw['scenarios'][0]['start_conditions'] = {
            'max_start_temperature_c': 70, 'observations': [{'temperature_c': 68}]}
        p.write_bytes(gzip.compress(json.dumps(raw).encode(), mtime=0))
    rows, groups, reports = chat_renderer.aggregate_chat([path])
    with pytest.raises(ValueError, match='conditions'):
        chat_renderer.add_flagos_comparison(rows, groups, reports, tmp_path,
                                           native_reference=reference)
