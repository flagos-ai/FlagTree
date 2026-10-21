# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Published graphs need matching sequences and correctly bounded throughput."""
import copy
import gzip
import importlib.util
import json
from pathlib import Path

import pytest


TUTORIAL = (Path(__file__).resolve().parents[3] / 'tutorials' / 'flagmega' /
            '01-qwen3-1.7b-bf16' / 'iluvator-bi-v150')


@pytest.fixture
def renderer(monkeypatch):
    monkeypatch.syspath_prepend(str(TUTORIAL))
    spec = importlib.util.spec_from_file_location('flagos_results_test', TUTORIAL / 'render_flagos_results.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def report(variant):
    native = variant == 'native'
    run = {'prompt_tokens': [1, 2], 'token_ids': [3, 4, 5],
           'decode_step_ms': [1.0, 9.0] if native else [2.0, 2.0],
           'e2e_ms': 30.0 if native else 20.0, 'ttft_ms': 10.0}
    result = {key: 'same' for key in ('checkpoint_files', 'compiler_sha256', 'visible_devices',
              'device', 'torch', 'triton', 'vllm', 'vllm_source', 'flaggems_source', 'plugin_source',
              'adapter_sha256', 'benchmark_sha256', 'boundary', 'prefill')}
    result.update(performance_valid=True, native_token_agreement=True, decode_tokens=3,
                  block_size=256, num_blocks=16, batch=1, concurrency=1, prefix_caching=False,
                  warmups=1, repeats=1, cuda_graph=native, cudagraph_mode='FULL' if native else 'NONE',
                  compilation_mode=0, enforce_eager=not native,
                  decode_execution='vllm_full_graph' if native else 'flagmega_worker_graph',
                  scenarios=[{'scenario': 'prompt2', 'prompt_length': 2, 'warmup_runs': [run],
                              'runs': [copy.deepcopy(run)], 'decode_ms': {'median': 5 if native else 2},
                              'e2e_ms': {'median': run['e2e_ms']}, 'ttft_ms': {'median': 10}}])
    if not native:
        result['executor_stats'] = [{'decode_calls': 4, 'graph_replays': 4, 'kv_copy_bytes_per_decode': 0,
                                     'source_sha256': 'cpp-source', 'codegen': {
                                         'backend': 'cpp', 'source_sha256': 'cpp-source',
                                         'library_sha256': 'cpp-library'}}]
    return result


def write_report(directory, variant, value, compressed=False):
    path = directory / f'round00.{variant}.json'
    data = json.dumps(value).encode()
    if compressed:
        path = path.with_suffix('.json.gz')
        data = gzip.compress(data, mtime=0)
    path.write_bytes(data)


def test_throughput_uses_whole_decode_and_request_intervals(tmp_path, renderer):
    write_report(tmp_path, 'native', report('native'), compressed=True)
    write_report(tmp_path, 'flagmega', report('flagmega'), compressed=True)
    summary = renderer.aggregate(tmp_path)
    row = summary['scenarios'][0]['variants']
    # Two decode intervals exclude the first of three output tokens.
    assert row['native']['decode_tokens_per_second'] == 200.0
    assert row['native']['request_tokens_per_second'] == 100.0
    assert row['flagmega']['decode_tokens_per_second'] == 500.0
    assert row['flagmega']['request_tokens_per_second'] == 150.0
    assert summary['codegen']['library_sha256'] == 'cpp-library'


@pytest.mark.parametrize('corruption', ['token', 'graph_replays', 'missing_round', 'source',
                                        'environment', 'fused_inputs', 'input_source'])
def test_reject_unverifiable_comparisons(tmp_path, renderer, corruption):
    native, candidate = report('native'), report('flagmega')
    if corruption == 'token':
        candidate['scenarios'][0]['warmup_runs'][0]['token_ids'][1] = 99
    elif corruption == 'graph_replays':
        candidate['executor_stats'][0]['graph_replays'] = 3
    elif corruption == 'source':
        other = copy.deepcopy(candidate)
        other['executor_stats'][0]['codegen']['library_sha256'] = 'different-library'
        (tmp_path / 'round01.flagmega.json').write_text(json.dumps(other))
        (tmp_path / 'round01.native.json').write_text(json.dumps(native))
    elif corruption == 'environment':
        candidate['compiler_sha256'] = 'different-compiler'
    elif corruption == 'fused_inputs':
        native['decode_inputs_sha256'] = candidate['decode_inputs_sha256'] = 'input-kernel'
        candidate['executor_stats'][0]['fused_decode_input_calls'] = 3
    elif corruption == 'input_source':
        native['decode_inputs_sha256'] = 'native-input-kernel'
        candidate['decode_inputs_sha256'] = 'different-input-kernel'
    write_report(tmp_path, 'native', native)
    if corruption != 'missing_round':
        write_report(tmp_path, 'flagmega', candidate)
    with pytest.raises((ValueError, AssertionError)):
        renderer.aggregate(tmp_path)
