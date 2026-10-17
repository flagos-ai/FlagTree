# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: MIT
"""The explicit official-prefix gate checks real greedy histories and state safety."""
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch


@pytest.fixture
def runner():
    path = Path(__file__).parents[3] / 'tutorials/flagmega/04-glm5.3-flash-fp8/nvidia-h800/run_tp4.py'
    spec = importlib.util.spec_from_file_location('glm_official_prefix_runner', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def reference(tokens=(3847, 284, 1373, 8282)):
    return {'versions': {'vllm': '0.31.0'}, 'requests': [{'tokens': list(tokens)}, {'tokens': list(tokens)}]}


def test_requested_prefix_does_not_mutate_the_independent_reference(runner, tmp_path):
    record = reference()
    path = tmp_path / 'reference.json'
    path.write_text(json.dumps(record))
    tokens, versions = runner.official_prefix(path, 3)
    assert tokens == [3847, 284, 1373] and versions['vllm'] == '0.31.0'
    assert json.loads(path.read_text()) == record


@pytest.mark.parametrize('case', ['version', 'empty', 'short', 'negative', 'boolean', 'disagreement'])
def test_missing_or_inconsistent_official_reference_is_rejected(runner, tmp_path, case):
    data = reference()
    if case == 'version':
        data['versions'] = {'oracle': 'CPU IR'}
    elif case == 'empty':
        data['requests'] = []
    elif case == 'short':
        data['requests'][0]['tokens'] = [3847]
    elif case == 'negative':
        data['requests'][0]['tokens'][0] = -1
    elif case == 'boolean':
        data['requests'][0]['tokens'][0] = True
    else:
        data['requests'][1]['tokens'][1] = 9
    path = tmp_path / 'reference.json'
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError):
        runner.official_prefix(path, 3)


def test_candidate_history_and_selected_token_are_both_checked(runner):
    prefix = [3847, 284, 1373]
    runner.check_greedy_step(11, 3847, prefix, 11, 0)
    runner.check_greedy_step(3847, 284, prefix, 11, 1)
    with pytest.raises(AssertionError):
        runner.check_greedy_step(11, 154822, prefix, 11, 0)
    with pytest.raises(AssertionError, match='histories diverged'):
        runner.check_greedy_step(154822, 284, prefix, 11, 1)
    assert prefix == [3847, 284, 1373]


@pytest.mark.parametrize('steps,requests', [(0, 2), (8, 0), (True, 2), (8, -1)])
def test_empty_workload_cannot_pass_or_initialize_a_runtime(runner, steps, requests):
    with pytest.raises(ValueError, match='positive integers'):
        runner.main(SimpleNamespace(steps=steps, requests=requests))


def state():
    return SimpleNamespace(
        tensor_states={0: (torch.zeros(3, 2), torch.zeros(2, 2))},
        paged=SimpleNamespace(kv_caches=torch.zeros(1, 1, 2, 4, 1, 1), seq_lens=torch.tensor([1]),
                              block_table=torch.tensor([[0]]), config=SimpleNamespace(block_size=4)),
        dsa=SimpleNamespace(packed_states=torch.zeros(1, 4, 2), seq_lens=torch.tensor([1])))


def test_token_mode_retains_state_length_finite_and_full_tail_checks(runner):
    assert runner.check_state(state(), None, 0) == {}


@pytest.mark.parametrize('case', ['conv_nan', 'rec_inf', 'kv_nan', 'kv_tail', 'dsa_nan', 'dsa_tail', 'length'])
def test_token_acceptance_does_not_hide_invalid_state(runner, case):
    actual = state()
    if case == 'conv_nan':
        actual.tensor_states[0][0][0, 0] = torch.nan
    elif case == 'rec_inf':
        actual.tensor_states[0][1][0, 0] = torch.inf
    elif case == 'kv_nan':
        actual.paged.kv_caches[0, 0, 0, 0, 0, 0] = torch.nan
    elif case == 'kv_tail':
        actual.paged.kv_caches[0, 0, 0, 3, 0, 0] = 1
    elif case == 'dsa_nan':
        actual.dsa.packed_states[0, 0, 0] = torch.nan
    elif case == 'dsa_tail':
        actual.dsa.packed_states[0, 3, 0] = 1
    else:
        actual.paged.seq_lens.fill_(2)
    with pytest.raises(AssertionError):
        runner.check_state(actual, None, 0)


def test_original_cpu_state_check_still_enforces_numerical_errors(runner):
    actual = state()
    wanted = {'tensor_states': {0: (torch.ones(3, 2), torch.zeros(2, 2))}}
    with pytest.raises(AssertionError, match='KDA convolution'):
        runner.check_state(actual, wanted, 0)
