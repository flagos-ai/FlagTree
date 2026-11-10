assert result['schema'] == 'flagmega.cpp-decode-benchmark/v2' and result['status'] == 'completed' and (result['numerical_status'] == 'passed')
assert result['steps'] == task['scenarios'][scenario]['steps'] == steps and result['repeats'] == 3 and (result['warmup_rounds'] == 1)
assert result['layers'] == 64 and result['model_contract'] == dict(revision='5a285a6217024a96be7064d6caf0a06121630ee1', numerical_contract='nncase', num_hidden_layers=64, scope='text-decoder', mode='decode-1', tokens_per_call=1)
assert result['target'] == 'nvidia-sm120' and result['codegen_backend'] == 'cpp' and (result['artifact_identity'] == identity)
assert result['readonly_data'] == side['readonly']['descriptor']
assert all((result[k] is True for k in ('eager_finite', 'eager_state_progression', 'graph_finite', 'graph_trajectory_equal', 'graph_state_progression', 'timed_trajectories_equal', 'repeats_reset_state', 'artifact_unchanged')))
assert result['harness']['sha256'] == 'b8fe232ab68e7f7a43fcd1f7489659345d0b9ba41aed12690a1da0655e62e2a8' and result['harness']['nbytes'] == 21459
assert result['recipe'] == side['recipe'] and result['measurement'] == 'Host wall clock: graph replay, GPU greedy, slot/token feedback, one CPU token return'
boot = read(path.with_suffix('.bootstrap.json'))
assert boot['benchmark_argv'] == result['argv'] and boot['artifact_manifest'] == identity['manifest'] and (boot['semantic_hash'] == semantic) and (boot['plan_sha256'] == plan_hash)
assert [boot[k] for k in ('original_layers', 'entries', 'compute', 'resets')] == [64, 33, 869, 275]
for q in result['golden_identity'].values():
    assert q == golden_pins[q['path']]
q = result['golden_identity']['metadata.json']
checked(q)
golden = read(q['path'])
assert result['golden_checkpoint_sha256'] == golden['checkpoint_sha256'] == task['checkpoint_sha256']
assert result['llama_commit'] == golden['llama_commit'] == task['reference_revision']
assert golden['steps'] == steps and golden['layers'] == 64 and (golden['vocab'] == 248320)
start = max(0, golden['prompt_token_count'] - 1)
assert start == (0 if scenario == 'synthetic32' else 16)
if scenario == 'chinese64':
    assert golden['prompt'] == task['scenarios'][scenario]['prompt'] and golden['prompt_token_count'] == 17
else:
    assert golden['input_tokens'][0] == task['scenarios'][scenario]['token_id'] == 1
prefix = result['prefix']
assert prefix['required'] == 8 and prefix['start_step'] == start and prefix['passed'] and prefix['independent_generation']
assert prefix['actual'] == prefix['golden'] == result['tokens'][start:start + 8] == golden['tokens'][start:start + 8] and len(result['tokens']) == steps
dp = path.with_suffix('.diagnostics.jsonl')
diagnostics = [json.loads(x) for x in raw(dp).splitlines()]
assert diagnostics == result['llama_comparison'] and len(diagnostics) == steps
history = True
for i, row in enumerate(diagnostics):
    consumed = golden['input_tokens'][i] if i <= start else result['tokens'][i - 1]
    assert row['step'] == i and row['input_token'] == consumed and (row['actual_token'] == result['tokens'][i])
    assert row['golden_token'] == golden['tokens'][i] and row['golden_input_token'] == golden['input_tokens'][i]
    history = history and consumed == golden['input_tokens'][i]
    assert row['same_input_history'] is history and row['actual_token'] == row['top5_actual'][0]
rp = path.with_suffix('.rounds.jsonl')
rounds_raw = raw(rp)
rounds = [json.loads(x) for x in rounds_raw.splitlines()]
assert len(rounds) == 4 and rounds[1:] == result['rounds']
for repeat, row in enumerate(rounds):
    finite(row)
    assert row['repeat'] == repeat and row['warmup'] is (repeat == 0)
    assert row['tokens'] == result['tokens'] and row['trajectory_equal'] and (row['sequence_length'] == steps)
    assert row['sample_count'] == len(row['latency_ms']) == steps and all((row[k] == v for k, v in stats(row['latency_ms']).items()))
assert [x['latency_ms'] for x in rounds[1:]] == result['latency_ms']
summary = stats([x for batch in result['latency_ms'] for x in batch])
means = list(map(statistics.mean, result['latency_ms']))
assert all((result[k] == v for k, v in summary.items())) and means == result['round_mean_ms'] and (summary['sample_count'] == 3 * steps)
monitor = read(folder / 'monitor/monitor.json')
telemetry_raw = raw(folder / 'monitor/telemetry.jsonl')
observations = [json.loads(x) for x in telemetry_raw.splitlines()]
policy_raw = raw(folder / 'policy.json')
assert policy_raw == raw(folder / 'monitor/isolation-policy.json')
assert monitor['argv'] == command['target_argv'] == plan['argv'][7:]
lease = boot['lease']
assert lease['status'] == 'verified'
assert lease['capture_pid'] == result['pid'] == monitor['pid'] and lease['holder_pid'] == monitor['monitor_pid'] == execution['monitor_pid']
assert lease['prelaunch_observation_sha256'] == hashlib.sha256(telemetry_raw.splitlines(keepends=True)[0]).hexdigest()
assert lease['prelaunch_summary']['clock'] == monitor['clock'] and lease['boot_id'] == monitor['clock']['boot_id']
assert lease['argv_prefix'] == plan['argv'][:7] and lease['lock_path'] == monitor['gpu_lease']['path'] == '/tmp/flagmega-gpu0.lock'
assert lease['monitor_source'] == workload['monitor_expected']['monitor'] and lease['helper_source'] == helper and (lease['lease_source'] == workload['lease_module']) and (lease['policy'] == pin(folder / 'policy.json'))
iso = isolation.classify_rounds(result, monitor, observations, telemetry_raw=telemetry_raw, policy_raw=policy_raw, rounds_raw=rounds_raw)
assert iso['all_measured_rounds_sampled_clear'] and all((r['eligible_isolation'] and (not r['continuous_absence_proven']) for r in iso['rounds']))
assert monitor['gpu_clear'] and monitor['foreign_cpu_compilation_clear']
save(scenario + '-isolation.json', dict(iso, inputs=dict(result=pin(path), rounds=pin(rp), monitor=pin(folder / 'monitor/monitor.json'), telemetry=pin(folder / 'monitor/telemetry.jsonl'), policy=pin(folder / 'policy.json'), classifier=helper)))
