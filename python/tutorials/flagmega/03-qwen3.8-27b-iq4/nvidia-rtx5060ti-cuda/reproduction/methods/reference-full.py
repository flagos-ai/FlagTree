rr = checked(review['reference_reviews'][Path(binding['binding_root']).name + '-' + scenario])
assert command['steps'] == steps
assert result['schema'] == 'flagmega.llama-performance/v2' and result['steps'] == steps and (result['repeats'] == 3) and (result['layers'] == 64) and (result['vocab'] == 248320)
assert result['checkpoint'] == task['checkpoint'] and result['checkpoint_sha256'] == task['checkpoint_sha256'] and (result['llama_commit'] == task['reference_revision'])
assert result['backend'] == 'CUDA SM120' and result['kv_dtype'] == 'BF16' and (result['intermediate_dtype'] == 'llama.cpp native F32')
assert result['n_batch'] == result['n_ubatch'] == 1 and result['requested_gpu_layers'] == -1 and result['flash_attention'] and (not result['mtp'])
assert result['measurement'] == 'Host steady_clock around decode, GPU greedy and one CPU token return'
assert result['sampling'] == 'Context backend greedy sampler; one CPU token returned, no full-vocabulary download'
assert len(result['tokens']) == len(result['input_tokens']) == steps
golden = read(R / 'reference' / ('golden-' + scenario) / 'metadata.json')
start = max(0, golden['prompt_token_count'] - 1)
assert result['tokens'][start:start + 8] == golden['tokens'][start:start + 8] and result['input_tokens'][:start + 1] == golden['input_tokens'][:start + 1]
if scenario == 'synthetic32':
    assert result['prompt'] is None and result['prompt_token_count'] == 0 and (result['input_tokens'][0] == 1)
else:
    assert result['prompt'] == task['scenarios'][scenario]['prompt'] and result['prompt_token_count'] == 17
assert all((result['input_tokens'][i] == result['tokens'][i - 1] for i in range(start + 1, steps)))
provenance = checked(result['reference_provenance'])
build_record = read(path.parent / 'runner-build.json')
assert provenance['status'] == 'verified' and provenance['returncode'] == 0 and (provenance['runtime_validation'] == 'stable_before_after')
assert provenance['reference_build'] == result['reference_build'] == ref_build['admission'] and provenance['runner_build'] == build_record and (build_record['returncode'] == 0)
checked(provenance['helper'])
assert provenance['helper'] == ref_build['helper']
assert checked(provenance['build_receipt_copy']) == checked(review['current_reference_receipt'])
checked(build_record['log'])
native_stdout = json.loads(checked(provenance['stdout']))
checked(provenance['stderr'])
assert all((result[k] == v for k, v in native_stdout.items()))
before = provenance['runner']
after = provenance['post_runner']
stable = lambda obj: {k: v for k, v in obj.items() if k not in ('ldd_stdout', 'ldd_stderr')}
assert stable(before) == stable(after)
assert before['binary'] == result['runner_binary'] and before['source']['sha256'] == result['runner_sha256']
checked(before['source'])
check_size(before['binary'])
assert not command['inherit_environment'] and before['loader_environment'] == command['loader_environment_expected'] == {'CUDA_VISIBLE_DEVICES': '0'}
assert all((k not in command['environment'] for k in command['forbidden_environment'])) and (not any((k.startswith('GGML_CUDA_') for k in command['environment'])))
libraries = ref_build['admission']['snapshot']['libraries']
assert result['resolved_reference_libraries'] == before['resolved_reference_libraries']
for name, row in before['resolved_reference_libraries'].items():
    assert row == libraries['aliases'][name]
    check_size(row)
assert {x['path'] for x in before['resolved_reference_libraries'].values()} == {x['path'] for x in libraries['objects']}
for run_identity in (before, after):
    assert not run_identity['ldd_stderr'] and 'not found' not in run_identity['ldd_stdout']
    observed = {}
    for ln in run_identity['ldd_stdout'].splitlines():
        mt = re.fullmatch('\\s*(\\S+)\\s+=>\\s+(.+?)\\s+\\(0x[0-9a-fA-F]+\\)\\s*', ln)
        if mt and mt.group(1).startswith(('libllama', 'libggml')):
            name, resolved = mt.groups()
            assert name not in observed
            observed[name] = str(Path(resolved).resolve())
    assert observed == {name: row['path'] for name, row in before['resolved_reference_libraries'].items()}
assert result['compile_command'] == build_record['command'] and result['run_command'] == provenance['run_command']
assert result['compile_command'][0] == ref_build['admission']['snapshot']['toolchain']['CMAKE_CXX_COMPILER']['binary']['path']
assert result['run_command'] == [before['binary']['path'], task['checkpoint'], str(steps), '3', '1'] + (['--prompt', task['scenarios'][scenario]['prompt']] if scenario == 'chinese64' else [])
assert result['cmake_cache_sha256'] == pin(path.parent / 'CMakeCache.txt')['sha256'] == ref_build['admission']['snapshot']['cmake_cache']['sha256']
assert execution['started_utc'] < provenance['started_utc'] < provenance['ended_utc'] < execution['ended_utc']
iso = isolation.classify_rounds(result, monitor, observations, telemetry_raw=telemetry_raw, policy_raw=policy_raw)
assert iso['all_measured_rounds_sampled_clear'] and len(iso['rounds']) == 4 and all((x['eligible_isolation'] and (not x['continuous_absence_proven']) for x in iso['rounds']))
save(scenario + '-isolation.json', dict(iso, inputs=dict(result=pin(path), monitor=pin(folder / 'monitor/monitor.json'), telemetry=pin(folder / 'monitor/telemetry.jsonl'), policy=pin(folder / 'policy.json'), classifier=helper)))
raw_batches = [result['latency_ms'][i * steps:(i + 1) * steps] for i in range(3)]
summary = stats(result['latency_ms'])
means = [statistics.mean(x) for x in raw_batches]
assert result['mean_ms'] == summary['mean_ms'] and result['median_ms'] == summary['median_ms'] and (result['tokens_per_second'] == summary['tokens_per_second'])
warmups.append(dict(measurement_id=measurement, backend=backend, scenario=scenario, process_index=process, repeat=0, timing=result['round_timing']['rounds'][0], excluded_reason='original warmup; original native method retains interval/count but does not emit warmup latencies', warmup_sample_count=steps, latency_ms=None))
scientific = dict(original_reference_runtime_qualified=True, current_build=review['current_reference_receipt'], provenance=result['reference_provenance'], runner=before, reference_review=review['reference_reviews'][Path(binding['binding_root']).name + '-' + scenario], golden8=dict(start_step=start, actual=result['tokens'][start:start + 8], golden=golden['tokens'][start:start + 8], passed=True))
