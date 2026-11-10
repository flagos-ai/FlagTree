gaps = {}
for scenario in ('synthetic32', 'chinese64'):
    both = {}
    for backend in ('t24', 'llama.cpp'):
        runs = sorted((x for x in all_results if x['scenario'] == scenario and x['backend'] == backend), key=lambda x: x['process_index'])
        assert len(runs) == 2 and [x['process_index'] for x in runs] == [1, 2]
        pm = [x['statistics']['mean_ms'] for x in runs]
        samples = [x['host_ms'] for x in raw_samples if x['backend'] == backend and x['scenario'] == scenario]
        assert len(samples) == decision['original_matrix']['samples_per_backend'][scenario]
        both[backend] = dict(statistics=stats(samples), process_mean_ms=pm, process_mean_distribution=distribution(pm), round_mean_ms=[x for run in runs for x in run['round_mean_ms']], independent_processes=2, measured_rounds=6, excluded_warmup_rounds=2)
    candidate = both['t24']['statistics']['mean_ms']
    reference = both['llama.cpp']['statistics']['mean_ms']
    gaps[scenario] = dict(backends=both, candidate_minus_reference_ms=candidate - reference, candidate_latency_change_pct=(candidate / reference - 1) * 100, speedup_reference_over_candidate=reference / candidate, strict_lower_mean=candidate < reference, paired_process_candidate_minus_reference_ms=[a - b for a, b in zip(both['t24']['process_mean_ms'], both['llama.cpp']['process_mean_ms'])])
strict_win = all((x['strict_lower_mean'] for x in gaps.values()))
assert len(raw_samples) == 1152 and len(all_rounds) == 24 and (len(warmups) == 8) and (len(resource_reports) == 4)
