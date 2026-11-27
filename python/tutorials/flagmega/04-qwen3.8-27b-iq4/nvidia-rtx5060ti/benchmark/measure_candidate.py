"""NUM first, then fresh rotated candidate/baseline/llama host-token pairing."""
import argparse
import datetime
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import traceback

from compare_backends import execute, SOURCE, BUILD
from reference_identity import file_pin, save_json, validate_build
from timing_isolation import (POLICY_SCHEMA, POLICY_ID, LIMITS, clock_identity,
                              monotonic_ns, validate_round_result, classify_rounds)

B = Path(__file__).resolve().parent
W = B.parent
REPO = W.parents[4]
AGENT = 'codex:/root:graph_profiler:measure_r000'
T95 = 12.706204736


def stats(means):
    mean = statistics.mean(means)
    sd = statistics.stdev(means)
    return dict(mean_ms=mean, process_mean_ms=means, process_min_ms=min(means),
                process_max_ms=max(means), process_spread_ms=max(means)-min(means),
                process_sd_ms=sd, ci95_half_ms=T95*sd/(2**.5))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--candidate', type=Path, required=True)
    p.add_argument('--baseline', type=Path, required=True)
    p.add_argument('--revision', default='r000')
    p.add_argument('--candidate-native-file', type=Path)
    p.add_argument('--candidate-default-path', action='store_true',
                   help='Use the compiler default encoders with no local registration')
    p.add_argument('--baseline-native-file', type=Path)
    p.add_argument('--candidate-status', default='unqualified')
    a = p.parse_args()
    if a.candidate_default_path and a.candidate_native_file:
        p.error('Default path cannot register a candidate native file')
    agent = 'codex:/root:graph_profiler:measure_' + a.revision
    a.output = a.output.resolve()
    a.candidate = a.candidate.resolve()
    a.baseline = a.baseline.resolve()
    a.output.mkdir(parents=True, exist_ok=False)
    task = json.loads((W/'.local/task.yaml').read_text())
    os.environ.update(PYTHONPATH=os.pathsep.join([str(W), str(REPO/'python')]),
                      PYTHONDONTWRITEBYTECODE='1', CUDA_VISIBLE_DEVICES='0',
                      TRITON_CACHE_DIR=str(W/'.local/cache/kernels'))
    import ortools
    assert ortools.__version__ == '9.10.4067'
    uuid = subprocess.check_output(['nvidia-smi', '--query-gpu=uuid', '--format=csv,noheader'], text=True).strip()
    policy = a.output/'policy.json'
    save_json(policy, dict(schema=POLICY_SCHEMA, id=POLICY_ID, limits=LIMITS,
                          clock=clock_identity(), created_ns=str(monotonic_ns()), target_gpu_uuid=uuid))
    order = []
    for scenario in task['workload']['scenarios']:
        backends = ['candidate', 'baseline', 'llama'] if scenario == 'synthetic32' else ['baseline', 'llama', 'candidate']
        for process, seq in [(0, backends), (1, list(reversed(backends)))]:
            order.extend(dict(scenario=scenario, backend=backend, process=process,
                              measurement_id=f'{a.output.name}-{scenario}-{backend}-p{process}') for backend in seq)
    protocol = dict(schema='flagmega.candidate-three-way-protocol/v1', agent_id=agent, revision_id=a.revision,
                    created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                    reserved_before_execution=True, reservation_author='graph profiler',
                    reviewer_reservation_status='pending; no reviewer-authored reservation supplied',
                    boundary_id='host-decode-greedy-token-input-outside-v3', task=file_pin(W/'.local/task.yaml'),
                    candidate=file_pin(a.candidate/'artifact.json'), baseline=file_pin(a.baseline/'artifact.json'),
                    candidate_directory=str(a.candidate), baseline_directory=str(a.baseline),
                    processes=2, measured_rounds=3, warmup_rounds=1, rotated_order=order,
                    num_gate='Both scenarios eager/graph/reset NUM before any ordinary timing; every Triton process rechecks before timing',
                    uncertainty='Per-process mean, sample SD/range, t95(df=1); paired process differences and their t95',
                    primary_history_scenario='synthetic32', primary_history_reason='One full-model revision point; chinese64 remains mandatory comparison in packet/backend view',
                    candidate_qualification_status=a.candidate_status,
                    candidate_native=file_pin(a.candidate_native_file) if a.candidate_native_file else None,
                    baseline_native=file_pin(a.baseline_native_file) if a.baseline_native_file else None,
                    candidate_load_path='compiler-default-no-local-registration' if a.candidate_default_path else 'registered-native',
                    default_encoder=file_pin(REPO/'python/triton/flagmega/codegen/triton/iq4_streaming_renderer.py') if a.candidate_default_path else None,
                    h001_threshold='U95 of paired process latency ratios <= 0.95 in both scenarios; latency reduction >=5%',
                    cohort='ordinary', measured_under_ncu=False, device_uuid=uuid,
                    harness=[file_pin(B/n) for n in ['measure_candidate.py', 'benchmark_model.py', 'llama_performance.cpp',
                             'llama_performance.py', 'llama_golden.py', 'monitor.py', 'timing_isolation.py', 'reference_identity.py']])
    save_json(a.output/'protocol.json', protocol)
    save_json(a.output/'reference-admission.json', validate_build(SOURCE, BUILD))
    numerical, runs = [], []
    def publish(status, **extra):
        save_json(a.output/'summary.json', dict(schema='flagmega.candidate-three-way-summary/v1',
            cohort='ordinary', status=status, numerical=numerical, runs=runs, groups=[],
            performance_accepted=False, reviewer_pending=True, **extra))
    publish('waiting-for-gpu-lease')
    # monitor.py holds the exclusive lease and checks idle before each child.
    for scenario, spec in task['workload']['scenarios'].items():
        golden = a.output/'golden'/scenario
        cmd = [sys.executable, B/'llama_golden.py', '--llama-source', SOURCE, '--llama-build', BUILD,
               '--checkpoint', task['workload']['checkpoint'], '--output', golden,
               '--steps', str(spec['steps']), '--repeats', '1']
        if 'prompt' in spec:
            cmd += ['--prompt', spec['prompt']]
        rc = execute(cmd, a.output/'monitors'/f'golden-{scenario}', policy)
        if rc:
            publish('golden-failed', failure_scenario=scenario)
            raise RuntimeError('Fresh golden failed; raw failure retained')
    for scenario, spec in task['workload']['scenarios'].items():
        result = a.output/'numerical'/scenario/'result.json'
        cmd = [sys.executable, B/'benchmark_model.py', '--artifact', a.candidate, '--golden', a.output/'golden'/scenario,
               '--scenario', scenario, '--steps', str(spec['steps']), '--output', result,
               '--numerical-only']
        if a.candidate_native_file:
            cmd += ['--native-file', a.candidate_native_file]
        elif not a.candidate_default_path:
            cmd += ['--native-module', 'agent_optimizations.iq4_streaming.native']
        monitor = a.output/'monitors'/f'NUM-{scenario}'
        rc = execute(cmd, monitor, policy)
        row = dict(scenario=scenario, exit=rc, result=str(result), monitor=str(monitor), numerical_status='failed')
        admission = json.loads((monitor/'monitor.json').read_text())
        if admission.get('prelaunch_admitted') is False:
            row.update(numerical_status='not-run', failure_kind='prelaunch-isolation-rejected')
        if result.exists():
            r = json.loads(result.read_text())
            row.update(result_pin=file_pin(result), prefix=r.get('prefix'), error=r.get('error'))
            if rc == 0 and r.get('status') == 'numerical-only-passed' and r.get('numerical_status') == 'passed':
                row['numerical_status'] = 'passed'
        numerical.append(row)
        save_json(a.output/'numerical-index.json', numerical)
        publish('numerical-checking')
    if not all(row['numerical_status'] == 'passed' for row in numerical):
        publish('NUM-not-run-isolation-rejected' if any(row['numerical_status']=='not-run' for row in numerical) else 'rejected-NUM', performance_measured=False)
        raise RuntimeError('Candidate NUM failed; no ordinary timing performed')
    publish('NUM-passed-timing-running')
    for row in order:
        row = dict(row)
        scenario, backend = row['scenario'], row['backend']
        spec = task['workload']['scenarios'][scenario]
        run = a.output/'ordinary'/row['measurement_id']
        run.mkdir(parents=True)
        if backend == 'llama':
            result = run/'native/metadata.json'
            cmd = [sys.executable, B/'llama_performance.py', '--llama-source', SOURCE, '--llama-build', BUILD,
                   '--checkpoint', task['workload']['checkpoint'], '--output', run/'native', '--steps', str(spec['steps']),
                   '--repeats', '3', '--measurement-id', row['measurement_id']]
            if 'prompt' in spec:
                cmd += ['--prompt', spec['prompt']]
        else:
            result = run/'result.json'
            cmd = [sys.executable, B/'benchmark_model.py', '--artifact', a.candidate if backend == 'candidate' else a.baseline,
                   '--golden', a.output/'golden'/scenario, '--scenario', scenario, '--steps', str(spec['steps']), '--output', result]
            native_file = a.candidate_native_file if backend == 'candidate' else a.baseline_native_file
            if native_file:
                cmd += ['--native-file', native_file]
            elif backend == 'candidate' and not a.candidate_default_path:
                cmd += ['--native-module', 'agent_optimizations.iq4_streaming.native']
        monitor = a.output/'monitors'/row['measurement_id']
        rc = execute(cmd, monitor, policy)
        row.update(exit=rc, result=str(result), monitor=str(monitor))
        try:
            assert rc == 0, 'Process failed; retained, no replacement selection'
            r = json.loads(result.read_text())
            raw = result.with_suffix('.rounds.jsonl').read_bytes() if backend != 'llama' else None
            validate_round_result(r, rounds_raw=raw)
            g = json.loads((a.output/'golden'/scenario/'metadata.json').read_text())
            if backend == 'llama':
                assert r['boundary_id'] == protocol['boundary_id']
                assert r['tokens'] == g['tokens'] and r['input_tokens'] == g['input_tokens']
                numerical_status = 'passed'
                batches = [r['latency_ms'][i*spec['steps']:(i+1)*spec['steps']] for i in range(3)]
            else:
                assert r['numerical_status'] == 'passed' and r['timed_trajectories_equal'] and r['repeats_reset_state']
                numerical_status = r['numerical_status']
                batches = r['latency_ms']
            mr = json.loads((monitor/'monitor.json').read_text())
            telemetry = (monitor/'telemetry.jsonl').read_bytes()
            isolation = classify_rounds(r, mr, [json.loads(x) for x in telemetry.splitlines()], telemetry_raw=telemetry,
                policy_raw=(monitor/'isolation-policy.json').read_bytes(), rounds_raw=raw)
            save_json(run/'isolation.json', isolation)
            assert isolation['all_measured_rounds_sampled_clear'], 'Sampled round isolation not clear'
            row.update(mean_ms=statistics.mean(x for batch in batches for x in batch),
                       round_mean_ms=[statistics.mean(batch) for batch in batches],
                       numerical_status=numerical_status, isolation_clear=True, result_pin=file_pin(result))
        except Exception as error:
            row.update(audit_error=repr(error), traceback=traceback.format_exc())
        runs.append(row)
        save_json(a.output/'run-index.json', runs)
        print(json.dumps(row, ensure_ascii=False), flush=True)
        if row.get('audit_error'):
            publish('timing-failed-incomplete', failure=row)
            raise RuntimeError('Invalid ordinary run retained; comparison incomplete')
    groups, comparisons = [], []
    for scenario in task['workload']['scenarios']:
        local = {}
        for backend in ['candidate', 'baseline', 'llama']:
            selected = sorted([r for r in runs if r['scenario'] == scenario and r['backend'] == backend], key=lambda r:r['process'])
            group = dict(scenario=scenario, backend=backend, **stats([r['mean_ms'] for r in selected]),
                         measurement_ids=[r['measurement_id'] for r in selected], status='measured-unreviewed', sampled_isolation_clear=True)
            group['bandwidth_utilization'] = task['secondary_metric']['bytes_per_token']/(group['mean_ms']*1e-3*task['secondary_metric']['device_peak_bytes_per_second'])
            groups.append(group)
            local[backend] = group
        delta = stats([c-l for c,l in zip(local['candidate']['process_mean_ms'], local['llama']['process_mean_ms'])])
        ratios = stats([c/b for c,b in zip(local['candidate']['process_mean_ms'], local['baseline']['process_mean_ms'])])
        gain = 1-local['candidate']['mean_ms']/local['baseline']['mean_ms']
        comparisons.append(dict(scenario=scenario, candidate_minus_llama=delta,
            candidate_strictly_beats_llama_beyond_uncertainty=delta['mean_ms']+delta['ci95_half_ms'] < 0,
            candidate_at_least_80_percent_bandwidth=local['candidate']['bandwidth_utilization'] >= .8,
            speedup_vs_baseline=local['baseline']['mean_ms']/local['candidate']['mean_ms'],
            latency_reduction_vs_baseline=gain, gain_vs_r000_percent=100*gain,
            throughput_gain_vs_baseline_percent=100*(local['baseline']['mean_ms']/local['candidate']['mean_ms']-1),
            paired_candidate_over_baseline_ratio=dict(process_ratios=ratios['process_mean_ms'],mean=ratios['mean_ms'],
                ci95_half=ratios['ci95_half_ms'],upper95=ratios['mean_ms']+ratios['ci95_half_ms']),
            h001_at_least_5_percent_point_estimate=gain>=.05,
            h001_at_least_5_percent_beyond_uncertainty=ratios['mean_ms']+ratios['ci95_half_ms']<=.95,
            fresh_g012_strict_latency_threshold_ms=local['llama']['mean_ms']-delta['ci95_half_ms']))
    save_json(a.output/'summary.json', dict(schema='flagmega.candidate-three-way-summary/v1', status='measured-unreviewed',
        cohort='ordinary', candidate_qualification_status=a.candidate_status, revision_id=a.revision,
        numerical=numerical, runs=runs, groups=groups, comparisons=comparisons,
        performance_accepted=False, reviewer_pending=True, reviewer_reservation_status=protocol['reviewer_reservation_status'],
        strictly_beats_llama_both=all(c['candidate_strictly_beats_llama_beyond_uncertainty'] for c in comparisons),
        bandwidth_80_both=all(c['candidate_at_least_80_percent_bandwidth'] for c in comparisons),
        h001_5_percent_both=all(c['h001_at_least_5_percent_beyond_uncertainty'] for c in comparisons)))
    print(json.dumps(dict(groups=groups, comparisons=comparisons), indent=2), flush=True)


if __name__ == '__main__':
    main()
