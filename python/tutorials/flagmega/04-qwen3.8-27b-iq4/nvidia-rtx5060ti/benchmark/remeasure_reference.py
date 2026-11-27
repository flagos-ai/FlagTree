"""Corrected reference cohort; retain prior candidate and superseded reference raw data."""
import argparse
import json
import os
import statistics
import sys
import traceback
from pathlib import Path

from compare_backends import execute, SOURCE, BUILD
from reference_identity import file_pin, save_json, validate_build
from timing_isolation import (POLICY_SCHEMA, POLICY_ID, LIMITS, clock_identity,
                              monotonic_ns, validate_round_result, classify_rounds)

B = Path(__file__).resolve().parent
W = B.parent
REPO = W.parents[4]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--prior-run', type=Path, required=True)
    p.add_argument('--reservation', type=Path, required=True)
    p.add_argument('--task', type=Path, default=W/'.local/task.yaml')
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=False)
    task = json.loads(a.task.read_text())
    reserved = json.loads(a.reservation.read_text())
    prefix = reserved['reservation_id']
    assert reserved['reserved_before_execution']
    order = [('synthetic32', 0), ('chinese64', 0), ('chinese64', 1), ('synthetic32', 1)]
    ids = [f'{prefix}-{scenario}-llama-p{process}' for scenario, process in order]
    assert set(ids).issubset(reserved['measurement_ids'])
    os.environ.update(PYTHONPATH=str(REPO/'python'), PYTHONDONTWRITEBYTECODE='1',
                      CUDA_VISIBLE_DEVICES='0', TRITON_CACHE_DIR=str(W/'.local/cache/kernels'))
    import subprocess
    uuid = subprocess.check_output(['nvidia-smi', '--query-gpu=uuid', '--format=csv,noheader'], text=True).strip()
    policy = a.output/'policy.json'
    save_json(policy, dict(schema=POLICY_SCHEMA, id=POLICY_ID, limits=LIMITS,
              clock=clock_identity(), created_ns=str(monotonic_ns()), target_gpu_uuid=uuid))
    save_json(a.output/'reference-admission.json', validate_build(SOURCE, BUILD))
    prior = json.loads((a.prior_run/'summary.json').read_text())
    save_json(a.output/'protocol.json', dict(schema='flagmega.reference-boundary-repair/v1',
        task=file_pin(a.task), reservation=file_pin(a.reservation), prior_summary=file_pin(a.prior_run/'summary.json'),
        boundary_id='host-decode-greedy-token-input-outside-v3',
        processes=2, measured_rounds=3, warmup_rounds=1, rotated_order=ids,
        candidate_boundary_changed=False, candidate_policy='Retain original selected ordinary candidate samples; no candidate A/B victory claim from this repair',
        uncertainty='Two process means; sample SD, min/max and 95% t(df=1), not per-token pseudoreplication',
        cohort='ordinary-corrected-reference', measured_under_ncu=False, device_uuid=uuid,
        harness=[file_pin(B/n) for n in ['remeasure_reference.py','llama_performance.cpp','llama_performance.py','monitor.py','timing_isolation.py','reference_identity.py']]))
    runs = []
    for (scenario, process), mid in zip(order, ids):
        spec = task['workload']['scenarios'][scenario]
        run = a.output/'ordinary'/mid
        run.mkdir(parents=True)
        cmd = [sys.executable, B/'llama_performance.py', '--llama-source', SOURCE,
               '--llama-build', BUILD, '--checkpoint', task['workload']['checkpoint'],
               '--output', run/'native', '--steps', str(spec['steps']), '--repeats', '3', '--measurement-id', mid]
        if 'prompt' in spec:
            cmd += ['--prompt', spec['prompt']]
        monitor = a.output/'monitors'/mid
        rc = execute(cmd, monitor, policy)
        result = run/'native/metadata.json'
        row = dict(scenario=scenario, backend='llama', process=process, measurement_id=mid,
                   exit=rc, result=str(result), monitor=str(monitor), cohort='ordinary-corrected-reference')
        if result.exists():
            try:
                r = json.loads(result.read_text())
                validate_round_result(r)
                assert r['boundary_id'] == 'host-decode-greedy-token-input-outside-v3'
                assert r['checkpoint_sha256'] == task['workload']['checkpoint_sha256']
                assert len(r['warmup_latency_ms']) == spec['steps']
                golden = a.prior_run/'golden'/scenario/'metadata.json'
                g = json.loads(golden.read_text())
                assert r['input_tokens'] == g['input_tokens'] and r['tokens'] == g['tokens']
                mr = json.loads((monitor/'monitor.json').read_text())
                telemetry = (monitor/'telemetry.jsonl').read_bytes()
                isolation = classify_rounds(r, mr, [json.loads(x) for x in telemetry.splitlines()],
                    telemetry_raw=telemetry, policy_raw=(monitor/'isolation-policy.json').read_bytes())
                save_json(run/'isolation.json', isolation)
                row.update(mean_ms=statistics.mean(r['latency_ms']),
                    round_mean_ms=[statistics.mean(r['latency_ms'][i*spec['steps']:(i+1)*spec['steps']]) for i in range(3)],
                    result_pin=file_pin(result), golden=file_pin(golden), golden_trajectory_equal=True,
                    numerical_status='passed', isolation_clear=isolation['all_measured_rounds_sampled_clear'])
                assert rc == 0 and row['isolation_clear']
            except Exception as e:
                row.update(audit_error=repr(e), traceback=traceback.format_exc())
        runs.append(row)
        save_json(a.output/'run-index.json', runs)
        print(json.dumps(row), flush=True)
        if 'audit_error' in row or 'mean_ms' not in row:
            raise RuntimeError('Invalid measurement retained; use reviewer-reserved contingency before retry')
    groups = [dict(g, retained_original_candidate=True) for g in prior['groups'] if g['backend']=='triton']
    changes = []
    for scenario in task['workload']['scenarios']:
        rows = [r for r in runs if r['scenario']==scenario]
        means = [r['mean_ms'] for r in rows]
        mean = statistics.mean(means)
        spread = max(means)-min(means)
        ci = 12.706204736*statistics.stdev(means)/(2**.5)
        utilization = task['secondary_metric']['bytes_per_token']/(mean*1e-3*task['secondary_metric']['device_peak_bytes_per_second'])
        group = dict(backend='llama', scenario=scenario, mean_ms=mean, process_mean_ms=means,
            process_min_ms=min(means), process_max_ms=max(means), process_spread_ms=spread,
            process_sd_ms=statistics.stdev(means), ci95_half_ms=ci, bandwidth_utilization=utilization,
            measurement_ids=[r['measurement_id'] for r in rows], aggregate_measurement_id=f'{prefix}-{scenario}-llama',
            status='measured-unreviewed', sampled_isolation_clear=True)
        groups.append(group)
        old = next(g for g in prior['groups'] if g['scenario']==scenario and g['backend']=='llama')
        changes.append(dict(scenario=scenario, superseded_ms=old['mean_ms'], corrected_ms=mean,
            delta_ms=mean-old['mean_ms'], delta_percent=100*(mean/old['mean_ms']-1),
            candidate_must_be_strictly_below_ms=mean, reference_ci95_half_ms=ci,
            secondary_target_ms=task['secondary_metric']['bytes_per_token']/(.8*task['secondary_metric']['device_peak_bytes_per_second'])*1000,
            win_rule='Future candidate requires freshly paired reference; gap must exceed paired uncertainty; this repair retains unchanged historical candidate'))
    candidate_runs = [r for r in prior['runs'] if r['backend']=='triton']
    save_json(a.output/'summary.json', dict(schema='flagmega.matched-summary/v1', cohort='ordinary-boundary-corrected',
        groups=groups, runs=candidate_runs+runs, changes=changes, performance_accepted=False, reviewer_pending=True,
        supersedes_reference_only=file_pin(a.prior_run/'summary.json'), original_samples_preserved=True))
    print(json.dumps(changes, indent=2), flush=True)


if __name__ == '__main__':
    main()
