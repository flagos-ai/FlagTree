"""Audit retained H001 samples and report NUM before ordinary results."""
import argparse
import json
import math
from pathlib import Path
import statistics

from measure_candidate import stats
from reference_identity import file_pin, save_json
from timing_isolation import validate_round_result, classify_rounds

W = Path(__file__).resolve().parents[1]
AGENT = 'codex:/root:graph_profiler:measure_r001'


def audit(run, summary):
    assert summary['status'] == 'measured-unreviewed'
    assert len(summary['runs']) == 12 and len(summary['groups']) == 6
    protocol = json.loads((run/'protocol.json').read_text())
    assert all(file_pin(p['path']) == p for p in protocol['harness'])
    numerical = []
    for row in summary['numerical']:
        path = Path(row['result'])
        assert row['exit'] == 0 and file_pin(path) == row['result_pin']
        raw = json.loads(path.read_text())
        golden_path = run/'golden'/row['scenario']/'metadata.json'
        golden = json.loads(golden_path.read_text())
        assert raw['status'] == 'numerical-only-passed' and raw['numerical_status'] == 'passed'
        assert raw['finite'] and raw['graph_trajectory_equal'] and raw['reset_trajectory_equal']
        assert raw['repeats_reset_state'] and raw['prefix']['independent_generation']
        assert raw['prefix']['actual'] == raw['prefix']['golden'] and len(raw['prefix']['actual']) == 10
        assert raw['artifact'] == protocol['candidate'] and raw['native_encoder'] == protocol['candidate_native']
        monitor = json.loads((Path(row['monitor'])/'monitor.json').read_text())
        assert monitor['prelaunch_admitted'] and monitor['exit'] == 0
        assert monitor['gpu_lease']['mode'] == 'exclusive'
        numerical.append(dict(scenario=row['scenario'],result=file_pin(path),golden=file_pin(golden_path),
            first10_equal=True,finite=True,reset_trajectory_equal=True,
            full_golden_trajectory_equal=raw['tokens']==golden['tokens'],
            full_golden_inputs_equal=raw['input_tokens']==golden['input_tokens']))
    assert len(numerical) == 2
    checked = []
    for row in summary['runs']:
        path = Path(row['result'])
        assert file_pin(path) == row['result_pin'] and row['exit'] == 0
        raw = json.loads(path.read_text())
        native = row['backend'] == 'llama'
        rounds = None if native else path.with_suffix('.rounds.jsonl').read_bytes()
        validate_round_result(raw, rounds_raw=rounds)
        samples = raw['latency_ms'] if native else [v for batch in raw['latency_ms'] for v in batch]
        assert len(samples) == 3 * raw['steps']
        assert math.isclose(statistics.mean(samples), row['mean_ms'], abs_tol=1e-12)
        monitor = Path(row['monitor'])
        telemetry = (monitor/'telemetry.jsonl').read_bytes()
        isolation = classify_rounds(raw, json.loads((monitor/'monitor.json').read_text()),
            [json.loads(line) for line in telemetry.splitlines()], telemetry_raw=telemetry,
            policy_raw=(monitor/'isolation-policy.json').read_bytes(), rounds_raw=rounds)
        assert isolation['all_measured_rounds_sampled_clear']
        golden = json.loads((run/'golden'/row['scenario']/'metadata.json').read_text())
        if native:
            assert raw['tokens'] == golden['tokens'] and raw['input_tokens'] == golden['input_tokens']
            assert raw['boundary_id'] == 'host-decode-greedy-token-input-outside-v3'
        else:
            assert raw['numerical_status'] == 'passed' and raw['timed_trajectories_equal']
            assert raw['repeats_reset_state'] and raw['graph_trajectory_equal'] and raw['finite']
            assert raw['prefix']['actual'] == raw['prefix']['golden']
            assert raw['artifact'] == protocol['candidate' if row['backend']=='candidate' else 'baseline']
            assert raw['native_encoder'] == protocol['candidate_native' if row['backend']=='candidate' else 'baseline_native']
        checked.append(dict(measurement_id=row['measurement_id'], samples=len(samples),
            result=file_pin(path), mean_ms=statistics.mean(samples), sampled_isolation_clear=True))
    for g in summary['groups']:
        rows = sorted([r for r in summary['runs'] if (r['backend'],r['scenario']) ==
                       (g['backend'],g['scenario'])], key=lambda r:r['process'])
        recomputed = stats([r['mean_ms'] for r in rows])
        for key, value in recomputed.items():
            assert g[key] == value, key
        assert g['bandwidth_utilization'] == 13622248096/(g['mean_ms']*.001*448e9)
    for c in summary['comparisons']:
        groups = {g['backend']:g for g in summary['groups'] if g['scenario']==c['scenario']}
        candidate, baseline, llama = [groups[k] for k in ['candidate','baseline','llama']]
        delta = stats([a-b for a,b in zip(candidate['process_mean_ms'],llama['process_mean_ms'])])
        assert delta == c['candidate_minus_llama']
        ratio = stats([a/b for a,b in zip(candidate['process_mean_ms'],baseline['process_mean_ms'])])
        assert ratio['mean_ms']+ratio['ci95_half_ms'] == c['paired_candidate_over_baseline_ratio']['upper95']
        assert c['gain_vs_r000_percent'] == 100*(1-candidate['mean_ms']/baseline['mean_ms'])
    receipt = dict(schema='flagmega.profiler-raw-audit/v1', status='passed-own-context-audit',
        independent_reviewer=False, numerical=numerical,measurement_harness_pins_unchanged=True,
        total_ordinary_samples=sum(r['samples'] for r in checked), runs=checked)
    save_json(run/'raw-audit.json', receipt)
    return receipt


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',type=Path,required=True)
    p.add_argument('--report-dir',type=Path,required=True)
    p.add_argument('--failure-run',type=Path,action='append',default=[])
    a = p.parse_args()
    run, out = a.run.resolve(), a.report_dir.resolve()
    out.mkdir(parents=True,exist_ok=True)
    assert not (out/'response.md').exists(), 'Use a fresh report directory'
    s = json.loads((run/'summary.json').read_text())
    protocol = json.loads((run/'protocol.json').read_text())
    num_pass = len(s['numerical'])==2 and all(n['numerical_status']=='passed' for n in s['numerical'])
    lines = ['# R001-5 / flow 10 — H001 graph profiler measurement','',
             '**NUM PASS** — both scenarios: first 10 independently generated tokens exactly match fresh llama.cpp; '
             'finite logits; eager/graph/reset trajectories equal.' if num_pass else
             '**NUM did not pass** — no ordinary timing result is admitted.','',
             '| Scenario | First10 start | Candidate tokens | Golden tokens | NUM |',
             '| --- | ---: | --- | --- | --- |']
    for n in s['numerical']:
        r = json.loads(Path(n['result']).read_text()) if Path(n['result']).exists() else {}
        prefix = r.get('prefix',{})
        lines.append(f"| {n['scenario']} | {prefix.get('start_step','pending')} | {prefix.get('actual','pending')} | {prefix.get('golden','pending')} | {n['numerical_status']} |")
    receipt = None
    historical = []
    if s['status']=='measured-unreviewed' and num_pass:
        receipt = audit(run,s)
        lines += ['', 'Ordinary host ms/decode-1 includes GPU greedy and one CPU token return. '
            'Load/setup/JIT/warmup/state reset/prompt injection/full-logit diagnostics are outside timing. '
            '`prepare_input()` remains before the llama.cpp start clock. Both scenarios use the frozen steps/prompt. '
            'Each backend/scenario has 2 independent processes × 3 rounds, with one excluded warmup round and rotated backend order. '
            '95% t intervals use the two process means (df=1).','',
            '| Backend / scenario | p0 ms | p1 ms | Mean ±95% t ms | Spread ms | Process SD ms | Bandwidth |',
            '| --- | ---: | ---: | ---: | ---: | ---: | ---: |']
        labels = {'candidate':'r001 H001','baseline':'r000 remeasured','llama':'llama.cpp fresh'}
        for g in s['groups']:
            lines.append(f"| {labels[g['backend']]} / {g['scenario']} | {g['process_mean_ms'][0]:.9f} | {g['process_mean_ms'][1]:.9f} | {g['mean_ms']:.9f} ± {g['ci95_half_ms']:.9f} | {g['process_spread_ms']:.9f} | {g['process_sd_ms']:.9f} | {100*g['bandwidth_utilization']:.6f}% |")
        lines += ['', '| Scenario | Latency reduction vs r000 | TPS gain vs r000 | Paired ratio U95 | ≥5% point / U95 gate | Strict fresh llama win | ≥80% bandwidth |',
            '| --- | ---: | ---: | ---: | --- | --- | --- |']
        for c in s['comparisons']:
            lines.append(f"| {c['scenario']} | {c['gain_vs_r000_percent']:.6f}% | {c['throughput_gain_vs_baseline_percent']:.6f}% | {c['paired_candidate_over_baseline_ratio']['upper95']:.9f} | {c['h001_at_least_5_percent_point_estimate']} / {c['h001_at_least_5_percent_beyond_uncertainty']} | {c['candidate_strictly_beats_llama_beyond_uncertainty']} | {c['candidate_at_least_80_percent_bandwidth']} |")
        for c in s['comparisons']:
            d, ratio = c['candidate_minus_llama'], c['paired_candidate_over_baseline_ratio']
            lines += ['', f"{c['scenario']}: paired T001/T000 ratios {ratio['process_ratios']}, mean {ratio['mean']:.9f} ± {ratio['ci95_half']:.9f}; "
                f"candidate − llama {d['mean_ms']:.9f} ± {d['ci95_half_ms']:.9f} ms. "
                f"Fresh paired strict-win comparison bound: <{c['fresh_g012_strict_latency_threshold_ms']:.9f} ms/token (fresh reference mean minus paired-difference uncertainty)."]
        lines += ['', 'Previous pairing thresholds 35.903922031 / 35.823221893 ms/token are historical. '
            'Per this measurement assignment, the fresh pairing supersedes those bounds for this A/B. '
            'The bandwidth requirement stays ≤38.008504732 ms/token. '
            'H001 gain uses the fixed accepted r000 artifact remeasured in this session.']
        for scenario, old_mean, old_ci in [('synthetic32',258.928,.012),('chinese64',259.027,.623)]:
            candidate = next(g for g in s['groups'] if g['backend']=='candidate' and g['scenario']==scenario)
            h = dict(scenario=scenario,accepted_r000_mean_ms=old_mean,accepted_r000_ci95_half_ms=old_ci,
                latency_reduction_percent=100*(1-candidate['mean_ms']/old_mean),
                comparison='Historical point estimate only; the matched comparison uses fresh r000 samples')
            historical.append(h)
            lines += ['', f"Against the supplied accepted r000 historical mean, {scenario} latency reduction is "
                f"{h['latency_reduction_percent']:.6f}% (contextual point estimate; not substituted for the matched uncertainty test)."]
        lines += ['', f"Both-scenario results: H001 ≥5% with U95(T001/T000)≤0.95 = **{s['h001_5_percent_both']}**; "
            f"strictly beats freshly paired llama.cpp beyond uncertainty = **{s['strictly_beats_llama_both']}**; "
            f"≥80% bandwidth = **{s['bandwidth_80_both']}**.", '',
            'Bandwidth = 13,622,248,096 B / (latency_s × 448e9); 80% ⇔ ≤38.008504732 ms/token. '
            'Historical r000 or llama.cpp numbers are excluded from these comparisons. The iteration view retains the sealed earlier baseline/r000 history and adds one r001 synthetic32 primary point.', '',
            f"Own-context raw audit: {receipt['total_ordinary_samples']} measured decode samples; all measured rounds sampled clear. "
            'All warmups, raw samples, reset trajectories, sidecars and telemetry are retained. No ordinary timing under ncu. GPU work uses exclusive `/tmp/flagmega-gpu0.lock`; OR-Tools remains 9.10.4067.']
    failures = []
    for failure in a.failure_run:
        f = failure.resolve()
        assessment = json.loads((f/'failure-assessment.json').read_text()) if (f/'failure-assessment.json').exists() else json.loads((f/'summary.json').read_text())
        failures.append(dict(run=str(f),assessment=assessment))
        lines += ['', f"Retained failed attempt [{f.name}]({f}/): {json.dumps(assessment,ensure_ascii=False)}"]
    attempt = Path(protocol['candidate_directory']).parent
    smoke = attempt/'smoke/smoke.json'
    native_manifest = attempt/'smoke-02/native-manifest.json'
    kernel_path = W/'.local/milestones/m000/revisions/r001/results/kernel-implementation.json'
    kernel_raw = kernel_path.read_bytes()
    kernel = json.loads(kernel_raw)
    assert kernel['artifact']['path'] == protocol['candidate']['path']
    assert kernel['artifact']['sha256'] == protocol['candidate']['sha256']
    for key in ['resource_audit','PC_proof']:
        assert file_pin(kernel[key]['path'])['sha256'] == kernel[key]['sha256']
    kernel_snapshot = out/'kernel-owner-report.json'
    kernel_snapshot.write_bytes(kernel_raw)
    lines += ['', 'Known retained smoke failure: `CodegenError: TIR call \'logits\' has no renderer for '
        'iq4_streaming_projection/stream`. The measured path explicitly registers the frozen candidate '
        '`native.py` family encoders before loading the exported artifact, as pinned in the protocol and '
        'each NUM/timing result. This measures that native registration path; it does not validate loading '
        'without registration or repair the original smoke failure. '
        f"Failure receipt: [{smoke.name}]({smoke}); smoke-02 manifest SHA256 `{file_pin(native_manifest)['sha256']}`.", '',
        f"Kernel optimizer owner report: integrated admission = `{kernel['gates']['integrated_admission']}`; "
        f"primary GLU register gate = `{kernel['gates']['primary_GLU_register_limit']}`; "
        f"zero spill = `{kernel['gates']['zero_spill']}`. "
        f"Pinned owner snapshot: [{kernel_snapshot.name}]({kernel_snapshot}); SHA256 `{file_pin(kernel_snapshot)['sha256']}`. "
        'These resource findings are retained from the separate owner report. Model latency gain measures the net result; '
        'ordinary-region/per-PC/traffic evidence remains pending and is not inferred from model timing.', '',
        'Candidate cohort: **exported-native-unqualified**. Timing and model NUM do not qualify the artifact. '
        'H001 ≥5% above is the final-model result; the separate full ordinary-region gain is pending. '
        'Independent reviewer audit, acceptance, ledger append and remaining reproduction gates remain pending. No acceptance or achieved milestone is declared.', '',
        f"Candidate artifact: `{protocol['candidate_directory']}`; SHA256 `{protocol['candidate']['sha256']}`. "
        f"Fixed r000 comparison artifact: `{protocol['baseline_directory']}`; SHA256 `{protocol['baseline']['sha256']}`. "
        'Pinned official llama.cpp tree: `3acab4ddbcde7a77ef2ab6bf13979f0137aaa86f`; source/config/toolchain/libraries and actual runner linkage checked before/after each reference execution.', '',
        f"Raw packet: [{run.name}]({run}/).", '',
        f"Reproduce from this workspace with `/home/sunnycase/miniconda3/envs/flagtree/bin/python benchmark/measure_candidate.py "
        f"--output FRESH --candidate {protocol['candidate_directory']} --baseline {protocol['baseline_directory']} "
        f"--revision r001 --candidate-status exported-native-unqualified "
        f"--candidate-native-file {protocol['candidate_native']['path']} --baseline-native-file {protocol['baseline_native']['path']}`. "
        'The protocol pins the exact harness and encoder bytes used for this session.', '',
        'Canonical views use retained ordinary data only; graph/operator instrumented cohorts remain separate. '
        'The profiler never changes dashboard.md, .local/task.yaml or the sibling CUDA workspace.', '']
    report = out/'response.md'
    report.write_text('\n'.join(lines)+'\n')
    save_json(out/'handoff.json',dict(schema='flagmega.graph-profiler-handoff/v1',agent_name='graph profiler',
        agent_id=AGENT,dashboard_row='R001-5',flow_step=10,status=s['status'],NUM='passed' if num_pass else 'not-passed',
        candidate_qualification_status='exported-native-unqualified',performance_accepted=False,reviewer_pending=True,
        groups=s['groups'],comparisons=s.get('comparisons',[]),historical_r000_context=historical,
        failures=failures,report=file_pin(report),
        summary=file_pin(run/'summary.json'),protocol=file_pin(run/'protocol.json'),
        raw_audit=file_pin(run/'raw-audit.json') if receipt else None,
        retained_smoke_failure=file_pin(smoke),native_manifest=file_pin(native_manifest),
        kernel_owner_report=file_pin(kernel_snapshot),
        measured_load_path='Frozen native.py family registration before exported artifact load',
        admission_preservation='failed (supplied resource audit)',full_ordinary_region_gain='pending'))
    print(report)


if __name__=='__main__':
    main()
