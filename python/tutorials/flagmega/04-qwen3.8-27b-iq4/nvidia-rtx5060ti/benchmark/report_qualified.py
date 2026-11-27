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
AGENT = 'codex:/root:graph_profiler:measure_r001_qualified'


def audit(run, summary):
    assert summary['status'] == 'measured-unreviewed'
    assert len(summary['runs']) == 12 and len(summary['groups']) == 6
    protocol = json.loads((run/'protocol.json').read_text())
    assert all(file_pin(p['path']) == p for p in protocol['harness'])
    assert protocol['candidate_load_path'] == 'compiler-default-no-local-registration'
    assert file_pin(protocol['default_encoder']['path']) == protocol['default_encoder']
    expected_registers = {e:80 for e in (2,4,6,9,12,14,18,21,24)}
    expected_registers.update({8:104,16:96,20:104})
    resource_checks = []
    def check_resources(raw, result_path):
        selected = {int(r['name'].rsplit('_',1)[1]):r for r in raw['resource_report']
                    if int(r['name'].rsplit('_',1)[1]) in expected_registers}
        assert set(selected) == set(expected_registers)
        for entry, registers in expected_registers.items():
            row = selected[entry]
            assert row['registers_per_thread'] == registers
            assert all(row[k] == 0 for k in ('shared_memory_bytes','stack_frame_bytes',
                'spill_store_bytes','spill_load_bytes','local_memory_bytes','spill_bytes'))
        resource_checks.append(dict(result=file_pin(result_path), selected_entries=selected))
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
        assert raw['artifact'] == protocol['candidate'] and raw.get('native_encoder') == protocol['candidate_native']
        check_resources(raw, path)
        monitor = json.loads((Path(row['monitor'])/'monitor.json').read_text())
        assert monitor['prelaunch_admitted'] and monitor['exit'] == 0
        assert monitor['gpu_lease']['mode'] == 'exclusive'
        assert not any(x.startswith('--native-') for x in monitor['argv'])
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
        if row['backend'] == 'candidate':
            admission = json.loads((monitor/'monitor.json').read_text())
            assert not any(x.startswith('--native-') for x in admission['argv'])
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
            assert raw.get('native_encoder') == protocol['candidate_native' if row['backend']=='candidate' else 'baseline_native']
            if row['backend'] == 'candidate':
                check_resources(raw, path)
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
        own_runtime_resource_checks=resource_checks,
        total_ordinary_samples=sum(r['samples'] for r in checked), runs=checked)
    save_json(run/'raw-audit.json', receipt)
    return receipt


def main():
    p = argparse.ArgumentParser(description='Audit and report the resource-qualified r001 fresh pairing')
    p.add_argument('--run', type=Path, required=True)
    p.add_argument('--report-dir', type=Path, required=True)
    p.add_argument('--superseded-run', type=Path, required=True)
    p.add_argument('--resource-owner', type=Path, required=True)
    a = p.parse_args()
    run, out, old = a.run.resolve(), a.report_dir.resolve(), a.superseded_run.resolve()
    out.mkdir(parents=True, exist_ok=True)
    assert not (out/'response.md').exists(), 'Use a fresh report directory'
    s = json.loads((run/'summary.json').read_text())
    protocol = json.loads((run/'protocol.json').read_text())
    assert protocol['candidate']['sha256'] == 'ae96264431d4f13e1918fbd9cbba180ec5c918c5070c82292f1da2726b77bd32'
    receipt = audit(run, s)
    (out/'default-encoder.py').write_bytes(Path(protocol['default_encoder']['path']).read_bytes())
    assert file_pin(out/'default-encoder.py')['sha256'] == protocol['default_encoder']['sha256']
    resource_raw = a.resource_owner.read_bytes()
    resource = json.loads(resource_raw)
    support = json.loads((run/'runtime-source-pins/manifest.json').read_text())
    for row in support['files']:
        if 'original' in row:
            assert file_pin(row['original']['path']) == row['original']
            assert file_pin(row['snapshot']['path']) == row['snapshot']
    assert file_pin(resource['compiler']['path']) == resource['compiler']
    assert file_pin(resource['ptxas']['path']) == resource['ptxas']
    assert resource['artifact_manifest']['sha256'] == protocol['candidate']['sha256']
    assert resource['resource_admission_passed'] and resource['default_candidate_NUM_passed']
    for pin in [resource['runtime_resource_report'], resource['bitwise'], resource['PC_proof']['result']]:
        assert file_pin(pin['path'])['sha256'] == pin['sha256']
    (out/'resource-owner-snapshot.json').write_bytes(resource_raw)
    supersession = dict(schema='flagmega.primary-result-supersession/v1', revision_id='r001',
        superseded_protocol=file_pin(old/'protocol.json'), superseded_summary=file_pin(old/'summary.json'),
        replacement_protocol=file_pin(run/'protocol.json'), replacement_summary=file_pin(run/'summary.json'),
        reason='Different artifact; old unqualified result retained as history and excluded from current primary point and A/B',
        old_raw_modified=False, r001_primary_point_count=1)
    retained = W/'.local/agents/graph_profiler/measure_r001_qualified/retained-harness/historical-binding.json'
    supersession['retained_original_harness_binding'] = file_pin(retained)
    save_json(out/'supersession.json', supersession)
    lines = ['# r001 resource-qualified candidate — fresh three-way measurement', '',
        '**NUM PASS, independently verified through the default compiler load path in clean processes.** '
        'Both scenarios: first 10 independently generated tokens equal freshly generated llama.cpp golden; '
        'all logits finite; eager/graph/reset trajectories agree. Every measured Triton process repeats NUM '
        'before timing and resets recurrent/KV state before every round.', '',
        '| Scenario | Generation start | Candidate first 10 = fresh golden | NUM |',
        '| --- | ---: | --- | --- |']
    for row in s['numerical']:
        r = json.loads(Path(row['result']).read_text())
        lines.append(f"| {row['scenario']} | {r['prefix']['start_step']} | {r['prefix']['actual']} | PASS |")
    lines += ['', 'Host ms per decode-1 includes GPU greedy and one CPU token return. '
        'Setup/load/JIT/warmup/state reset/prompt injection/full-logit diagnostics are outside timing; '
        '`prepare_input()` is before the llama.cpp start clock. Frozen synthetic32 = 32 steps from token 1; '
        'chinese64 = 64 steps with the frozen Chinese prompt. Each backend/scenario has 2 independent '
        'processes × 3 measured rounds plus one excluded warmup round. Backend order rotates and reverses; '
        'all three backends were remeasured in this session. Uncertainty is 95% t(df=1) from the two process means.', '',
        '| Backend / scenario | p0 ms | p1 ms | Mean ±95% t ms | Spread ms | Process SD ms | Bandwidth |',
        '| --- | ---: | ---: | ---: | ---: | ---: | ---: |']
    labels = {'candidate':'r001 resource-qualified', 'baseline':'accepted r000 remeasured', 'llama':'fresh llama.cpp'}
    for g in s['groups']:
        lines.append(f"| {labels[g['backend']]} / {g['scenario']} | {g['process_mean_ms'][0]:.9f} | "
            f"{g['process_mean_ms'][1]:.9f} | {g['mean_ms']:.9f} ± {g['ci95_half_ms']:.9f} | "
            f"{g['process_spread_ms']:.9f} | {g['process_sd_ms']:.9f} | {100*g['bandwidth_utilization']:.6f}% |")
    lines += ['', '| Scenario | Latency reduction vs r000 | TPS gain vs r000 | Paired ratio U95 | ≥5% point / U95 gate | Strict llama win | ≥80% bandwidth |',
        '| --- | ---: | ---: | ---: | --- | --- | --- |']
    for c in s['comparisons']:
        lines.append(f"| {c['scenario']} | {c['gain_vs_r000_percent']:.6f}% | {c['throughput_gain_vs_baseline_percent']:.6f}% | "
            f"{c['paired_candidate_over_baseline_ratio']['upper95']:.9f} | "
            f"{c['h001_at_least_5_percent_point_estimate']} / {c['h001_at_least_5_percent_beyond_uncertainty']} | "
            f"{c['candidate_strictly_beats_llama_beyond_uncertainty']} | {c['candidate_at_least_80_percent_bandwidth']} |")
        d, ratio = c['candidate_minus_llama'], c['paired_candidate_over_baseline_ratio']
        lines += ['', f"{c['scenario']}: paired r001/r000 ratios {ratio['process_ratios']}, "
            f"mean {ratio['mean']:.9f} ± {ratio['ci95_half']:.9f}; r001 − llama "
            f"{d['mean_ms']:.9f} ± {d['ci95_half_ms']:.9f} ms. Strict win requires the paired difference upper95 < 0; "
            'the H001 falsifier requires paired latency-ratio upper95 ≤0.95.']
    lines += ['', f"Both scenarios: ≥5% whole-model gain beyond uncertainty = **{s['h001_5_percent_both']}**; "
        f"strict fresh llama win beyond uncertainty = **{s['strictly_beats_llama_both']}**; "
        f"≥80% bandwidth = **{s['bandwidth_80_both']}**.", '',
        'Bandwidth = 13,622,248,096 B / (latency_s × 448e9); 80% ⇔ ≤38.008504732 ms/token. '
        'Gain means latency reduction: 100 × (1 − r001/r000). TPS gain is separately reported.', '',
        f"The single r001 primary point belongs to `{Path(protocol['candidate_directory']).parent.name}`; "
        f"artifact.json SHA256 `{protocol['candidate']['sha256']}`. Candidate loading uses compiler defaults "
        'without local native registration. The default encoder is pinned in protocol.json and checked again in the raw audit.', '',
        f"Accepted r000 artifact: `{protocol['baseline_directory']}`; SHA256 `{protocol['baseline']['sha256']}`. "
        'Its frozen native v4 encoder is registered for the r000 comparison; the encoder pin is in protocol.json. '
        'Reference source is the unmodified pinned llama.cpp commit `5ce06fc71f5e938301f7dc51697a59ae308bd6b7`; '
        'prebuilt libraries are from `20261105T031000Z-cpp-decode/rebuild-r1/bin`. Source/config/toolchain/library '
        'identities and actual runner linkage are checked before and after each reference run.', '',
        f"**Superseded retained history:** `{old.name}` measures the different unqualified artifact "
        '`full64-h001-20261116T142200Z` (215.731 / 216.190 ms). Those raw samples, failures, reports and archived '
        'figures remain unchanged. They are excluded from this session’s comparisons and replaced in the canonical '
        'r001 primary point. See supersession.json; the sealed baseline/r000 iteration history remains intact.', '',
        'Resource admission evidence belongs to the kernel optimizer’s separate pinned owner snapshot: '
        'nine projections 80 registers/6 CTA; entry 8 104/4; Q16 96/5; GLU20 104/4. Driver/cuobjdump agree; '
        'spill/local/stack/shared and SASS LDS/STS/LDL/STL/BAR all zero. '
        'Resource admission and measured NUM/performance do not declare reviewer acceptance.', '',
        f"Own-context raw audit passed: {receipt['total_ordinary_samples']} measured decode samples; all measured "
        'rounds sampled clear. All raw samples, excluded warmups, reset trajectories, sidecars, stdout/stderr, '
        'prelaunch/postexit observations and telemetry remain in the packet. GPU work held exclusive '
        '`/tmp/flagmega-gpu0.lock`; ordinary timing was never under ncu. OR-Tools remains 9.10.4067.', '',
        'Retained report-only failures: a report f-string syntax error and three incomplete pending-render attempts '
        '(missing renderer dependency, required model_status, then required x field). Their source snapshots and '
        'logs remain under `.local/agents/graph_profiler/measure_r001_qualified/` and '
        '`.local/reports/r001-qualified-running-20261116T1622Z`, `…1623Z`, `…1623Z-03`. '
        'The corrected pending view was synced by `…1623Z-04`. These CPU publication failures did not alter or '
        'replace any GPU run or raw sample.', '',
        'Still open: in-model per-PC independent audit; full-model dynamic NCU/placement gates; '
        'K9; K6/K10; B007. Independent reviewer acceptance and ledger append remain pending. '
        'm000 is open; the milestone figure has no achieved point. Graph/operator instrumented evidence stays pending.', '',
        f"Raw packet: [{run.name}]({run}/).", '',
        f"Reproduce from this work directory: `/home/sunnycase/miniconda3/envs/flagtree/bin/python "
        f"benchmark/measure_candidate.py --output FRESH --candidate {protocol['candidate_directory']} "
        f"--baseline {protocol['baseline_directory']} --revision r001 --candidate-status resource-qualified-review-pending "
        f"--candidate-default-path --baseline-native-file {protocol['baseline_native']['path']}`.", '']
    report = out/'response.md'
    report.write_text('\n'.join(lines)+'\n')
    save_json(out/'handoff.json', dict(schema='flagmega.graph-profiler-handoff/v1', agent_name='graph profiler',
        agent_id=AGENT, dashboard_row='R001-5', flow_step=10, status=s['status'], NUM='passed-own-default-path',
        candidate_qualification_status='resource-qualified-review-pending', performance_accepted=False,
        reviewer_pending=True, groups=s['groups'], comparisons=s['comparisons'],
        report=file_pin(report), summary=file_pin(run/'summary.json'), protocol=file_pin(run/'protocol.json'),
        raw_audit=file_pin(run/'raw-audit.json'), supersession=file_pin(out/'supersession.json'),
        resource_owner=file_pin(out/'resource-owner-snapshot.json'), r001_primary_points=1,
        runtime_source_manifest=file_pin(run/'runtime-source-pins/manifest.json'),
        open_gates=['in-model independent per-PC audit','full-model dynamic NCU/placement','K9','K6/K10','B007']))
    print(report)


if __name__ == '__main__':
    main()


