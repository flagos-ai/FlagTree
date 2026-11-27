"""Independently recompute raw ordinary statistics and write the profiler handoff."""
import argparse
import json
import math
from pathlib import Path
import statistics
import sys

from reference_identity import file_pin, save_json
from timing_isolation import validate_round_result, classify_rounds

W = Path(__file__).resolve().parents[1]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run', type=Path, required=True)
    p.add_argument('--report-dir', type=Path, required=True)
    a = p.parse_args()
    a.run = a.run.resolve()
    a.report_dir = a.report_dir.resolve()
    a.report_dir.mkdir(parents=True,exist_ok=True)
    s = json.loads((a.run/'summary.json').read_text())
    protocol = json.loads((a.run/'protocol.json').read_text())
    assert s['status']=='measured-unreviewed' and len(s['runs'])==12
    checked = []
    for row in s['runs']:
        path = Path(row['result'])
        r = json.loads(path.read_text())
        assert row['result_pin']==file_pin(path) and row['exit']==0
        native = row['backend']=='llama'
        raw = None if native else path.with_suffix('.rounds.jsonl').read_bytes()
        validate_round_result(r,rounds_raw=raw)
        samples = r['latency_ms'] if native else [v for batch in r['latency_ms'] for v in batch]
        assert len(samples)==3*r['steps'] and math.isclose(statistics.mean(samples),row['mean_ms'],abs_tol=1e-12)
        m = Path(row['monitor'])
        telemetry = (m/'telemetry.jsonl').read_bytes()
        iso = classify_rounds(r,json.loads((m/'monitor.json').read_text()),[json.loads(x) for x in telemetry.splitlines()],
            telemetry_raw=telemetry,policy_raw=(m/'isolation-policy.json').read_bytes(),rounds_raw=raw)
        assert iso['all_measured_rounds_sampled_clear']
        checked.append(dict(measurement_id=row['measurement_id'],measured_samples=len(samples),
                            mean_ms=statistics.mean(samples),result=file_pin(path),sampled_isolation_clear=True))
    for g in s['groups']:
        rows = sorted([r for r in s['runs'] if r['scenario']==g['scenario'] and r['backend']==g['backend']],key=lambda r:r['process'])
        means = [r['mean_ms'] for r in rows]
        assert means==g['process_mean_ms'] and math.isclose(statistics.mean(means),g['mean_ms'],abs_tol=1e-12)
        assert math.isclose(12.706204736*statistics.stdev(means)/(2**.5),g['ci95_half_ms'],abs_tol=1e-12)
        assert math.isclose(13622248096/(g['mean_ms']*.001*448e9),g['bandwidth_utilization'],abs_tol=1e-12)
    audit = dict(schema='flagmega.profiler-raw-audit/v1',status='passed-own-context-audit',
        independent_reviewer=False,total_ordinary_samples=sum(r['measured_samples'] for r in checked),runs=checked)
    save_json(a.run/'raw-audit.json',audit)
    lines = ['# R000-6 / flow 10 — R4 candidate measurement','',
        '**NUM PASS** — synthetic32 and chinese64: first 10 independently generated decode tokens exactly equal the fresh llama.cpp golden; finite logits; eager/graph trajectories equal; reset repeat trajectory equal. Each ordinary Triton process revalidated NUM and reset/timed trajectories.', '',
        '| Scenario | Candidate first10 | Golden first10 |', '| --- | --- | --- |']
    for n in s['numerical']:
        lines.append(f"| {n['scenario']} (start {n['prefix']['start_step']}) | {n['prefix']['actual']} | {n['prefix']['golden']} |")
    lines += ['', 'All numbers below are ordinary host ms/decode-1 including GPU greedy and one CPU token return. Setup/load/JIT/warmup/reset/prompt injection/full-logit diagnostics are outside timing. `prepare_input()` remains before the native start clock. Each backend/scenario: 2 independent processes × 3 rounds; one warmup excluded; order prospectively declared in protocol.json.', '',
        '| Backend / scenario | p0 ms | p1 ms | Mean ±95% t ms (df=1) | Spread ms | Process SD ms | Bandwidth |',
        '| --- | --- | --- | --- | --- | --- | --- |']
    for g in s['groups']:
        lines.append(f"| {g['backend']} / {g['scenario']} | {g['process_mean_ms'][0]:.9f} | {g['process_mean_ms'][1]:.9f} | {g['mean_ms']:.9f} ± {g['ci95_half_ms']:.9f} | {g['process_spread_ms']:.9f} | {g['process_sd_ms']:.9f} | {100*g['bandwidth_utilization']:.6f}% |")
    lines += ['', '| Scenario | Speedup vs fresh baseline | Paired candidate − llama ms, ±95% t | Strict llama win beyond uncertainty | ≥80% bandwidth |',
              '| --- | --- | --- | --- | --- |']
    for c in s['comparisons']:
        d=c['candidate_minus_llama']
        lines.append(f"| {c['scenario']} | {c['speedup_vs_baseline']:.6f}× | {d['mean_ms']:.9f} ± {d['ci95_half_ms']:.9f} | {'yes' if c['candidate_strictly_beats_llama_beyond_uncertainty'] else 'no'} | {'yes' if c['candidate_at_least_80_percent_bandwidth'] else 'no'} |")
    lines += ['',f"Both-scenario conditions: strictly beats freshly paired llama.cpp beyond uncertainty = **{s['strictly_beats_llama_both']}**; reaches ≥80% bandwidth = **{s['bandwidth_80_both']}**. This is a measurement report; reviewer owns acceptance.", '',
        'Utilization = 13,622,248,096 B / (latency_s × 448e9). 80% ⇔ ≤38.008504732 ms/token. Earlier paired reference numbers were not reused. Fresh baseline was remeasured in the same session; original baseline remains the invariant starting history point.', '',
        f"Candidate artifact SHA256: `{protocol['candidate']['sha256']}`. Original R4 artifact stayed unchanged. Pinned llama source tree: `3acab4ddbcde7a77ef2ab6bf13979f0137aaa86f`; rebuild-r1 prebuilt official libraries were verified in this session. Actual GPU UUID: `{protocol['device_uuid']}`.", '',
        f"Raw sample audit passed for {audit['total_ordinary_samples']} measured ordinary decode calls across 12 processes; all sampled round isolation checks clear. GPU work serialized under `/tmp/flagmega-gpu0.lock`; no ordinary run under ncu. Warmup samples are retained separately. OR-Tools remains 9.10.4067.", '',
        'Retained failure: r000-r4-20261116T103400Z was rejected before launching a GPU child because optimizer PID 1310972 remained visible just after lease release. No candidate NUM failure and no ordinary samples were produced. Full telemetry retained; a fresh attempt began after the card became idle.', '',
        'Canonical figures/root readme updated through workspace sync, with immutable prior snapshots. Exactly one r000 primary synthetic32 full-model point; chinese64 stays in backend comparison and this packet. No achieved milestone point. Candidate graph/operator instrumentation remains pending; baseline instrumented figures remain historical evidence in their own packet and do not enter ordinary history.', '',
        'Pending: independent reviewer audit/ledger append, reviewer pre-execution reservation receipt (profiler IDs/order were declared prospectively), candidate attribution, K9 standalone industry-derived comparison, literal tracked clean-checkout reproduction. No acceptance is declared.', '',
        f"Raw packet: [{a.run.name}]({a.run}/). Figures: [performance-vs-iteration]({W/'figures/performance-vs-iteration.svg'}), [backend-comparison]({W/'figures/backend-comparison.svg'}).", '',
        f"Reproduction: `{sys.executable} benchmark/measure_candidate.py --output FRESH --candidate {protocol['candidate_directory']} --baseline {protocol['baseline_directory']}`. Render via `benchmark/render_candidate.py --run RAW_PACKET`; raw audit/report via `benchmark/report_candidate.py --run RAW_PACKET --report-dir REPORT_DIR`.",'']
    report = a.report_dir/'response.md'
    if report.exists(): raise RuntimeError('Retain existing report; use a fresh report directory')
    report.write_text('\n'.join(lines)+'\n')
    handoff = dict(schema='flagmega.graph-profiler-handoff/v1',agent_name='graph profiler',agent_id='codex:/root:graph_profiler:measure_r000',
        dashboard_row='R000-6',flow_step=10,status='measured-unreviewed',NUM='passed',groups=s['groups'],comparisons=s['comparisons'],
        strictly_beats_llama_both=s['strictly_beats_llama_both'],bandwidth_80_both=s['bandwidth_80_both'],
        performance_accepted=False,reviewer_pending=True,report=file_pin(report),summary=file_pin(a.run/'summary.json'),
        protocol=file_pin(a.run/'protocol.json'),raw_audit=file_pin(a.run/'raw-audit.json'),
        root_views=[file_pin(W/f'.local/views/root-{kind}.json') for kind in ['figures','readme']])
    save_json(a.report_dir/'handoff.json',handoff)
    print(report)


if __name__=='__main__': main()
