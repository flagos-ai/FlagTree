"""Publish current r000 figures from audited ordinary data, one primary point."""
import argparse
import datetime
import json
from pathlib import Path
import shutil
import statistics
import subprocess
import sys

from reference_identity import file_pin, save_json

B = Path(__file__).resolve().parent
W = B.parent
SK = W.parents[4]/'skills/flagmega-developer'
AGENT = 'codex:/root:graph_profiler:measure_r000'
NAMES = {'graph':'graph-trace', 'operators':'operator-latency', 'history':'performance-vs-iteration',
         'milestones':'performance-vs-milestone', 'backends':'backend-comparison'}
CHANGE = 'IQ4 R4 streaming at all 12 projection sites; Attention Q owner duplication fixed'


def point(group, *, throughput=False, **extra):
    values = (dict(value=1000 / group['mean_ms'], low=1000 / group['process_max_ms'],
                   high=1000 / group['process_min_ms']) if throughput else
              dict(value=group['mean_ms'], low=group['process_min_ms'], high=group['process_max_ms']))
    return dict(**values, measurement_ids=group['measurement_ids'], status='measured', **extra)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run', type=Path, required=True)
    a = p.parse_args()
    a.run = a.run.resolve()
    stamp = datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    frame = a.run/'renders'/stamp
    frame.mkdir(parents=True)
    (frame/'workspace-figures').mkdir()
    for name in ['render_workspace_figures.py', 'render_kernel_swimlane.py']:
        shutil.copy2(SK/'scripts'/name, frame/name)
    shutil.copy2(__file__, frame/'render_results.py')
    shutil.copy2(B/'render_snapshot.py', frame/'render_snapshot.py')
    summary = json.loads((a.run/'summary.json').read_text())
    protocol = json.loads((a.run/'protocol.json').read_text())
    measured = summary['status'] == 'measured-unreviewed'
    rejected = summary['status'] == 'rejected-NUM'
    numerical_pass = len(summary['numerical']) == 2 and all(r['numerical_status'] == 'passed' for r in summary['numerical'])
    # Completed groups are actual outputs even while another scenario is running.
    # Snapshot the audited index; never modify the live driver summary or IDs.
    if not measured and numerical_pass and (a.run/'run-index.json').exists():
        summary['runs']=json.loads((a.run/'run-index.json').read_text())
        for row in summary['runs']:
            assert row['exit']==0 and row['numerical_status']=='passed' and row['isolation_clear']
            assert row['result_pin']==file_pin(row['result'])
        summary['groups']=[]
        for scenario in ['synthetic32','chinese64']:
            for backend in ['candidate','baseline','llama']:
                rows=sorted([r for r in summary['runs'] if r['scenario']==scenario and r['backend']==backend],key=lambda r:r['process'])
                if len(rows)!=2: continue
                means=[r['mean_ms'] for r in rows];mean=statistics.mean(means)
                summary['groups'].append(dict(scenario=scenario,backend=backend,process_mean_ms=means,mean_ms=mean,
                    process_min_ms=min(means),process_max_ms=max(means),process_spread_ms=max(means)-min(means),
                    ci95_half_ms=12.706204736*statistics.stdev(means)/2**.5,
                    bandwidth_utilization=13622248096/(mean*.001*448e9),measurement_ids=[r['measurement_id'] for r in rows]))
    primary_measured=numerical_pass and any(g['backend']=='candidate' and g['scenario']=='synthetic32' for g in summary['groups'])
    source = Path(protocol['candidate_directory']).parent/'sources.json'
    start_file = W/'.local/baseline/measurements/b006-replication-20261116T081633Z/summary.json'
    start = next(g for g in json.loads(start_file.read_text())['groups'] if g['backend'] == 'triton' and g['scenario'] == 'synthetic32')
    for name, doc in [('summary-snapshot.json',summary), ('protocol-snapshot.json',protocol)]:
        save_json(frame/name, doc)
    provenance = [file_pin(W/'.local/task.yaml'),file_pin(source),file_pin(start_file),
                  file_pin(frame/'summary-snapshot.json'),file_pin(frame/'protocol-snapshot.json')]
    status = dict(build='built-and-verified', NUM='failed' if rejected else 'passed' if numerical_pass else 'pending/running',
        measurement=summary['status'], review='pending', trace='candidate capture pending; historical baseline trace retained in its own packet',
        visible_context=[f"Current candidate r000 R4 / artifact {protocol['candidate']['sha256'][:16]}",
                         f"NUM: {'FAILED; candidate rejected' if rejected else 'passed' if numerical_pass else 'pending/running'}; ordinary: {summary['status']}; review pending",
                         'Host decode-1 + GPU greedy + one CPU token return; all setup/injection/diagnostics excluded',
                         'Fresh same-session three-way pairing; 2 processes x 3 measured rounds; separate instrumented cohort'])
    files = []
    for kind, name in NAMES.items():
        template = frame/'workspace-figures'/f'{kind}.json'
        shutil.copy2(SK/f'assets/workspace-figures/{kind}.json',template)
        d = json.loads(template.read_text())
        for key in ['timebase','window_ns','milestones','revision_table','observed_window_ns']:
            d.pop(key,None)
        d.update(synthetic=False, title=d['title']+' | current r000 R4 | review pending',
            provenance=provenance, working_status=status, cohort='RTX 5060 Ti / full64 IQ4_XS / ordinary host-token',
            uncertainty_method='Bars show range of two process means; t95(df=1) in measurement packet',
            template_binding=dict(path=str(template),template_id=kind,sha256=file_pin(template)['sha256'],
                                  renderer_sha256=file_pin(frame/'render_workspace_figures.py')['sha256']),
            rows=[dict(label=name,value=None,reason='Current candidate capture/measurement pending')],
            pending_only=True,pending_reason='Current candidate data pending; historical baseline evidence is separately retained.')
        if kind in ['history','milestones']:
            baseline = point(start,throughput=True,label='FlagMega / synthetic32',x=0,point_id='baseline',main_improvement='Starting default full64 baseline')
            d.update(pending_only=False,metric='Full-model host decode throughput',unit='tokens/s',direction='higher',rows=[baseline],
                     x_labels={'0':'baseline'},milestones=[])
            if kind == 'history':
                candidate_group = next((g for g in summary['groups'] if g['backend']=='candidate' and g['scenario']=='synthetic32'),None)
                row = dict(label=baseline['label'],x=1,point_id='r000',value=None,main_improvement=CHANGE,
                           model_status='NUM failed; no timing result' if rejected else '整模未测 / '+summary['status'],reason=summary['status'])
                if primary_measured:
                    row = point(candidate_group,throughput=True,label=baseline['label'],x=1,point_id='r000',main_improvement=CHANGE)
                    d['rows'].append(row)
                d.update(revision_table=[baseline,row],x_labels={'0':'baseline','1':'r000'})
        if kind == 'backends' and summary['groups']:
            d.update(pending_only=False,metric='Host decode-1 latency',unit='ms/token',direction='lower',
                     rows=[point(g,label=f"{g['backend']} / {g['scenario']}") for g in summary['groups']])
            for scenario in ['synthetic32','chinese64']:
                for backend in ['candidate','baseline','llama']:
                    if not any(g['scenario']==scenario and g['backend']==backend for g in summary['groups']):
                        d['rows'].append(dict(label=f'{backend} / {scenario}',value=None,reason='Two-process result pending'))
        if kind in ['graph','operators']:
            d['cohort']='Current candidate instrumented cohort: not captured; ordinary timing is separate'
            d['pending_reason']='R4 candidate trace/operator attribution pending. Baseline trace is historical, never substituted for candidate data.'
        save_json(frame/(name+'.json'),d)
        out = frame/(name+'.svg')
        subprocess.run([sys.executable,str(frame/'render_workspace_figures.py'),str(frame/(name+'.json')),'--output',str(out)],check=True,stdout=subprocess.DEVNULL)
        files.append(dict(source=str(out.relative_to(W)),path='figures/'+out.name,sha256=file_pin(out)['sha256']))
    lines = ['# Qwen3.8-27B IQ4_XS / r000 R4 / RTX 5060 Ti','',
        f"Current artifact: `{protocol['candidate_directory']}`; artifact.json SHA256 `{protocol['candidate']['sha256']}`.",'',
        f"NUM: **{status['NUM']}**. Measurement: **{summary['status']}**. Independent review pending.",'',
        'Frozen boundary: host ms/decode-1 including GPU greedy and one CPU token return. Load/JIT/warmup/state reset/prompt injection/full-logit diagnostics are outside timing.', '',
        'Protocol: per backend/scenario, 2 independent processes × 3 rounds, one excluded warmup round; rotated order. Means and uncertainty use process means, 95% t with df=1.', '',
        '| Backend / scenario | Process means ms | Mean ± t95 ms | Spread ms | Bandwidth |',
        '| --- | --- | --- | --- | --- |']
    for g in summary['groups']:
        lines.append(f"| {g['backend']} / {g['scenario']} | {g['process_mean_ms']} | {g['mean_ms']:.9f} ± {g['ci95_half_ms']:.9f} | {g['process_spread_ms']:.9f} | {100*g['bandwidth_utilization']:.6f}% |")
    if not summary['groups']:
        lines.append('| candidate / both scenarios | pending | pending | pending | pending |')
    lines += ['', 'Bandwidth = 13,622,248,096 / (latency_s × 448e9); 80% requires ≤38.008504732 ms/token.','']
    for c in summary.get('comparisons',[]):
        lines.append(f"{c['scenario']}: speedup vs fresh baseline {c['speedup_vs_baseline']:.6f}×; strictly beats fresh llama beyond paired uncertainty: {c['candidate_strictly_beats_llama_beyond_uncertainty']}; ≥80% bandwidth: {c['candidate_at_least_80_percent_bandwidth']}.")
    lines += ['', 'The iteration and milestone figures use decode TPS (tokens/s), higher-is-better, calculated as 1000 / mean host ms/decode-1. The iteration figure has one synthetic32 primary full-model r000 point only after validated timing; chinese64 remains a mandatory result in this packet. m000 remains open, with no achieved milestone point.', '',
        'Candidate graph/operator capture remains pending. Baseline instrumented evidence and ordinary timings retain separate packets and are excluded from the candidate primary history.', '',
        f"Raw packet: [{a.run.name}]({a.run.relative_to(W)}/). All failures and samples are retained; reviewer owns ledger append and acceptance. Reviewer pre-execution reservation receipt remains pending; profiler IDs/order were declared before execution.",'',
        'Reproduce with repo-pinned Python (torch 2.8.0+cu128, ortools 9.10.4067): `benchmark/measure_candidate.py --output FRESH --candidate CANDIDATE --baseline BASELINE`; then `benchmark/render_candidate.py --run RAW_PACKET`.', '',
        'Literal tracked clean-checkout reproduction, candidate attribution, K9 industry-derived standalone comparison and independent review remain pending.','']
    for name in NAMES.values():
        lines.append(f'- [{name}](figures/{name}.svg)')
    readme = frame/'readme.md'
    readme.write_text('\n'.join(lines)+'\n')
    input_pins = [dict(path=str(Path(pin['path']).relative_to(W)),sha256=pin['sha256']) for pin in provenance]
    source_pin = next(pin for pin in input_pins if pin['path']==str(source.relative_to(W)))
    for kind, chosen in [('figures',files), ('readme',[dict(source=str(readme.relative_to(W)),path='readme.md',sha256=file_pin(readme)['sha256'])])]:
        ptr = W/f'.local/views/root-{kind}.json'
        record = dict(schema='flagmega.working-sync/v1',sync_id=f'r000-gp-{stamp}-{kind}',kind=kind,
            agent_name='graph profiler',agent_id=AGENT,source_id=a.run.name,status=status,input_pins=input_pins,
            source_manifest=source_pin,base_view_sha256=file_pin(ptr)['sha256'] if ptr.exists() else None,files=chosen)
        path = frame/(kind+'-sync.json')
        save_json(path,record)
        subprocess.run([sys.executable,str(SK/'scripts/workspace.py'),'sync','--work-dir',str(W),
                        '--agent','graph_profiler','--agent-id',AGENT,'--kind',kind,'--record',str(path)],check=True)
    subprocess.run([sys.executable,str(B/'render_snapshot.py'),'--frame',str(frame),'--output',str(frame/'standalone')],check=True)
    save_json(frame/'render-receipt.json',dict(status='rendered-synced-and-exact-redrawn',
        primary_points=1 if primary_measured else 0,revision_id='r000',files=[file_pin(frame/(name+'.svg')) for name in NAMES.values()]))
    print(frame)


if __name__ == '__main__':
    main()
