"""Render maintained canonical figures and synchronize the actual working view."""
import argparse, datetime, hashlib, json, subprocess, sys, shutil, copy, xml.etree.ElementTree as ET
import platform
from pathlib import Path
from reference_identity import file_pin, save_json
B=Path(__file__).resolve().parent;W=B.parent;REPO=W.parents[4];SK=REPO/'skills/flagmega-developer';RENDER=SK/'scripts/render_workspace_figures.py'
AGENT='codex:/root:graph_profiler:fix_baseline'
def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',type=Path,required=True);p.add_argument('--capture',type=Path);a=p.parse_args()
    a.run=a.run.resolve()
    if a.capture:a.capture=a.capture.resolve()
    stamp=datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%S%fZ');frame=a.run/'renders'/stamp;frame.mkdir(parents=True)
    (frame/'workspace-figures').mkdir()
    shutil.copy2(RENDER,frame/'render_workspace_figures.py')
    shutil.copy2(SK/'scripts/render_kernel_swimlane.py',frame/'render_kernel_swimlane.py')
    shutil.copy2(B/'render_snapshot.py',frame/'render_snapshot.py')
    shutil.copy2(__file__,frame/'render_results.py')
    for k in ['graph','operators','history','milestones','backends']:
        shutil.copy2(SK/f'assets/workspace-figures/{k}.json',frame/'workspace-figures'/f'{k}.json')
    summary=json.loads((a.run/'summary.json').read_text()) if (a.run/'summary.json').exists() else None
    corrected=bool(summary and summary.get('supersedes_reference_only'))
    status=dict(build='smoke-verified',measurement='measured-unreviewed' if summary else 'running',NUM='first10/finite/reset passed in all selected ordinary processes; independent review pending' if summary and all(r.get('numerical_status')=='passed' for r in summary['runs'] if r['backend']=='triton' and r.get('exit')==0) else 'running; see per-process receipts',review='pending',trace='pending')
    task=file_pin(W/'.local/task.yaml');source=file_pin(W/'.local/baseline/artifacts/full64-attempt2-20261116T070204Z/sources.json')
    provenance=[task,source];
    if summary:
        save_json(frame/'summary-snapshot.json',summary)
        provenance.append(file_pin(frame/'summary-snapshot.json'))
    status['visible_context']=['Current default Triton full64 baseline / source 9ce51588d98ed3b9 / artifact 5dfd43dfe17fc591',f"Build: {status['build']}; ordinary measurement: {status['measurement']}; reviewer: pending",f"NUM: {status['NUM']}",'Host decode-1 includes GPU greedy and one CPU token return; profile and ordinary cohorts are separate']
    if corrected:status['visible_context'].append('B006 corrected llama input/position outside clock; original reference superseded, candidate unchanged')
    plot_files=[]
    names={'graph':'graph-trace','operators':'operator-latency','history':'performance-vs-iteration','milestones':'performance-vs-milestone','backends':'backend-comparison'}
    groups=summary['groups'] if summary else []
    trace=json.loads((a.capture/'figure-packet.json').read_text()) if a.capture and (a.capture/'figure-packet.json').exists() else None
    for kind,name in names.items():
        template=frame/'workspace-figures'/f'{kind}.json';d=json.loads(template.read_text())
        d.update(synthetic=False,uncertainty_method='Pending capture / matched per-process statistics; no synthetic data',provenance=provenance,cohort='RTX 5060 Ti; current full64 IQ4_XS baseline; ordinary host decode-1 cohorts; independent review pending',working_status=status,
                 title=d['title']+' | current full64 baseline | review pending',template_binding=dict(path=str(template),template_id=kind,sha256=file_pin(template)['sha256'],renderer_sha256=file_pin(RENDER)['sha256']),rows=[dict(label=name,x=0,value=None,reason='Capture / matched ordinary measurements pending')],
                 pending_only=True,pending_reason='No fabricated values: measurements / trace not yet available.')
        for k in ['timebase','window_ns','milestones','revision_table','observed_window_ns']:d.pop(k,None)
        if kind in ['history','milestones','backends'] and groups and all(g.get('mean_ms') for g in groups):
            d.update(pending_only=False,metric='Host decode-1 latency',unit='ms/token',direction='lower',uncertainty_method='min/max of two independent process means (spread; not a confidence interval)',rows=[])
            for g in groups:
                if kind!='backends' and g['backend']!='triton': continue
                ids=g['measurement_ids']
                row=dict(label=f"{g['backend']} / {g['scenario']}",value=g['mean_ms'],low=g['process_min_ms'],high=g['process_max_ms'],status='measured',measurement_ids=ids,
                         main_improvement='Starting default full64 Triton baseline; no optimization')
                if kind!='backends':row.update(x=0,point_id=f"baseline/{g['scenario']}")
                d['rows'].append(row)
            if kind in ['history','milestones']:
                d.update(metric='Host decode throughput',unit='tokens/s',direction='higher')
                for row in d['rows']:
                    row.update(value=1000/row['value'],low=1000/row['high'],high=1000/row['low'])
            if kind=='history':d.update(revision_table=[dict(r) for r in d['rows']],x_labels={'0':'baseline'})
            if kind=='milestones':d.update(milestones=[],x_labels={'0':'baseline'})
        if kind in ['graph','operators'] and trace:
            d.update(trace[kind]);status['trace']='captured-unreviewed'
            cap=json.loads((a.capture/'manifest.json').read_text())
            d['cohort']=f"Instrumented synthetic32 step0 / full64 IQ4_XS / 32 consumer CTAs / measured eager-control overhead +{(cap['instrumentation_overhead_ratio']-1)*100:.1f}% / review pending"
            d['metric']='TIR caller boundary timestamps' if kind=='graph' else d['metric']
            d['uncertainty_method']='One full synthetic32 step0 instrumented trace; measured clock controls; no sampling confidence interval'
            d['provenance']=provenance+trace['provenance']
            if (a.capture/'graph-overhead-control.json').exists():d['provenance'].append(file_pin(a.capture/'graph-overhead-control.json'))
            if kind=='operators':
                d['rows']=[dict(row,label=row['label'].replace('decode_linear_0_', 'GDN/').replace('decode_attention_1_', 'attn/').replace('decode_linear_0.', 'GDN/').replace('decode_attention_1.', 'attn/').replace('.vectorized.compute','').replace('hidden.distributed_boundary_restore','hidden.restore')) for row in d['rows']]
        inp=frame/(name+'.json');out=frame/(name+'.svg');save_json(inp,d)
        subprocess.run([sys.executable,str(frame/'render_workspace_figures.py'),str(inp),'--output',str(out)],check=True,stdout=subprocess.DEVNULL)
        if kind=='graph' and trace and not d.get('pending_only'):
            selected=[event for lane in d['rows'] for event in lane['intervals'] if event.get('cta')==0 and event.get('layer')==0]
            if selected:
                detail=copy.deepcopy(d)
                detail.update(title='Detail: text layer 0, consumer CTA0; same capture; host mapping in separate per-point receipt',scope='observed_role_leaders',
                              leader_capture=dict(health_validated=True,observer='role_local_leader',cta=0,observed_engine_busy=False,full_model_window=False,
                                                  source_pin=provenance[1],clock_pin=trace['provenance'][-1]),
                              observed_window_ns=[min(x['start_ns'] for x in selected),max(x['end_ns'] for x in selected)],
                              timebase=dict(source='PTX globaltimer; device scale verified by CUDA events; host alignment pending',nominal_ns_verified=True,resolution_ns=d['timebase']['resolution_ns'],alignment='pending'),
                              rows=[dict(label=f"S{x['site']} {x['callsite'].split(' -> ')[0].replace('# tir.call ','').replace('decode_linear_0_', '').replace('decode_linear_0.', '').replace('.vectorized.compute','')}",status='captured',intervals=[x]) for x in selected])
                detail.pop('window_ns',None);detail.pop('working_status',None)
                if (a.capture/'host-device-correlation.json').exists():
                    detail['leader_capture']['calibration_from_full_capture']=True
                    detail['timebase'].update(alignment='validated_common',source='PTX globaltimer; calibrated full capture; per-point host bounds in retained correlation receipt')
                dp=frame/'graph-trace-detail.json';ds=frame/'graph-trace-detail.svg';save_json(dp,detail)
                subprocess.run([sys.executable,str(frame/'render_workspace_figures.py'),str(dp),'--output',str(ds)],check=True,stdout=subprocess.DEVNULL)
                # Preserve canonical overview bytes and append another canonical panel.
                shutil.copy2(out,frame/'graph-trace-overview.svg')
                overview=ET.fromstring(out.read_text());panel=ET.fromstring(ds.read_text());h0=int(overview.get('height'));h1=int(panel.get('height'))
                panel.set('y',str(h0));overview.append(panel);overview.set('height',str(h0+h1));overview.set('viewBox',f"0 0 {overview.get('width')} {h0+h1}")
                ET.register_namespace('', 'http://www.w3.org/2000/svg');out.write_text(ET.tostring(overview,encoding='unicode')+'\n')
                save_json(frame/'graph-composition.json',dict(overview=file_pin(frame/'graph-trace-overview.svg'),detail=file_pin(ds),combined=file_pin(out),method='Canonical full overview plus scoped canonical detail from same raw spans; original overview window/endpoints preserved'))
        plot_files.append(dict(source=str(out.relative_to(W)),path='figures/'+out.name,sha256=file_pin(out)['sha256']))
    lines=['# Qwen3.8-27B IQ4_XS / Triton decode-1 / RTX 5060 Ti','','## 1. Contract and current state','','| Field | Actual value / receipt |','| --- | --- |',
           '| Task / reference | [.local/task.yaml](.local/task.yaml); official llama.cpp 5ce06fc71f5e938301f7dc51697a59ae308bd6b7 / tree 3acab4ddbcde7a77ef2ab6bf13979f0137aaa86f |',
           '| Scope | Full64: 48 GDN + 16 full attention; output head + GPU greedy; F32 recurrent state, BF16 KV; no MTP |',
           '| Latest artifact | full64-attempt2-20261116T070204Z; artifact.json SHA256 5dfd43dfe17fc5910b6a399e3c21a0693a4ac303d74bbcd90a3e5d8404f35031 |',
           f'| Working freshness | {stamp}; graph profiler working view; independent review pending |',
           f'| Build / NUM / measurement / review | {status} |',
           '| Primary boundary | Host ms/decode-1 including GPU greedy + one CPU token return. Reset/load/JIT/warmup/injection/full-logit diagnostics excluded |',
           '| Statistics | 2 independent processes × 3 rounds; warmup excluded. Corrected reference rotated S0,C0,C1,S1; candidate original matched order retained. Range is across process means; t95(df=1) uncertainty in raw summary. |',
           '| Milestone/revision | m000 open; r000 plan not implemented/measured. Baseline boundary repair B006/B007; independent follow-up review pending. |',
           '', '## 2. Results and figures','','| Backend / scenario | Process means ms/token | Spread ms | Effective bandwidth |','| --- | --- | --- | --- |']
    for g in groups:
        lines.append(f"| {g['backend']} / {g['scenario']} | {g.get('process_mean_ms')} | {g.get('process_spread_ms','pending')} | {g.get('bandwidth_utilization','pending')} |")
    if summary:
        lines+=['', 'Retained failed attempt: synthetic32 Triton p0 completed samples but sidecar admission failed (missing sample_count). It remains excluded. Matched synthetic results select pairs p1 (llama then Triton) and p2 (Triton then llama); llama p0 is a retained control.']
    if corrected:
        lines+=['', 'B006: old reference timing included batch token/position setup; corrected reference excludes it. Candidate boundary and ordinary baseline numbers are unchanged. All old reference samples/figures remain immutable and are explicitly superseded. Future candidate wins need fresh paired reference timing beyond uncertainty.', '', '| Scenario | Old reference ms/token | Corrected ms/token | Delta ms | t95 half-width ms | Candidate must beat ms/token |', '| --- | --- | --- | --- | --- | --- |']
        for x in summary['changes']:
            lines.append(f"| {x['scenario']} | {x['superseded_ms']:.6f} | {x['corrected_ms']:.6f} | {x['delta_ms']:+.6f} | {x['reference_ci95_half_ms']:.6f} | < {x['candidate_must_be_strictly_below_ms']:.6f} |")
        lines+=['', 'The initial corrected cohort with a 162.565340 ms synthetic sample remains valid noisy history. A prospectively reserved full replication is selected chronologically regardless speed; no fastest-sample selection.'] if summary.get('prior_corrected_noisy') else []
    if not groups:lines.append('| Both backends / both scenarios | running | pending | pending |')
    if trace:
        cap=json.loads((a.capture/'manifest.json').read_text());ranking=json.loads((a.capture/'operator-ranking.json').read_text())
        lines+=['',f"TIR capture [{a.capture.name}]({a.capture.relative_to(W)}/): {cap['captured_invocations']} actual invocation/CTA spans, 44 source-bound callsites, full64 synthetic32 step0. Consumer-only schedule; producer not used. Measured eager-control overhead +{(cap['instrumentation_overhead_ratio']-1)*100:.1f}%; resources 140 reg/thread versus ordinary 113; ordinary dynamic shared 4096 B + static 1024 B = 5120 B, 0 spill. Instrumented shared is separately recorded, not inferred from ordinary. Operator durations are instrumented caller-envelope unions, not ordinary inner-kernel time or engine activity.",'', '| TIR operator | Same-trace envelope union ms |','| --- | --- |']
        if (a.capture/'host-device-correlation.json').exists():
            corr=json.loads((a.capture/'host-device-correlation.json').read_text())
            lines+=[f"Host-device correlation: {corr['endpoint_count']} endpoints, {corr['anchor_count']} execution brackets; max endpoint bound width {corr['max_endpoint_bound_width_ns']} ns under measured affine model. Original historical trace remains device-only; reviewer audit pending."]
        if (a.capture/'graph-overhead-control.json').exists():
            control=json.loads((a.capture/'graph-overhead-control.json').read_text())
            lines+=[f"Separate same-boundary CUDA graph control: ordinary {control['results']['ordinary']['mean_excluding_first_ms']:.6f} ms / instrumented {control['results']['instrumented']['mean_excluding_first_ms']:.6f} ms, overhead +{(control['ratio']-1)*100:.6f}%; excluded from ordinary acceptance."]
        lines += [f"| {r['label']} | {r['value']:.6f} |" for r in ranking[:5]]
    lines+=['','Bandwidth = 13,622,248,096 / (latency_s × 448e9). Target 80% requires ≤38.0085 ms/token. These are working measurements; acceptance requires reviewer.','']
    if summary and summary.get('reproduction'):
        lines+=['Dirty source capsule measurement reproduction: '+summary['reproduction']['status']+'. Full32/64 scenarios; separate reproduction cohort, original baseline unchanged. Literal tracked clean-checkout gate remains pending.','']
    for name in names.values():lines.append(f'- [{name}](figures/{name}.svg)')
    lines+=['',f'Raw packet: [{a.run.name}]({a.run.relative_to(W)}/). The iteration and milestone figures use decode TPS (tokens/s), higher-is-better, calculated as 1000 / mean host ms/decode-1. Figures use maintained renderer and pinned inputs. Graph/operator figures share the same span dataset when available. Ordinary and instrumented cohorts remain separate.','',
            '## 3. Reproduction','',
            '1. Use repo-pinned environment /home/sunnycase/miniconda3/envs/flagtree/bin/python; ortools 9.10.4067. Build via benchmark/build_artifacts.py from frozen GGUF into a fresh trial.',
            '2. Run benchmark/compare_backends.py --output FRESH_DIRECTORY --artifact ARTIFACT_DIRECTORY. It freezes isolation policy, takes /tmp/flagmega-gpu0.lock and stores golden, raw samples, clocks, process lineage and monitoring.',
            '3. Run benchmark/remeasure_reference.py --output FRESH --prior-run ORIGINAL --reservation REVIEWER_RESERVATION to repeat corrected reference. Run benchmark/profile_operators.py for separate instrumented capture; render_results.py --run RAW_PACKET [--capture CAPTURE_PACKET] redraws and syncs figures/readme.',
            '4. benchmark/ scripts are currently untracked. Clean-checkout acceptance remains pending; dirty source capsule reproduction is separately identified. ortools stays at 9.10.4067.',
            '4. Graph profiler owns this measurement packet and working publication; reviewer owns ledger append, independent validation and release.','',
            '## 4. Remaining work','',
            'B006/B007 follow-up reviewer audit and clean-checkout reproduction remain pending. Original B004 review is retained separately. Kernel SOTA counterpart and detailed NCU occupancy evidence belong to later profiling; this assignment has measured neither. Baseline is slower than reference and below the 80% bandwidth target in both scenarios. No milestone gain or release is claimed.','']
    readme=frame/'readme.md';readme.write_text('\n'.join(lines))
    for kind,files in [('figures',plot_files),('readme',[dict(source=str(readme.relative_to(W)),path='readme.md',sha256=file_pin(readme)['sha256'])])]:
        ptr=W/f'.local/views/root-{kind}.json'
        record=dict(schema='flagmega.working-sync/v1',sync_id='gp-'+stamp+'-'+kind,kind=kind,agent_name='graph profiler',agent_id=AGENT,source_id=a.run.name,
                    status=status,base_view_sha256=file_pin(ptr)['sha256'] if ptr.exists() else None,input_pins=[dict(task,path=str(Path(task['path']).relative_to(W))),dict(source,path=str(Path(source['path']).relative_to(W)))],source_manifest=dict(source,path=str(Path(source['path']).relative_to(W))),files=files)
        path=frame/(kind+'-sync.json');save_json(path,record)
        subprocess.run([sys.executable,str(SK/'scripts/workspace.py'),'sync','--work-dir',str(W),'--agent','graph_profiler','--agent-id',AGENT,'--kind',kind,'--record',str(path)],check=True)
    save_json(frame/'render-receipt.json',dict(status='rendered-and-synced',input_provenance=provenance,renderer=file_pin(RENDER),files=[file_pin(frame/(n+'.svg')) for n in names.values()],readme=file_pin(readme)))
    subprocess.run([sys.executable,str(B/'render_snapshot.py'),'--frame',str(frame),'--output',str(frame/'standalone')],check=True)
    print(frame)
if __name__=='__main__':main()
