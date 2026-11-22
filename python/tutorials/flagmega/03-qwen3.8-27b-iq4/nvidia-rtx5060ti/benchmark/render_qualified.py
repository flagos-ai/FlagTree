"""Render qualified r001 data, replacing the superseded primary point."""
import argparse
import copy
import json
from pathlib import Path
import shutil
import subprocess
import sys
import xml.etree.ElementTree as ET

from reference_identity import file_pin, save_json
from render_candidate import point

W = Path(__file__).resolve().parents[1]
SK = W.parents[4]/'skills/flagmega-developer'
AGENT = 'codex:/root:graph_profiler:measure_r001_qualified'
CHANGE = 'H001 warp-local; Q96/GLU104; R4 exact FP32'
NAMES = {'history':'performance-vs-iteration', 'milestones':'performance-vs-milestone',
         'backends':'backend-comparison', 'graph':'graph-trace', 'operators':'operator-latency'}


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument('--run',type=Path,required=True)
    cli.add_argument('--report',type=Path,required=True)
    cli.add_argument('--output',type=Path,required=True)
    cli.add_argument('--history',type=Path,required=True)
    args = cli.parse_args()
    run, frame = args.run.resolve(),args.output.resolve()
    assert frame.is_relative_to(W)
    summary = json.loads((run/'summary.json').read_text())
    protocol = json.loads((run/'protocol.json').read_text())
    assert summary['status']=='measured-unreviewed' and summary['candidate_qualification_status']=='resource-qualified-review-pending'
    assert json.loads((run/'raw-audit.json').read_text())['status']=='passed-own-context-audit'
    assert len(summary['numerical'])==2 and all(n['numerical_status']=='passed' for n in summary['numerical'])
    history = json.loads(args.history.read_text())
    superseded_rows = [r for r in history['rows'] if r.get('point_id')=='r001']
    assert len(superseded_rows)==1
    history['rows'] = [r for r in history['rows'] if r.get('point_id')!='r001']
    assert sum(r.get('point_id')=='r000' for r in history['rows'])==1
    frame.mkdir(parents=True,exist_ok=False)
    for folder in ['workspace-figures','redraw']:
        (frame/folder).mkdir()
    for name in ['render_workspace_figures.py','render_kernel_swimlane.py','workspace.py']:
        shutil.copy2(SK/'scripts'/name,frame/name)
    shutil.copy2(__file__,frame/'render_qualified.py')
    sources = [run/'summary.json',run/'protocol.json',run/'raw-audit.json',args.history.resolve(),
        args.report.resolve(),W/'.local/task.yaml',Path(protocol['candidate_directory']).parent/'sources.json',
        Path(protocol['baseline_directory']).parent/'sources.json',
        Path(protocol['candidate']['path']),Path(protocol['baseline']['path']),
        args.report.resolve().parent/'default-encoder.py',Path(protocol['baseline_native']['path']),
        args.report.resolve().parent/'supersession.json',run/'runtime-source-pins/manifest.json']
    kernel_snapshot = args.report.resolve().parent/'resource-owner-snapshot.json'
    if kernel_snapshot.exists():
        kernel = json.loads(kernel_snapshot.read_text())
        sources += [kernel_snapshot,Path(kernel['runtime_resource_report']['path']),Path(kernel['PC_proof']['result']['path'])]
    sources += [Path(r['result']) for r in summary['runs']]
    sources += [Path(r['result']).with_suffix('.rounds.jsonl') for r in summary['runs'] if r['backend']!='llama']
    # Retain prior history provenance too; earlier points keep their original sample IDs.
    sources += [Path(p['path']) for p in history['provenance']]
    records=[]
    for source in dict.fromkeys(sources):
        pin = file_pin(source)
        records.append(dict(source=pin,storage='pinned-in-place; retained evidence is not copied'))
    provenance=[r['source'] for r in records]
    status=dict(build='resource-qualified-review-pending',NUM='passed',measurement='measured-unreviewed',review='pending',
        candidate_qualification_status='resource-qualified-review-pending',milestone_achieved=False,milestone_status='open',
        h001_5_percent_both=summary['h001_5_percent_both'],strict_llama_win_both=summary['strictly_beats_llama_both'],
        bandwidth_80_both=summary['bandwidth_80_both'],
        visible_context=['r001 resource admission + default NUM passed; review pending',
            'Fresh three-way ordinary pairing: r001 vs fixed r000 vs llama.cpp; 2 processes x 3 rounds',
            'Artifact cap96-104 155144Z / ae96264431d4; old r001 superseded'])
    status['candidate_artifact'] = protocol['candidate']
    status['superseded_primary_rows'] = superseded_rows
    groups={(g['backend'],g['scenario']):g for g in summary['groups']}
    files,rendered=[] ,[]

    def render(kind,name,rows,**extra):
        template=frame/'workspace-figures'/(kind+'.json')
        if not template.exists():shutil.copy2(SK/'assets/workspace-figures'/(kind+'.json'),template)
        d=json.loads(template.read_text())
        for k in ['milestones','revision_table','timebase','window_ns','observed_window_ns']:
            d.pop(k,None)
        cohort=extra.pop('cohort','RTX 5060 Ti / full64 IQ4_XS / ordinary / resource-qualified-review-pending')
        d.update(synthetic=False,provenance=provenance,working_status=status,rows=rows,
            title=name+' | r001 H001 | review pending',
            cohort=cohort,
            uncertainty_method='Process range whiskers; 95% t(df=1) and paired ratio uncertainty in report',
            template_binding=dict(path=str(template),template_id=kind,sha256=file_pin(template)['sha256'],
                renderer_sha256=file_pin(frame/'render_workspace_figures.py')['sha256']),**extra)
        inp=frame/(name+'.json'); save_json(inp,d)
        out=frame/(name+'.svg')
        for dest in [out,frame/'redraw'/out.name]:
            subprocess.run([sys.executable,frame/'render_workspace_figures.py',inp,'--output',dest],check=True,stdout=subprocess.DEVNULL)
        assert out.read_bytes()==(frame/'redraw'/out.name).read_bytes()
        rendered.append(dict(input=file_pin(inp),output=file_pin(out),exact_redraw=True))
        return out

    rows=copy.deepcopy(history['rows'])
    rows.append(point(groups['candidate','synthetic32'],throughput=True,label=rows[-1]['label'],x=2,
                      point_id='r001',main_improvement=CHANGE,artifact=protocol['candidate']))
    out=render('history',NAMES['history'],rows,metric='Full-model decode throughput',unit='tokens/s',direction='higher',
        revision_table=rows,x_labels={'0':'baseline','1':'r000','2':'r001'},milestones=[],
        aggregation='r001: full64-h001-cap96-104-20261116T155144Z / ae96264431d4; old unqualified r001 superseded. Sealed baseline/r000; gain uses fresh r000.')
    assert len(rows)==len(history['rows'])+1 and sum(r.get('point_id')=='r001' for r in rows)==1
    files.append(out)
    baseline = copy.deepcopy(history['rows'][0])
    label = baseline['label']
    baseline['x'] = 0
    pending = dict(label=label, x=1, point_id='m000', value=None, status='open',
        reason='Milestone open; independent reviewer and remaining gates pending',
        main_improvement=CHANGE)
    out=render('milestones',NAMES['milestones'],[baseline,pending],
        metric='Full-model decode throughput',unit='tokens/s',direction='higher',
        x_labels={'0':'baseline','1':'m000 open'},
        milestones=[dict(milestone_id='m000',revision_id='r001',series=label,point_x=1,
            baseline_point_x=0,baseline_id=baseline['point_id'],baseline_value=baseline['value'],
            value=None,main_improvement=CHANGE,status='open',reason=pending['reason'])])
    files.append(out)
    labels={'candidate':'r001 H001','baseline':'r000 remeasured','llama':'llama.cpp fresh'}
    out=render('backends',NAMES['backends'],[point(g,label=labels[g['backend']]+' / '+g['scenario']) for g in summary['groups']],
        metric='Host decode-1 latency',unit='ms/token',direction='lower');files.append(out)
    for kind in ['graph','operators']:
        files.append(render(kind,NAMES[kind],[dict(label='r001 instrumented trace',value=None,reason='Capture pending in this graph-profiler packet')],
            cohort='RTX 5060 Ti / full64 IQ4_XS / instrumented evidence pending',
            pending_only=True,pending_reason='r001 instrumented trace/operator evidence is separate; ordinary timings do not populate this cohort.'))
    readme=frame/'readme.md'
    readme.write_text(args.report.read_text()+'\n'+'\n'.join(f'- [{name}](figures/{name}.svg)' for name in NAMES.values())+'\n')
    manifest=Path(protocol['candidate_directory']).parent/'sources.json'
    source_pin=dict(path=str(manifest.relative_to(W)),sha256=file_pin(manifest)['sha256'])
    input_pins=[dict(path=str(Path(r['source']['path']).relative_to(W)),sha256=r['source']['sha256']) for r in records]+[source_pin]
    for kind,chosen in [('figures',[dict(source=str(p.relative_to(W)),path='figures/'+p.name,sha256=file_pin(p)['sha256']) for p in files]),
                       ('readme',[dict(source=str(readme.relative_to(W)),path='readme.md',sha256=file_pin(readme)['sha256'])])]:
        ptr=W/f'.local/views/root-{kind}.json'
        request=frame/(kind+'-sync.json')
        save_json(request,dict(schema='flagmega.working-sync/v1',sync_id=run.name+'-'+frame.name+'-'+kind,kind=kind,
            agent_name='graph profiler',agent_id=AGENT,source_id=run.name,status=status,input_pins=input_pins,
            source_manifest=source_pin,base_view_sha256=file_pin(ptr)['sha256'],files=chosen))
        subprocess.run([sys.executable,SK/'scripts/workspace.py','sync','--work-dir',W,'--agent','graph_profiler',
            '--agent-id',AGENT,'--kind',kind,'--record',request],check=True)
    save_json(frame/'render-receipt.json',dict(status='canonical-rendered-synced-exact-redraw',
        revision_id='r001',r001_primary_points=1,cohort='ordinary / resource-qualified-review-pending',instrumented_separate=True,
        history_retained_points=history['rows'], superseded_primary_rows=superseded_rows,source_records=records,panels=rendered,files=[file_pin(p) for p in files],
        root_views=[file_pin(W/f'.local/views/root-{kind}.json') for kind in ['figures','readme']]))
    print(frame)


if __name__=='__main__':
    main()
