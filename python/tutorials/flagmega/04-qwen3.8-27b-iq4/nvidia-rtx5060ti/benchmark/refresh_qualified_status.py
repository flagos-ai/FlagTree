"""Refresh canonical status while the NUM-passed fresh pairing is running."""
import argparse
import copy
import json
from pathlib import Path
import shutil
import subprocess
import sys

from reference_identity import file_pin, save_json

W = Path(__file__).resolve().parents[1]
SK = W.parents[4]/'skills/flagmega-developer'
AGENT = 'codex:/root:graph_profiler:measure_r001_qualified'
NAMES = {'history':'performance-vs-iteration', 'milestones':'performance-vs-milestone',
         'backends':'backend-comparison', 'graph':'graph-trace', 'operators':'operator-latency'}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run', type=Path, required=True)
    p.add_argument('--history', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    run, out = a.run.resolve(), a.output.resolve()
    s = json.loads((run/'summary.json').read_text())
    assert len(s['numerical']) == 2 and all(n['numerical_status']=='passed' for n in s['numerical'])
    protocol = json.loads((run/'protocol.json').read_text())
    out.mkdir(parents=True, exist_ok=False)
    (out/'workspace-figures').mkdir()
    shutil.copy2(__file__, out/'refresh_qualified_status.py')
    shutil.copy2(SK/'scripts/render_workspace_figures.py', out/'render_workspace_figures.py')
    shutil.copy2(SK/'scripts/render_kernel_swimlane.py', out/'render_kernel_swimlane.py')
    save_json(out/'status-snapshot.json', s)
    history = json.loads(a.history.read_text())
    old = [r for r in history['rows'] if r.get('point_id')=='r001']
    assert len(old)==1
    rows = [copy.deepcopy(r) for r in history['rows'] if r.get('point_id')!='r001']
    rows.append(dict(label=rows[-1]['label'], x=2, point_id='r001', value=None,
        model_status='Fresh full-model timing running',
        reason='Qualified artifact fresh timing running; old unqualified point superseded',
        main_improvement='H001 warp-local; Q96/GLU104; R4 exact FP32', artifact=protocol['candidate']))
    sources = [out/'status-snapshot.json', run/'protocol.json', a.history.resolve(),
               *[Path(n['result']) for n in s['numerical']],
               *[Path(pin['path']) for pin in history['provenance']]]
    provenance = [file_pin(path) for path in dict.fromkeys(sources)]
    status = dict(build='resource-qualified-review-pending', NUM='passed-own-default-path',
        measurement='fresh-three-way-running', review='pending', milestone_achieved=False,
        candidate_artifact=protocol['candidate'], superseded_primary_rows=old,
        visible_context=['r001 resource admission + default NUM passed; fresh timing running',
            'Artifact cap96-104 155144Z / ae96264431d4; old r001 superseded',
            'Review pending; m000 open; graph/operator instrumented evidence pending'])
    files = []
    for kind, name in NAMES.items():
        template = out/'workspace-figures'/(kind+'.json')
        shutil.copy2(SK/'assets/workspace-figures'/(kind+'.json'), template)
        d = json.loads(template.read_text())
        for key in ['milestones','revision_table','timebase','window_ns','observed_window_ns']:
            d.pop(key, None)
        d.update(synthetic=False, title=name+' | qualified r001 | timing running',
            cohort='RTX 5060 Ti / full64 IQ4_XS / fresh measurement running',
            working_status=status, provenance=provenance,
            uncertainty_method='Sealed history process range; fresh pairing pending',
            template_binding=dict(path=str(template),template_id=kind,sha256=file_pin(template)['sha256'],
                renderer_sha256=file_pin(out/'render_workspace_figures.py')['sha256']))
        if kind=='history':
            d.update(rows=rows, revision_table=rows, milestones=[], unit='tokens/s', direction='higher',
                     metric='Full-model decode throughput', x_labels={'0':'baseline','1':'r000','2':'r001'})
        else:
            reason = 'Fresh three-way timing running' if kind=='backends' else (
                'Milestone open; independent review and remaining gates pending' if kind=='milestones' else
                'Current artifact instrumented capture pending; ordinary timing does not supply inner spans')
            d.update(rows=[dict(label=name, x=0, value=None, reason=reason)], pending_only=True, pending_reason=reason)
            if kind=='milestones':
                d.update(unit='tokens/s', direction='higher', metric='Full-model decode throughput', milestones=[])
        inp = out/(name+'.json'); save_json(inp, d)
        svg = out/(name+'.svg')
        subprocess.run([sys.executable,out/'render_workspace_figures.py',inp,'--output',svg],check=True,
                       stdout=subprocess.DEVNULL)
        files.append(svg)
    readme = out/'readme.md'
    readme.write_text('NUM PASS — independently verified through compiler defaults in both scenarios; '
        'first 10 generated tokens equal fresh llama.cpp golden, finite logits, eager/graph/reset trajectories equal.\n\n'
        'Fresh matched r001 / accepted r000 / llama.cpp measurement is running: 2 independent processes × '
        '3 rounds per backend/scenario, excluded warmup, rotated order, exclusive GPU lease.\n\n'
        f"Current candidate: `{protocol['candidate_directory']}`; artifact.json SHA256 `{protocol['candidate']['sha256']}`.\n\n"
        'The previous unqualified r001 point is superseded and retained unchanged in its original packet/archives. '
        'It is excluded from current comparisons; fresh r001 timing remains pending.\n\n'
        'Acceptance remains with the reviewer. Open gates: in-model independent per-PC audit, full-model '
        'dynamic NCU/placement, K9, K6/K10, B007.\n\n'+
        '\n'.join(f'- [{n}](figures/{n}.svg)' for n in NAMES.values())+'\n')
    source = Path(protocol['candidate_directory']).parent/'sources.json'
    source_pin = dict(path=str(source.relative_to(W)),sha256=file_pin(source)['sha256'])
    pins = [dict(path=str(Path(pin['path']).relative_to(W)),sha256=pin['sha256']) for pin in provenance]+[source_pin]
    for kind, payload in [('figures',files),('readme',[readme])]:
        request = out/(kind+'-sync.json')
        save_json(request, dict(schema='flagmega.working-sync/v1', sync_id=out.name+'-'+kind,kind=kind,
            agent_name='graph profiler',agent_id=AGENT, source_id=run.name, status=status, input_pins=pins,
            source_manifest=source_pin, base_view_sha256=file_pin(W/f'.local/views/root-{kind}.json')['sha256'],
            files=[dict(source=str(f.relative_to(W)), path='figures/'+f.name if kind=='figures' else 'readme.md',
                        sha256=file_pin(f)['sha256']) for f in payload]))
        subprocess.run([sys.executable,SK/'scripts/workspace.py','sync','--work-dir',W,'--agent','graph_profiler',
            '--agent-id',AGENT,'--kind',kind,'--record',request],check=True)
    print(out)


if __name__=='__main__':
    main()
