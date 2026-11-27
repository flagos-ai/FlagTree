"""Refresh the two TPS working figures from retained ordinary records, without GPU work."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import shutil
import statistics
import subprocess
import sys
import xml.etree.ElementTree as ET

W = Path(__file__).resolve().parents[1]
SK = W.parents[4] / 'skills/flagmega-developer'
AGENT = 'codex:/root:graph_profiler:tps_figures'
CHANGE = 'IQ4 R4 streaming at all 12 projection sites; Attention Q owner duplication fixed'


def pin(path):
    path = Path(path).resolve()
    return dict(path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest())


def save(path, data):
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + '\n')


def audited_groups(run, summary):
    groups, sources = {}, []
    assert summary['cohort'] == 'ordinary' and len(summary['runs']) == 12
    for group in summary['groups']:
        rows = sorted((row for row in summary['runs'] if
                       (row['backend'], row['scenario']) == (group['backend'], group['scenario'])),
                      key=lambda row: row['process'])
        assert len(rows) == 2
        means = []
        for row in rows:
            path = Path(row['result'])
            assert pin(path)['sha256'] == row['result_pin']['sha256'] and row['exit'] == 0
            raw = json.loads(path.read_text())
            samples = raw['latency_ms']
            if row['backend'] != 'llama':
                samples = [value for batch in samples for value in batch]
                sources.append(path.with_suffix('.rounds.jsonl'))
            assert len(samples) == 3 * raw['steps']
            means.append(statistics.mean(samples))
            assert means[-1] == row['mean_ms']
            sources.append(path)
        assert means == group['process_mean_ms'] and statistics.mean(means) == group['mean_ms']
        assert [row['measurement_id'] for row in rows] == group['measurement_ids']
        groups[group['backend'], group['scenario']] = dict(
            backend=group['backend'], scenario=group['scenario'], mean_ms=statistics.mean(means),
            value=1000 / statistics.mean(means), low=1000 / max(means), high=1000 / min(means),
            measurement_ids=group['measurement_ids'])
    assert set(groups) == {(backend, scenario) for backend in ['baseline', 'candidate', 'llama']
                           for scenario in ['synthetic32', 'chinese64']}
    return groups, sources


def point(group, label, x, point_id, improvement):
    return dict(value=group['value'], low=group['low'], high=group['high'],
                measurement_ids=group['measurement_ids'], status='measured', label=label,
                x=x, point_id=point_id, main_improvement=improvement)


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument('--run', type=Path, required=True)
    cli.add_argument('--report-id', required=True)
    args = cli.parse_args()
    assert args.report_id and all(c.isalnum() or c in '-_' for c in args.report_id)
    run = args.run.resolve()
    assert run.is_relative_to(W)
    frame = W / '.local/reports' / args.report_id
    assert not frame.exists(), 'Use a new report ID; existing evidence is immutable'
    summary = json.loads((run / 'summary.json').read_text())
    review_path = W / '.local/milestones/m000/revisions/r000/results/review.json'
    review = json.loads(review_path.read_text())
    assert review['revision_id'] == 'r000' and review['candidate_accepted'] is True
    assert review['milestone_achieved'] is False
    groups, raw_sources = audited_groups(run, summary)
    previous = {kind: json.loads((W / f'.local/views/root-{kind}.json').read_text())
                for kind in ['figures', 'readme']}
    source_manifest = W / previous['figures']['source_manifest']['path']
    frame.mkdir(parents=True)
    (frame / 'workspace-figures').mkdir()
    (frame / 'input-snapshots').mkdir()
    (frame / 'redraw').mkdir()
    for name in ['render_workspace_figures.py', 'render_kernel_swimlane.py', 'workspace.py']:
        shutil.copy2(SK / 'scripts' / name, frame / name)
    shutil.copy2(__file__, frame / 'refresh_tps_figures.py')
    sources = [run / 'summary.json', run / 'protocol.json', run / 'raw-audit.json',
               review_path, W / '.local/task.yaml', source_manifest, W / 'readme.md', *raw_sources]
    snapshots = []
    for source in dict.fromkeys(sources):
        original = pin(source)
        destination = frame / 'input-snapshots' / (original['sha256'] + '-' + source.name)
        shutil.copy2(source, destination)
        snapshots.append(dict(source=original, snapshot=pin(destination)))
    provenance = [item['snapshot'] for item in snapshots]
    status = copy.deepcopy(previous['figures']['status'])
    status.update(measurement='measured', review=review['status'], candidate_accepted=True,
                  milestone_achieved=False, milestone_status='open',
                  figure_unit='tokens/s', figure_direction='higher')
    status['visible_context'] = [
        'r000 measured; candidate_accepted=true (scoped baseline gain); milestone_achieved=false',
        'm000 open / not achieved; acceptance timing boundary and bandwidth rule unchanged',
        'TPS = 1000 / mean host decode-1 latency; retained ordinary records only']
    gains = {scenario: groups['candidate', scenario]['value'] / groups['baseline', scenario]['value'] - 1
             for scenario in ['synthetic32', 'chinese64']}
    receipts, files = [], []
    for kind, name in [('history', 'performance-vs-iteration'), ('milestones', 'performance-vs-milestone')]:
        template = frame / 'workspace-figures' / (kind + '.json')
        maintained_template = SK / 'assets/workspace-figures' / template.name
        shutil.copy2(maintained_template, template)
        panels, inputs, outputs = [], [], []
        scenarios = ['synthetic32'] if kind == 'history' else ['synthetic32', 'chinese64']
        for scenario in scenarios:
            data = json.loads(template.read_text())
            label = 'FlagMega / ' + scenario
            baseline = point(groups['baseline', scenario], label, 0, 'baseline/' + scenario,
                             'Fresh paired default full64 baseline')
            candidate = point(groups['candidate', scenario], label, 1,
                              'r000' if kind == 'history' else 'r000/' + scenario, CHANGE)
            rows = [baseline, candidate]
            if kind == 'milestones':
                rows.append(point(groups['llama', scenario], 'llama.cpp / ' + scenario, 1,
                                  'fresh-llama.cpp/' + scenario, 'Fresh paired reference'))
            data.update(synthetic=False, unit='tokens/s', direction='higher',
                        metric='Full-model host decode throughput (higher is better)',
                        title=('Full-model performance versus revision' if kind == 'history' else
                               'Performance versus milestone') + f' | {scenario} | m000 open / not achieved',
                        cohort='RTX 5060 Ti / full64 IQ4_XS / fresh paired ordinary host-token',
                        provenance=provenance, working_status=status, rows=rows,
                        x_labels={'0': 'baseline', '1': 'r000' if kind == 'history' else 'm000 (open)'},
                        aggregation=f'Inverse of mean latency across two process means; whiskers: inverse process range. TPS gain vs baseline {gains[scenario]:+.6%}; r000 scoped accepted.',
                        uncertainty_method='Invert retained min/max process means and swap bounds; t95 latency uncertainty remains in the raw packet',
                        template_binding=dict(path=str(template), template_id=kind,
                                              sha256=pin(template)['sha256'],
                                              renderer_sha256=pin(frame / 'render_workspace_figures.py')['sha256']),
                        milestones=[dict(milestone_id='m000', revision_id='r000', series=label,
                                         point_x=1, baseline_point_x=0, baseline_id='fresh paired baseline/' + scenario,
                                         baseline_value=baseline['value'], value=candidate['value'],
                                         gain=gains[scenario], main_improvement=CHANGE, status='open')])
            if kind == 'history':
                data['revision_table'] = rows
            inp = frame / f'{name}-{scenario}.json'
            out = frame / f'{name}-{scenario}.svg'
            save(inp, data)
            for renderer, destination in [(SK / 'scripts/render_workspace_figures.py', out),
                                          (frame / 'render_workspace_figures.py', frame / 'redraw' / out.name)]:
                subprocess.run([sys.executable, str(renderer), str(inp), '--output', str(destination)],
                               check=True, stdout=subprocess.DEVNULL)
            assert out.read_bytes() == (frame / 'redraw' / out.name).read_bytes()
            panels.append(ET.fromstring(out.read_text()))
            inputs.append(pin(inp))
            outputs.append(pin(out))
        out = frame / (name + '.svg')
        if len(panels) == 1:
            shutil.copy2(Path(outputs[0]['path']), out)
        else:
            # Compose two unchanged canonical panels, keeping each point and its table row.
            root, detail = panels
            h0, h1 = int(root.get('height')), int(detail.get('height'))
            detail.set('y', str(h0))
            root.append(detail)
            root.set('height', str(h0 + h1))
            root.set('viewBox', f"0 0 {root.get('width')} {h0 + h1}")
            ET.register_namespace('', 'http://www.w3.org/2000/svg')
            out.write_text(ET.tostring(root, encoding='unicode') + '\n')
        receipt = dict(schema='flagmega.tps-figure-receipt/v1', unit='tokens/s', direction='higher',
                       candidate_accepted=True, milestone_achieved=False, milestone_status='open',
                       template=pin(maintained_template), template_snapshot=pin(template),
                       renderer=pin(SK / 'scripts/render_workspace_figures.py'),
                       renderer_snapshot=pin(frame / 'render_workspace_figures.py'),
                       renderer_helper=pin(frame / 'render_kernel_swimlane.py'),
                       wrapper=pin(frame / 'refresh_tps_figures.py'), inputs=inputs, panels=outputs,
                       figure=pin(out), source_records=snapshots, exact_panel_redraw=True,
                       composition='One canonical panel' if len(panels) == 1 else
                                   'Two canonical scenario panels; one table row per plotted point')
        rp = frame / (name + '-receipt.json')
        save(rp, receipt)
        receipts.append(pin(rp))
        files.append(dict(source=str(out.relative_to(W)), path='figures/' + out.name,
                          sha256=pin(out)['sha256']))
    readme = (W / 'readme.md').read_text()
    readme = readme.replace('NUM: **passed**. Measurement: **measured-unreviewed**. Independent review pending.',
                            'NUM: **passed**. Measurement: **measured**. Independent reviewer: '
                            '**candidate_accepted: true** (scoped baseline gain); **milestone_achieved: false**. m000 remains **open / not achieved**.')
    description = ('The iteration and milestone figures use **Decode TPS (tokens/s), higher-is-better**. '
                   'TPS = 1000 / mean host ms/decode-1: average the retained process latency means before inversion; '
                   'invert the process range and swap its bounds. The iteration curve carries exactly one synthetic32 r000 primary point. '
                   'The milestone figure has a scenario panel for each set of three fresh paired results, with one table row per plotted point; '
                   'the candidate is measured and scoped accepted, while m000 is open with no achieved milestone point.\n\n'
                   '| Point | synthetic32 TPS | chinese64 TPS |\n| --- | ---: | ---: |\n')
    for backend, label in [('baseline', 'Fresh paired baseline'), ('candidate', 'r000 candidate'), ('llama', 'Fresh paired llama.cpp')]:
        description += f"| {label} | {groups[backend, 'synthetic32']['value']!r} | {groups[backend, 'chinese64']['value']!r} |\n"
    description += ('\nTPS gains vs fresh baseline: synthetic32 ' + f'{gains["synthetic32"]:+.6%}' +
                    '; chinese64 ' + f'{gains["chinese64"]:+.6%}' + '. Acceptance remains based on the frozen latency/bandwidth rule.\n\n'
                    'Historical render snapshots and frozen standalone redraw fixtures intentionally retain ms/token to preserve sealed evidence byte for byte.')
    old = 'The iteration figure has one synthetic32 primary full-model r000 point only after validated timing; chinese64 remains a mandatory result in this packet. m000 remains open, with no achieved milestone point.'
    assert old in readme, 'The working readme changed; inspect it before a fresh refresh'
    readme = readme.replace(old, description)
    readme = readme.replace('Literal tracked clean-checkout reproduction, candidate attribution, K9 industry-derived standalone comparison and independent review remain pending.',
                            'Literal tracked clean-checkout reproduction, candidate attribution and K9 industry-derived standalone comparison remain separate delivery gates. The scoped r000 gain review is accepted; the milestone is not achieved.')
    readme_path = frame / 'readme.md'
    readme_path.write_text(readme)
    save(frame / 'raw-to-tps.json', dict(groups=list(groups.values()), gains=gains,
                                       aggregation='1000 / mean(two raw process latency means)',
                                       source_records=snapshots))
    input_pins = [dict(path=str(Path(item['snapshot']['path']).relative_to(W)), sha256=item['snapshot']['sha256'])
                  for item in snapshots]
    source_pin = dict(path=str(source_manifest.relative_to(W)), sha256=pin(source_manifest)['sha256'])
    input_pins += [source_pin, *[dict(path=str(Path(item['path']).relative_to(W)), sha256=item['sha256'])
                                for item in receipts]]
    sync_receipts = []
    for kind, chosen in [('figures', files), ('readme', [dict(source=str(readme_path.relative_to(W)),
                                                          path='readme.md', sha256=pin(readme_path)['sha256'])])]:
        request = frame / (kind + '-sync.json')
        save(request, dict(schema='flagmega.working-sync/v1', sync_id=args.report_id + '-' + kind,
                           kind=kind, agent_name='graph profiler', agent_id=AGENT, source_id=run.name,
                           status=status, input_pins=input_pins, source_manifest=source_pin,
                           base_view_sha256=pin(W / f'.local/views/root-{kind}.json')['sha256'],
                           retain_previous=kind == 'figures', files=chosen))
        subprocess.run([sys.executable, str(SK / 'scripts/workspace.py'), 'sync', '--work-dir', str(W),
                        '--agent', 'graph_profiler', '--agent-id', AGENT, '--kind', kind, '--record', str(request)], check=True)
        sync_receipts.append(pin(W / '.local/working' / kind / (args.report_id + '-' + kind) / 'receipt.json'))
    save(frame / 'render-receipt.json', dict(status='synced', figure_receipts=receipts,
                                            sync_receipts=sync_receipts, raw_to_tps=pin(frame / 'raw-to-tps.json'),
                                            readme=pin(readme_path), unit='tokens/s', direction='higher'))
    print(frame)


if __name__ == '__main__':
    main()
