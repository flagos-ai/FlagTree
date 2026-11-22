"""Freeze render dependencies and redraw using only retained bytes, including historical frames."""
import argparse
import hashlib
import importlib.util
import json
import platform
import shutil
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

NAMES = ['graph-trace', 'operator-latency', 'performance-vs-iteration',
         'performance-vs-milestone', 'backend-comparison']


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def pin(path):
    p = Path(path)
    return dict(path=str(p.resolve()), sha256=sha(p), nbytes=p.stat().st_size)


def freeze(frame, output):
    output.mkdir(parents=True, exist_ok=False)
    for name in ['workspace-figures', 'inputs', 'data']:
        (output/name).mkdir()
    scripts = Path(__file__).resolve().parents[6]/'skills/flagmega-developer/scripts'
    # Explicitly record post-hoc recovery for old packets; never modify originals.
    helper = frame/'render_kernel_swimlane.py'
    recovered = not helper.exists()
    if recovered:
        helper = scripts/'render_kernel_swimlane.py'
    for source in [frame/'render_workspace_figures.py', helper, frame/'render_results.py', Path(__file__)]:
        shutil.copy2(source, output/source.name)
    manifest = dict(schema='flagmega.standalone-render/v1', python=platform.python_version(),
        original_frame=str(frame.resolve()), helper_posthoc_recovered=recovered,
        original_helper_provenance='Not recorded historically; recovered current bytes reproduce exact SVGs' if recovered else 'Captured before render',
        relocation_method='Validate identical snapshot data at local paths; render unchanged original input metadata; preserve original SVG bytes',
        history_label_correction=(frame/'history-metric-label-correction.json').exists(),
        pin_map={}, template_map={}, inputs={}, expected={})
    for name in NAMES+(['graph-trace-detail'] if (frame/'graph-trace-detail.json').exists() else []):
        source = frame/(name+'.json')
        original = json.loads(source.read_text())
        shutil.copy2(source, output/'inputs'/source.name)
        manifest['inputs'][name] = pin(source)
        binding = original['template_binding']
        template = Path(binding['path'])
        assert sha(template) == binding['sha256']
        shutil.copy2(template, output/'workspace-figures'/template.name)
        manifest['template_map'][str(template)] = 'workspace-figures/'+template.name
        for item in original['provenance']:
            path = (source.parent/item['path']).resolve()
            assert sha(path) == item['sha256']
            destination = 'data/'+item['sha256']+'.bin'
            if not (output/destination).exists():
                shutil.copy2(path, output/destination)
            manifest['pin_map'][item['path']] = destination
        if (frame/(name+'.svg')).exists():
            manifest['expected'][name] = pin(frame/(name+'.svg'))
    manifest['files'] = {str(p.relative_to(output)):pin(p) for p in sorted(output.rglob('*')) if p.is_file()}
    (output/'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    replay(output, output/'redraw')


def replay(packet, output):
    manifest = json.loads((packet/'manifest.json').read_text())
    for relative, item in manifest['files'].items():
        assert sha(packet/relative) == item['sha256'], relative
    output.mkdir(parents=True, exist_ok=False)
    sys.path.insert(0, str(packet.resolve()))
    spec = importlib.util.spec_from_file_location('frozen_renderer', packet/'render_workspace_figures.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    def relocate(value):
        if isinstance(value, list):
            return [relocate(x) for x in value]
        if not isinstance(value, dict):
            return value
        result = {k:relocate(v) for k,v in value.items()}
        if result.get('path') in manifest['pin_map']:
            result['path'] = str((packet/manifest['pin_map'][result['path']]).resolve())
        elif result.get('path') in manifest['template_map']:
            result['path'] = str((packet/manifest['template_map'][result['path']]).resolve())
        return result
    for name in manifest['inputs']:
        source = packet/'inputs'/(name+'.json')
        original = json.loads(source.read_text())
        # All runtime validation reads are relocated to the immutable packet.
        module.validate(relocate(original), source)
        result = module.render(original)
        if name == 'performance-vs-iteration' and manifest.get('history_label_correction'):
            result = result.replace('整模 Decode TPS / 状态','整模 host ms/token / 状态')
        (output/(name+'.svg')).write_text(result)
    if 'graph-trace-detail' in manifest['inputs']:
        overview = ET.fromstring((output/'graph-trace.svg').read_text())
        detail = ET.fromstring((output/'graph-trace-detail.svg').read_text())
        h0, h1 = int(overview.get('height')), int(detail.get('height'))
        detail.set('y',str(h0));overview.append(detail)
        overview.set('height',str(h0+h1));overview.set('viewBox',f"0 0 {overview.get('width')} {h0+h1}")
        ET.register_namespace('', 'http://www.w3.org/2000/svg')
        (output/'graph-trace.svg').write_text(ET.tostring(overview,encoding='unicode')+'\n')
    receipts = []
    for name in NAMES:
        target = output/(name+'.svg')
        expected = manifest['expected'][name]['sha256']
        actual = sha(target)
        assert actual == expected, f'{name}: {actual} != {expected}'
        receipts.append(dict(name=name, expected_sha256=expected, actual_sha256=actual, exact=True))
    (output/'receipt.json').write_text(json.dumps(dict(status='exact-frozen-only-redraw', files=receipts),indent=2)+'\n')
    print(str(output))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--frame',type=Path)
    p.add_argument('--packet',type=Path)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    if a.packet:
        replay(a.packet,a.output)
    else:
        freeze(a.frame,a.output)


if __name__=='__main__':
    main()
