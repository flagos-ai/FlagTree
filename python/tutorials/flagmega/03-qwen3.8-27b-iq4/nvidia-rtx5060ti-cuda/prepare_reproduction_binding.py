"""Bind fresh reviewed artifact metadata to the unchanged public runtime method.

The --side file retains the original artifact/native/resource schemas and must
come from independent review of this reconstruction. This command does not
issue native review or replace it with hashes of the artifact under test.
"""
import argparse
import json
from pathlib import Path
import sys

from reproduction.provider import REPO, DELIVERY, checked, pin, verify_binding


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--side', type=Path, required=True)
    parser.add_argument('--build', type=Path, required=True, help='Fresh successful compile.py build.json; source freeze must already be independently reviewed')
    parser.add_argument('--golden-synthetic32', type=Path, required=True)
    parser.add_argument('--golden-chinese64', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error('Use a fresh binding path')
    side_pin = pin(args.side)
    build_pin = pin(args.build)
    sources_path = Path(build_pin['path']).parent / 'sources.json'
    binding = dict(schema='flagmega.public-t27-execution-binding/v2', side=json.loads(checked(side_pin)),
        side_input=side_pin, public_repo=str(REPO), public_build=build_pin,
        public_sources=pin(sources_path), goldens={},
        monitor_expected=dict(python_sha256=[pin(Path(sys.executable).resolve())['sha256']],
            monitor=pin(DELIVERY/'reproduction/monitor.py'), helper=pin(DELIVERY/'timing_isolation.py')))
    for name,root in (('synthetic32',args.golden_synthetic32),('chinese64',args.golden_chinese64)):
        root=root.resolve()
        binding['goldens'][name]=dict(root=str(root),pins={f:pin(root/f) for f in ('metadata.json','logits.f32')})
        verify_binding(binding,name)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open('x') as stream:
        json.dump(binding,stream,indent=2);stream.write('\n')
    print(json.dumps(pin(args.output)))


if __name__ == '__main__':
    main()
