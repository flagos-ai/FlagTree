"""Run a vendor-neutral script of the sibling NVIDIA tutorial with AMD-specific labels."""

import argparse
import json
import runpy
import sys
from pathlib import Path

NVIDIA = Path(__file__).resolve().parent.parent / 'nvidia-rtx5060ti'
ARCHES = ('gfx1201', 'gfx1100')


def split_arch_argument(argv):
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--arch', choices=ARCHES, default='gfx1201')
    known, rest = parser.parse_known_args(argv)
    return known.arch, rest


def output_argument(argv):
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--output', type=Path)
    return parser.parse_known_args(argv)[0].output


def run_nvidia_script(name, argv):
    sys.argv = [str(NVIDIA / name), *argv]
    if str(NVIDIA) not in sys.path:
        sys.path.insert(0, str(NVIDIA))
    runpy.run_path(str(NVIDIA / name), run_name='__main__')


def relabel_llama_backend(output, arch):
    metadata = output / 'metadata.json'
    report = json.loads(metadata.read_text())
    if report.get('backend') != 'CPU':
        report['backend'] = f'HIP {arch}'
    metadata.write_text(json.dumps(report, indent=2) + '\n')
