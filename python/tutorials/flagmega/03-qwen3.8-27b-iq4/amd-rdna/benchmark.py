"""Decode timing and llama.cpp comparison on AMD RDNA.

Runs the NVIDIA tutorial's protocol (stateful decode, graph replay, one CPU token
return) unchanged, then reports bandwidth against the AMD card and applies the
near-tie rule: the required generated prefix must equal llama.cpp's, except that
the first differing step may be a genuine near-tie of the golden logits (top-2
margin below --near-tie-margin, with our token as the golden's second choice).
After such a step the independently generated contexts differ, so later prefix
tokens are not compared.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from _shared import output_argument, run_nvidia_script, split_arch_argument

NOMINAL_GB_S = {'gfx1201': 644.6, 'gfx1100': 864.0}


def near_tie_acceptance(comparison, reference, margin):
    metadata = json.loads((reference / 'metadata.json').read_text())
    logits = np.fromfile(reference / 'logits.f32', dtype=np.float32).reshape(metadata['steps'], metadata['vocab'])
    start = comparison['prefix_start_step']
    for offset, (actual, golden) in enumerate(zip(comparison['actual_prefix'], comparison['golden_prefix'])):
        if actual == golden:
            continue
        step = start + offset
        first, second = np.argsort(logits[step])[::-1][:2]
        gap = float(logits[step][first] - logits[step][second])
        tie = {'step': step, 'actual': actual, 'golden': golden, 'golden_top2': [int(first), int(second)],
               'golden_margin': gap}
        return gap < margin and actual == int(second), tie
    return True, None


def main():
    arch, argv = split_arch_argument(sys.argv[1:])
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--near-tie-margin', type=float, default=0.1)
    parser.add_argument('--llama-reference', type=Path)
    known, _ = parser.parse_known_args(argv)
    argv = [value for index, value in enumerate(argv)
            if value != '--near-tie-margin' and (index == 0 or argv[index - 1] != '--near-tie-margin')]
    mismatch = None
    try:
        run_nvidia_script('benchmark.py', argv)
    except RuntimeError as error:
        if 'differ from llama.cpp golden' not in str(error):
            raise
        mismatch = error
    output = output_argument(argv)
    report = json.loads(output.read_text())
    peak = NOMINAL_GB_S[arch]
    report.update(gpu_arch=arch, theoretical_memory_bandwidth_gb_s=peak,
                  model_bandwidth_utilization=report['model_bandwidth_gb_s'] / peak)
    comparison = report.get('llama_comparison')
    if comparison is not None:
        accepted, tie = near_tie_acceptance(comparison, known.llama_reference, known.near_tie_margin)
        comparison.update(accepted=accepted, near_tie=tie, near_tie_margin=known.near_tie_margin,
                          acceptance='Generated prefix equal to llama.cpp, except a first differing step that is a '
                          'golden-logit near-tie with our token as its second choice')
    output.write_text(json.dumps(report, indent=2) + '\n')
    if comparison is not None and not comparison['accepted']:
        raise SystemExit(f'Generated tokens differ from llama.cpp beyond a near-tie: {mismatch}')


if __name__ == '__main__':
    main()
