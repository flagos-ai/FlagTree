"""Freeze complete source/build provenance before independent native review.

This command issues no native, numerical or performance acceptance. Pass the
resulting pin to the existing independent native review as source_freeze.
"""
import argparse
import json
from pathlib import Path

from reproduction.build_source import FREEZE_NAME, build_source_record
from reproduction.provider import pin


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--build', type=Path, required=True, help='Successful fresh compile.py build.json')
    args = parser.parse_args()
    build = pin(args.build)
    output = Path(build['path']).parent / FREEZE_NAME
    if output.exists():
        parser.error('Use a fresh source-freeze output; prior records are immutable')
    record = build_source_record(build)
    with output.open('x') as stream:
        json.dump(record, stream, indent=2); stream.write('\n')
    print(json.dumps(pin(output)))


if __name__ == '__main__':
    main()
