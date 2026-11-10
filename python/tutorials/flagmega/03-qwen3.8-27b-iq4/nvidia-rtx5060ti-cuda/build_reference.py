"""Build clean pinned llama.cpp and issue a revalidated library provenance receipt.

Existing source/config-compatible directories are supported. The first receipt
uses --clean-first to establish the provenance of every linked object. A later
incremental build requires an unchanged validated --previous-receipt. Explicit
--adopt-fresh-build records a trusted manual checkout/build chain and its limits
without rerunning or claiming ownership of those historical commands.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess

from reference_identity import (
    EXPECTED_CONFIG, LLAMA_COMMIT, RECEIPT_NAME, build_snapshot, cmake_cache,
    file_pin, manual_build_evidence, save_json, source_identity, tool_identity, utc_now, validate_build, validate_config,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--llama-source', type=Path, required=True)
    parser.add_argument('--llama-build', type=Path, required=True)
    parser.add_argument('--receipt', type=Path, help='Fresh receipt path; default: build/reference-build-receipt.json')
    parser.add_argument('--previous-receipt', type=Path, help='Validated same-source/config/library receipt permits incremental rebuild')
    parser.add_argument('--adopt-fresh-build', type=Path,
                        help='Evidence directory with manual-build.json, configure.log and build.log; executes no build')
    parser.add_argument('--jobs', type=int, default=12)
    args = parser.parse_args()
    if args.jobs < 1:
        parser.error('jobs must be positive')
    if args.previous_receipt and args.adopt_fresh_build:
        parser.error('--previous-receipt and --adopt-fresh-build are mutually exclusive')
    source, build = args.llama_source.resolve(), args.llama_build.resolve()
    receipt = (args.receipt or build / RECEIPT_NAME).resolve()
    if receipt.exists() or receipt.parent.joinpath(receipt.stem + '.logs').exists():
        parser.error('Use a fresh receipt and log path; previous build records are preserved')
    initial_source = source_identity(source)
    cache_path = build / 'CMakeCache.txt'
    existing = cache_path.exists()
    if existing:
        validate_config(cmake_cache(cache_path), source)
    elif build.exists() and any(build.iterdir()):
        parser.error('Existing build directory has no admitted CMake cache')
    previous = validate_build(source, build, args.previous_receipt) if args.previous_receipt else None
    receipt.parent.mkdir(parents=True, exist_ok=True)
    if args.adopt_fresh_build:
        adoption = manual_build_evidence(args.adopt_fresh_build, source=source, build=build)
        snapshot = build_snapshot(source, build)
        if snapshot['source'] != initial_source:
            raise ValueError('Source changed during manual-build admission')
        record = adoption['record']
        report = {'schema': 'flagmega.llama-reference-build/v1', 'status': 'verified',
                  'expected_commit': LLAMA_COMMIT, 'build_mode': 'adopted_trusted_manual_fresh',
                  'source_before': None, 'adoption': adoption, 'snapshot': snapshot,
                  'created_utc': utc_now(), 'entrypoint': file_pin(__file__),
                  'helper': file_pin(Path(__file__).with_name('reference_identity.py')),
                  'commands': [
                      {'step': 'configure', 'command': record.get('configure_command'), 'returncode': 0,
                       'log': adoption['configure_log'], 'executed_by': record['actor'], 'backfilled': True},
                      {'step': 'build', 'command': record.get('build_command'), 'returncode': 0,
                       'log': adoption['build_log'], 'executed_by': record['actor'], 'backfilled': True}]}
        save_json(receipt, report)
        validate_build(source, build, receipt)
        print(json.dumps({'receipt': file_pin(receipt), 'status': 'verified',
                          'build_mode': report['build_mode'], 'limitation': adoption['limitation']}, indent=2))
        return
    logs = receipt.with_name(receipt.stem + '.logs')
    logs.mkdir()
    cmake = tool_identity('cmake')
    report = {'schema': 'flagmega.llama-reference-build/v1', 'status': 'building', 'expected_commit': LLAMA_COMMIT,
              'started_utc': utc_now(), 'source_before': initial_source, 'existing_build_directory': existing,
              'build_mode': 'incremental_admitted' if previous else 'clean_first', 'previous_admission': previous,
              'entrypoint': file_pin(__file__), 'helper': file_pin(Path(__file__).with_name('reference_identity.py')),
              'cmake': cmake, 'commands': []}
    save_json(receipt, report)
    configure = [cmake['binary']['path'], '-S', str(source), '-B', str(build),
                 *[f'-D{key}={value}' for key, value in EXPECTED_CONFIG.items()]]
    compile_command = [cmake['binary']['path'], '--build', str(build), '--target', 'llama', '--parallel', str(args.jobs)]
    if previous is None:
        compile_command.append('--clean-first')
    try:
        for step, command in [('configure', configure), ('build', compile_command)]:
            log = logs / (step + '.log')
            print(json.dumps({'step': step, 'command': command, 'log': str(log)}), flush=True)
            with log.open('w') as stream:
                result = subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT)
            report['commands'].append({'step': step, 'command': command, 'returncode': result.returncode,
                                       'log': file_pin(log)})
            save_json(receipt, report)
            result.check_returncode()
        snapshot = build_snapshot(source, build)
        if snapshot['source'] != initial_source:
            raise ValueError('Clean pinned source changed during the build')
        report.update(status='verified', snapshot=snapshot, ended_utc=utc_now())
        save_json(receipt, report)
        # Exercise the same admission routine reference runners will use.
        validate_build(source, build, receipt)
    except BaseException as error:
        report.update(status='failed', error=f'{type(error).__name__}: {error}', ended_utc=utc_now())
        save_json(receipt, report)
        raise
    print(json.dumps({'receipt': file_pin(receipt), 'status': 'verified', 'build_mode': report['build_mode']}, indent=2))


if __name__ == '__main__':
    main()
