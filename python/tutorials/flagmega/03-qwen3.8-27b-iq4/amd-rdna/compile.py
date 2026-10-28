"""Build an immutable native-IQ4 trial for an AMD RDNA GPU through the FlagMega workflow."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from time import perf_counter

TUTORIAL = Path(__file__).resolve().parent
sys.path.insert(0, str(TUTORIAL.parent / 'nvidia-rtx5060ti'))
from download_checkpoint import FILENAME, REVISION  # noqa: E402  (shared checkpoint pin)

REPO = TUTORIAL.parents[4]
LOCAL = TUTORIAL / '.local'
SKILL = REPO / 'skills/flagmega-developer/SKILL.md'
# gfx1100 has no FP8 capability, so the portable IQ4 GEMV candidates are not offered there yet.
TARGETS = {'gfx1201': 'amd-gfx1201'}


def source_identity():
    paths = [SKILL, *sorted(path for path in TUTORIAL.iterdir() if path.suffix == '.py')]
    paths.extend(
        sorted(path for path in (REPO / 'python/triton/flagmega').rglob('*')
               if path.suffix in ('.py', '.jinja') and '__pycache__' not in path.parts))
    paths.append(REPO / 'third_party/amd/backend/tle_supported.py')
    paths.append(REPO / 'third_party/amd/backend/compiler.py')
    paths.append(REPO / 'python/triton/experimental/tle/language/pipe.py')
    paths.append(REPO / 'lib/Conversion/TritonToTritonGPU/TritonToTritonGPUPass.cpp')
    sources = {str(path.relative_to(REPO)): hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}
    digest = hashlib.sha256(json.dumps(sources, sort_keys=True).encode()).hexdigest()
    return digest, sources


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, default=LOCAL / 'checkpoint' / FILENAME)
    parser.add_argument('--arch', choices=('gfx1201', 'gfx1100'), default='gfx1201')
    destination = parser.add_mutually_exclusive_group()
    destination.add_argument('--trial', type=Path, help='Fresh trial directory; default: .local/trials/<UTC>-<arch>')
    destination.add_argument('--output', type=Path,
                             help='Explicit artifact directory; build records use <output>.trial')
    parser.add_argument('--input', type=Path, help='Resume trusted Python IR or a complete function dump directory')
    parser.add_argument('--revision', help='Checkpoint revision; the default tutorial file uses the pinned HF revision')
    parser.add_argument('--rdata-cache-dir', type=Path, default=LOCAL / 'rdata-cache')
    parser.add_argument('--work-dir', type=Path, help='Fresh directory for compiler stage dumps')
    args = parser.parse_args()
    if args.arch not in TARGETS:
        parser.error(f'{args.arch} is not supported yet: its target has no FP8 capability, so the IQ4 GEMV '
                     'candidates are not offered')
    target = TARGETS[args.arch]
    checkpoint = args.checkpoint.expanduser().resolve()
    if not checkpoint.is_file() or checkpoint.suffix.lower() != '.gguf':
        parser.error(f'GGUF checkpoint does not exist: {checkpoint}; run download_checkpoint.py first')
    if args.input is not None and args.revision is not None:
        parser.error('--input retains its imported revision')
    if args.output is not None:
        artifact = args.output.expanduser().resolve()
        trial = artifact.with_name(artifact.name + '.trial')
        if artifact.exists():
            parser.error(f'Artifact already exists; use a fresh output: {artifact}')
    else:
        stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
        trial = (args.trial or LOCAL / 'trials' / f'{stamp}-{args.arch}').expanduser().resolve()
        artifact = trial / 'artifact'
    dumps = (args.work_dir or trial / 'dumps').expanduser().resolve()
    if trial.exists() or dumps.exists():
        parser.error(f'Use a fresh trial and dump directory; existing trials are preserved: {trial}')
    trial.mkdir(parents=True, exist_ok=False)
    logs = trial / 'logs'
    logs.mkdir()
    inputs = trial / 'input'
    inputs.mkdir()
    imported = inputs / 'imported.py'
    digest, sources = source_identity()
    (trial / 'sources.json').write_text(json.dumps(sources, indent=2) + '\n')
    revision = args.revision or (REVISION if checkpoint.name == FILENAME else None)
    env = dict(os.environ)
    env['PYTHONPATH'] = os.pathsep.join([str(REPO / 'python'), *([env['PYTHONPATH']] if env.get('PYTHONPATH') else [])])
    commands = []
    report = {
        'workflow': 'flagmega-developer', 'skill': str(SKILL.relative_to(REPO)), 'checkpoint': str(checkpoint),
        'checkpoint_nbytes': checkpoint.stat().st_size, 'checkpoint_revision': revision, 'target': target,
        'compiler_source_sha256': digest, 'trial': str(trial), 'artifact': str(artifact), 'rdata_cache_dir':
        str(args.rdata_cache_dir.expanduser().resolve()), 'resumed_from':
        None if args.input is None else str(args.input.expanduser().resolve()), 'commands': commands,
        'status': 'running'
    }
    started = perf_counter()
    command_name = None

    def save_report():
        (trial / 'build.json').write_text(json.dumps(report, indent=2) + '\n')

    def run(name, arguments):
        nonlocal command_name
        command_name = name
        command = [sys.executable, '-m', 'triton.flagmega', *arguments]
        log = logs / (name + '.json')
        print(f'[{name}] {log}', flush=True)
        before = perf_counter()
        with log.open('w') as stream:
            process = subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT, env=env)
        commands.append({
            'step': name, 'argv': command, 'log': str(log.relative_to(trial)), 'seconds': perf_counter() - before,
            'returncode': process.returncode
        })
        save_report()
        if process.returncode:
            raise RuntimeError(f'FlagMega {name} failed; see {log}')
        return json.loads(log.read_text())

    save_report()
    try:
        if args.input is None:
            # The GGUF importer builds the complete decoder; single-layer import is safetensors-only.
            options = [
                'import', '--model',
                str(checkpoint), '--output',
                str(imported), '--numerical-profile', 'nncase', '--mode', 'decode-1', '--full-model', '--json'
            ]
            if revision is not None:
                options.extend(['--revision', revision])
            run('import', options)
        else:
            sys.path.insert(0, str(REPO / 'python'))
            from triton.flagmega.ir import emit_module, load_module
            emit_module(load_module(args.input), imported)
        run('verify-ir', ['verify', str(imported), '--json'])
        run('inspect-ir', ['inspect', str(imported), '--json'])
        compiled = run('compile', [
            'compile', '--input',
            str(imported), '--checkpoint',
            str(checkpoint), '--target', target, '--emit-executable', '--dump-flags', 'compile,pass-ir', '--work-dir',
            str(dumps), '--output',
            str(artifact), '--rdata-cache-dir',
            str(args.rdata_cache_dir.expanduser().resolve()), '--json'
        ])
        verified = run('verify-artifact', ['artifact', 'verify', str(artifact), '--json'])
        if source_identity()[0] != digest:
            raise RuntimeError('Compiler/tutorial sources changed during this build; create a new trial')
        manifest = json.loads((artifact / 'artifact.json').read_text())
        report.update({
            'status': 'verified', 'input_semantic_hash': compiled['input_semantic_hash'], 'semantic_hash':
            compiled['output_semantic_hash'], 'imported': str(imported), 'dumps': str(dumps),
            'generated_source_sha256': hashlib.sha256((artifact / 'generated_kernels.py').read_bytes()).hexdigest(),
            'artifact_verified': bool(verified['ok']), 'scope': manifest['metadata'].get('scope'),
            'num_hidden_layers': manifest['metadata'].get('num_hidden_layers'), 'build_seconds':
            perf_counter() - started
        })
        print(json.dumps({key: value for key, value in report.items() if key != 'commands'}, indent=2), flush=True)
    except Exception as error:
        report.update({
            'status': 'failed', 'failed_step': command_name, 'error': str(error), 'build_seconds':
            perf_counter() - started
        })
        raise
    finally:
        save_report()


if __name__ == '__main__':
    main()
