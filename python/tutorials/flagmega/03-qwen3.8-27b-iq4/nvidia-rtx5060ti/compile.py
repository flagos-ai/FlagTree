"""Build an immutable native-Q4 trial through the FlagMega developer workflow."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from time import perf_counter

from download_checkpoint import FILENAME, REVISION

TUTORIAL = Path(__file__).resolve().parent
REPO = TUTORIAL.parents[4]
LOCAL = TUTORIAL / '.local'
SKILL = REPO / 'skills/flagmega-developer/SKILL.md'


def source_identity():
    paths = [
        SKILL, *sorted(path for path in TUTORIAL.iterdir()
                       if path.suffix in ('.py', '.cpp') and path.name != 'generated_kernels.py')
    ]
    paths.extend(
        sorted(path for path in (TUTORIAL / 'agent_optimizations').rglob('*')
               if path.suffix in ('.py', '.jinja') and '__pycache__' not in path.parts))
    paths.extend(
        sorted(path for path in (REPO / 'python/triton/flagmega').rglob('*')
               if path.suffix in ('.py', '.jinja') and '__pycache__' not in path.parts))
    paths.append(REPO / 'third_party/nvidia/backend/tle_supported.py')
    paths.append(REPO / 'third_party/nvidia/backend/compiler.py')
    paths.append(REPO / 'python/triton/experimental/tle/language/pipe.py')
    paths.append(REPO / 'lib/Conversion/TritonToTritonGPU/TritonToTritonGPUPass.cpp')
    sources = {str(path.relative_to(REPO)): hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}
    digest = hashlib.sha256(json.dumps(sources, sort_keys=True).encode()).hexdigest()
    return digest, sources


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, default=LOCAL / 'checkpoint' / FILENAME)
    destination = parser.add_mutually_exclusive_group()
    destination.add_argument('--trial', type=Path, help='Fresh trial directory; default: .local/trials/<UTC>-baseline')
    destination.add_argument('--output', type=Path,
                             help='Explicit artifact directory; build records use <output>.trial')
    parser.add_argument('--input', type=Path, help='Resume trusted Python IR or a complete function dump directory')
    parser.add_argument('--revision', help='Checkpoint revision; the default tutorial file uses the pinned HF revision')
    parser.add_argument('--rdata-cache-dir', type=Path, default=LOCAL / 'rdata-cache')
    parser.add_argument('--layer', type=int, help='Compile one layer; the default imports all 64 decoder layers')
    parser.add_argument('--work-dir', type=Path, help='Fresh directory for compiler stage dumps')
    parser.add_argument('--optimization',
                        choices=('paired', 'word', 'explicit', 'accurate', 'refined', 'pc_tma', 'pc_tma_inline'),
                        help='Install the local IQ4 template override')
    parser.add_argument('--iq4-chunk', type=int, default=512)
    parser.add_argument('--tile-n', type=int, default=16)
    parser.add_argument('--warps', type=int, choices=(4, 8, 16, 32), default=4)
    parser.add_argument('--grid', type=int, choices=(32, 36), default=32)
    parser.add_argument('--gdn-value-tile', type=int, choices=(4, 8, 16, 32, 64), default=4)
    parser.add_argument('--gdn-fast-head', action='store_true')
    parser.add_argument('--pc-segments', action='store_true')
    parser.add_argument('--q8-group-tile', type=int, choices=(8, 16, 32, 64, 128, 256), default=8)
    parser.add_argument('--fp32-residual', action='store_true',
                        help='Explicitly retain FP32 residual storage from a fresh import')
    parser.add_argument('--pc-dual-activation', action='store_true')
    parser.add_argument('--pc-dual-all', action='store_true')
    parser.add_argument('--balanced-rows', action='store_true')
    parser.add_argument('--pc-dense-dual', action='store_true')
    parser.add_argument('--pc-global-activation', action='store_true')
    parser.add_argument('--pc-packed-activation', action='store_true')
    parser.add_argument('--pc-down-chunk', type=int, choices=(1024, 2048, 4096), default=1024)
    parser.add_argument('--pc-precision-scope', choices=('all', 'ffn', 'glu', 'down', 'attention'), default='all')
    parser.add_argument('--pc-nibble-pair', action='store_true')
    parser.add_argument('--gdn-linear-owners', action='store_true')
    parser.add_argument('--gdn-uniform-norm', action='store_true')
    parser.add_argument('--gdn-cache-heads', action='store_true')
    parser.add_argument('--q8-vector-quant', action='store_true',
                        help='Quantize four contiguous activation values per lane')
    parser.add_argument('--refine-guard', type=int, default=128)
    args = parser.parse_args()
    checkpoint = args.checkpoint.expanduser().resolve()
    if not checkpoint.is_file() or checkpoint.suffix.lower() != '.gguf':
        parser.error(f'GGUF checkpoint does not exist: {checkpoint}; run download_checkpoint.py first')
    if args.input is not None and (args.layer is not None or args.revision is not None):
        parser.error('--input retains its imported layer scope and revision')
    if args.output is not None:
        artifact = args.output.expanduser().resolve()
        trial = artifact.with_name(artifact.name + '.trial')
        if artifact.exists():
            parser.error(f'Artifact already exists; use a fresh output: {artifact}')
    else:
        stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
        trial = (args.trial or LOCAL / 'trials' / (stamp + '-baseline')).expanduser().resolve()
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
    local_sources = [
        SKILL, *sorted(path for path in TUTORIAL.iterdir()
                       if path.suffix in ('.py', '.cpp') and path.name != 'generated_kernels.py')
    ]
    local_sources.extend(
        sorted(path for path in (TUTORIAL / 'agent_optimizations').rglob('*')
               if path.suffix in ('.py', '.jinja') and '__pycache__' not in path.parts))
    for path in local_sources:
        target = trial / 'sources' / path.relative_to(REPO)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
    revision = args.revision or (REVISION if checkpoint.name == FILENAME else None)
    env = dict(os.environ)
    env['PYTHONPATH'] = os.pathsep.join(
        [str(REPO / 'python'),
         str(TUTORIAL), *([env['PYTHONPATH']] if env.get('PYTHONPATH') else [])])
    commands = []
    report = {
        'workflow': 'flagmega-developer', 'skill': str(SKILL.relative_to(REPO)), 'checkpoint': str(checkpoint),
        'checkpoint_nbytes': checkpoint.stat().st_size, 'checkpoint_revision': revision, 'target': 'nvidia-sm120',
        'compiler_source_sha256': digest, 'trial': str(trial), 'artifact': str(artifact), 'rdata_cache_dir':
        str(args.rdata_cache_dir.expanduser().resolve()), 'resumed_from':
        None if args.input is None else str(args.input.expanduser().resolve()), 'optimization': args.optimization,
        'iq4_chunk': args.iq4_chunk, 'tile_n': args.tile_n, 'warps': args.warps, 'grid': args.grid, 'gdn_value_tile':
        args.gdn_value_tile, 'gdn_fast_head': args.gdn_fast_head, 'pc_segments': args.pc_segments, 'q8_group_tile':
        args.q8_group_tile, 'fp32_residual': args.fp32_residual, 'pc_dual_activation': args.pc_dual_activation,
        'pc_dual_all': args.pc_dual_all, 'balanced_rows': args.balanced_rows, 'pc_dense_dual': args.pc_dense_dual,
        'pc_global_activation': args.pc_global_activation, 'pc_packed_activation': args.pc_packed_activation,
        'pc_down_chunk': args.pc_down_chunk, 'pc_precision_scope': args.pc_precision_scope, 'pc_nibble_pair':
        args.pc_nibble_pair, 'gdn_linear_owners': args.gdn_linear_owners, 'gdn_uniform_norm': args.gdn_uniform_norm,
        'gdn_cache_heads': args.gdn_cache_heads, 'q8_vector_quant': args.q8_vector_quant, 'refinement_guard':
        args.refine_guard, 'commands': commands, 'status': 'running'
    }
    started = perf_counter()
    command_name = None

    def save_report():
        (trial / 'build.json').write_text(json.dumps(report, indent=2) + '\n')

    def run(name, arguments):
        nonlocal command_name
        command_name = name
        if args.optimization:
            bootstrap = (f'from agent_optimizations import install; install(chunk={args.iq4_chunk}, '
                         f'tile_n={args.tile_n}, warps={args.warps}, mode={args.optimization!r}, '
                         f'grid={args.grid}, guard={args.refine_guard}, gdn_value_tile={args.gdn_value_tile}, '
                         f'gdn_fast_head={args.gdn_fast_head}, pc_segments={args.pc_segments}, '
                         f'q8_group_tile={args.q8_group_tile}, pc_dual_activation={args.pc_dual_activation}, '
                         f'pc_dual_all={args.pc_dual_all}, balanced_rows={args.balanced_rows}, '
                         f'pc_dense_dual={args.pc_dense_dual}, pc_global_activation={args.pc_global_activation}, '
                         f'pc_packed_activation={args.pc_packed_activation}, pc_down_chunk={args.pc_down_chunk}, '
                         f'pc_precision_scope={args.pc_precision_scope!r}, pc_nibble_pair={args.pc_nibble_pair}, '
                         f'gdn_linear_owners={args.gdn_linear_owners}, gdn_uniform_norm={args.gdn_uniform_norm}, '
                         f'gdn_cache_heads={args.gdn_cache_heads}, q8_vector_quant={args.q8_vector_quant}); '
                         'from triton.flagmega.cli import main; raise SystemExit(main())')
            command = [sys.executable, '-c', bootstrap, *arguments]
        else:
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
            options = [
                'import', '--model',
                str(checkpoint), '--output',
                str(imported), '--numerical-profile', 'nncase', '--mode', 'decode-1', '--json'
            ]
            options.extend(['--full-model'] if args.layer is None else ['--layer', str(args.layer)])
            if revision is not None:
                options.extend(['--revision', revision])
            run('import', options)
            if args.fp32_residual:
                sys.path.insert(0, str(REPO / 'python'))
                from triton.flagmega.ir import emit_module, load_module
                from agent_optimizations.numerics import fp32_residual
                shutil.copy2(imported, inputs / 'original.nncase.py')
                emit_module(fp32_residual(load_module(imported)), imported)
        else:
            sys.path.insert(0, str(REPO / 'python'))
            from triton.flagmega.ir import emit_module, load_module
            module = load_module(args.input)
            if args.fp32_residual and module.metadata.get('numerical_contract') != 'nncase-fp32-residual':
                parser.error('FP32 residual resume requires a checkpoint already imported with that profile')
            emit_module(module, imported)
        run('verify-ir', ['verify', str(imported), '--json'])
        run('inspect-ir', ['inspect', str(imported), '--json'])
        compiled = run('compile', [
            'compile', '--input',
            str(imported), '--checkpoint',
            str(checkpoint), '--target', 'nvidia-sm120', '--emit-executable', '--dump-flags',
            'compile,pass-ir,egraph-cost', '--work-dir',
            str(dumps), '--output',
            str(artifact), '--rdata-cache-dir',
            str(args.rdata_cache_dir.expanduser().resolve()), '--json'
        ])
        verified = run('verify-artifact', ['artifact', 'verify', str(artifact), '--json'])
        if source_identity()[0] != digest:
            raise RuntimeError('Compiler/tutorial sources changed during this build; create a new trial')
        manifest = json.loads((artifact / 'artifact.json').read_text())
        report.update({
            'status':
            'verified', 'input_semantic_hash':
            compiled['input_semantic_hash'], 'semantic_hash':
            compiled['output_semantic_hash'], 'imported':
            str(imported), 'dumps':
            str(dumps), 'generated_source_sha256':
            hashlib.sha256((artifact / 'generated_kernels.py').read_bytes()).hexdigest(), 'artifact_verified':
            bool(verified['ok']), 'scope':
            manifest['metadata'].get('scope'), 'num_hidden_layers':
            manifest['metadata'].get('num_hidden_layers'), 'inspect_log':
            'logs/inspect-ir.json', 'build_seconds':
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
