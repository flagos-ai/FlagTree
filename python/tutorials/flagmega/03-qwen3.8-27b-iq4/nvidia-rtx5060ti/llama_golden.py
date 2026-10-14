"""Run an independent pinned llama.cpp golden with the original packed GGUF."""

import argparse
import hashlib
import json
from pathlib import Path
import statistics
import subprocess
import shutil

from download_checkpoint import FILENAME

TUTORIAL = Path(__file__).resolve().parent
LLAMA_COMMIT = '4413a192410fad1f054e9216114087a3e01527a0'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--llama-source', type=Path, required=True)
    parser.add_argument('--llama-build', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path, default=TUTORIAL / '.local/checkpoint' / FILENAME)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--steps', type=int, default=32)
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--token-id', type=int, default=1)
    inputs = parser.add_mutually_exclusive_group()
    inputs.add_argument('--input-tokens', type=Path, help='Whitespace separated token IDs, one per decode step')
    inputs.add_argument('--prompt', help='Tokenize a real prompt, then continue greedy single-token decode')
    parser.add_argument('--cpu', action='store_true')
    args = parser.parse_args()
    revision = subprocess.check_output(['git', '-C', str(args.llama_source), 'rev-parse', 'HEAD'], text=True).strip()
    if revision != LLAMA_COMMIT:
        parser.error(f'Expected pinned llama.cpp commit {LLAMA_COMMIT}, got {revision}')
    if args.output.exists():
        parser.error('Use a fresh output directory')
    args.output.mkdir(parents=True)
    source = TUTORIAL / 'llama_golden.cpp'
    binary = args.output / 'llama-golden'
    libraries = args.llama_build.resolve() / 'bin'
    command = [
        'g++', '-O3', '-std=c++17',
        str(source), '-I' + str(args.llama_source / 'include'), '-I' + str(args.llama_source / 'ggml/include'),
        '-L' + str(libraries), '-Wl,-rpath,' + str(libraries), '-lllama', '-o',
        str(binary)
    ]
    subprocess.run(command, check=True)
    run = [
        str(binary.resolve()),
        str(args.checkpoint.resolve()),
        str((args.output / 'logits.f32').resolve()),
        str(args.steps),
        str(args.repeats),
        str(args.token_id),
        str(int(args.cpu))
    ]
    if args.input_tokens:
        run.append(str(args.input_tokens.resolve()))
    elif args.prompt:
        run.extend(['--prompt', args.prompt])
    with (args.output / 'llama.log').open('w') as log:
        completed = subprocess.run(run, stdout=subprocess.PIPE, stderr=log, text=True)
    (args.output / 'stdout.log').write_text(completed.stdout)
    completed.check_returncode()
    report = json.loads(completed.stdout)
    checkpoint_hash = hashlib.sha256()
    with args.checkpoint.open('rb') as checkpoint:
        for chunk in iter(lambda: checkpoint.read(8 * 1024 * 1024), b''):
            checkpoint_hash.update(chunk)
    report.update(
        llama_commit=revision, checkpoint=str(args.checkpoint.resolve()), checkpoint_sha256=checkpoint_hash.hexdigest(),
        runner_sha256=hashlib.sha256(source.read_bytes()).hexdigest(), backend='CPU' if args.cpu else 'CUDA SM120',
        kv_dtype='BF16', intermediate_dtype='llama.cpp native F32', n_batch=1, n_ubatch=1,
        requested_gpu_layers=0 if args.cpu else -1, flash_attention=not args.cpu, mtp=False, prompt=args.prompt,
        measurement='Host steady_clock around llama_decode, synchronize and CPU logits retrieval; excludes CPU argmax',
        compile_command=command, run_command=run, mean_ms=statistics.mean(report['latency_ms']))
    report['tokens_per_second'] = 1000 / report['mean_ms']
    report['cmake_cache_sha256'] = hashlib.sha256((args.llama_build / 'CMakeCache.txt').read_bytes()).hexdigest()
    shutil.copy2(args.llama_build / 'CMakeCache.txt', args.output / 'CMakeCache.txt')
    (args.output / 'metadata.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
