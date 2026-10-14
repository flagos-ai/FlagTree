"""Measure official llama.cpp with GPU greedy sampling and CUDA graphs enabled."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import statistics
import subprocess
from download_checkpoint import FILENAME
from llama_golden import LLAMA_COMMIT

TUTORIAL = Path(__file__).resolve().parent


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
    inputs.add_argument('--input-tokens', type=Path)
    inputs.add_argument('--prompt')
    args = parser.parse_args()
    revision = subprocess.check_output(['git', '-C', str(args.llama_source), 'rev-parse', 'HEAD'], text=True).strip()
    if revision != LLAMA_COMMIT:
        parser.error(f'Expected pinned llama.cpp commit {LLAMA_COMMIT}, got {revision}')
    if args.output.exists():
        parser.error('Use a fresh output directory')
    args.output.mkdir(parents=True)
    source = TUTORIAL / 'llama_performance.cpp'
    binary = args.output / 'llama-performance'
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
        str(args.steps),
        str(args.repeats),
        str(args.token_id)
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
        runner_sha256=hashlib.sha256(source.read_bytes()).hexdigest(), backend='CUDA SM120', kv_dtype='BF16',
        intermediate_dtype='llama.cpp native F32', n_batch=1, n_ubatch=1, requested_gpu_layers=-1, flash_attention=True,
        mtp=False, sampling='Context backend greedy sampler; one CPU token returned, no full-vocabulary download',
        prompt=args.prompt, measurement='Host steady_clock around decode, GPU greedy and one CPU token return',
        compile_command=command, run_command=run, mean_ms=statistics.mean(report['latency_ms']),
        median_ms=statistics.median(report['latency_ms']))
    report['tokens_per_second'] = 1000 / report['mean_ms']
    report['cmake_cache_sha256'] = hashlib.sha256((args.llama_build / 'CMakeCache.txt').read_bytes()).hexdigest()
    shutil.copy2(args.llama_build / 'CMakeCache.txt', args.output / 'CMakeCache.txt')
    (args.output / 'metadata.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
