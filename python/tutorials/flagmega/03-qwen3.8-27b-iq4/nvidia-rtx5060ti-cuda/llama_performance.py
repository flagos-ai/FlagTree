"""Measure official llama.cpp with GPU greedy sampling and CUDA graphs enabled."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import statistics
from download_checkpoint import FILENAME
from reference_identity import LLAMA_COMMIT, execute_checked, file_pin, prepare_runner
from timing_isolation import clock_identity, observed_policy, validate_round_result

TUTORIAL = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--llama-source', type=Path, required=True)
    parser.add_argument('--llama-build', type=Path, required=True)
    parser.add_argument('--build-receipt', type=Path, help='Validated build_reference.py receipt; default is in --llama-build')
    parser.add_argument('--checkpoint', type=Path, default=TUTORIAL / '.local/checkpoint' / FILENAME)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--steps', type=int, default=32)
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--token-id', type=int, default=1)
    inputs = parser.add_mutually_exclusive_group()
    inputs.add_argument('--input-tokens', type=Path)
    inputs.add_argument('--prompt')
    args = parser.parse_args()
    if args.steps < 1 or args.repeats < 1:
        parser.error('steps and repeats must be positive')
    if args.output.exists():
        parser.error('Use a fresh output directory')
    source = TUTORIAL / 'llama_performance.cpp'
    binary, command, provenance = prepare_runner(source=args.llama_source, build=args.llama_build,
        output=args.output, cpp_source=source, binary_name='llama-performance', receipt=args.build_receipt)
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
    completed = execute_checked(run, source=args.llama_source, build=args.llama_build, output=args.output,
                                cpp_source=source, preflight=provenance, receipt=args.build_receipt)
    report = json.loads(completed.stdout)
    validate_round_result(report)
    if report['steps'] != args.steps or report['repeats'] != args.repeats:
        raise RuntimeError('Native sample dimensions differ from requested rounds')
    if report['round_timing']['policy'] != observed_policy():
        raise RuntimeError('Native policy differs from the monitored wrapper')
    if report['round_timing']['clock'] != clock_identity() or report['round_timing']['parent_pid'] != os.getpid():
        raise RuntimeError('Reference round clock/process lineage differs from its observing wrapper')
    checkpoint_hash = hashlib.sha256()
    with args.checkpoint.open('rb') as checkpoint:
        for chunk in iter(lambda: checkpoint.read(8 * 1024 * 1024), b''):
            checkpoint_hash.update(chunk)
    report.update(
        llama_commit=LLAMA_COMMIT, checkpoint=str(args.checkpoint.resolve()), checkpoint_sha256=checkpoint_hash.hexdigest(),
        runner_sha256=hashlib.sha256(source.read_bytes()).hexdigest(), backend='CUDA SM120', kv_dtype='BF16',
        intermediate_dtype='llama.cpp native F32', n_batch=1, n_ubatch=1, requested_gpu_layers=-1, flash_attention=True,
        mtp=False, sampling='Context backend greedy sampler; one CPU token returned, no full-vocabulary download',
        prompt=args.prompt, measurement='Host steady_clock around decode, GPU greedy and one CPU token return',
        compile_command=command, run_command=run, mean_ms=statistics.mean(report['latency_ms']),
        median_ms=statistics.median(report['latency_ms']))
    report['reference_provenance'] = file_pin(args.output / 'provenance.json')
    report['reference_build'] = provenance['reference_build']
    report['runner_binary'] = provenance['runner']['binary']
    report['resolved_reference_libraries'] = provenance['runner']['resolved_reference_libraries']
    report['tokens_per_second'] = 1000 / report['mean_ms']
    report['cmake_cache_sha256'] = hashlib.sha256((args.llama_build / 'CMakeCache.txt').read_bytes()).hexdigest()
    shutil.copy2(args.llama_build / 'CMakeCache.txt', args.output / 'CMakeCache.txt')
    (args.output / 'metadata.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
