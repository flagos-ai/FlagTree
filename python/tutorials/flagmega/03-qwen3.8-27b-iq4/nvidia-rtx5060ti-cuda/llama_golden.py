"""Run an independent pinned llama.cpp golden with the original packed GGUF."""

import argparse
import hashlib
import json
from pathlib import Path
import statistics
import shutil

from download_checkpoint import FILENAME
from reference_identity import LLAMA_COMMIT, execute_checked, file_pin, prepare_runner

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
    inputs.add_argument('--input-tokens', type=Path, help='Whitespace separated token IDs, one per decode step')
    inputs.add_argument('--prompt', help='Tokenize a real prompt, then continue greedy single-token decode')
    parser.add_argument('--cpu', action='store_true')
    args = parser.parse_args()
    if args.steps < 1 or args.repeats < 1:
        parser.error('steps and repeats must be positive')
    if args.output.exists():
        parser.error('Use a fresh output directory')
    source = TUTORIAL / 'llama_golden.cpp'
    binary, command, provenance = prepare_runner(source=args.llama_source, build=args.llama_build,
        output=args.output, cpp_source=source, binary_name='llama-golden', receipt=args.build_receipt)
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
    completed = execute_checked(run, source=args.llama_source, build=args.llama_build, output=args.output,
                                cpp_source=source, preflight=provenance, receipt=args.build_receipt)
    report = json.loads(completed.stdout)
    checkpoint_hash = hashlib.sha256()
    with args.checkpoint.open('rb') as checkpoint:
        for chunk in iter(lambda: checkpoint.read(8 * 1024 * 1024), b''):
            checkpoint_hash.update(chunk)
    report.update(
        llama_commit=LLAMA_COMMIT, checkpoint=str(args.checkpoint.resolve()), checkpoint_sha256=checkpoint_hash.hexdigest(),
        runner_sha256=hashlib.sha256(source.read_bytes()).hexdigest(), backend='CPU' if args.cpu else 'CUDA SM120',
        kv_dtype='BF16', intermediate_dtype='llama.cpp native F32', n_batch=1, n_ubatch=1,
        requested_gpu_layers=0 if args.cpu else -1, flash_attention=not args.cpu, mtp=False, prompt=args.prompt,
        measurement='Host steady_clock around llama_decode, synchronize and CPU logits retrieval; excludes CPU argmax',
        compile_command=command, run_command=run, mean_ms=statistics.mean(report['latency_ms']))
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
