"""Verify a rebuilt full decode artifact, then optionally compare fresh vLLM."""

import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--trial', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--sdk', type=Path, required=True)
    parser.add_argument('--jit-device', type=int, default=5)
    parser.add_argument('--checkpoint', type=Path)
    parser.add_argument('--native-python', type=Path)
    parser.add_argument('--native-triton-cache', type=Path)
    parser.add_argument('--native-inductor-cache', type=Path)
    args = parser.parse_args()
    native = (args.checkpoint, args.native_python, args.native_triton_cache,
              args.native_inductor_cache)
    if any(native) and not all(native):
        parser.error('A native comparison requires all four native options')
    trial, output, sdk = args.trial.resolve(), args.output.resolve(), args.sdk.resolve()
    output.mkdir(exist_ok=False)
    env = dict(os.environ, BASH_ENV='/dev/null', ENV='/dev/null',
               PYTHONDONTWRITEBYTECODE='1', OMP_NUM_THREADS='4', MKL_NUM_THREADS='4',
               GLOO_SOCKET_IFNAME='lo', FLAGCX_MODULE_PATH=str(sdk))
    env.pop('TRITON_PTXAS_PATH', None)
    env.pop('TRITON_CACHE_DIR', None)
    receipts = []

    def run(label, command):
        begin = time.monotonic()
        with (output/f'{label}.log').open('w') as log:
            child = subprocess.Popen(command, env=env, stdout=log,
                                     stderr=subprocess.STDOUT)
            print(json.dumps(dict(event='started', label=label, pid=child.pid)), flush=True)
            code = child.wait()
        receipt = dict(label=label, pid=child.pid, actual_exit_code=code,
                       real_child_wait=True, seconds=time.monotonic()-begin,
                       timestamp=datetime.datetime.now(datetime.timezone.utc).isoformat())
        receipts.append(receipt)
        (output/'receipts.json').write_text(json.dumps(receipts, indent=2)+'\n')
        print(json.dumps(receipt), flush=True)
        if code:
            raise SystemExit(code)

    run('jit', [sys.executable, str(trial/'local_optimizations/run_independent_jit.py'),
                '--device', str(args.jit_device)])
    inputs = []
    for directory in ('local_optimizations', 'patches'):
        inputs.extend(p for p in (trial/directory).rglob('*')
                      if p.is_file() and '__pycache__' not in p.parts)
    from triton import flagmega
    compiler = Path(flagmega.__file__).resolve().parent
    inputs.extend(p for p in compiler.rglob('*') if p.is_file() and p.suffix in {'.py', '.jinja'} and '__pycache__' not in p.parts)
    inputs.extend(trial/name for name in ('dag-final.py', 'dag-audit.json',
                  'build-receipt.json', 'warmup.json', 'benchmark_gpu.py',
                  'benchmark_vllm.py', 'reproduce.py', 'accuracy.py',
                  'local_optimizations/full-state-reference-digests.json',
                  'local_optimizations/reproduction-runtime-dependencies.json',
                  'local_optimizations/delivery-contract.json',
                  'artifact/generated_kernels.py', 'artifact/artifact.json',
                  'artifact/ir/final.py', 'artifact/assets/rdata.index.json'))
    identities = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs}
    (trial/'source-identities-v3.json').write_text(json.dumps(dict(files=identities), indent=2)+'\n')
    numerics = output/'numerics'
    run('numerics', [sys.executable, str(trial/'local_optimizations/run_reserved.py'), '--mode', 'numerics',
                     '--output', str(numerics), '--sdk', str(sdk)])
    verified = json.loads((numerics/'verified.json').read_text())
    assert verified['passed'] and verified['state_byte_digests_verified']
    result = dict(numerical_acceptance=True, performance_acceptance=False,
                  source_sha256=verified['source_sha256'], receipts=receipts)
    if all(native):
        comparison_dir = output/'native-comparison'
        run('native-comparison', [sys.executable,
            str(trial/'local_optimizations/compare_native.py'),
            '--output', str(comparison_dir), '--trial', str(trial),
            '--validation', str(numerics), '--sdk', str(sdk),
            '--checkpoint', str(args.checkpoint.resolve()),
            '--native-python', str(args.native_python.absolute()),
            '--native-triton-cache', str(args.native_triton_cache.resolve()),
            '--native-inductor-cache', str(args.native_inductor_cache.resolve())])
        comparison = json.loads((comparison_dir/'comparison.json').read_text())
        assert comparison['hardware_acceptance'] and comparison['all_official8_passed']
        result['fresh_comparison'] = comparison
        # Preserve the complete comparison; a positive point estimate alone
        # does not establish a repeatable performance win.
    assert all(hashlib.sha256(Path(p).read_bytes()).hexdigest() == h
               for p, h in identities.items()), 'Reproduction inputs changed during validation'
    result['source_inputs_unchanged'] = True
    (output/'validation.json').write_text(json.dumps(result, indent=2)+'\n')


if __name__ == '__main__':
    main()
