"""Run the accepted ordinary full NUM/timing method with fresh public bindings.

Launch under reproduction/monitor.py. Each invocation validates eager, graph
and timed trajectories once; final qualification uses the original eight-process
order and independent predicates, not this command's successful exit alone.
"""
import argparse
import json
import os
from pathlib import Path
import sys

from reproduction.provider import checked, install, pin, require, verify_binding


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--binding', type=Path, required=True,
                        help='Root/Validator current artifact/native/source/golden/monitor bindings')
    parser.add_argument('--scenario', choices=('synthetic32', 'chinese64'), required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--preflight-only', action='store_true')
    args = parser.parse_args()
    binding_pin = pin(args.binding)
    binding = json.loads(checked(binding_pin))
    side, manifest, programs = verify_binding(binding, args.scenario)
    require(not args.output.exists() and not args.output.with_suffix('.bootstrap.json').exists(),
            'Use fresh immutable output paths')
    if args.preflight_only:
        print(json.dumps(dict(binding=binding_pin, side=side['schedule'], CPU_preflight=True,
                              GPU=False, native_loaded=False, accepted=False), indent=2))
        return
    require(os.environ.get('FLAGMEGA_GPU_LEASE_GRANTED') == '1', 'Root monitored lease required')
    from reproduction.lease import validate_open_lock
    handle = os.fdopen(os.open('/tmp/flagmega-gpu0.lock', os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW), 'r')
    try:
        lease = validate_open_lock(handle, '/tmp/flagmega-gpu0.lock', binding['monitor_expected'])
        import benchmark
        argv = [str(Path(benchmark.__file__).resolve()), '--artifact', side['artifact']['root'],
                '--output', str(args.output.resolve()), '--llama-reference', binding['goldens'][args.scenario]['root'],
                '--steps', str(32 if args.scenario == 'synthetic32' else 64), '--repeats', '3',
                '--token-id', '1', '--golden-token-prefix', '8', '--recipe', side['recipe']['path'],
                '--device', 'cuda:0', '--measurement', 'host-token']
        bootstrap = dict(schema='flagmega.i0-u4-workload-bootstrap/v1', stage=args.scenario,
            artifact=side['artifact']['root'], artifact_manifest=side['artifact']['manifest'],
            semantic_hash=manifest['semantic_hash'], plan_sha256=side['plan_sha256'],
            counter_epoch_plan_sha256=side['counter_epoch_plan_sha256'],
            original_layers=64, entries=33, compute=869, resets=side['reset_count'],
            benchmark_argv=argv, lease=lease, public_binding=binding_pin)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        benchmark.save_json(args.output.with_suffix('.bootstrap.json'), bootstrap)
        install(benchmark, side, programs, binding_pin, args.output)
        sys.argv = argv
        benchmark.main()
    finally:
        handle.close()


if __name__ == '__main__':
    main()
