"""Build the current FlagOS profile and collect three rotated comparison rounds."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--artifact', type=Path)
    parser.add_argument('--work-dir', type=Path)
    parser.add_argument('--results', type=Path, required=True)
    parser.add_argument('--rounds', type=int, default=3)
    parser.add_argument('--repeats', type=int, default=3)
    args = parser.parse_args()
    if args.rounds < 1 or args.repeats < 1:
        parser.error('Require positive rounds/repeats')
    if args.artifact is None and args.work_dir is None:
        parser.error('Provide --artifact or a fresh --work-dir for compilation')
    args.results.mkdir(parents=True, exist_ok=False)
    tutorial = Path(__file__).resolve().parent
    env = dict(os.environ)
    env.update(CUDA_MODULE_LOADING='0', VLLM_PLUGINS='fl',
               FLAGOS_DEVICE_CONTROL_ENV_VAR='CUDA_VISIBLE_DEVICES',
               VLLM_FL_FLAGOS_BLACKLIST_APPEND='sort,sort_stable',
               VLLM_NO_USAGE_STATS='1', HF_HUB_OFFLINE='1', TOKENIZERS_PARALLELISM='false',
               OMP_NUM_THREADS='1')
    env['PYTHONPATH'] = str(tutorial) + os.pathsep + env.get('PYTHONPATH', '')

    def run(command, logfile):
        with logfile.open('w') as log:
            subprocess.run(command, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)

    artifact = args.artifact
    if artifact is None:
        run([sys.executable, str(tutorial/'optimize.py'), '--checkpoint', str(args.checkpoint),
             '--work-dir', str(args.work_dir), '--profile', 'serving',
             '--numerical-profile', 'vllm-flagos-eager-bf16'], args.results/'compile.log')
        artifact = args.work_dir/'serving/artifact'
    run([sys.executable, '-m', 'triton.flagmega', 'artifact', 'verify', str(artifact)],
        args.results/'artifact_verify.json')
    reference = args.results/'round00.native.json'
    for round_id in range(args.rounds):
        order = ('native', 'flagmega') if round_id % 2 == 0 else ('flagmega', 'native')
        for variant in order:
            output = args.results/f'round{round_id:02d}.{variant}.json'
            cmd = [sys.executable, str(tutorial/'benchmark_flagos.py'), '--checkpoint', str(args.checkpoint),
                   '--output', str(output), '--repeats', str(args.repeats), '--rotation', str(round_id)]
            if variant == 'flagmega': cmd.extend(['--artifact', str(artifact)])
            if variant == 'flagmega' or round_id: cmd.extend(['--reference', str(reference)])
            run(cmd, output.with_suffix('.log'))
            report = json.loads(output.read_text())
            if not report['performance_valid']: raise AssertionError(f'Unaccepted result: {output}')
            print(f'Completed round {round_id} {variant}', flush=True)
    run([sys.executable, str(tutorial/'render_flagos_results.py'), '--results', str(args.results)],
        args.results/'render.log')


if __name__ == '__main__':
    main()
