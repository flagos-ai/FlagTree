"""Serialize a GPU child; retain complete, policy-bound process observations."""
import argparse
import csv
import datetime
import fcntl
import json
import os
import pathlib
import subprocess
import sys
import time

# Import the public helper adjacent to this reproduction package.
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from timing_isolation import (MONITOR_SCHEMA, OBSERVATION_SCHEMA, byte_pin, clock_identity,
                              derive_processes, load_policy, monotonic_ns, policy_environment, require, timestamp)


def utc():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def observe(root, *, observation_id, phase, policy, policy_ref, monitor_pid):
    started = monotonic_ns()
    clock = clock_identity()
    cpu_start = monotonic_ns()
    ps = subprocess.run(['ps', '-eo', 'pid=,ppid=,comm=,args='], capture_output=True, text=True)
    cpu_end = monotonic_ns()
    gpu_start = monotonic_ns()
    gpu = subprocess.run(['nvidia-smi', '--query-compute-apps=pid,gpu_uuid,used_gpu_memory',
                          '--format=csv,noheader,nounits'], capture_output=True, text=True)
    gpu_end = monotonic_ns()
    row = {'schema': OBSERVATION_SCHEMA, 'observation_id': observation_id, 'phase': phase,
           'clock': clock, 'policy': policy_ref, 'policy_limits': policy['limits'],
           'start_ns': str(started), 'end_ns': str(monotonic_ns()), 'utc': utc(),
           'cpu_start_ns': str(cpu_start), 'cpu_end_ns': str(cpu_end),
           'gpu_start_ns': str(gpu_start), 'gpu_end_ns': str(gpu_end),
           'root_pid': root, 'cpu_query_exit': ps.returncode, 'cpu_query_stderr': ps.stderr,
           'ps_raw': ps.stdout, 'gpu_query_exit': gpu.returncode, 'gpu_query_stderr': gpu.stderr,
           'gpu_raw': gpu.stdout, 'parse_error': None}
    try:
        require(clock == policy['clock'], 'Observed clock differs from frozen policy')
        row.update(derive_processes(row, policy, monitor_pid))
    except (ValueError, TypeError, KeyError, csv.Error) as error:
        # Preserve failed raw observations rather than turning them into empty
        # success lists. Independent classification reparses everything.
        row['parse_error'] = f'{type(error).__name__}: {error}'
    return row


def gpu_clear(row):
    return row.get('parse_error') is None and row.get('foreign_gpu_pids') == []


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', required=True)
    parser.add_argument('--policy', type=pathlib.Path, required=True, help='Frozen before prelaunch with timing_isolation.py --freeze-policy')
    parser.add_argument('command', nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command and args.command[0] == '--' else args.command
    if not command:
        parser.error('A child command is required')
    policy_raw = args.policy.read_bytes()
    policy, policy_ref = load_policy(policy_raw)
    require(policy['clock'] == clock_identity(), 'Frozen policy belongs to a different clock')
    out = pathlib.Path(args.output)
    out.mkdir(parents=True, exist_ok=False)
    (out / 'isolation-policy.json').write_bytes(policy_raw)
    with open('/tmp/flagmega-gpu0.lock', 'a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        clock = clock_identity()
        with (out / 'stdout.log').open('w') as stdout, (out / 'stderr.log').open('w') as stderr, (out / 'telemetry.jsonl').open('w') as log:
            observations = []
            def record(root, phase):
                row = observe(root, observation_id=len(observations), phase=phase, policy=policy,
                              policy_ref=policy_ref, monitor_pid=os.getpid())
                observations.append(row)
                log.write(json.dumps(row, allow_nan=False) + '\n')
                log.flush()
                return row
            pre = record(os.getpid(), 'prelaunch')
            common = {'schema': MONITOR_SCHEMA, 'clock': clock, 'monitor_pid': os.getpid(),
                      'policy': policy_ref, 'policy_limits': policy['limits'], 'argv': command,
                      'gpu_lease': {'path': '/tmp/flagmega-gpu0.lock', 'mode': 'exclusive', 'scope': 'prelaunch through postexit'},
                      'poll_interval_ns': policy['limits']['poll_interval_ns']}
            if timestamp(policy['created_ns']) > timestamp(pre['start_ns']) or not gpu_clear(pre):
                result = {**common, 'pid': None, 'exit': None, 'prelaunch_admitted': False,
                          'status': 'prelaunch_rejected', 'observations': len(observations),
                          'telemetry': byte_pin((out / 'telemetry.jsonl').read_bytes()),
                          'reason': 'GPU busy, incomplete process evidence, or invalid prelaunch policy; raw telemetry retained'}
                (out / 'monitor.json').write_text(json.dumps(result, indent=2) + '\n')
                raise SystemExit('GPU prelaunch admission rejected; telemetry saved')
            env = {**os.environ, **policy_environment(policy_ref)}
            started, launch_start = utc(), monotonic_ns()
            child = subprocess.Popen(command, stdout=stdout, stderr=stderr, env=env)
            launch_end = monotonic_ns()
            while True:
                observation_start_ns = monotonic_ns()
                record(child.pid, 'child')
                if child.poll() is not None:
                    break
                remaining_ns = observation_start_ns + policy['limits']['poll_interval_ns'] - monotonic_ns()
                if remaining_ns > 0:
                    time.sleep(remaining_ns / 1e9)
            terminal_observed = monotonic_ns()
            record(child.pid, 'postexit')
            result = {**common, 'pid': child.pid, 'started_utc': started, 'ended_utc': utc(),
                      'launch_start_ns': str(launch_start), 'launch_end_ns': str(launch_end),
                      'terminal_observed_ns': str(terminal_observed), 'prelaunch_admitted': True,
                      'exit': child.returncode, 'observations': len(observations),
                      'gpu_clear': all(gpu_clear(o) for o in observations),
                      'foreign_cpu_compilation_clear': all(o.get('parse_error') is None for o in observations)
                          and not any(c for o in observations for c in o['cpu_compilers'] if not c['owned']),
                      'scope': 'Whole child plus prelaunch/postexit; complete raw PS/GPU and separate command intervals. Poll sleep is not the complete CPU sampling interval.',
                      'telemetry': byte_pin((out / 'telemetry.jsonl').read_bytes())}
            (out / 'monitor.json').write_text(json.dumps(result, indent=2) + '\n')
            print(json.dumps(result, indent=2))
            raise SystemExit(child.returncode)


if __name__ == '__main__':
    main()
