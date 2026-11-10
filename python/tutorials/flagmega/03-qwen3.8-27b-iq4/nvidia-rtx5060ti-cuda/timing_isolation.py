"""Prospective, sampled round isolation; never numerical/performance acceptance.

A policy must be frozen before monitor prelaunch. Old timestamp-free or v1/v2
captures remain ineligible. Raw process polls cannot prove continuous absence.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import io
import json
import math
import os
from pathlib import Path
import re
import time

CLOCK_SCHEMA = 'flagmega.clock-monotonic/v1'
ROUND_SCHEMA = 'flagmega.round-timing/v2'
WINDOW_SCHEMA = 'flagmega.round-window/v2'
OBSERVATION_SCHEMA = 'flagmega.process-observation/v3'
MONITOR_SCHEMA = 'flagmega.process-monitor/v3'
ISOLATION_SCHEMA = 'flagmega.round-isolation/v2'
POLICY_SCHEMA = 'flagmega.round-isolation-policy/v1'
POLICY_ID = 'cpp-host-token-sampled-v2'
# Changing a limit requires a new reviewed policy ID and new captures.
LIMITS = {'poll_interval_ns': 500_000_000, 'max_observation_gap_ns': 1_000_000_000,
          'max_query_duration_ns': 1_000_000_000, 'max_cpu_sample_span_ns': 1_000_000_000,
          'max_clock_resolution_ns': 1_000}
PYTHON_RESULT = 'flagmega.cpp-decode-benchmark/v2'
NATIVE_RESULT = 'flagmega.llama-performance/v2'
LAYOUTS = {PYTHON_RESULT: 'python-repeats-by-steps', NATIVE_RESULT: 'native-flat-six-decimal-ms'}
COMPILERS = ('nvcc', 'ptxas', 'cicc', 'cc1plus', 'clang++', 'g++', 'cc1')
GPU_UUID = r'GPU-[0-9a-fA-F]{8}(?:-[0-9a-fA-F]{4}){3}-[0-9a-fA-F]{12}'
ENV_PREFIX = 'FLAGMEGA_TIMING_POLICY_'


def require(condition, message):
    if not condition:
        raise ValueError(message)


def byte_pin(raw):
    return {'sha256': hashlib.sha256(raw).hexdigest(), 'nbytes': len(raw)}


def monotonic_ns():
    return time.clock_gettime_ns(time.CLOCK_MONOTONIC)


def clock_identity():
    return {'schema': CLOCK_SCHEMA, 'name': 'CLOCK_MONOTONIC', 'clock_id': int(time.CLOCK_MONOTONIC),
            'unit': 'ns', 'resolution_ns': round(time.clock_getres(time.CLOCK_MONOTONIC) * 1e9),
            'boot_id': Path('/proc/sys/kernel/random/boot_id').read_text().strip(),
            'time_namespace': os.readlink('/proc/self/ns/time')}


def timestamp(value):
    require(isinstance(value, str) and re.fullmatch(r'[1-9][0-9]*', value)
            and int(value) < 1 << 64, 'Missing or invalid canonical monotonic timestamp')
    return int(value)


def validate_clock(clock):
    require(isinstance(clock, dict) and clock.get('schema') == CLOCK_SCHEMA
            and clock.get('name') == 'CLOCK_MONOTONIC' and clock.get('unit') == 'ns'
            and type(clock.get('clock_id')) is int and clock['clock_id'] == 1
            and type(clock.get('resolution_ns')) is int
            and 0 < clock['resolution_ns'] <= LIMITS['max_clock_resolution_ns'],
            'Unsupported or missing observed clock identity/resolution')
    require(isinstance(clock.get('boot_id'), str) and re.fullmatch(
        r'[0-9a-f]{8}(?:-[0-9a-f]{4}){3}-[0-9a-f]{12}', clock['boot_id'])
        and isinstance(clock.get('time_namespace'), str)
        and re.fullmatch(r'time:\[[0-9]+\]', clock['time_namespace']), 'Missing boot/time-namespace identity')


def load_policy(raw):
    require(isinstance(raw, bytes) and raw, 'Missing prelaunch policy bytes')
    doc = json.loads(raw)
    require(set(doc) == {'schema', 'id', 'limits', 'clock', 'created_ns', 'target_gpu_uuid'}
            and doc['schema'] == POLICY_SCHEMA and doc['id'] == POLICY_ID
            and doc['limits'] == LIMITS
            and all(type(v) is int for v in doc['limits'].values()), 'Unknown or modified frozen policy')
    validate_clock(doc['clock'])
    timestamp(doc['created_ns'])
    require(isinstance(doc['target_gpu_uuid'], str) and re.fullmatch(GPU_UUID, doc['target_gpu_uuid']),
            'Policy must bind one concrete GPU UUID')
    return doc, {'id': doc['id'], **byte_pin(raw)}


def policy_environment(reference):
    return {ENV_PREFIX + 'ID': reference['id'], ENV_PREFIX + 'SHA256': reference['sha256'],
            ENV_PREFIX + 'NBYTES': str(reference['nbytes'])}


def observed_policy():
    values = [os.environ.get(ENV_PREFIX + key) for key in ('ID', 'SHA256', 'NBYTES')]
    if all(value is None for value in values):
        return None  # Ordinary unmonitored runs are allowed, but cannot classify clear.
    name, digest, size = values
    require(name == POLICY_ID and isinstance(digest, str) and re.fullmatch('[0-9a-f]{64}', digest)
            and isinstance(size, str) and re.fullmatch('[1-9][0-9]*', size), 'Incomplete policy environment binding')
    return {'id': name, 'sha256': digest, 'nbytes': int(size)}


def timing_metadata():
    clock = clock_identity()
    info = time.get_clock_info('perf_counter')
    require(info.monotonic and not info.adjustable and info.implementation == 'clock_gettime(CLOCK_MONOTONIC)'
            and round(info.resolution * 1e9) == clock['resolution_ns'], 'perf_counter clock differs from outer clock')
    return {'schema': ROUND_SCHEMA, 'clock': clock, 'policy': observed_policy(),
            'policy_limits': LIMITS, 'sample_layout': LAYOUTS[PYTHON_RESULT], 'pid': os.getpid(), 'parent_pid': os.getppid(), 'rounds': []}


def positive_samples(samples, count):
    require(isinstance(samples, list) and len(samples) == count
            and all(type(v) in (int, float) and math.isfinite(v) and v > 0 for v in samples),
            'Samples must have the exact shape and finite positive milliseconds')


def duration_check(window, samples, clock, *, native):
    """Bound clock read quantization, float conversion and native printf rounding.

Each token has two clock reads; outer timestamps have two more. Python absolute
perf_counter floats add at most one ULP at each endpoint. Native %.6f ms adds
0.5 ns per printed token. No caller-supplied tolerance can relax containment.
"""
    start, end = timestamp(window['start_ns']), timestamp(window['end_ns'])
    resolution = clock['resolution_ns']
    absolute_ulp_ns = 0 if native else math.ulp(end / 1e9) * 1e9
    allowance = 2 * resolution + len(samples) * (2 * resolution + 2 * absolute_ulp_ns + (0.5 if native else 0))
    allowance += math.fsum(math.ulp(float(v)) * 1e6 for v in samples)
    total_ns = math.fsum(samples) * 1e6
    require(math.isfinite(total_ns) and total_ns <= end - start + math.ceil(allowance),
            'Summed token durations do not fit their recorded round window')
    return {'sum_sample_ns': total_ns, 'window_duration_ns': end - start,
            'derived_error_bound_ns': math.ceil(allowance), 'native_print_decimal_places_ms': 6 if native else None}


def validate_round_result(result, *, rounds_raw=None):
    metadata = result.get('round_timing')
    require(isinstance(metadata, dict) and metadata.get('schema') == ROUND_SCHEMA,
            'Round timestamps unavailable; legacy timing cannot be reconstructed')
    validate_clock(metadata['clock'])
    require(all(type(metadata.get(k)) is int and metadata[k] > 0 for k in ('pid', 'parent_pid')),
            'Missing timing process lineage')
    steps, repeats = result['steps'], result['repeats']
    require(type(steps) is int and steps > 0 and type(repeats) is int and repeats > 0, 'Invalid round counts')
    schema = result.get('schema')
    require(schema in LAYOUTS and metadata.get('sample_layout') == LAYOUTS[schema], 'Unknown backend sample layout')
    native = schema == NATIVE_RESULT
    samples = result['latency_ms']
    if native:
        positive_samples(samples, steps * repeats)
        batches = [samples[i * steps:(i + 1) * steps] for i in range(repeats)]
        require(all(abs(v * 1e6 - round(v * 1e6)) <= max(math.ulp(v * 1e6), 1e-9)
                    for v in samples), 'Native latency does not match six-decimal-ms output')
    else:
        require(isinstance(samples, list) and len(samples) == repeats, 'Python repeats dimension differs')
        for batch in samples:
            positive_samples(batch, steps)
        batches = samples
    windows = metadata['rounds']
    require(isinstance(windows, list) and len(windows) == repeats + 1, 'Incomplete warmup/measurement windows')
    previous = 0
    checks = []
    for repeat, window in enumerate(windows):
        require(isinstance(window, dict) and window.get('schema') == WINDOW_SCHEMA
                and type(window.get('repeat')) is int and window['repeat'] == repeat
                and window.get('warmup') is (repeat == 0), 'Invalid round identity')
        start, end = timestamp(window['start_ns']), timestamp(window['end_ns'])
        require(previous <= start < end, 'Round windows overlap or reverse time')
        previous = end
        require(type(window.get('sample_count')) is int and window['sample_count'] == steps
                and (window.get('sample_offset') is None if repeat == 0 else
                     type(window.get('sample_offset')) is int and window['sample_offset'] == (repeat - 1) * steps),
                'Round windows do not map to the measured samples')
        checks.append(duration_check(window, batches[repeat - 1], metadata['clock'], native=native) if repeat else None)
    if not native:
        require(isinstance(rounds_raw, bytes) and rounds_raw and result.get('rounds_sidecar') == byte_pin(rounds_raw),
                'Missing or changed complete Python rounds.jsonl bytes')
        rows = [json.loads(line) for line in rounds_raw.splitlines()]
        require(len(rows) == repeats + 1 and result.get('rounds') == rows[1:], 'Duplicate measured round records differ')
        for repeat, row in enumerate(rows):
            require(type(row.get('repeat')) is int and row['repeat'] == repeat
                    and row.get('warmup') is (repeat == 0) and row.get('round_window') == windows[repeat]
                    and row.get('clock') == metadata['clock'] and row.get('policy') == metadata.get('policy')
                    and row.get('policy_limits') == metadata.get('policy_limits'),
                    'Sidecar round window/clock/policy differs')
            positive_samples(row['latency_ms'], steps)
            require(type(row.get('sample_count')) is int and row['sample_count'] == steps, 'Sidecar sample count differs')
            if repeat:
                require(row['latency_ms'] == batches[repeat - 1], 'Sidecar measured samples differ')
            checks[repeat] = duration_check(windows[repeat], row['latency_ms'], metadata['clock'], native=False)
    return windows, checks


def parse_ps(raw):
    require(isinstance(raw, str) and raw.strip(), 'Missing/empty full PS snapshot')
    rows, seen = [], set()
    for line in raw.splitlines():
        require(bool(line.strip()), 'Unexpected empty row in PS snapshot')
        fields = line.split(None, 3)
        require(len(fields) == 4 and re.fullmatch('[1-9][0-9]*', fields[0])
                and re.fullmatch('0|[1-9][0-9]*', fields[1]) and fields[2] and fields[3].strip(),
                'Unparseable full PS snapshot row')
        pid, ppid = int(fields[0]), int(fields[1])
        require(pid not in seen and pid != ppid, 'Duplicate/self-parent PS PID')
        seen.add(pid)
        rows.append({'pid': pid, 'ppid': ppid, 'comm': fields[2], 'argv': fields[3]})
    by_pid = {row['pid']: row for row in rows}
    for row in rows:
        chain, pid = set(), row['pid']
        while pid in by_pid:
            require(pid not in chain, 'Cyclic process ownership graph')
            chain.add(pid)
            pid = by_pid[pid]['ppid']
    return rows


def parse_gpu(raw, target_uuid):
    require(isinstance(raw, str), 'Missing GPU raw output')
    rows, seen = [], set()
    for line in raw.splitlines():
        require(bool(line.strip()), 'Unexpected empty GPU CSV row')
        cells = next(csv.reader([line], strict=True))
        fields = [cell.strip() for cell in cells]
        require(len(fields) == 3 and re.fullmatch('[1-9][0-9]*', fields[0])
                and re.fullmatch(GPU_UUID, fields[1]) and re.fullmatch('0|[1-9][0-9]*', fields[2]),
                'Unparseable GPU process CSV row')
        require(fields[1] == target_uuid, 'GPU query contains a device outside the frozen policy')
        pid = int(fields[0])
        require(pid not in seen, 'Duplicate GPU process row')
        seen.add(pid)
        rows.append({'pid': pid, 'gpu_uuid': fields[1], 'used_gpu_memory_mib': int(fields[2])})
    return rows


def derive_processes(row, policy, monitor_pid):
    require(type(row.get('root_pid')) is int and row['root_pid'] > 0, 'Missing observed root PID')
    for prefix in ('cpu', 'gpu'):
        require(type(row.get(prefix + '_query_exit')) is int and row[prefix + '_query_exit'] == 0
                and row.get(prefix + '_query_stderr') == '', 'CPU/GPU query failed or emitted unknown diagnostics')
    processes = parse_ps(row.get('ps_raw'))
    require(any(p['pid'] == monitor_pid for p in processes), 'Full PS snapshot omits the active monitor')
    by_pid = {p['pid']: p for p in processes}
    # Do not invent ownership by seeding an absent/exited root. An absent root
    # with no GPU rows needs no GPU owner; any unresolved active PID fails closed.
    owned = {row['root_pid']} if row['root_pid'] in by_pid else set()
    while True:
        grown = owned | {p['pid'] for p in processes if p['ppid'] in owned}
        if grown == owned:
            break
        owned = grown
    gpu = parse_gpu(row.get('gpu_raw'), policy['target_gpu_uuid'])
    active = {p['pid'] for p in gpu}
    require(active <= set(by_pid), 'Active GPU PID has no owner evidence in the full PS snapshot')
    return {'processes': processes, 'owned_pids': sorted(owned), 'gpu_processes': gpu,
            'foreign_gpu_pids': sorted(active - owned),
            'gpu_process_details': [p for p in processes if p['pid'] in active],
            'cpu_compilers': [{'pid': p['pid'], 'owned': p['pid'] in owned, 'comm': p['comm'], 'argv': p['argv']}
                              for p in processes if p['comm'] in COMPILERS]}


def classify_rounds(result, monitor, observations, *, telemetry_raw, policy_raw, rounds_raw=None):
    """Keep inclusive boundary samples; do not infer when interference departed."""
    policy, policy_ref = load_policy(policy_raw)
    windows, duration_checks = validate_round_result(result, rounds_raw=rounds_raw)
    metadata = result['round_timing']
    require(metadata.get('policy') == policy_ref and metadata.get('policy_limits') == policy['limits'], 'Timing result is not bound to this prelaunch policy')
    require(monitor.get('schema') == MONITOR_SCHEMA, 'Timestamped monitor v3 required; old exclusions remain unchanged')
    require(isinstance(telemetry_raw, bytes) and telemetry_raw and monitor.get('telemetry') == byte_pin(telemetry_raw)
            and [json.loads(line) for line in telemetry_raw.splitlines()] == observations, 'Raw telemetry differs from monitor byte pin/decoded observations')
    require(monitor.get('policy') == policy_ref and monitor.get('policy_limits') == policy['limits'], 'Monitor frozen policy differs')
    require(monitor.get('clock') == metadata['clock'] == policy['clock'], 'Monitor and round clocks are not comparable')
    require(type(monitor.get('pid')) is int and monitor['pid'] > 0
            and monitor['pid'] in (metadata['pid'], metadata['parent_pid']), 'Round process does not belong to monitored child')
    require(type(monitor.get('exit')) is int and monitor['exit'] == 0, 'Child did not finish successfully')
    require(type(monitor.get('monitor_pid')) is int and monitor['monitor_pid'] > 0
            and monitor.get('gpu_lease') == {'path': '/tmp/flagmega-gpu0.lock', 'mode': 'exclusive',
                                             'scope': 'prelaunch through postexit'}, 'Missing shared GPU lease provenance')
    require(monitor.get('poll_interval_ns') == policy['limits']['poll_interval_ns'], 'Monitor polling policy differs')
    require(isinstance(observations, list) and len(observations) >= 3, 'No complete raw monitor observations')
    bounds, cpu_bounds, previous, phases = [], [], 0, []
    for index, row in enumerate(observations):
        require(row.get('schema') == OBSERVATION_SCHEMA and row.get('clock') == metadata['clock']
                and row.get('policy') == policy_ref and row.get('policy_limits') == policy['limits']
                and type(row.get('observation_id')) is int and row['observation_id'] == index,
                'Observation schema/clock/policy/sequence mismatch')
        start, end = timestamp(row['start_ns']), timestamp(row['end_ns'])
        cpu_start, cpu_end = timestamp(row['cpu_start_ns']), timestamp(row['cpu_end_ns'])
        gpu_start, gpu_end = timestamp(row['gpu_start_ns']), timestamp(row['gpu_end_ns'])
        require(previous <= start <= cpu_start < cpu_end <= gpu_start < gpu_end <= end,
                'Observation/query intervals overlap or reverse time')
        previous = end
        phases.append(row.get('phase'))
        require(row.get('root_pid') == (monitor['pid'] if index else monitor['monitor_pid']), 'Observation belongs to another process')
        derived = derive_processes(row, policy, monitor['monitor_pid'])
        require(row.get('parse_error') is None and all(row.get(key) == value for key, value in derived.items()),
                'Derived process lists differ from complete raw snapshots')
        bounds.append((start, end))
        cpu_bounds.append((cpu_start, cpu_end))
    require(phases[0] == 'prelaunch' and phases[-1] == 'postexit'
            and all(phase == 'child' for phase in phases[1:-1]), 'Incomplete prelaunch/child/postexit coverage')
    require(timestamp(policy['created_ns']) <= bounds[0][0], 'Policy was created after prelaunch observation began')
    require(type(monitor.get('observations')) is int and monitor['observations'] == len(observations), 'Monitor observation count differs')
    launch_start, launch_end = timestamp(monitor['launch_start_ns']), timestamp(monitor['launch_end_ns'])
    terminal = timestamp(monitor['terminal_observed_ns'])
    require(bounds[0][1] <= launch_start <= launch_end <= bounds[1][0]
            and bounds[-2][1] <= terminal <= bounds[-1][0], 'Launch/exit timing lacks enclosing observations')
    require(all(launch_start <= timestamp(w['start_ns']) < timestamp(w['end_ns']) <= terminal for w in windows), 'Round outside child lifetime')
    require(any(metadata['pid'] in row['owned_pids'] for row in observations[1:]), 'Timing process never observed owned by child')
    lineage = [p for row in observations[1:] for p in row['processes'] if p['pid'] == metadata['pid']]
    require(lineage and all(p['ppid'] == metadata['parent_pid'] for p in lineage),
            'Timing process lineage differs from complete PS snapshots')
    for row in observations[1:]:
        # During a complete PS interval inside an active round, its timing
        # process cannot already have exited. Post-round boundary samples may
        # legitimately omit the exited child when the GPU process list is empty.
        if any(timestamp(w['start_ns']) <= timestamp(row['cpu_start_ns'])
               and timestamp(row['cpu_end_ns']) <= timestamp(w['end_ns']) for w in windows):
            require(metadata['pid'] in row['owned_pids'], 'Active round lacks observed timing-process ownership')
    gpu_clear = all(not row['foreign_gpu_pids'] for row in observations)
    whole_cpu_clear = not any(c for row in observations for c in row['cpu_compilers'] if not c['owned'])
    require(monitor.get('gpu_clear') is gpu_clear and monitor.get('foreign_cpu_compilation_clear') is whole_cpu_clear,
            'Whole-child summary flags differ from raw observations')
    require(not observations[0]['foreign_gpu_pids'] and monitor.get('prelaunch_admitted') is True, 'Prelaunch GPU admission failed')
    verdicts = []
    for window, duration in zip(windows, duration_checks):
        start, end = timestamp(window['start_ns']), timestamp(window['end_ns'])
        left = next((i for i in range(len(bounds) - 1, -1, -1) if bounds[i][1] <= start), None)
        right = next((i for i, bound in enumerate(bounds) if bound[0] >= end), None)
        row = {'repeat': window['repeat'], 'warmup': window['warmup'], 'window': window, 'duration_check': duration,
               'whole_child_gpu_clear': gpu_clear, 'eligible_isolation': False,
               'continuous_absence_proven': False, 'status': 'unavailable', 'reasons': []}
        if left is None or right is None or left >= right:
            row['reasons'].append('missing_fully_enclosing_observations')
            verdicts.append(row)
            continue
        selected = observations[left:right + 1]
        gap = max(bounds[i + 1][0] - bounds[i][1] for i in range(left, right))
        query_duration = max(b - a for a, b in bounds[left:right + 1])
        # PS can sample anywhere inside its own command interval. This upper
        # bound includes both endpoint query durations plus the intervening gap.
        cpu_span = max(cpu_bounds[i + 1][1] - cpu_bounds[i][0] for i in range(left, right))
        compilers = [{'observation_id': observation['observation_id'], **compiler}
                     for observation in selected for compiler in observation['cpu_compilers']]
        row.update(enclosing_observation_ids=[left, right], checked_observation_ids=list(range(left, right + 1)),
                   cpu_compilers=compilers,
                   uncertainty={'enclosing_start_ns': str(bounds[left][0]), 'enclosing_end_ns': str(bounds[right][1]),
                                'left_margin_ns': start - bounds[left][0], 'right_margin_ns': bounds[right][1] - end,
                                'maximum_observation_gap_ns': gap, 'maximum_full_query_envelope_ns': query_duration,
                                'maximum_cpu_sample_span_ns': cpu_span, 'limits': policy['limits']})
        if not gpu_clear:
            row['reasons'].append('whole_child_gpu_isolation_failed')
        if gap > LIMITS['max_observation_gap_ns'] or query_duration > LIMITS['max_query_duration_ns'] or cpu_span > LIMITS['max_cpu_sample_span_ns']:
            row['reasons'].append('sampling_uncertainty_exceeds_policy')
        if compilers:
            row['reasons'].append('compiler_observed_in_enclosing_window')
        row.update(status='sampled_clear' if not row['reasons'] else 'excluded', eligible_isolation=not row['reasons'])
        verdicts.append(row)
    return {'schema': ISOLATION_SCHEMA, 'clock': metadata['clock'], 'policy': policy_ref, 'policy_limits': policy['limits'],
            'rounds': verdicts, 'all_measured_rounds_sampled_clear': all(v['eligible_isolation'] for v in verdicts if not v['warmup']),
            'performance_acceptance': False, 'numerical_acceptance': False,
            'limitation': 'Inclusive enclosing sampled process queries cannot prove continuous absence; old results remain excluded.'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--freeze-policy', type=Path)
    parser.add_argument('--target-gpu-uuid')
    parser.add_argument('--result', type=Path)
    parser.add_argument('--monitor-dir', type=Path)
    parser.add_argument('--rounds', type=Path, help='Exact Python rounds.jsonl sidecar; default next to result')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    if args.freeze_policy:
        require(args.target_gpu_uuid and not any((args.result, args.monitor_dir, args.rounds, args.output)), 'Policy freezing is a separate prelaunch operation')
        policy = {'schema': POLICY_SCHEMA, 'id': POLICY_ID, 'limits': LIMITS, 'clock': clock_identity(),
                  'created_ns': str(monotonic_ns()), 'target_gpu_uuid': args.target_gpu_uuid}
        raw = (json.dumps(policy, indent=2, allow_nan=False) + '\n').encode()
        _, ref = load_policy(raw)
        with args.freeze_policy.open('xb') as stream:
            stream.write(raw)
        print(json.dumps(ref))
        return
    require(args.result and args.monitor_dir and args.output and not args.target_gpu_uuid, 'Classification needs result, monitor-dir and output')
    require(not args.output.exists(), 'Use a fresh classification path; old exclusions are immutable')
    paths = {'result': args.result, 'monitor': args.monitor_dir / 'monitor.json',
             'telemetry': args.monitor_dir / 'telemetry.jsonl', 'policy': args.monitor_dir / 'isolation-policy.json'}
    raw = {}
    try:
        for key, path in paths.items():
            raw[key] = path.read_bytes()
        result = json.loads(raw['result'])
        if result.get('schema') == PYTHON_RESULT:
            paths['rounds'] = args.rounds or args.result.with_suffix('.rounds.jsonl')
            raw['rounds'] = paths['rounds'].read_bytes()
        report = classify_rounds(result, json.loads(raw['monitor']), [json.loads(line) for line in raw['telemetry'].splitlines()],
                                 telemetry_raw=raw['telemetry'], policy_raw=raw['policy'], rounds_raw=raw.get('rounds'))
    except (ValueError, KeyError, TypeError, OSError, csv.Error) as error:
        report = {'schema': ISOLATION_SCHEMA, 'status': 'unavailable', 'all_measured_rounds_sampled_clear': False,
                  'performance_acceptance': False, 'numerical_acceptance': False, 'reason': f'{type(error).__name__}: {error}'}
    report['inputs'] = {key: {'path': str(paths[key].resolve()), **byte_pin(data)} for key, data in raw.items()}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x') as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write('\n')
    print(json.dumps(report, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
