"""Collect and audit the admitted CPP operator diagnostic outside acceptance timing.

The v2 compiler protocol admits static flat uniform SM120 schedules and derives
CTA/event counts from the artifact. The v1 reviewed96-CTA protocol remains supported.
Unsupported control/role schedules are rejected rather than assigned operator times.
This file can also render an existing raw archive with no CUDA initialization.
"""

from __future__ import annotations

# A public file named profile.py must remain compatible with cProfile imports
# reached through torch. Delegate those three stdlib symbols to their real file.
if __name__ == 'profile':
    import importlib.util as _importlib_util
    import sysconfig as _sysconfig
    from pathlib import Path as _Path
    _spec = _importlib_util.spec_from_file_location('_stdlib_profile', _Path(_sysconfig.get_path('stdlib')) / 'profile.py')
    _stdlib = _importlib_util.module_from_spec(_spec)
    _spec.loader.exec_module(_stdlib)
    run, runctx, _Utils = _stdlib.run, _stdlib.runctx, _stdlib._Utils

import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics
import struct
import sys

from benchmark import DecodeContext, Golden, append_json, artifact_identity, file_pin, install_recipe, save_json, utc_now


def write_bytes(path, data):
    with Path(path).open('xb') as stream:
        stream.write(data)
        stream.flush()
        import os
        os.fsync(stream.fileno())
    return file_pin(path)


def check_pin(pin):
    actual = file_pin(pin['path'])
    if actual != pin:
        raise ValueError(f'Pinned data changed: {pin["path"]}')
    return Path(pin['path'])


def audit_frame(raw, *, event_count, cta_count, epoch):
    """Validate every raw cell before integer subtraction or aggregation."""
    import numpy as np
    if len(raw) != event_count * cta_count * 32:
        raise ValueError('Incomplete raw frame')
    cells = np.frombuffer(raw, dtype='<u8').reshape(event_count, cta_count, 4)
    if (not np.all(cells[:, :, 2] == epoch)
            or not np.all(cells[:, :, 3] == np.arange(event_count, dtype=np.uint64)[:, None])
            or not np.all(cells[:, :, 1] >= cells[:, :, 0])):
        raise ValueError('Stale tags/epochs or reversed raw intervals')
    if not np.all(cells[1:, :, 0] >= cells[:-1, :, 1]):
        raise ValueError('Per-CTA sequential event ordering failed')
    return (cells[:, :, 1] - cells[:, :, 0]).astype(np.float64)


def durable_capture_sink(output, archive, runtime, location):
    """Acknowledge complete raw cells only after bytes and index are durable."""
    index = output / 'archive.json'

    def sink(epoch, raw):
        binary = write_bytes(output / f'frame-{epoch:06d}.bin', raw)
        frame = {'launch_epoch': str(epoch), 'known_GPU_completion': True, 'binary': binary,
                 'event_observation': runtime.profile_event_observation, **location}
        archive['frames'].append(frame)
        save_json(index, archive)
        return file_pin(index)

    def failed(epoch, raw, error, status):
        row = {'launch_epoch': str(epoch), 'known_GPU_completion': status['known_GPU_completion'],
               'error': str(error), 'failure': status, **location}
        if raw is not None:
            row['binary'] = write_bytes(output / f'failed-frame-{epoch:06d}.bin', raw)
        archive['failed_frames'].append(row)
        archive['status'] = 'failed'
        save_json(index, archive)
        return file_pin(index)

    return sink, failed


def calibrate(context, output):
    """Use the compiler's bounded integer clock probe and unmodified fit rule."""
    from triton.flagmega.runtime.profile_clock import checksum, fit_clock
    samples = []
    long_lengths = []
    calibration_dir = output / 'clock'
    calibration_dir.mkdir()
    for length in [1 << 12, *[1 << k for k in range(20, 27)]]:
        batch = []
        for repetition in range(1 if length == 1 << 12 else 3):
            epoch = len(samples) + 1
            observation = context.runtime.calibrate_profile_clock(length, 123 + repetition, epoch, stream=context.stream)
            raw = observation.pop('raw')
            raw_pin = write_bytes(calibration_dir / f'{epoch:03d}.bin', raw)
            document = {'schema': 'flagmega.profile-clock-observation/v1', **observation,
                        'raw_sha256': raw_pin['sha256'], 'binary': raw_pin}
            path = calibration_dir / f'{epoch:03d}.json'
            save_json(path, document)
            context.runtime.acknowledge_clock(epoch, file_pin(path))
            local_check = (int(observation['checksum']) == checksum(length, observation['seed'])
                           and observation['event_status'] == 0
                           and int(observation['end_tick']) > int(observation['start_tick']))
            sample = {**observation, 'guard_pass': local_check, 'observation': file_pin(path),
                      'guard_scope': 'bounded checksum, event status, owned-stream completion, monotone ticks; external isolation separately required'}
            samples.append(sample)
            batch.append(sample)
            append_json(calibration_dir / 'samples.jsonl', sample)
        if len(batch) == 3 and min(s['event_elapsed_ms'] for s in batch) >= 1:
            long_lengths.append(length)
        if len(long_lengths) >= 3:
            break
    fit = fit_clock(samples, data_origin='actual_execution')
    save_json(calibration_dir / 'fit.json', fit)
    if not fit['criteria_satisfied'] or fit['ns_per_tick'] is None:
        raise RuntimeError('Device clock calibration failed the predeclared fit; raw data retained without latency bars')
    return {'fit': file_pin(calibration_dir / 'fit.json'), 'ns_per_tick': fit['ns_per_tick'],
            'independent_validation': 'pending', 'external_isolation': 'must be supplied by Verifier'}


def audit_loaded_identity(archive):
    """Bind runtime/clock provenance to the pinned compiled artifact and device.

    Recompute artifact fields from their actual files, not an archive's digest
    labels. The runtime device observation is the anchor for every calibration
    sample; external device-isolation validation remains a separate gate.
    """
    from triton.flagmega.runtime.operator_profile import loaded_identity, digest
    manifest_path = check_pin(archive['artifact_identity']['manifest'])
    manifest, actual_pins = artifact_identity(manifest_path.parent)
    if archive['artifact_identity'] != actual_pins:
        raise ValueError('Archive artifact pins differ from the actual compiled artifact')
    codegen = manifest['codegen']
    if codegen.get('operator_profile') != archive['contract']:
        raise ValueError('Archive profile contract differs from its pinned artifact')
    contract = archive['contract']
    if digest(contract.get('implementation_sources')) != contract.get('implementation_sha256'):
        raise ValueError('Profile implementation digest differs from its source inventory')
    expected = loaded_identity(manifest_path.parent, manifest, manifest_path.parent / codegen['library'])
    identity = archive.get('loaded_profile_identity')
    device_keys = {'device_UUID', 'device_ordinal', 'compute_capability', 'SM_count', 'driver_version'}
    if not isinstance(identity, dict) or set(identity) != set(expected) | device_keys:
        raise ValueError('Complete loaded artifact/device identity is required')
    if any(identity[key] != value for key, value in expected.items()):
        raise ValueError('Loaded profile identity differs from the pinned artifact')
    import re
    if (not isinstance(identity['device_UUID'], str)
            or re.fullmatch(r'GPU-[0-9a-fA-F]{8}(?:-[0-9a-fA-F]{4}){3}-[0-9a-fA-F]{12}', identity['device_UUID']) is None
            or type(identity['device_ordinal']) is not int or identity['device_ordinal'] < 0
            or type(identity['SM_count']) is not int or identity['SM_count'] <= 0
            or type(identity['driver_version']) is not int or identity['driver_version'] <= 0
            or identity['compute_capability'] != codegen['launch_resource_limits']['compute_capability']
            or not isinstance(identity['compute_capability'], list)
            or any(type(value) is not int for value in identity['compute_capability'])):
        raise ValueError('Invalid loaded device identity or target capability mismatch')
    if archive.get('resources', {}).get('available_sm_count') != identity['SM_count']:
        raise ValueError('Loaded device identity differs from the captured resource observation')
    return manifest, identity


def audit_archive(path):
    """Recompute every bar from every raw CTA cell; aggregate flags are ignored."""
    sys.path.insert(0, str(Path(__file__).resolve().parents[4]))
    from triton.flagmega.runtime.profile_clock import checksum, fit_clock
    path = Path(path).resolve()
    archive = json.loads(path.read_text())
    if archive.get('data_origin') != 'actual_execution':
        raise ValueError('Synthetic or unspecified origins cannot render actual latency bars')
    if archive.get('schema') != 'flagmega.provisional-profile-archive/v1' or archive['status'] != 'complete_raw':
        raise ValueError('Only a complete raw archive can be rendered')
    if archive.get('failed_frames'):
        raise ValueError('Failed profile frames require a separate repair; no actual latency bars')
    contract = archive['contract']
    from triton.flagmega.runtime.operator_profile import ProfileSession, digest
    manifest, identity = audit_loaded_identity(archive)
    codegen = manifest['codegen']
    ProfileSession(contract, grid=codegen['grid'], block_threads=codegen['block_threads'])
    for name in ('site_inventory', 'invocation_manifest', 'sync_manifest'):
        if digest(contract[name]) != contract[name + '_sha256']:
            raise ValueError(f'Profile contract {name} hash mismatch')
    for pin in archive['artifact_identity'].values():
        check_pin(pin)
    fit_doc = json.loads(check_pin(archive['clock']['fit']).read_text())
    samples = fit_doc['attempted_samples']
    for sample in samples:
        observation = json.loads(check_pin(sample['observation']).read_text())
        if (observation.get('schema') != 'flagmega.profile-clock-observation/v1'
                or observation.get('identity') != identity or sample.get('identity') != identity):
            raise ValueError('Clock calibration identity differs from the profiled artifact/device')
        raw = check_pin(observation['binary']).read_bytes()
        start, end, epoch, actual_checksum = struct.unpack('<QQQQ', raw)
        if (str(start) != observation['start_tick'] or str(end) != observation['end_tick']
                or str(epoch) != observation['epoch'] or str(actual_checksum) != observation['checksum']
                or hashlib.sha256(raw).hexdigest() != observation['raw_sha256']
                or actual_checksum != checksum(observation['iterations'], observation['seed'])):
            raise ValueError('Clock raw cells differ from observation')
        for key, value in observation.items():
            if key not in ('schema', 'raw_sha256', 'binary') and sample.get(key) != value:
                raise ValueError(f'Clock fit input differs from observed {key}')
    fit = fit_clock(samples, data_origin='actual_execution')
    if not fit['criteria_satisfied'] or fit['ns_per_tick'] != 1.0:
        raise ValueError('Actual raw clock evidence does not establish latency units')
    events = contract['event_count']
    ctas = contract['grid_total']
    site_map = {site['site_id']: site for site in contract['site_inventory']}
    invocations = contract['invocation_manifest']
    syncs = contract['sync_manifest']
    all_indices = [row['event_index'] for row in (*invocations, *syncs)]
    if len(set(all_indices)) != events or sorted(all_indices) != list(range(events)):
        raise ValueError('Invocation/synchronization census does not cover every event exactly once')
    totals = {name: [] for name in site_map}
    sync_totals = {}
    frame_receipts = []
    seen_epochs = set()
    for frame in archive['frames']:
        epoch = int(frame['launch_epoch'])
        if epoch in seen_epochs or not frame['known_GPU_completion']:
            raise ValueError('Repeated epoch or unknown stream completion')
        seen_epochs.add(epoch)
        raw = check_pin(frame['binary']).read_bytes()
        durations = audit_frame(raw, event_count=events, cta_count=ctas, epoch=epoch)
        event = frame['event_observation']
        if (event['epoch'] != str(epoch) or event['event_status'] != 0
                or not all(math.isfinite(event[k]) and event[k] >= 0 for k in ('reset_inclusive_ms', 'kernel_only_ms', 'reset_ms'))):
            raise ValueError('Frame CUDA event observation failed')
        frame_receipts.append({'epoch': epoch, 'phase': frame['phase'], 'raw': frame['binary'],
                               'kernel_only_ms': event['kernel_only_ms'],
                               'CTA_mean_instrumented_event_ms': float(durations.sum(0).mean() / 1e6)})
        if frame['phase'] != 'profile':
            continue
        for name in site_map:
            selected = [row['event_index'] for row in invocations if row['site_id'] == name]
            if len(selected) != site_map[name]['expected_invocations_per_token']:
                raise ValueError('Executed operator invocation count differs from static census')
            totals[name].append(float(durations[selected].sum(0).mean() / 1e6))
        for scope in sorted({row['scope'] for row in syncs}):
            selected = [row['event_index'] for row in syncs if row['scope'] == scope]
            sync_totals.setdefault(scope, []).append(float(durations[selected].sum(0).mean() / 1e6))
    count = sum(frame['phase'] == 'profile' for frame in archive['frames'])
    if count != archive['steps'] or not count:
        raise ValueError('Raw archive lacks every requested profile token')
    operators = []
    for name, site in site_map.items():
        operators.append({**site, 'samples_ms_per_token': totals[name], 'mean_ms_per_token': statistics.mean(totals[name]),
                          'profile_tokens': count, 'total_calls': count * site['expected_invocations_per_token']})
    operators.sort(key=lambda row: row['mean_ms_per_token'], reverse=True)
    return {'schema': 'flagmega.cpp-operator-profile-summary/v1', 'data_origin': 'actual_execution',
            'archive': file_pin(path), 'artifact_identity': archive['artifact_identity'],
            'clock_audited': True, 'raw_frames_audited': frame_receipts, 'all_operator_sites': len(operators),
            'profile_tokens': count, 'steps': archive['steps'], 'CTA_count': ctas,
            'aggregation': 'Per token: sum this operator site over all layer calls for each CTA, then mean across CTAs; mean across tokens',
            'unit': 'ms per token (CTA mean)', 'includes': 'Instrumented leaf wrapper intervals, including internal waits',
            'excludes': 'Caller schedule barriers shown separately; parent containers excluded to avoid nested double counting',
            'attribution_limits': 'Not end-to-end latency or critical-path attribution; CTA skew/overlap is not summed as kernel time; extra instrumentation changes timing',
            'performance_acceptance': False, 'instrumentation_overhead': 'unmeasured; separately built diagnostic artifact',
            'independent_validation': 'pending; this runner performs a full raw audit, not independent acceptance',
            'monitor_status': archive['monitor_status'], 'numerical_status': archive.get('numerical_status'),
            'operators': operators, 'schedule_barriers': [
                {'scope': name, 'samples_ms_per_token': values, 'mean_ms_per_token': statistics.mean(values)}
                for name, values in sync_totals.items()]}


def render(summary, path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    rows = summary['operators']
    labels = [row['site_id'] + '  ' + row['family'] for row in rows]
    fig, ax = plt.subplots(figsize=(12, max(7, len(rows) * .24 + 2)))
    ax.barh(range(len(rows)), [row['mean_ms_per_token'] for row in rows], color='#247aa8')
    ax.set_yticks(range(len(rows)), labels, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel('Accumulated operator time per token, mean across CTAs (ms)')
    ax.set_title(f'CPP diagnostic: all {len(rows)} sites, {summary["profile_tokens"]} tokens\nInstrumented eager execution; not acceptance latency')
    fig.text(.02, .008, 'Caller barriers excluded; internal waits included. CTA skew/overlap prevents interpreting sums as end-to-end latency.', fontsize=8)
    fig.tight_layout(rect=(0, .025, 1, 1))
    fig.savefig(path, format='svg')
    plt.close(fig)


def capture(args):
    install_recipe(args.recipe)
    manifest, pins = artifact_identity(args.artifact)
    contract = manifest['codegen'].get('operator_profile')
    if not contract:
        raise ValueError('Artifact has no admitted CPP operator profile; rebuild using --cpp-operator-profile cta-globaltimer-v2 if its schedule is supported')
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    archive = {'schema': 'flagmega.provisional-profile-archive/v1', 'status': 'capture_pending',
               'data_origin': 'actual_execution',
               'started_utc': utc_now(), 'artifact_identity': pins, 'contract': contract,
               'steps': args.steps, 'frames': [], 'failed_frames': [], 'harness': file_pin(__file__),
               'monitor_status': 'not_checked_by_harness; Verifier must supply external CPU/GPU isolation evidence',
               'numerical_status': 'not_run', 'accepted_timing_eligible': False}
    save_json(output / 'archive.json', archive)
    import torch
    torch.set_num_threads(1)
    context = DecodeContext(args.artifact, token_id=args.token_id, device=args.device)
    location = {'phase': 'warmup', 'step': 0}
    sink, failure_sink = durable_capture_sink(output, archive, context.runtime, location)
    context.runtime.attach_profile_sink(sink, failure_sink=failure_sink, stream_id=int(context.stream.cuda_stream))
    try:
        archive['loaded_profile_identity'] = context.runtime.profile_identity
        archive['resources'] = context.runtime.resource_report
        archive['clock'] = calibrate(context, output)
        golden = Golden(args.llama_reference, steps=args.steps, prefix=8, context=context) if args.llama_reference else None
        tokens = []
        context.reset()
        for step in range(args.warmup):
            location.update(phase='warmup', step=step)
            context.launch()
        context.reset()
        for step in range(args.steps):
            location.update(phase='profile', step=step)
            if golden:
                golden.inject(context, step)
            context.launch()
            if not torch.isfinite(context.logits).all().item():
                raise RuntimeError(f'Non-finite diagnostic logits at step {step}')
            tokens.append(int(context.token.item()))
        context.check_state(args.steps)
        archive['tokens'] = tokens
        archive['numerical_status'] = 'finite_state_and_logits_no_golden'
        if golden:
            archive['golden_identity'] = golden.pins
            archive['prefix'] = {'start_step': golden.start, 'required': 8,
                                  'actual': tokens[golden.start:golden.end],
                                  'golden': golden.metadata['tokens'][golden.start:golden.end]}
            passed = archive['prefix']['actual'] == archive['prefix']['golden']
            archive['prefix']['passed'] = passed
            archive['numerical_status'] = 'independent_prefix_passed' if passed else 'independent_prefix_failed'
        archive.update(status='complete_raw', ended_utc=utc_now())
        save_json(output / 'archive.json', archive)
        summary = audit_archive(output / 'archive.json')
        save_json(output / 'operator-latency.json', summary)
        render(summary, output / 'operator-latency.svg')
        return output / 'operator-latency.svg'
    except BaseException as error:
        archive.update(status='failed', error=f'{type(error).__name__}: {error}', ended_utc=utc_now())
        save_json(output / 'archive.json', archive)
        raise
    finally:
        context.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--artifact', type=Path)
    parser.add_argument('--output', type=Path, required=True, help='Fresh capture directory, or SVG path with --render-only')
    parser.add_argument('--render-only', type=Path, metavar='ARCHIVE_JSON', help='Audit all raw cells and render without CUDA')
    parser.add_argument('--llama-reference', type=Path)
    parser.add_argument('--recipe', type=Path)
    parser.add_argument('--steps', type=int, default=32)
    parser.add_argument('--warmup', type=int, default=1)
    parser.add_argument('--token-id', type=int, default=1)
    parser.add_argument('--device', default='cuda:0')
    args = parser.parse_args()
    if args.output.exists():
        parser.error('Use a fresh output path')
    if args.steps < 1 or args.warmup < 0:
        parser.error('steps must be positive and warmup nonnegative')
    if args.render_only:
        if args.artifact:
            parser.error('--artifact and --render-only are mutually exclusive')
        summary = audit_archive(args.render_only)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        save_json(args.output.with_suffix('.json'), summary)
        render(summary, args.output)
        print(args.output)
    else:
        if not args.artifact:
            parser.error('--artifact is required for capture')
        print(capture(args))


if __name__ == '__main__':
    main()
