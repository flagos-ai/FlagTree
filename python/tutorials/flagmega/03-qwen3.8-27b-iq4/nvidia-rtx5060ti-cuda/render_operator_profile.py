"""Render independently verified compact CPP operator data without private files.

Input schemas and the Verifier export contract are documented in the tutorial.
This command checks the independent full-raw receipt and recomputes every bar
from public per-token samples. It never treats a raw-summary 'audited' flag as
evidence, and never opens paths in a receipt or initializes CUDA.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
from pathlib import Path
import re
import statistics

DATA_SCHEMA = 'flagmega.cpp-operator-profile-compact/v1'
RECEIPT_SCHEMA = 'flagmega.cpp-operator-profile-export-verification/v1'
METRIC = 'instrumented_eager_cta_mean_ms_per_token'


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def byte_pin(data):
    return {'sha256': hashlib.sha256(data).hexdigest(), 'nbytes': len(data)}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def positive(value):
    return type(value) is int and value > 0


def sha(value):
    return isinstance(value, str) and re.fullmatch('[0-9a-f]{64}', value) is not None


def pin(value):
    return (isinstance(value, dict) and set(value) == {'sha256', 'nbytes'}
            and sha(value['sha256']) and positive(value['nbytes']))


def read_json(path):
    def unique(pairs):
        result = {}
        for key, value in pairs:
            require(key not in result, 'Duplicate JSON key: ' + key)
            result[key] = value
        return result
    def invalid_constant(value):
        raise ValueError('Non-finite JSON constant: ' + value)
    data = Path(path).read_bytes()
    return json.loads(data, object_pairs_hook=unique, parse_constant=invalid_constant), byte_pin(data)


def validate_structure(data):
    """Check representation/census/samples only; this does not authorize a chart."""
    require(data.get('schema') == DATA_SCHEMA and data.get('metric') == METRIC, 'Unsupported compact schema/metric')
    require(data.get('performance_acceptance') is False, 'Instrumented data cannot carry performance acceptance')
    tokens = data['profile_tokens']
    require(positive(tokens), 'Profile token count must be a positive integer')
    identity, census, provenance = data['identity'], data['census'], data['provenance']
    required_hashes = ('artifact_manifest_sha256', 'source_sha256', 'library_sha256', 'final_IR_sha256',
                       'recipe_sha256', 'model_ABI_sha256', 'profile_contract_sha256', 'implementation_sha256')
    require(all(sha(identity.get(key)) for key in required_hashes), 'Incomplete artifact/implementation identity')
    grid, threads = identity.get('grid'), identity.get('block_threads')
    require(isinstance(grid, list) and len(grid) == 3 and all(positive(v) for v in grid), 'Invalid launch grid')
    require(positive(threads) and threads % 32 == 0, 'Invalid launch block size')
    require(identity.get('target') == 'nvidia-sm120' and identity.get('compute_capability') == [12, 0]
            and all(type(v) is int for v in identity['compute_capability']),
            'This exporter contract requires the measured SM120 target')
    require('max_registers' in identity and (identity['max_registers'] is None or
            positive(identity['max_registers']) and identity['max_registers'] <= 255), 'Invalid register-cap identity')
    require(isinstance(identity.get('device_UUID'), str) and re.fullmatch(
        r'GPU-[0-9a-fA-F]{8}(?:-[0-9a-fA-F]{4}){3}-[0-9a-fA-F]{12}', identity['device_UUID']), 'Invalid device UUID')
    require(type(identity.get('device_ordinal')) is int and identity['device_ordinal'] >= 0
            and positive(identity.get('SM_count')) and positive(identity.get('driver_version')), 'Invalid device identity')
    ctas = math.prod(grid)
    require(data['CTA_count'] == ctas and type(data['CTA_count']) is int, 'CTA count differs from pinned grid')
    require(census['launch_grid'] == grid and census['block_threads'] == threads, 'Census launch differs from artifact')
    require(census['profile_mode'] in ('cta-globaltimer-v1', 'cta-globaltimer-v2'), 'Unknown profile clock protocol')
    events = census['event_count']
    require(positive(events), 'Invalid static event count')
    inventory = census['site_inventory']
    require(isinstance(inventory, list) and inventory, 'Missing static site inventory')
    sites = {}
    for site in inventory:
        name = site['site_id']
        require(isinstance(name, str) and name and name not in sites, 'Duplicate/invalid operator site')
        require(all(isinstance(site.get(k), str) and site[k] for k in ('call', 'family', 'enclosing_function')),
                'Incomplete static operator identity')
        require(positive(site['expected_invocations_per_token']), 'Invalid static invocation count')
        sites[name] = site
    require(data['all_operator_sites'] == len(sites) and type(data['all_operator_sites']) is int,
            'Site count does not cover the pinned inventory')
    event_owners, site_counts, barrier_counts = {}, {}, {}
    for kind, rows in [('operator', census['site_invocations']), ('barrier', census['schedule_barriers'])]:
        owners = site_counts if kind == 'operator' else barrier_counts
        for row in rows:
            name = row['site_id'] if kind == 'operator' else row['scope']
            require(name not in owners and (name in sites if kind == 'operator' else name in ('grid', 'cta')),
                    'Invalid/duplicate event owner')
            indices = row['event_indices']
            require(isinstance(indices, list) and indices, 'Missing event indices')
            for index in indices:
                require(type(index) is int and 0 <= index < events and index not in event_owners,
                        'Duplicate/out-of-range event index')
                event_owners[index] = (kind, name)
            owners[name] = len(indices)
    require(set(event_owners) == set(range(events)), 'Static event census has gaps')
    require(set(site_counts) == set(sites), 'Invocation census omits an operator site')
    require(all(site_counts[name] == sites[name]['expected_invocations_per_token'] for name in sites),
            'Invocation census differs from the pinned site inventory')

    def samples(rows, owners, operator):
        checked, seen = [], set()
        for row in rows:
            name = row['site_id'] if operator else row['scope']
            require(name in owners and name not in seen, 'Duplicate/unknown sample owner')
            seen.add(name)
            values = row['samples_ms_per_token']
            require(isinstance(values, list) and len(values) == tokens, 'Incomplete per-token sample array')
            require(all(type(x) in (float, int) and math.isfinite(x) and x >= 0 for x in values),
                    'Non-finite/negative sample')
            mean = statistics.mean(values)
            require(math.isfinite(mean), 'Sample aggregate overflow')
            reported = row['mean_ms_per_token']
            require(type(reported) in (float, int) and math.isfinite(reported)
                    and math.isclose(mean, reported, rel_tol=1e-12, abs_tol=1e-15), 'Reported mean differs from samples')
            require(row['profile_tokens'] == tokens and type(row['profile_tokens']) is int
                    and row['total_calls'] == tokens * owners[name] and type(row['total_calls']) is int,
                    'Sample call/token count differs from static census')
            checked.append({**(sites[name] if operator else {'scope': name}),
                            'samples_ms_per_token': values, 'mean_ms_per_token': mean,
                            'profile_tokens': tokens, 'total_calls': tokens * owners[name]})
        require(seen == set(owners), 'Samples omit executed operators or caller barriers')
        return sorted(checked, key=lambda row: row['mean_ms_per_token'], reverse=True)

    operators = samples(data['operators'], site_counts, True)
    barriers = samples(data['schedule_barriers'], barrier_counts, False)
    clock = provenance['clock']
    require(pin(clock['fit']) and pin(clock['raw_observations'])
            and clock['identity_sha256'] == digest(identity) and type(clock['ns_per_tick']) in (float, int)
            and clock['ns_per_tick'] == 1.0,
            'Clock evidence is missing or bound to another artifact/device')
    require(pin(provenance['archive']), 'Missing raw archive provenance')
    frames, last_epoch, profile_steps = provenance['raw_frames'], 0, []
    require(isinstance(frames, list) and frames, 'Missing complete raw frame inventory')
    seen_raw, profiling_started = set(), False
    for frame in frames:
        epoch = frame['epoch']
        require(isinstance(epoch, str) and epoch.isdecimal() and str(int(epoch)) == epoch
                and last_epoch < int(epoch) < 1 << 64, 'Raw epochs are duplicated, reordered or invalid')
        last_epoch = int(epoch)
        require(frame['phase'] in ('warmup', 'profile') and type(frame['step']) is int and frame['step'] >= 0,
                'Invalid raw phase or step')
        require(pin(frame['raw']) and frame['raw']['nbytes'] == events * ctas * 32, 'Incomplete raw CTA frame provenance')
        require(frame['raw']['sha256'] not in seen_raw, 'Raw frame provenance reuses the same epoch-bearing bytes')
        seen_raw.add(frame['raw']['sha256'])
        if frame['phase'] == 'profile':
            profiling_started = True
            profile_steps.append(frame['step'])
        else:
            require(not profiling_started, 'Warmup frames cannot interrupt the profiled token sequence')
    require(profile_steps == list(range(tokens)), 'Raw profile frames do not cover samples in token order')
    actors = data['actors']
    require(set(actors) == {'capture', 'implementation', 'renderer'}, 'Complete actor lineage is required')
    require(all(isinstance(v, list) and v and all(isinstance(x, str) and x for x in v) for v in actors.values()),
            'Invalid actor lineage')
    for field in ('numerical_status', 'instrumentation_overhead', 'monitor_status'):
        require(isinstance(data.get(field), str) and data[field], 'Missing explicit diagnostic gate: ' + field)
    return {'operators': operators, 'schedule_barriers': barriers, 'profile_tokens': tokens,
            'CTA_count': ctas, 'all_operator_sites': len(sites),
            'operator_invocations_per_token': sum(site_counts.values()), 'schedule_barriers_per_token': sum(barrier_counts.values()),
            'raw_frame_count': len(frames), 'raw_cells_checked': len(frames) * events * ctas}


def validate_compact(data, data_pin, receipt):
    checked = validate_structure(data)
    require(receipt.get('schema') == RECEIPT_SCHEMA and receipt.get('status') == 'passed',
            'Passed independent full-raw export receipt required')
    require(receipt.get('compact_data') == data_pin, 'Receipt does not bind these exact data bytes')
    verifier = receipt['verifier']
    require(verifier.get('role') == 'Verifier' and isinstance(verifier.get('agent_id'), str) and verifier['agent_id'],
            'Receipt must identify its independent Verifier')
    owners = sorted({name for names in data['actors'].values() for name in names})
    require(verifier['agent_id'] not in owners and verifier.get('independent_of') == owners,
            'Verifier is not independent of capture/implementation/renderer lineage')
    audit = receipt['full_raw_audit']
    expected_digests = {'artifact_identity_sha256': digest(data['identity']), 'census_sha256': digest(data['census']),
                        'clock_sha256': digest(data['provenance']['clock']),
                        'raw_provenance_sha256': digest(data['provenance'])}
    require(all(audit.get(key) == value for key, value in expected_digests.items()),
            'Full-raw receipt is bound to different artifact/clock/census/raw evidence')
    require(pin(audit.get('report')) and pin(audit.get('aggregation_source')), 'Pinned full-raw audit report and aggregation source required')
    for key in ('all_operator_sites', 'operator_invocations_per_token', 'schedule_barriers_per_token',
                'profile_tokens', 'CTA_count', 'raw_frame_count', 'raw_cells_checked'):
        require(type(audit.get(key)) is int and audit[key] == checked[key], 'Full-raw audit coverage differs: ' + key)
    require(audit.get('checks') == {key: 'passed' for key in
            ('every_raw_cta_cell', 'artifact_and_clock_identity', 'complete_census', 'per_token_aggregation')},
            'Full-raw audit did not pass all required checks')
    require(data.get('data_origin') == receipt.get('data_origin') == 'actual_execution',
            'Synthetic/unspecified data cannot authorize actual latency bars')
    # The flags above are never sufficient: byte binding, independent lineage,
    # exact provenance/counts and recomputed public arrays all had to pass.
    return checked


def render(checked):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    operators, barriers = checked['operators'], checked['schedule_barriers']
    height = max(7, len(operators) * .24 + 2)
    fig, axes = plt.subplots(2, 1, figsize=(12, height + 1.5),
                             gridspec_kw={'height_ratios': [max(len(operators), 8), max(len(barriers), 2)]})
    axes[0].barh(range(len(operators)), [x['mean_ms_per_token'] for x in operators], color='#247aa8')
    axes[0].set_yticks(range(len(operators)), [x['site_id'] + '  ' + x['family'] for x in operators], fontsize=8)
    axes[0].invert_yaxis()
    axes[0].set_title(f'CPP diagnostic: all {len(operators)} sites, {checked["profile_tokens"]} tokens, {checked["CTA_count"]} CTAs\nInstrumented eager execution; not acceptance latency')
    axes[1].barh(range(len(barriers)), [x['mean_ms_per_token'] for x in barriers], color='#858d95')
    axes[1].set_yticks(range(len(barriers)), ['Caller ' + x['scope'] + ' barriers' for x in barriers], fontsize=8)
    axes[1].invert_yaxis()
    axes[1].set_xlabel('Accumulated time per token, mean across CTAs (ms)')
    fig.text(.02, .008, 'Leaf intervals include internal waits. Caller barriers are separate; parent containers excluded.\nCTA skew/overlap prevents interpreting sums as end-to-end latency. Instrumentation overhead requires separate comparison.', fontsize=8)
    fig.tight_layout(rect=(0, .045, 1, 1))
    output = io.BytesIO()
    fig.savefig(output, format='svg')
    plt.close(fig)
    return output.getvalue()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--receipt', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--validate-only', action='store_true', help='Check a verified compact export without drawing a chart')
    args = parser.parse_args()
    report_path = args.output.with_suffix('.render.json')
    require(not args.output.exists() and not report_path.exists(), 'Use fresh SVG/report paths')
    data, data_pin = read_json(args.data)
    receipt, receipt_pin = read_json(args.receipt)
    checked = validate_compact(data, data_pin, receipt)
    report = {'schema': 'flagmega.cpp-operator-profile-render/v1', 'validation_scope': 'verified_compact_summary',
              'data': data_pin, 'independent_receipt': receipt_pin,
              'raw_audit_performed_here': False, 'actual_chart_rendered': False,
              'profile_tokens': checked['profile_tokens'], 'all_operator_sites': checked['all_operator_sites'],
              'CTA_count': checked['CTA_count'], 'performance_acceptance': False,
              'numerical_status': data['numerical_status'], 'instrumentation_overhead': data['instrumentation_overhead'],
              'monitor_status': data['monitor_status'], 'renderer': byte_pin(Path(__file__).read_bytes())}
    if not args.validate_only:
        svg = render(checked)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open('xb') as stream:
            stream.write(svg)
        report.update(actual_chart_rendered=True, svg=byte_pin(svg))
        with report_path.open('x') as stream:
            json.dump(report, stream, indent=2, allow_nan=False)
            stream.write('\n')
    print(json.dumps(report, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
