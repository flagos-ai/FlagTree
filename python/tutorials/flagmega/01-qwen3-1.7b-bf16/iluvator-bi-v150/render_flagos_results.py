"""Recheck paired raw requests and draw the same three metrics as the H800 tutorial."""
import argparse
from html import escape
import gzip
import json
from pathlib import Path
import statistics

from benchmark_flagos import require_agreement, summarize

COLORS = {'native': '#556579', 'flagmega': '#198269', 'chat_cli': '#8252b3'}
CHARTS = (
    ('decode_latency.svg', 'decode_median_ms', 'Decode step latency (lower is better)', 'ms/token'),
    ('decode_throughput.svg', 'decode_tokens_per_second', 'Decode throughput (higher is better)', 'tokens/s'),
    ('request_throughput.svg', 'request_tokens_per_second',
     'Request throughput incl. prefill (higher is better)', 'tokens/s'),
)


def aggregate(directory):
    paths = {variant: sorted([*directory.glob(f'round*.{variant}.json'),
                              *directory.glob(f'round*.{variant}.json.gz')])
             for variant in ('native', 'flagmega')}
    if not paths['native'] or [p.name.split('.')[0] for p in paths['native']] != [
            p.name.split('.')[0] for p in paths['flagmega']]:
        raise ValueError('Require complete paired native/FlagMega rounds')
    reports = {variant: [json.loads(gzip.decompress(p.read_bytes()) if p.suffix == '.gz'
                                  else p.read_text()) for p in values]
               for variant, values in paths.items()}
    first = reports['native'][0]
    expected = {s['scenario']: s['runs'][0] for s in first['scenarios']}
    rows, identities = {}, set()
    for variant, values in reports.items():
        for report in values:
            if not report['performance_valid']:
                raise ValueError('Failed correctness run cannot be plotted')
            for key in ('checkpoint_files', 'compiler_sha256', 'visible_devices', 'device',
                        'decode_tokens', 'block_size', 'num_blocks', 'batch', 'concurrency',
                        'prefix_caching', 'torch', 'triton', 'vllm', 'vllm_source',
                        'flaggems_source', 'plugin_source', 'adapter_sha256', 'benchmark_sha256',
                        'warmups', 'repeats', 'boundary', 'prefill'):
                if report[key] != first[key]:
                    raise ValueError(f'Incomparable reports at {key}')
            if report.get('decode_inputs_sha256') != first.get('decode_inputs_sha256'):
                raise ValueError('Incomparable reports at decode_inputs_sha256')
            for key in ('max_start_temperature_c', 'conditions_sha256'):
                if report.get(key) != first.get(key):
                    raise ValueError(f'Incomparable scenario conditions at {key}')
            # Native FULL graphs and extension-owned decode graphs have different
            # capture boundaries. Enforce each variant's contract across rounds.
            for key in ('cuda_graph', 'cudagraph_mode', 'compilation_mode', 'enforce_eager',
                        'decode_execution'):
                if report.get(key) != values[0].get(key):
                    raise ValueError(f'Different {variant} execution modes at {key}')
            execution = report.get('decode_execution')
            if execution == 'vllm_full_graph' and (not report['cuda_graph'] or
                    report['cudagraph_mode'] != 'FULL' or report['enforce_eager']):
                raise ValueError('Native FULL graph configuration differs')
            if execution == 'flagmega_worker_graph' and (report['cuda_graph'] or
                    report['cudagraph_mode'] != 'NONE' or not report['enforce_eager']):
                raise ValueError('Worker graph requires eager native prefill')
            if variant == 'flagmega':
                if not report['native_token_agreement']:
                    raise ValueError('Missing native token verification')
                stats = report['executor_stats'][0]
                identity = stats.get('codegen', {'source_sha256': stats['source_sha256']})
                identities.add(json.dumps(identity, sort_keys=True))
                calls = len(expected) * (report['warmups'] + report['repeats']) * (report['decode_tokens'] - 1)
                if stats['decode_calls'] != calls or stats['kv_copy_bytes_per_decode'] != 0:
                    raise ValueError('Compiled decode coverage or zero-copy contract differs')
                if report.get('decode_execution') == 'flagmega_worker_graph' and stats['graph_replays'] != calls:
                    raise ValueError('Missing FlagMega graph replay')
                if report.get('decode_inputs_sha256') and stats.get('fused_decode_input_calls') != calls:
                    raise ValueError('Missing fused decode input preparation')
            if {s['scenario'] for s in report['scenarios']} != set(expected):
                raise ValueError('Scenario sets differ')
            for scenario in report['scenarios']:
                if report.get('max_start_temperature_c') is not None:
                    conditions = scenario['start_conditions']
                    if (conditions['max_start_temperature_c'] != report['max_start_temperature_c'] or
                            conditions['observations'][-1]['temperature_c'] > report['max_start_temperature_c']):
                        raise ValueError('Scenario start temperature exceeds its measurement contract')
                if len(scenario['runs']) != report['repeats'] or len(scenario['warmup_runs']) != report['warmups']:
                    raise ValueError('Incomplete requests')
                for run in scenario['warmup_runs'] + scenario['runs']:
                    require_agreement(run, expected[scenario['scenario']], scenario['scenario'])
                    if (len(run['token_ids']) != report['decode_tokens'] or
                            len(run['decode_step_ms']) != report['decode_tokens'] - 1 or
                            len(run['prompt_tokens']) != scenario['prompt_length'] or
                            min(run['decode_step_ms']) <= 0 or run['e2e_ms'] <= 0):
                        raise ValueError('Invalid timing boundary or token count')
                bucket = rows.setdefault((variant, scenario['prompt_length']),
                                         {'decode': [], 'e2e': [], 'ttft': [], 'decode_rate': [],
                                          'request_rate': [], 'rounds': []})
                bucket['rounds'].append({name: scenario[name]['median']
                                        for name in ('decode_ms', 'e2e_ms', 'ttft_ms')})
                for run in scenario['runs']:
                    bucket['decode'].extend(run['decode_step_ms'])
                    bucket['e2e'].append(run['e2e_ms'])
                    bucket['ttft'].append(run['ttft_ms'])
                    bucket['decode_rate'].append(len(run['decode_step_ms']) * 1000 / sum(run['decode_step_ms']))
                    bucket['request_rate'].append(len(run['token_ids']) * 1000 / run['e2e_ms'])
    if len(identities) != 1:
        raise ValueError('Different compiled artifacts cannot share a final label')
    identity = json.loads(next(iter(identities)))
    labels = {'native': 'Native vLLM (FULL CUDA Graph)' if first['cuda_graph'] else 'Native vLLM (eager)',
              'flagmega': 'FlagMega C++ decode (worker CUDA Graph)' if identity.get('backend') == 'cpp'
              else 'FlagMega decode (worker CUDA Graph)'}
    summaries = []
    for length in sorted(s['prompt_length'] for s in first['scenarios']):
        row = {'prompt_length': length, 'decode_tokens': first['decode_tokens'], 'variants': {}}
        for variant in reports:
            bucket = rows[variant, length]
            row[variant] = {name: summarize(bucket[name]) for name in ('decode', 'e2e', 'ttft')}
            row[variant]['rounds'] = bucket['rounds']
            row['variants'][variant] = {
                'decode_median_ms': row[variant]['decode']['median'],
                'decode_p95_ms': row[variant]['decode']['p95'],
                'decode_tokens_per_second': statistics.median(bucket['decode_rate']),
                'request_tokens_per_second': statistics.median(bucket['request_rate']),
                'request_e2e_median_ms': row[variant]['e2e']['median'],
                'ttft_median_ms': row[variant]['ttft']['median'],
                'runs': len(bucket['e2e']), 'decode_samples': len(bucket['decode']),
                'native_token_agreement': 1.0,
            }
        row['decode_speedup'] = row['native']['decode']['median'] / row['flagmega']['decode']['median']
        row['request_speedup'] = row['native']['e2e']['median'] / row['flagmega']['e2e']['median']
        summaries.append(row)
    return {'accepted': True, 'rounds': len(reports['native']), 'codegen': identity,
            'source_sha256': identity['source_sha256'], 'native_prefill': True,
            'labels': labels, 'sources': [p.name for values in paths.values() for p in values],
            'environment': {key: first[key] for key in ('checkpoint_files', 'device', 'visible_devices',
                                                       'torch', 'triton', 'vllm', 'compiler_sha256')},
            'conditions': {key: first.get(key) for key in ('max_start_temperature_c', 'conditions_sha256')},
            'execution': {variant: {key: values[0].get(key) for key in
                                   ('decode_execution', 'cuda_graph', 'cudagraph_mode', 'enforce_eager')}
                          for variant, values in reports.items()}, 'scenarios': summaries}


def chart(rows, metric, title, unit, labels, conditions=None):
    if not rows or any(set(row["variants"]) != set(rows[0]["variants"]) for row in rows):
        raise ValueError("Every chart scenario must contain the same variants")
    if not labels or set(labels) != set(rows[0]["variants"]):
        raise ValueError("Unknown chart variant")
    width, height = 1200, 690
    left, right, top, bottom = 85, 30, 145, 150
    plot_width, plot_height = width-left-right, height-top-bottom
    maximum = max(row["variants"][label][metric] for row in rows for label in labels) * 1.18
    output_tokens = "/".join(str(value) for value in sorted({row["decode_tokens"] for row in rows}))
    description = "Qwen3-1.7B BF16, one BI-V150, batch one. Bars are measured medians."
    start_note = ""
    if conditions and conditions.get('max_start_temperature_c') is not None:
        start_note = f" / scenario start ≤{conditions['max_start_temperature_c']}°C"
        description += f" Each scenario starts at most {conditions['max_start_temperature_c']}°C; cooling before warmup is outside timing."
    if 'chat_cli' in labels:
        description += " CLI uses compiled token-scan prefill; vLLM uses batched prefill. Terminal I/O excluded."
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}" role="img">',
             f'<title>{escape(title)}</title>',
             f'<desc>{escape(description)}</desc>',
             '<rect width="100%" height="100%" fill="#ffffff"/>',
             '<g font-family="sans-serif" fill="#243343">',
             f'<text x="{left}" y="37" font-size="24">{escape(title)}</text>',
             f'<text x="{left}" y="64" font-size="14">BF16 / BI-V150 / batch 1 / {output_tokens} output tokens / host-visible timing{escape(start_note)}</text>']
    boundary = ("Decode intervals exclude the first output token; not end-to-end request latency."
                if metric == "decode_median_ms" else
                "Median per-request decode tokens / decode time; excludes prefill and the first output token."
                if metric == "decode_tokens_per_second" else
                "Output tokens / complete request time, including prefill; not inverse decode latency.")
    parts.append(f'<text x="{left}" y="89" font-size="14">{escape(boundary)}</text>')
    note = ("CLI: token-scan prefill; vLLM: batched prefill. BF16 outputs may differ; CLI terminal I/O excluded."
            if 'chat_cli' in labels else "Both use native vLLM prefill, scheduling and sampling; no prefix reuse.")
    parts.append(f'<text x="{left}" y="113" font-size="14">{escape(note)}</text>')
    for tick in range(6):
        value = maximum*tick/5
        y = top+plot_height*(1-tick/5)
        parts += [f'<path d="M {left} {y:.1f} H {width-right}" stroke="#dfe5eb"/>',
                  f'<text x="{left-12}" y="{y+5:.1f}" text-anchor="end" font-size="13">{value:.1f}</text>']
    group_width = plot_width/len(rows)
    bar_width = group_width*.82/(len(labels)+(len(labels)-1)*.15)
    for group, row in enumerate(rows):
        for variant, label in enumerate(labels):
            value = row["variants"][label][metric]
            h = plot_height*value/maximum
            x = left+group_width*(group+.09)+variant*bar_width*1.15
            y = top+plot_height-h
            number = f"{value:.3f}" if metric == "decode_median_ms" else f"{value:.2f}"
            parts += [f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_width:.1f}" height="{h:.1f}" fill="{COLORS[label]}"/>',
                      f'<text x="{x+bar_width/2:.1f}" y="{y-8:.1f}" text-anchor="middle" font-size="12">{number}</text>']
        parts.append(f'<text x="{left+group_width*(group+.5):.1f}" y="{top+plot_height+28}" text-anchor="middle" font-size="14">Prompt {row["prompt_length"]}</text>')
    parts.append(f'<text x="16" y="{top-16}" font-size="13">{escape(unit)}</text>')
    for i, (label, text_label) in enumerate(labels.items()):
        x, y = left + (i % 2)*460, height-90+(i//2)*25
        parts += [f'<rect x="{x}" y="{y-12}" width="14" height="14" fill="{COLORS[label]}"/>',
                  f'<text x="{x+22}" y="{y}" font-size="13">{escape(text_label)}</text>']
    return "\n".join(parts + ['</g>', '</svg>', ''])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results', type=Path, required=True)
    parser.add_argument('--figure-dir', type=Path)
    parser.add_argument('--chat-results', type=Path, help='Add standalone chat CLI measurements')
    parser.add_argument('--chat-native-reference', type=Path, help='Native token-scan correctness report for CLI')
    args = parser.parse_args()
    summary = aggregate(args.results)
    if args.chat_results:
        from render_chat_cli_results import aggregate_chat, add_flagos_comparison
        paths = sorted([*args.chat_results.glob('round*.chat_cli.json'),
                        *args.chat_results.glob('round*.chat_cli.json.gz')])
        if (args.chat_results / 'failure.json').exists():
            raise ValueError('Cannot publish a failed chat CLI experiment')
        rows, groups, reports = aggregate_chat(paths)
        summary = add_flagos_comparison(rows, groups, reports, args.results, comparison=summary,
                                        native_reference=args.chat_native_reference)
        summary['chat_cli']['sources'] = [str(p.relative_to(args.results))
                                         if p.is_relative_to(args.results) else str(p) for p in paths]
    (args.results / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    figure_dir = args.figure_dir or args.results
    figure_dir.mkdir(parents=True, exist_ok=True)
    for filename, metric, title, unit in CHARTS:
        (figure_dir / filename).write_text(chart(summary['scenarios'], metric, title, unit, summary['labels'],
                                                summary.get('conditions')))
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
