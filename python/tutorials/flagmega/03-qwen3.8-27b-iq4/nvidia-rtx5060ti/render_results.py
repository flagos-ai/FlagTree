"""Render verified decode latency and throughput in the tutorial SVG style."""

import argparse
import hashlib
from html import escape
import json
import math
from pathlib import Path
import statistics

LABELS = {'native': 'llama.cpp', 'baseline': 'FlagMega Initial', 'agent': 'FlagMega Agent'}
COLORS = {'native': '#556579', 'baseline': '#c47725', 'agent': '#198269'}


def decode_statistics(latencies):
    values = [value for row in latencies
              for value in row] if latencies and isinstance(latencies[0], list) else latencies
    if not values or any(not math.isfinite(value) or value <= 0 for value in values):
        raise ValueError('Charts require positive finite measured decode times')
    return {'median_ms': statistics.median(values), 'tokens_per_second': 1000 / statistics.mean(values)}


def chart(variants, metric, *, gpu, steps, measurement, required_prefix=None):
    """Use the geometry, palette and statistics of the other FlagMega tutorials."""
    labels = {key: label for key, label in LABELS.items() if key in variants}
    latency = metric == 'median_ms'
    title = 'Decode latency (lower is better)' if latency else 'Decode throughput (higher is better)'
    unit = 'ms/token' if latency else 'tokens/s'
    width, height, left, top, plot_width, plot_height = 1120, 650, 85, 135, 1000, 355
    maximum = max(variants[key][metric] for key in labels) * 1.2
    gpu = gpu.removeprefix('NVIDIA GeForce ')
    boundary = ('Host timing: graph execution + GPU greedy sampling + one CPU token return'
                if measurement.startswith('Host wall clock') else measurement)
    description = f'Qwen3.8-27B IQ4_XS, full 64-layer decoder, {gpu}, batch 1. {boundary}.'
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" role="img">', f'<title>{escape(title)}</title>',
        f'<desc>{escape(description)}</desc>', '<rect width="100%" height="100%" fill="white"/>',
        '<g font-family="sans-serif" fill="#243343">', f'<text x="{left}" y="36" font-size="24">{escape(title)}</text>',
        f'<text x="{left}" y="63" font-size="14">Qwen3.8-27B IQ4_XS / {escape(gpu)} / batch 1 / {steps} decode-1 calls</text>',
        f'<text x="{left}" y="86" font-size="14">{escape(boundary)}</text>',
        f'<text x="{left}" y="108" font-size="14">Loading, JIT, warmup and state reset excluded.</text>'
    ]
    for tick in range(6):
        y = top + plot_height * (1 - tick / 5)
        parts.extend([
            f'<path d="M {left} {y:.1f} H {left + plot_width}" stroke="#e2e8ef"/>',
            f'<text x="{left - 12}" y="{y + 5:.1f}" text-anchor="end" font-size="13">{maximum * tick / 5:.1f}</text>'
        ])
    bar_width = plot_width * .66 / len(labels)
    for offset, variant in enumerate(labels):
        value = variants[variant][metric]
        bar_height = plot_height * value / maximum
        x = left + plot_width * .12 + offset * bar_width * 1.15
        y = top + plot_height - bar_height
        parts.extend([
            f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_width:.1f}" height="{bar_height:.1f}" fill="{COLORS[variant]}"/>',
            f'<text x="{x + bar_width / 2:.1f}" y="{y - 8:.1f}" text-anchor="middle" font-size="13">{value:.3f}</text>'
        ])
    parts.append(
        f'<text x="{left + plot_width / 2:.1f}" y="{top + plot_height + 29}" text-anchor="middle" font-size="14">{steps} decode-1 calls</text>'
    )
    parts.append(f'<text x="15" y="{top - 12}" font-size="13">{unit}</text>')
    for index, (variant, label) in enumerate(labels.items()):
        x = left + index * 310
        parts.extend([
            f'<rect x="{x}" y="555" width="15" height="15" fill="{COLORS[variant]}"/>',
            f'<text x="{x + 23}" y="568" font-size="14">{escape(label)}</text>'
        ])
    statistic = 'Median per-token latency.' if latency else '1000 / mean per-token latency.'
    acceptance = (
        f'First {required_prefix} generated tokens match llama.cpp; timed rounds reproduce the independent trajectory.'
        if required_prefix is not None else 'Full-decoder smoke validation.')
    parts.append(f'<text x="{left}" y="602" font-size="13">{escape(statistic)}</text>')
    parts.append(f'<text x="{left}" y="626" font-size="13">{escape(acceptance)}</text>')
    return '\n'.join(parts + ['</g>', '</svg>', ''])


def write_charts(output, variants, **context):
    for filename, metric in (('decode_latency.svg', 'median_ms'), ('decode_throughput.svg', 'tokens_per_second')):
        (output / filename).write_text(chart(variants, metric, **context))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--trial', type=Path, required=True)
    parser.add_argument('--benchmark', type=Path,
                        help='Accepted complete-model benchmark.py results with llama.cpp golden')
    parser.add_argument('--llama-performance', type=Path,
                        help='llama_performance.py directory or metadata.json with the same timing boundary')
    parser.add_argument('--baseline-benchmark', type=Path,
                        help='Fresh current-compiler baseline using the same timing boundary')
    parser.add_argument('--output', type=Path, default=Path(__file__).resolve().parent / 'figures')
    args = parser.parse_args()
    trial = args.trial
    build = json.loads((trial / 'build.json').read_text())
    if args.benchmark:
        render_gpu(build, trial, args.benchmark, args.output, args.llama_performance, args.baseline_benchmark)
        return
    run = json.loads((trial / 'results/run.json').read_text())
    if (build['status'] != 'verified' or not build['artifact_verified'] or not run['finite'] or run['layers'] != 64
            or run['sequence_length'] != run['steps']):
        raise ValueError('Only an accepted complete-decoder trial can be published')
    if str(Path(run['artifact']).resolve()) != str((trial / 'artifact').resolve()):
        raise ValueError('Measurements must belong to the supplied trial artifact')
    generated = trial / 'artifact/generated_kernels.py'
    if hashlib.sha256(generated.read_bytes()).hexdigest() != build['generated_source_sha256']:
        raise ValueError('Generated source differs from the measured artifact')
    latencies = run['latency_ms']
    mean = statistics.mean(latencies)
    summary = {
        'scope':
        '64-layer decode-1 smoke validation', 'steps':
        len(latencies), 'gpu':
        run['gpu'], 'mean_ms_per_token':
        mean, 'median_ms_per_token':
        statistics.median(latencies), 'tokens_per_second':
        1000 / mean, 'peak_allocated_GiB':
        run['peak_allocated_bytes'] / 2**30, 'artifact_semantic_hash':
        build['semantic_hash'], 'generated_source_sha256':
        build['generated_source_sha256'], 'measurement_boundary':
        'Prepared run_into wall time, including Python binding checks and CUDA synchronization; loading and JIT excluded.'
    }
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    write_charts(args.output, {'baseline': decode_statistics(latencies)}, gpu=run['gpu'], steps=run['steps'],
                 measurement=summary['measurement_boundary'])
    print(json.dumps(summary, indent=2))


def render_gpu(build, trial, benchmark, output, llama_performance=None, baseline_benchmark=None):
    run = json.loads(benchmark.read_text())
    comparison = run['llama_comparison']
    source_hash = hashlib.sha256((trial / 'artifact/generated_kernels.py').read_bytes()).hexdigest()
    if (build['status'] != 'verified' or not build['artifact_verified'] or not run['finite'] or run['layers'] != 64
            or not comparison['prefix_match'] or source_hash != run['source_sha256']
            or source_hash != build['generated_source_sha256']
            or Path(run['artifact']).resolve() != (trial / 'artifact').resolve()):
        raise ValueError('Only measured verified full-model artifacts passing llama.cpp token prefix can be published')
    summary = {
        key: run[key]
        for key in ('gpu', 'steps', 'repeats', 'mean_ms', 'tokens_per_second', 'packed_model_bytes_per_token',
                    'model_bandwidth_gb_s', 'model_bandwidth_utilization', 'resource_report')
    }
    summary.update(generated_source_sha256=source_hash, artifact_semantic_hash=build['semantic_hash'],
                   measurement_boundary=run['measurement'], llama_commit=comparison['llama_commit'],
                   llama_token_agreement=comparison['token_agreement'], golden_prefix=comparison['golden_prefix'],
                   required_generated_prefix=comparison['required_generated_prefix'],
                   prefix_match=comparison['prefix_match'])
    fair = None
    if llama_performance is not None:
        path = llama_performance / 'metadata.json' if llama_performance.is_dir() else llama_performance
        fair = json.loads(path.read_text())
        if (not run['measurement'].startswith('Host wall clock')
                or fair['measurement'] != 'Host steady_clock around decode, GPU greedy and one CPU token return'
                or fair['layers'] != 64 or fair['steps'] != run['steps']
                or fair['llama_commit'] != comparison['llama_commit'] or not comparison['independent_generation']):
            raise ValueError('llama.cpp performance comparison requires matched host-token timing and workload')
        start = comparison['prefix_start_step']
        end = start + comparison['required_generated_prefix']
        if fair['tokens'][start:end] != comparison['golden_prefix']:
            raise ValueError('Performance reference and independent golden disagree on the required prefix')
        summary['llama_performance'] = {
            key: fair[key]
            for key in ('mean_ms', 'median_ms', 'tokens_per_second', 'measurement', 'llama_commit', 'checkpoint_sha256',
                        'flash_attention', 'sampling', 'kv_dtype', 'intermediate_dtype', 'steps', 'repeats',
                        'prompt_token_count')
        }
        summary['speedup_vs_llama_cpp'] = fair['mean_ms'] / run['mean_ms']
    baseline = None
    if baseline_benchmark is not None:
        baseline = json.loads(baseline_benchmark.read_text())
        baseline_artifact = Path(baseline['artifact'])
        baseline_build = json.loads((baseline_artifact.parent / 'build.json').read_text())
        if (not baseline['finite'] or baseline['layers'] != 64 or baseline['steps'] != run['steps']
                or baseline['measurement'] != run['measurement'] or baseline_build['status'] != 'verified'
                or baseline_build['optimization'] is not None
                or compiler_core_sources(baseline_artifact.parent) != compiler_core_sources(trial)
                or baseline_build['checkpoint'] != build['checkpoint']
                or baseline_build['fp32_residual'] != build['fp32_residual'] or hashlib.sha256(
                    (baseline_artifact / 'generated_kernels.py').read_bytes()).hexdigest()
                != baseline['source_sha256']):
            raise ValueError('Agent gains require a fresh baseline from the same compiler, input and numerical profile')
        summary['current_compiler_baseline'] = {
            key: baseline[key]
            for key in ('mean_ms', 'tokens_per_second', 'measurement', 'source_sha256', 'resource_report')
        }
        summary['speedup_vs_current_compiler_baseline'] = baseline['mean_ms'] / run['mean_ms']
        summary['compiler_core_source_sha256'] = hashlib.sha256(
            json.dumps(compiler_core_sources(trial), sort_keys=True).encode()).hexdigest()
    variants = {'agent': decode_statistics(run['latency_ms'])}
    if fair:
        variants['native'] = decode_statistics(fair['latency_ms'])
    if baseline:
        variants['baseline'] = decode_statistics(baseline['latency_ms'])
    summary['median_ms'] = variants['agent']['median_ms']
    summary['chart_variants'] = {key: variants[key] for key in LABELS if key in variants}
    summary['chart_statistics'] = {
        'decode_latency': 'Pooled median per-token latency', 'decode_throughput': '1000 / mean per-token latency'
    }
    output.mkdir(parents=True, exist_ok=True)
    (output / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    write_charts(output, variants, gpu=run['gpu'], steps=run['steps'], measurement=run['measurement'],
                 required_prefix=comparison['required_generated_prefix'])
    print(json.dumps(summary, indent=2))


def compiler_core_sources(trial):
    """Agent-only templates may evolve without changing the default compiler."""
    sources = json.loads((trial / 'sources.json').read_text())
    prefixes = ('python/triton/flagmega/', 'third_party/nvidia/backend/', 'python/triton/experimental/tle/',
                'lib/Conversion/')
    return {path: digest for path, digest in sources.items() if path.startswith(prefixes)}


if __name__ == '__main__':
    main()
