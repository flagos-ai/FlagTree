"""Independent stateful CPP decode validation and matched host-token timing.

Ported from the sibling native-IQ4 tutorial's benchmark. The generated C++
artifact owns sampling; this runner captures model, slot and token feedback on
one CUDA stream. Setup, state reset and prompt injection are outside timing.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import sys
from time import perf_counter

from timing_isolation import PYTHON_RESULT, WINDOW_SCHEMA, byte_pin, monotonic_ns, timing_metadata


class NumericalError(RuntimeError):
    """A completed execution violated the frozen numerical/state contract."""


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def file_pin(path):
    path = Path(path).resolve()
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b''):
            digest.update(chunk)
    return {'path': str(path), 'nbytes': path.stat().st_size, 'sha256': digest.hexdigest()}


def save_json(path, value):
    """Atomically replace a live receipt; evaluated artifact inputs stay immutable."""
    path = Path(path)
    temporary = path.with_name(path.name + '.tmp')
    with temporary.open('w') as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def append_json(path, value):
    with Path(path).open('a') as stream:
        stream.write(json.dumps(value, ensure_ascii=False, allow_nan=False) + '\n')
        stream.flush()
        os.fsync(stream.fileno())


def install_recipe(path):
    sys.path.insert(0, str(Path(__file__).resolve().parents[4]))
    if path is None:
        return None
    recipe = json.loads(Path(path).read_text())
    if not isinstance(recipe, dict):
        raise ValueError('A local recipe must be a JSON object of installer arguments')
    from local_optimizations import install
    install(**recipe)
    return file_pin(path)


def artifact_identity(artifact):
    root = Path(artifact).resolve()
    manifest = json.loads((root / 'artifact.json').read_text())
    codegen = manifest['codegen']
    if manifest.get('codegen_backend') != 'cpp':
        raise ValueError('This benchmark requires a normal CPP artifact')
    pins = {'manifest': file_pin(root / 'artifact.json')}
    for key, relative in [('source', codegen['source']), ('library', codegen['library']),
                          ('final_ir', manifest['final_ir'])]:
        path = (root / relative).resolve()
        if not path.is_relative_to(root):
            raise ValueError(f'Artifact {key} escapes its root')
        pins[key] = file_pin(path)
    for key in ('source', 'library'):
        if pins[key]['sha256'] != codegen[key + '_sha256']:
            raise ValueError(f'Artifact {key} changed after publication')
    return manifest, pins


def quantile(samples, fraction):
    """Linear interpolation on sorted samples, matching NumPy's default."""
    values = sorted(samples)
    if not values:
        return None
    index = (len(values) - 1) * fraction
    lo, hi = math.floor(index), math.ceil(index)
    return values[lo] + (values[hi] - values[lo]) * (index - lo)


def summarize(samples):
    if not samples:
        return {'sample_count': 0, 'mean_ms': None, 'median_ms': None, 'p95_ms': None,
                'max_ms': None, 'tokens_per_second': None}
    if any(not math.isfinite(x) or x <= 0 for x in samples):
        raise ValueError('Timings must be finite positive milliseconds')
    mean = statistics.mean(samples)
    return {'sample_count': len(samples), 'mean_ms': mean, 'median_ms': statistics.median(samples),
            'p95_ms': quantile(samples, .95), 'max_ms': max(samples), 'tokens_per_second': 1000 / mean}


class DecodeContext:
    """One runtime, its bound stream, and its external state storage."""

    def __init__(self, artifact, *, token_id=1, device='cuda:0'):
        import torch
        from triton.flagmega.ir import VectorType
        from triton.flagmega.runtime import load

        self.torch = torch
        self.device = device
        self.stream = torch.cuda.Stream(device=device)
        torch.cuda.set_stream(self.stream)
        self.runtime = load(artifact, device=device)
        self.values = {}
        for argument in self.runtime.external_arguments:
            buffer = self.runtime.buffer_plan.buffer_map[argument['buffer']]
            dtype = buffer.dtype
            lanes = dtype.lanes if isinstance(dtype, VectorType) else ()
            scalar = dtype.elem_type if lanes else dtype
            self.values[buffer.id] = torch.zeros((*buffer.shape, *lanes),
                                                 dtype=getattr(torch, scalar.value), device=device)
        self.values['input_ids'].fill_(token_id)
        self.values['paged_state.query_start_loc'].copy_(torch.tensor([0, 1], dtype=torch.int32, device=device))
        table = self.values['paged_state.block_table']
        table.copy_(torch.arange(table.numel(), dtype=table.dtype, device=device).reshape_as(table))
        self.arguments = tuple(self.values[a['buffer']] for a in self.runtime.external_arguments)
        outputs = self.runtime.buffer_plan.function_map[self.runtime.ir_module.entry].outputs
        self.logits, self.token = (self.values[outputs[i][1][0]] for i in (0, 1))
        if self.token.numel() != 1 or self.values['input_ids'].numel() != 1:
            raise ValueError('This contract is single-sequence decode-1')
        self.initial = {name: value.clone() for name, value in self.values.items()}
        self.runtime.prepare(*self.arguments)
        self.stream.synchronize()

    def reset(self):
        for name, value in self.values.items():
            value.copy_(self.initial[name])

    def launch(self):
        self.values['paged_state.slot_mapping'].copy_(self.values['paged_state.seq_lens'])
        self.runtime.run_into(*self.arguments)
        self.values['input_ids'].copy_(self.token.reshape_as(self.values['input_ids']))

    def check_state(self, steps):
        length = int(self.values['paged_state.seq_lens'].item())
        if length != steps:
            raise NumericalError(f'Sequence length is {length}, expected {steps}')
        if int(self.values['paged_state.slot_mapping'].item()) != steps - 1:
            raise NumericalError('Slot mapping did not follow sequence progression')
        for name, value in self.values.items():
            if value.is_floating_point() and not self.torch.isfinite(value).all().item():
                raise NumericalError(f'Non-finite external state/output {name}')

    def capture(self):
        self.reset()
        self.stream.synchronize()
        graph = self.torch.cuda.CUDAGraph()
        with self.torch.cuda.graph(graph, stream=self.stream):
            self.launch()
        self.graph = graph
        return graph

    def close(self):
        self.stream.synchronize()
        # Destroy graph references before their captured argument storage.
        self.graph = None
        self.runtime.close()


class Golden:
    """Only prompt inputs are injected; continuation is always independent."""

    def __init__(self, path, *, steps, prefix, context):
        import numpy as np
        self.path = Path(path).resolve()
        self.metadata = json.loads((self.path / 'metadata.json').read_text())
        self.start = max(0, self.metadata.get('prompt_token_count', 0) - 1)
        self.end = self.start + prefix
        if self.metadata['steps'] < steps or self.end > steps:
            raise ValueError('Golden/request has too few steps to validate the generated prefix')
        if len(self.metadata['input_tokens']) < steps or len(self.metadata['tokens']) < steps:
            raise ValueError('Golden token arrays are incomplete')
        logits = np.fromfile(self.path / 'logits.f32', dtype=np.float32)
        if logits.size != self.metadata['steps'] * self.metadata['vocab']:
            raise ValueError('Golden logit file has an invalid byte count')
        self.logits = context.torch.from_numpy(logits.reshape(self.metadata['steps'], self.metadata['vocab']))
        if context.logits.numel() != self.metadata['vocab'] or not context.torch.isfinite(self.logits).all().item():
            raise ValueError('Golden logits have wrong shape or non-finite values')
        self.inputs = context.torch.tensor(self.metadata['input_tokens'][:steps], dtype=context.values['input_ids'].dtype,
                                          device=context.device)
        self.pins = {name: file_pin(self.path / name) for name in ('metadata.json', 'logits.f32')}

    def inject(self, context, step):
        if step <= self.start:
            context.values['input_ids'].copy_(self.inputs[step].reshape_as(context.values['input_ids']))

    def compare(self, context, step, actual_token, consumed_token, *, same_history):
        actual = context.logits.detach().float().cpu().flatten()
        wanted = self.logits[step].float().flatten()
        error = actual - wanted
        wanted_norm = wanted.norm().item()
        p_log, q_log = wanted.log_softmax(0), actual.log_softmax(0)
        av, ai = actual.topk(5)
        gv, gi = wanted.topk(5)
        return {'step': step, 'input_token': consumed_token, 'golden_input_token': self.metadata['input_tokens'][step],
                'same_input_history': same_history, 'actual_token': actual_token, 'golden_token': self.metadata['tokens'][step],
                'max_absolute_error': error.abs().max().item(),
                'relative_l2_error': error.norm().item() / wanted_norm if wanted_norm else None,
                'kl_golden_to_actual': (p_log.exp() * (p_log - q_log)).sum().item(),
                'top5_actual': ai.tolist(), 'top5_actual_logits': av.tolist(),
                'top5_golden': gi.tolist(), 'top5_golden_logits': gv.tolist(),
                'actual_top_margin': (av[0] - av[1]).item(), 'golden_top_margin': (gv[0] - gv[1]).item()}


def weight_traffic(artifact):
    index = json.loads((Path(artifact) / 'assets/rdata.index.json').read_text())
    embeddings = [e for e in index['entries'] if 'embed_tokens_weight' in e['key']]
    if len(embeddings) != 1:
        return {'packed_model_bytes_per_token': None, 'traffic_reason': 'Embedding identity is not unique'}
    embedding = embeddings[0]
    packed = sum(e['nbytes'] for e in index['entries'] if e['dtype'] == 'uint8')
    consumed = packed - embedding['nbytes'] + embedding['nbytes'] // embedding['shape'][0]
    return {'packed_model_bytes_per_token': consumed,
            'traffic_definition': 'Encoded model weights, full output head, one embedding row; requested bytes, not DRAM counters'}


def run(args, report, save):
    import torch
    torch.set_num_threads(1)
    manifest, pins = artifact_identity(args.artifact)
    if manifest['codegen'].get('operator_profile') is not None:
        raise ValueError('Diagnostic instrumented artifacts cannot be used for acceptance timing; use profile.py')
    report.update(artifact_identity=pins, target=manifest['target'], codegen_backend='cpp')
    report['model_contract'] = {key: manifest.get('metadata', {}).get(key) for key in
                                ('revision', 'numerical_contract', 'num_hidden_layers', 'scope', 'mode', 'tokens_per_call')}
    report['readonly_data'] = manifest.get('rdata')
    report['checkpoint_linkage'] = 'Verifier must match original checkpoint through the frozen compile receipt'
    if args.llama_reference and (report['model_contract']['num_hidden_layers'] != 64
                                or report['model_contract']['tokens_per_call'] != 1
                                or report['model_contract']['numerical_contract'] != 'nncase'):
        raise ValueError('Golden acceptance requires all 64 layers, decode-1 and the frozen nncase numerical profile')
    context = DecodeContext(args.artifact, token_id=args.token_id, device=args.device)
    try:
        report.update(gpu=torch.cuda.get_device_name(args.device), torch_version=torch.__version__,
                      cuda_version=torch.version.cuda, resource_report=context.runtime.resource_report,
                      layers=context.runtime.ir_module.metadata.get('num_hidden_layers'))
        golden = Golden(args.llama_reference, steps=args.steps, prefix=args.golden_token_prefix,
                        context=context) if args.llama_reference else None
        if golden is not None:
            report.update(golden_identity=golden.pins, llama_commit=golden.metadata['llama_commit'],
                          golden_checkpoint_sha256=golden.metadata['checkpoint_sha256'])
        comparison, expected = [], []
        same_history = True
        context.reset()
        for step in range(args.steps):
            if golden:
                golden.inject(context, step)
            consumed = int(context.values['input_ids'].item())
            context.launch()
            context.stream.synchronize()
            if not torch.isfinite(context.logits).all().item():
                raise NumericalError(f'Non-finite eager logits at step {step}')
            token = int(context.token.item())
            expected.append(token)
            if golden:
                same_history = same_history and consumed == golden.metadata['input_tokens'][step]
                row = golden.compare(context, step, token, consumed, same_history=same_history)
                comparison.append(row)
                append_json(args.output.with_suffix('.diagnostics.jsonl'), row)
        context.check_state(args.steps)
        report.update(tokens=expected, eager_finite=True, eager_state_progression=True, llama_comparison=comparison)
        if golden:
            actual, wanted = expected[golden.start:golden.end], golden.metadata['tokens'][golden.start:golden.end]
            report['prefix'] = {'required': args.golden_token_prefix, 'start_step': golden.start,
                                'actual': actual, 'golden': wanted, 'passed': actual == wanted,
                                'independent_generation': True}
        save()
        graph = context.capture()
        # A separate graph trajectory proves finite logits on the same captured
        # path later timed; reference forwards never touch candidate storage.
        context.reset()
        graph_tokens = []
        for step in range(args.steps):
            if golden:
                golden.inject(context, step)
            graph.replay()
            context.stream.synchronize()
            if not torch.isfinite(context.logits).all().item():
                raise NumericalError(f'Non-finite graph logits at step {step}')
            graph_tokens.append(int(context.token.item()))
        context.check_state(args.steps)
        if graph_tokens != expected:
            report['graph_tokens'] = graph_tokens
            raise NumericalError('Graph/eager independent trajectories differ')
        report.update(graph_finite=True, graph_trajectory_equal=True, graph_state_progression=True)
        save()
        latencies, rounds = [], []
        report['round_timing'] = timing_metadata()
        for repeat in range(args.repeats + 1):
            context.reset()
            measured, trajectory = [], []
            round_start_ns = monotonic_ns()
            for step in range(args.steps):
                if golden:
                    golden.inject(context, step)
                context.stream.synchronize()
                start = perf_counter()
                graph.replay()
                returned_token = int(context.token.item())
                measured.append((perf_counter() - start) * 1000)
                trajectory.append(returned_token)
            round_end_ns = monotonic_ns()
            window = {'schema': WINDOW_SCHEMA, 'repeat': repeat, 'warmup': repeat == 0,
                      'start_ns': str(round_start_ns), 'end_ns': str(round_end_ns),
                      'sample_count': args.steps, 'sample_offset': None if repeat == 0 else (repeat - 1) * args.steps}
            report['round_timing']['rounds'].append(window)
            row = {'repeat': repeat, 'warmup': repeat == 0, 'latency_ms': measured,
                   'tokens': trajectory, 'trajectory_equal': trajectory == expected,
                   'sequence_length': int(context.values['paged_state.seq_lens'].item()),
                   'round_window': window, 'clock': report['round_timing']['clock'],
                   'policy': report['round_timing']['policy'], 'policy_limits': report['round_timing']['policy_limits'], **summarize(measured)}
            # Preserve each completed sample batch before subsequent validation.
            append_json(args.output.with_suffix('.rounds.jsonl'), row)
            if repeat:
                latencies.append(measured)
                rounds.append(row)
                report.update(latency_ms=latencies, rounds=rounds)
                save()
            context.check_state(args.steps)
            if trajectory != expected:
                raise NumericalError(f'Timed graph trajectory changed in repeat {repeat}')
        report['rounds_sidecar'] = byte_pin(args.output.with_suffix('.rounds.jsonl').read_bytes())
        samples = [x for batch in latencies for x in batch]
        report.update(**summarize(samples), timed_trajectories_equal=True, repeats_reset_state=True,
                      warmup_rounds=1, round_mean_ms=[statistics.mean(batch) for batch in latencies],
                      **weight_traffic(args.artifact), peak_allocated_bytes=torch.cuda.max_memory_allocated(args.device))
        if report.get('packed_model_bytes_per_token'):
            report['model_bandwidth_gb_s'] = report['packed_model_bytes_per_token'] / (report['mean_ms'] * 1e6)
        report['artifact_unchanged'] = artifact_identity(args.artifact)[1] == pins
        if not report['artifact_unchanged']:
            raise RuntimeError('Artifact inputs changed during measurement')
        prefix_ok = report.get('prefix', {}).get('passed')
        report['numerical_status'] = 'passed' if prefix_ok else 'failed' if golden else 'not_qualified_no_golden'
        report['performance_acceptance'] = 'pending_external_matched_comparison_and_monitor_review'
        if golden and not prefix_ok:
            raise NumericalError('Independent generated prefix differs from llama.cpp; samples retained as diagnostics')
    finally:
        context.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--artifact', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--llama-reference', type=Path)
    parser.add_argument('--recipe', type=Path, help='JSON keyword arguments to local_optimizations.install')
    parser.add_argument('--steps', type=int, default=32)
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--token-id', type=int, default=1)
    parser.add_argument('--golden-token-prefix', type=int, default=8)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--independent-generation', action='store_true', default=True,
                        help='Always enabled; only prompt tokens are injected')
    parser.add_argument('--measurement', choices=('host-token',), default='host-token')
    args = parser.parse_args()
    if args.steps < 1 or args.repeats < 1 or args.golden_token_prefix < 8:
        parser.error('steps/repeats must be positive; the frozen contract requires at least 8 generated tokens')
    paths = [args.output, args.output.with_suffix('.diagnostics.jsonl'), args.output.with_suffix('.rounds.jsonl')]
    if any(path.exists() for path in paths):
        parser.error('Use fresh result and diagnostic paths; earlier attempts are preserved')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    report = {'schema': PYTHON_RESULT, 'status': 'running', 'started_utc': utc_now(),
              'argv': sys.argv, 'pid': os.getpid(), 'artifact': str(args.artifact.resolve()),
              'steps': args.steps, 'repeats': args.repeats, 'harness': file_pin(__file__),
              'measurement': 'Host wall clock: graph replay, GPU greedy, slot/token feedback, one CPU token return',
              'excluded': 'load/prepare, state reset, prompt injection, eager/graph diagnostics, one full warmup round',
              'monitor_status': 'not_checked_by_harness; attach Orchestrator/Verifier CPU/GPU monitor evidence',
              'numerical_status': 'not_run', 'performance_acceptance': 'not_run'}
    save = lambda: save_json(args.output, report)
    save()
    try:
        report['recipe'] = install_recipe(args.recipe)
        run(args, report, save)
        report['status'] = 'completed'
    except BaseException as error:
        if isinstance(error, NumericalError):
            report['numerical_status'] = 'failed'
        report.update(status='failed', error_type=type(error).__name__, error=str(error), ended_utc=utc_now())
        save()
        raise
    report['ended_utc'] = utc_now()
    save()
    print(json.dumps({k: v for k, v in report.items() if k not in ('latency_ms', 'rounds', 'llama_comparison')},
                     indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
