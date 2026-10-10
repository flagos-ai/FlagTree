"""Compare native FlagOS vLLM and native-prefill/FlagMega-decode requests.

Both variants use BF16, batch/concurrency one, 256-token cache pages, 16
pages, no prefix reuse, and identical native scheduling and sampling. The
native FlagOS path can optionally use a vLLM FULL CUDA graph; FlagMega keeps
the eager worker-extension contract until graph capture is moved after the
extension is installed. All warmup and measured independent greedy
sequences must agree.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import statistics
import subprocess
import time


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def source_identity(module_file):
    path = Path(module_file).resolve()
    root = subprocess.check_output(['git', '-C', str(path.parent), 'rev-parse', '--show-toplevel'], text=True).strip()
    head = subprocess.check_output(['git', '-C', root, 'rev-parse', 'HEAD'], text=True).strip()
    diff = subprocess.check_output(['git', '-C', root, 'diff', 'HEAD', '--binary'])
    return {'path': str(path), 'revision': head, 'diff_sha256': hashlib.sha256(diff).hexdigest()}


def summarize(values):
    ordered = sorted(values)
    return {'median': statistics.median(ordered), 'mean': statistics.mean(ordered),
            'p95': ordered[min(len(ordered)-1, int(.95 * len(ordered)))], 'count': len(ordered)}


def require_agreement(actual, expected, context):
    if actual['prompt_tokens'] != expected['prompt_tokens']:
        raise AssertionError(f'{context}: prompts differ')
    if len(actual['token_ids']) != len(expected['token_ids']):
        raise AssertionError(f'{context}: output lengths differ')
    for index, (token, reference) in enumerate(zip(actual['token_ids'], expected['token_ids'], strict=True)):
        if token != reference:
            raise AssertionError(f'{context}: token {index}: {token} != {reference}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--artifact', type=Path)
    parser.add_argument('--reference', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--prompt-lengths', nargs='+', type=int, default=[32, 128, 1024, 2048])
    parser.add_argument('--decode-tokens', type=int, default=64)
    parser.add_argument('--warmups', type=int, default=1)
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--rotation', type=int, default=0)
    parser.add_argument('--chat-smoke', action='store_true', help='Use Chinese and arithmetic chat prompts')
    parser.add_argument('--cuda-graph', action=argparse.BooleanOptionalAction, default=False,
                        help='Use vLLM FULL CUDA Graph for native FlagOS (default: eager)')
    args = parser.parse_args()
    if args.output.exists():
        parser.error('Use a fresh output; measurements are immutable')
    if args.decode_tokens < 2 or args.warmups < 1 or args.repeats < 1:
        parser.error('Require >=2 output tokens, >=1 warmup and >=1 measured request')
    if min(args.prompt_lengths) < 2 or max(args.prompt_lengths) + args.decode_tokens > 3072:
        parser.error('Require prompt>=2 and prompt+decode<=3072')
    if args.cuda_graph and args.artifact:
        parser.error('--cuda-graph is currently supported only for native FlagOS; '
                     'the FlagMega worker extension is installed after vLLM graph capture')
    os.environ['VLLM_ENABLE_V1_MULTIPROCESSING'] = '0'
    os.environ['CUDA_MODULE_LOADING'] = '0'
    import torch
    import triton
    import vllm
    import flag_gems
    import vllm_fl
    from triton._C import libtriton
    from vllm import LLM, SamplingParams

    torch.set_num_threads(1)
    checkpoint = args.checkpoint.resolve()
    expected_report = json.loads(args.reference.read_text()) if args.reference else None
    checkpoint_id = {p.name: sha256(p) for p in sorted(checkpoint.glob('*.safetensors'))}
    checkpoint_id['config.json'] = sha256(checkpoint/'config.json')
    if expected_report:
        if not expected_report['performance_valid'] or expected_report['checkpoint_files'] != checkpoint_id:
            raise ValueError('Reference is unaccepted or uses different checkpoint files')
        for key, value in [('decode_tokens', args.decode_tokens), ('batch', 1), ('block_size', 256)]:
            if expected_report[key] != value: raise ValueError(f'Reference differs at {key}')
    extra = {'worker_extension_cls': 'flagos_adapter.FlagMegaExtension'} if args.artifact else {}
    compilation_config = {
        'mode': 0,
        'cudagraph_mode': 'FULL' if args.cuda_graph else 'NONE',
        'cudagraph_capture_sizes': [1] if args.cuda_graph else [],
    }
    llm = LLM(model=str(checkpoint), dtype='bfloat16', max_model_len=3072, max_num_seqs=1,
              max_num_batched_tokens=3072, gpu_memory_utilization=.35,
              enforce_eager=not args.cuda_graph, compilation_config=compilation_config,
              block_size=256, num_gpu_blocks_override=16, enable_prefix_caching=False,
              enable_chunked_prefill=False, disable_log_stats=True, seed=0, **extra)
    engine = llm.llm_engine
    report = {'schema': 'flagmega.flagos-benchmark/v2',
              'label': 'flagmega_decode' if args.artifact else 'vllm_flagos',
              'boundary': 'engine.step: scheduler+forward+sampler+output processing; initialization excluded',
              'prefill': 'native FlagOS vLLM', 'decode': 'FlagMega' if args.artifact else 'native FlagOS vLLM',
              'batch': 1, 'concurrency': 1, 'block_size': 256, 'num_blocks': 16,
              'cuda_graph': args.cuda_graph,
              'cudagraph_mode': 'FULL' if args.cuda_graph else 'NONE',
              'compilation_mode': 0, 'enforce_eager': not args.cuda_graph,
              'prefix_caching': False, 'decode_tokens': args.decode_tokens,
              'checkpoint': str(checkpoint), 'checkpoint_files': checkpoint_id,
              'torch': importlib.metadata.version('torch'), 'triton': triton.__version__, 'vllm': vllm.__version__,
              'vllm_source': source_identity(vllm.__file__), 'flaggems_source': source_identity(flag_gems.__file__),
              'plugin_source': source_identity(vllm_fl.__file__), 'compiler_sha256': sha256(libtriton.__file__),
              'adapter_sha256': sha256(Path(__file__).with_name('flagos_adapter.py')),
              'benchmark_sha256': sha256(__file__), 'device': torch.cuda.get_device_name(),
              'visible_devices': os.environ.get('CUDA_VISIBLE_DEVICES'),
              'timestamp_utc': datetime.now(timezone.utc).isoformat(), 'warmups': args.warmups,
              'repeats': args.repeats, 'rotation': args.rotation, 'scenarios': [],
              'native_token_agreement': False if expected_report else None, 'performance_valid': False}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    current = None
    try:
        if args.artifact:
            llm.collective_rpc('flagmega_install', args=(str(args.artifact.resolve()),))
        tokenizer = llm.get_tokenizer()
        if args.chat_smoke:
            scenarios = []
            for name, content in [('chat_chinese', '请用一句话介绍你自己。'),
                                  ('chat_arithmetic', 'What is 2 + 3? Answer with the number only.')]:
                rendered = tokenizer.apply_chat_template(
                    [{'role': 'user', 'content': content}], tokenize=False,
                    add_generation_prompt=True, enable_thinking=False)
                scenarios.append((name, tokenizer.encode(rendered, add_special_tokens=False)))
        else:
            source = tokenizer.encode('Explain why the sky is blue, and how sunlight interacts with air. ', add_special_tokens=False)
            scenarios = [(f'prompt{length}', (source * ((length+len(source)-1)//len(source)))[:length])
                         for length in args.prompt_lengths]
        rotation = args.rotation % len(scenarios)
        scenarios = scenarios[rotation:] + scenarios[:rotation]
        expected = {s['scenario']:s['runs'][0] for s in expected_report['scenarios']} if expected_report else None
        sampling = SamplingParams(temperature=0, max_tokens=args.decode_tokens, ignore_eos=True)
        counter = 0

        def request(prompt):
            nonlocal counter
            counter += 1
            started = time.perf_counter()
            engine.add_request(str(counter), {'prompt_token_ids':prompt}, sampling)
            previous = started
            steps = []
            output = None
            while engine.has_unfinished_requests():
                values = engine.step()
                now = time.perf_counter()
                if values:
                    if len(values) != 1: raise RuntimeError('Unexpected concurrent output')
                    output = values[0]
                    steps.append((now-previous)*1000)
                    previous = now
            elapsed = (time.perf_counter()-started)*1000
            if output is None: raise RuntimeError('No output')
            tokens = list(output.outputs[0].token_ids)
            if len(tokens) != args.decode_tokens or len(steps) != args.decode_tokens:
                raise RuntimeError('Expected exactly one token per engine step')
            return {'prompt_tokens':prompt, 'token_ids':tokens, 'text':output.outputs[0].text,
                    'ttft_ms':steps[0], 'decode_step_ms':steps[1:], 'e2e_ms':elapsed}

        for name, prompt in scenarios:
            current = {'scenario':name, 'prompt_length':len(prompt), 'warmup_runs':[], 'runs':[]}
            truth = expected[name] if expected else None
            for _ in range(args.warmups):
                run = request(prompt)
                current['warmup_runs'].append(run)
                if truth is None: truth = run
                require_agreement(run, truth, f'{name} warmup')
            for _ in range(args.repeats):
                run = request(prompt)
                current['runs'].append(run)
                require_agreement(run, truth, f'{name} measured')
            current.update(decode_ms=summarize([x for r in current['runs'] for x in r['decode_step_ms']]),
                           ttft_ms=summarize([r['ttft_ms'] for r in current['runs']]),
                           e2e_ms=summarize([r['e2e_ms'] for r in current['runs']]))
            report['scenarios'].append(current)
            args.output.write_text(json.dumps(report, indent=2)+'\n')
            print(json.dumps({k:v for k,v in current.items() if k not in ('runs','warmup_runs')}), flush=True)
            current = None
        if args.artifact:
            report['executor_stats'] = llm.collective_rpc('flagmega_stats')
            calls = sum((args.decode_tokens-1)*(args.warmups+args.repeats) for _ in scenarios)
            if report['executor_stats'][0]['decode_calls'] != calls:
                raise AssertionError('Not every decode step used FlagMega')
            if report['executor_stats'][0]['native_prefill_calls'] != len(scenarios)*(args.warmups+args.repeats):
                raise AssertionError('Not every prompt used native prefill')
        report['performance_valid'] = True
        report['native_token_agreement'] = True if expected_report else None
    except BaseException as error:
        report['failure'] = {'error':str(error), 'scenario':current}
        raise
    finally:
        args.output.write_text(json.dumps(report, indent=2)+'\n')
        engine.engine_core.shutdown()
        from vllm.distributed.parallel_state import destroy_model_parallel, destroy_distributed_environment
        destroy_model_parallel()
        destroy_distributed_environment()


if __name__ == '__main__':
    main()
