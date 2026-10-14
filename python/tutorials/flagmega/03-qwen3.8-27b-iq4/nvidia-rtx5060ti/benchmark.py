"""Stateful decode timing with CUDA events, graph replay and packed weight traffic."""

import argparse
import hashlib
import json
import statistics
import sys
from time import perf_counter
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--artifact', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    golden = parser.add_mutually_exclusive_group()
    golden.add_argument('--reference', type=Path, help='Internal FlagMega regression reference, not model golden')
    golden.add_argument('--llama-reference', type=Path, help='Independent llama_golden.py output directory')
    parser.add_argument('--golden-token-prefix', type=int, default=8,
                        help='Required equal initial generated tokens; logits are diagnostic')
    parser.add_argument('--independent-generation', action='store_true',
                        help='Inject only prompt tokens; generate the accepted prefix independently')
    parser.add_argument('--optimization', choices=('pc_tma', 'pc_tma_inline'),
                        help='Load the local candidate catalog before artifact verification')
    parser.add_argument('--measurement', choices=('cuda-event', 'host-token'), default='cuda-event',
                        help='host-token matches llama.cpp decode + GPU greedy + one CPU token return')
    parser.add_argument('--grid', type=int, choices=(32, 36), default=32)
    parser.add_argument('--gdn-value-tile', type=int, choices=(4, 8, 16, 32, 64), default=4)
    parser.add_argument('--gdn-fast-head', action='store_true')
    parser.add_argument('--pc-segments', action='store_true')
    parser.add_argument('--q8-group-tile', type=int, choices=(8, 16, 32, 64, 128, 256), default=8)
    parser.add_argument('--pc-dual-activation', action='store_true')
    parser.add_argument('--pc-dual-all', action='store_true')
    parser.add_argument('--balanced-rows', action='store_true')
    parser.add_argument('--pc-dense-dual', action='store_true')
    parser.add_argument('--pc-global-activation', action='store_true')
    parser.add_argument('--pc-packed-activation', action='store_true')
    parser.add_argument('--pc-down-chunk', type=int, choices=(1024, 2048, 4096), default=1024)
    parser.add_argument('--pc-precision-scope', choices=('all', 'ffn', 'glu', 'down', 'attention'), default='all')
    parser.add_argument('--pc-nibble-pair', action='store_true')
    parser.add_argument('--gdn-linear-owners', action='store_true')
    parser.add_argument('--gdn-uniform-norm', action='store_true')
    parser.add_argument('--gdn-cache-heads', action='store_true')
    parser.add_argument('--q8-vector-quant', action='store_true')
    parser.add_argument('--save-reference', type=Path)
    parser.add_argument('--steps', type=int, default=32)
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--token-id', type=int, default=1)
    args = parser.parse_args()
    if args.steps < 1 or args.repeats < 1 or args.golden_token_prefix < 1:
        parser.error('steps, repeats and golden token prefix must be positive')
    if args.output.exists():
        parser.error('use a fresh result path')
    sys.path.insert(0, str(Path(__file__).resolve().parents[4]))
    import torch
    if args.optimization:
        from agent_optimizations import install
        install(mode=args.optimization, warps=8, tile_n=32, chunk=512, grid=args.grid,
                gdn_value_tile=args.gdn_value_tile, gdn_fast_head=args.gdn_fast_head, pc_segments=args.pc_segments,
                q8_group_tile=args.q8_group_tile, pc_dual_activation=args.pc_dual_activation,
                pc_dual_all=args.pc_dual_all, balanced_rows=args.balanced_rows, pc_dense_dual=args.pc_dense_dual,
                pc_global_activation=args.pc_global_activation, pc_packed_activation=args.pc_packed_activation,
                pc_down_chunk=args.pc_down_chunk, pc_precision_scope=args.pc_precision_scope,
                pc_nibble_pair=args.pc_nibble_pair, gdn_linear_owners=args.gdn_linear_owners,
                gdn_uniform_norm=args.gdn_uniform_norm, gdn_cache_heads=args.gdn_cache_heads,
                q8_vector_quant=args.q8_vector_quant)
    from triton.flagmega.ir import VectorType
    from triton.flagmega.runtime import load

    torch.set_num_threads(1)
    # Runtime scratch is bound to its first stream. Keep eager and graph launches
    # on the same non-default stream, including graph capture.
    stream = torch.cuda.Stream()
    torch.cuda.set_stream(stream)
    runtime = load(args.artifact, device='cuda:0')
    values = {}
    for argument in runtime.external_arguments:
        buffer = runtime.buffer_plan.buffer_map[argument['buffer']]
        dtype = buffer.dtype
        lanes = dtype.lanes if isinstance(dtype, VectorType) else ()
        scalar = dtype.elem_type if lanes else dtype
        values[buffer.id] = torch.zeros((*buffer.shape, *lanes), dtype=getattr(torch, scalar.value), device='cuda:0')
    values['input_ids'].fill_(args.token_id)
    values['paged_state.query_start_loc'].copy_(torch.tensor([0, 1], dtype=torch.int32, device='cuda:0'))
    table = values['paged_state.block_table']
    table.copy_(torch.arange(table.numel(), dtype=table.dtype, device=table.device).reshape_as(table))
    arguments = tuple(values[a['buffer']] for a in runtime.external_arguments)
    outputs = runtime.buffer_plan.function_map[runtime.ir_module.entry].outputs
    logits, token = (values[outputs[i][1][0]] for i in (0, 1))
    initial = {name: value.clone() for name, value in values.items()}

    def reset():
        for name, value in values.items():
            value.copy_(initial[name])

    def launch():
        values['paged_state.slot_mapping'].copy_(values['paged_state.seq_lens'])
        runtime.run_into(*arguments)
        values['input_ids'].copy_(token.reshape_as(values['input_ids']))

    runtime.prepare(*arguments)
    print(json.dumps({'prepared_resources': runtime.resource_report}), flush=True)
    reset()
    reference = torch.load(args.reference, weights_only=True) if args.reference else None
    llama_reference, forced_inputs = None, None
    comparison = []
    if args.llama_reference:
        import numpy as np
        llama_reference = json.loads((args.llama_reference / 'metadata.json').read_text())
        if llama_reference['steps'] < args.steps:
            parser.error('llama.cpp golden has fewer steps than requested')
        prefix_start = max(0, llama_reference.get('prompt_token_count', 0) - 1)
        prefix_end = prefix_start + args.golden_token_prefix
        if prefix_end > args.steps:
            parser.error('Need more steps to compare the requested generated token prefix after the prompt')
        golden_logits = torch.from_numpy(
            np.fromfile(args.llama_reference / 'logits.f32',
                        dtype=np.float32).reshape(llama_reference['steps'], llama_reference['vocab']))
        if logits.numel() != llama_reference['vocab'] or not torch.isfinite(golden_logits).all():
            raise RuntimeError('invalid llama.cpp golden shape or values')
        forced_inputs = torch.tensor(llama_reference['input_tokens'][:args.steps], dtype=torch.int64, device='cuda:0')
    expected, saved_logits = [], []
    max_absolute_error, max_relative_l2_error = 0., 0.

    def inject_token(step):
        if forced_inputs is not None and (not args.independent_generation or step <= prefix_start):
            values['input_ids'].copy_(forced_inputs[step].reshape_as(values['input_ids']))

    for step in range(args.steps):
        inject_token(step)
        consumed_token = int(values['input_ids'].item())
        launch()
        torch.cuda.synchronize()
        if not torch.isfinite(logits).all():
            raise RuntimeError(f'non-finite logits at step {step}')
        expected.append(int(token.item()))
        actual = logits.cpu()
        if llama_reference is not None:
            wanted = golden_logits[step].reshape_as(actual)
            error = (actual - wanted).float()
            p_log = wanted.float().flatten().log_softmax(0)
            q_log = actual.float().flatten().log_softmax(0)
            top_actual = actual.flatten().topk(5).indices.tolist()
            top_wanted = wanted.flatten().topk(5).indices.tolist()
            comparison.append({
                'step':
                step, 'input_token':
                consumed_token, 'golden_input_token':
                int(forced_inputs[step].item()), 'same_input_token':
                consumed_token == int(forced_inputs[step].item()), 'actual_token':
                expected[-1], 'golden_token':
                llama_reference['tokens'][step], 'max_absolute_error':
                error.abs().max().item(), 'relative_l2_error': (error.norm() / wanted.float().norm()).item(),
                'cosine_similarity':
                torch.nn.functional.cosine_similarity(actual.flatten(), wanted.flatten(), dim=0).item(),
                'kl_golden_to_actual': (p_log.exp() * (p_log - q_log)).sum().item(), 'top5_actual':
                top_actual, 'top5_golden':
                top_wanted, 'top5_overlap':
                len(set(top_actual) & set(top_wanted))
            })
            max_absolute_error = max(max_absolute_error, comparison[-1]['max_absolute_error'])
            max_relative_l2_error = max(max_relative_l2_error, comparison[-1]['relative_l2_error'])
            print(json.dumps({'llama_comparison': comparison[-1]}), flush=True)
        if reference is not None:
            wanted = reference['logits'][step]
            error = (actual - wanted).float()
            max_absolute_error = max(max_absolute_error, error.abs().max().item())
            max_relative_l2_error = max(max_relative_l2_error, (error.norm() / wanted.float().norm()).item())
            try:
                torch.testing.assert_close(actual, wanted, rtol=0.02, atol=0.15)
            except AssertionError:
                print(
                    json.dumps({
                        'failed_step': step, 'token': expected[-1], 'max_absolute_error': error.abs().max().item(),
                        'relative_l2_error': (error.norm() / wanted.float().norm()).item()
                    }), flush=True)
                raise
        if args.save_reference:
            saved_logits.append(actual.clone())
    if reference is not None and expected != reference['tokens'][:args.steps]:
        raise RuntimeError('autoregressive tokens differ from the independent baseline')
    if args.save_reference:
        args.save_reference.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                'tokens': expected, 'logits': saved_logits, 'input_tokens':
                None if llama_reference is None else llama_reference['input_tokens'][:args.steps]
            }, args.save_reference)
    if int(values['paged_state.seq_lens'].item()) != args.steps:
        raise RuntimeError('sequence length did not advance correctly')
    reset()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        launch()
    latencies = []
    for repeat in range(args.repeats + 1):
        reset()
        measured, trajectory = [], []
        for step in range(args.steps):
            inject_token(step)
            if args.measurement == 'host-token':
                stream.synchronize()  # input injection/reset is outside both timing boundaries
                start_time = perf_counter()
                graph.replay()
                returned_token = int(token.item())
                measured.append((perf_counter() - start_time) * 1000)
                trajectory.append(returned_token)
            else:
                start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                start.record()
                graph.replay()
                end.record()
                end.synchronize()
                measured.append(start.elapsed_time(end))
                trajectory.append(int(token.item()))
        if trajectory != expected or int(values['paged_state.seq_lens'].item()) != args.steps:
            raise RuntimeError('captured graph changed the decode trajectory')
        if repeat:
            latencies.append(measured)
    index = json.loads((args.artifact / 'assets/rdata.index.json').read_text())
    embedding = next(e for e in index['entries'] if 'embed_tokens_weight' in e['key'])
    # One embedding row is read; the entire embedding table is not per-token traffic.
    packed_bytes = sum(e['nbytes'] for e in index['entries'] if e['dtype'] == 'uint8')
    consumed = packed_bytes - embedding['nbytes'] + embedding['nbytes'] // embedding['shape'][0]
    samples = [sample for repeat in latencies for sample in repeat]
    mean = statistics.mean(samples)
    bandwidth = consumed / (mean / 1000) / 1e9
    report = {
        'artifact':
        str(args.artifact.resolve()), 'gpu':
        torch.cuda.get_device_name(), 'source_sha256':
        hashlib.sha256((args.artifact / 'generated_kernels.py').read_bytes()).hexdigest(), 'layers':
        runtime.ir_module.metadata['num_hidden_layers'], 'steps':
        args.steps, 'repeats':
        args.repeats, 'tokens':
        expected, 'finite':
        True, 'measurement':
        ('Host wall clock around graph replay, GPU greedy and one CPU token return' if args.measurement == 'host-token'
         else 'CUDA events around graph replay: model, sampling, device token/slot feedback'), 'latency_ms':
        latencies, 'mean_ms':
        mean, 'median_ms':
        statistics.median(samples), 'tokens_per_second':
        1000 / mean, 'packed_model_bytes_per_token':
        consumed, 'model_bandwidth_gb_s':
        bandwidth, 'theoretical_memory_bandwidth_gb_s':
        448, 'model_bandwidth_utilization':
        bandwidth / 448, 'max_absolute_logit_error':
        max_absolute_error, 'max_relative_l2_logit_error':
        max_relative_l2_error, 'resource_report':
        runtime.resource_report, 'peak_allocated_bytes':
        torch.cuda.max_memory_allocated()
    }
    if llama_reference is not None:
        prefix_match = expected[prefix_start:prefix_end] == llama_reference['tokens'][prefix_start:prefix_end]
        report['llama_comparison'] = {
            'reference': str(args.llama_reference.resolve()), 'llama_commit': llama_reference['llama_commit'],
            'forced_same_input_tokens': not args.independent_generation, 'independent_generation':
            args.independent_generation, 'reference_measurement': llama_reference['measurement'], 'reference_mean_ms':
            llama_reference['mean_ms'], 'reference_tokens_per_second': llama_reference['tokens_per_second'],
            'token_agreement': sum(row['actual_token'] == row['golden_token']
                                   for row in comparison) / args.steps, 'per_step': comparison, 'acceptance':
            'User contract: initial generated token prefix equal; logits errors are diagnostic',
            'required_generated_prefix': args.golden_token_prefix, 'prefix_start_step': prefix_start, 'prefix_match':
            prefix_match, 'actual_prefix': expected[prefix_start:prefix_end], 'golden_prefix':
            llama_reference['tokens'][prefix_start:prefix_end]
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({k: v for k, v in report.items() if k != 'latency_ms'}, indent=2), flush=True)
    if llama_reference is not None and not prefix_match:
        raise RuntimeError('Initial generated tokens differ from llama.cpp golden; diagnostic results retained')


if __name__ == '__main__':
    main()
