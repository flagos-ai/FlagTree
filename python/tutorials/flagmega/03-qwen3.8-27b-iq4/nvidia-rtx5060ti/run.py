"""Execute the compiled native IQ4_XS/Q4_K decoder on RTX 5060 Ti."""

import argparse
import json
import sys
from pathlib import Path
from time import perf_counter


def run_full(module, args, torch):
    from triton.flagmega.ir import VectorType
    values = {}
    for argument in module.external_arguments:
        buffer = module.buffer_plan.buffer_map[argument['buffer']]
        dtype = buffer.dtype
        lanes = dtype.lanes if isinstance(dtype, VectorType) else ()
        scalar = dtype.elem_type if lanes else dtype
        values[buffer.id] = torch.zeros((*buffer.shape, *lanes), dtype=getattr(torch, scalar.value), device='cuda:0')
    values['input_ids'].fill_(args.token_id)
    values['paged_state.query_start_loc'].copy_(torch.tensor([0, 1], dtype=torch.int32, device='cuda:0'))
    table = values['paged_state.block_table']
    table.copy_(torch.arange(table.numel(), dtype=table.dtype, device=table.device).reshape_as(table))
    arguments = tuple(values[a['buffer']] for a in module.external_arguments)
    outputs = module.buffer_plan.function_map[module.ir_module.entry].outputs
    logits, token = (values[outputs[i][1][0]] for i in (0, 1))
    module.prepare(*arguments)
    tokens, latencies = [], []
    for step in range(args.steps):
        values['paged_state.slot_mapping'].copy_(values['paged_state.seq_lens'])
        started = perf_counter()
        module.run_into(*arguments)
        torch.cuda.synchronize()
        latencies.append((perf_counter() - started) * 1000)
        if not torch.isfinite(logits).all():
            raise SystemExit(f'Native quantized decoder produced non-finite logits at step {step}')
        tokens.append(int(token.item()))
        values['input_ids'].copy_(token.reshape_as(values['input_ids']))
    return {
        'scope': 'text-decoder', 'layers': module.ir_module.metadata['num_hidden_layers'], 'output_shape':
        list(logits.shape), 'output_norm': logits.norm().item(), 'tokens': tokens, 'sequence_length':
        int(values['paged_state.seq_lens'].item()), 'latency_ms': latencies
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--artifact', type=Path, required=True)
    parser.add_argument('--token-id', type=int, default=1)
    parser.add_argument('--steps', type=int, default=4)
    args = parser.parse_args()
    if args.steps < 1:
        parser.error('--steps must be positive')
    sys.path.insert(0, str(Path(__file__).resolve().parents[4]))
    import torch
    from triton.flagmega.runtime import load

    torch.set_num_threads(1)
    module = load(args.artifact, device='cuda:0')
    if getattr(module, 'result_kind', None) == 'tensor_state':
        state = module.create_state()
        ids = torch.tensor([args.token_id], dtype=torch.int32, device='cuda:0')
        output = module.create_outputs()
        module.prepare(ids, state, output=output)
        for _ in range(args.steps):
            module.run_into(output, ids, state)
            torch.cuda.synchronize()
            if not torch.isfinite(output).all():
                raise SystemExit('Native quantized layer produced a non-finite output')
        result = {'scope': 'layer0', 'output_shape': list(output.shape), 'output_norm': output.float().norm().item()}
    else:
        result = run_full(module, args, torch)
    result.update({
        'artifact': str(args.artifact.resolve()), 'gpu': torch.cuda.get_device_name(), 'steps': args.steps, 'finite':
        True, 'peak_allocated_bytes': torch.cuda.max_memory_allocated()
    })
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
