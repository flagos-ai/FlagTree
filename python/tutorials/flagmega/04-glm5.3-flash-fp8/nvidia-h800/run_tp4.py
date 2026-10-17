# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Execute the original complete TP4 model; compare independent greedy requests."""
import argparse
import hashlib
import json
import os
import time
import traceback
from pathlib import Path

import torch
import torch.distributed as dist
from triton.flagmega.runtime import load


def emit(event, **fields):
    row = {'rank': int(os.environ['RANK']), 'event': event, **fields}
    print(json.dumps(row), flush=True)


def reset_request(state):
    for kda in (state.kda, state.next_kda):
        kda.convolution.zero_()
        kda.recurrent.zero_()
    state.paged.kv_caches.zero_()
    state.paged.seq_lens.zero_()
    state.paged.slot_mapping.zero_()
    state.paged.query_start_loc.copy_(torch.tensor([0, 1], dtype=torch.int32, device=state.paged.seq_lens.device))
    state.paged.block_table.copy_(torch.arange(state.paged.config.num_blocks, dtype=torch.int32,
                                              device=state.paged.seq_lens.device).reshape(1, -1))
    if state.dsa is not None:
        state.dsa.packed_states.zero_()
        state.dsa.seq_lens.zero_()


def compare(got, expected, *, label, atol=1e-4, rtol=.005):
    assert torch.isfinite(got).all(), f'{label} contains non-finite values'
    torch.testing.assert_close(got, expected, atol=atol, rtol=rtol, msg=label)
    return float((got.float() - expected.float()).abs().max())


def official_prefix(path, steps):
    """Read the independently generated official sequence without modifying it."""
    if type(steps) is not int or steps < 1:
        raise ValueError('The token prefix length must be positive.')
    report = json.loads(path.read_text())
    if report.get('versions', {}).get('vllm') != '0.31.0':
        raise ValueError('The official reference must identify vLLM 0.31.0.')
    requests = report.get('requests', ())
    if not requests:
        raise ValueError('The official reference has no independent requests.')
    prefixes = []
    for request in requests:
        tokens = request.get('tokens', ())
        if len(tokens) < steps or any(type(token) is not int or token < 0 for token in tokens):
            raise ValueError('The official reference has an invalid or incomplete token sequence.')
        prefixes.append(tokens[:steps])
    if any(prefix != prefixes[0] for prefix in prefixes[1:]):
        raise ValueError('Independent official requests disagree on the requested prefix.')
    return prefixes[0], report['versions']


def check_greedy_step(input_token, selected, prefix, initial_token, step):
    expected_input = initial_token if step == 0 else prefix[step - 1]
    assert input_token == expected_input, 'independent greedy histories diverged'
    assert selected == prefix[step], (step, selected, prefix[step])


def count_nonzero_bounded(value, *, chunk_elements=64 << 10):
    """Count every element while bounding the temporary boolean mask size."""
    if chunk_elements < 1:
        raise ValueError('chunk_elements must be positive')
    total = 0
    pending = [value]
    while pending:
        part = pending.pop()
        if part.numel() <= chunk_elements:
            total += int(torch.count_nonzero(part).item())
            continue
        # Split views instead of flattening: flatten can copy a strided cache.
        axis = next(i for i, extent in enumerate(part.shape) if extent > 1)
        per_index = part.numel() // part.shape[axis]
        pending.extend(part.split(max(1, chunk_elements // per_index), dim=axis))
    return total


def check_state(state, reference, step):
    errors = {'convolution': 0., 'recurrent': 0.}
    for layer, (conv, rec) in state.tensor_states.items():
        if reference is None:
            assert torch.isfinite(conv).all(), f'KDA convolution layer {layer} contains non-finite values'
            assert torch.isfinite(rec).all(), f'KDA recurrence layer {layer} contains non-finite values'
        else:
            want_conv, want_rec = reference['tensor_states'][layer]
            errors['convolution'] = max(errors['convolution'], compare(
                conv.cpu(), want_conv, label=f'KDA convolution layer {layer}'))
            errors['recurrent'] = max(errors['recurrent'], compare(
                rec.cpu(), want_rec, label=f'KDA recurrence layer {layer}', atol=1e-5))
    paged = state.paged
    assert int(paged.seq_lens.item()) == step + 1
    if reference is not None:
        for name, expected in reference['paged_metadata'].items():
            torch.testing.assert_close(getattr(paged, name).cpu(), expected, atol=0, rtol=0, msg=name)
    active = torch.stack([
        paged.kv_caches[int(paged.block_table[0, pos // paged.config.block_size]),
                        :, :, pos % paged.config.block_size]
        for pos in range(step + 1)
    ])
    assert torch.isfinite(active).all(), 'active KV cache contains non-finite values'
    if reference is not None:
        errors['kv'] = compare(active.cpu(), reference['active_kv'], label='active KV cache')
    assert count_nonzero_bounded(paged.kv_caches) == count_nonzero_bounded(active), 'KV tail was written'
    if state.dsa is not None:
        packed = state.dsa.packed_states[:, :step + 1]
        assert torch.isfinite(packed).all(), 'active DSA cache contains non-finite values'
        assert int(state.dsa.seq_lens.item()) == step + 1
        if reference is not None:
            errors['dsa'] = compare(packed.cpu(), reference['dsa_packed'], label='active DSA cache')
            torch.testing.assert_close(state.dsa.seq_lens.cpu(), reference['dsa_seq_lens'], atol=0, rtol=0)
        assert count_nonzero_bounded(state.dsa.packed_states) == count_nonzero_bounded(packed), 'DSA tail was written'
    return errors if reference is not None else {}


def main(args):
    if any(type(value) is not int or value < 1 for value in (args.steps, args.requests)):
        raise ValueError('steps and requests must be positive integers.')
    acceptance = getattr(args, 'acceptance', 'strict')
    prefix, reference_versions = (official_prefix(args.reference, args.steps)
                                  if acceptance == 'official-prefix' else (None, None))
    if prefix is not None and args.initial_token != 11:
        raise ValueError('The pinned GLM official reference uses initial token 11.')
    rank = int(os.environ['RANK'])
    device = f"cuda:{os.environ['LOCAL_RANK']}"
    torch.cuda.set_device(device)
    output = args.output / f'rank-{rank}'
    output.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    report = {'artifact': str(args.artifact.resolve()), 'rank': rank, 'requests': [], 'passed': False,
              'acceptance': acceptance,
              'oracle': ('official vLLM 0.31.0 independent greedy prefix' if prefix is not None
                         else 'CPU IR semantics; not native serving-runtime acceptance')}
    if prefix is not None:
        report.update(reference_prefix=prefix, reference_versions=reference_versions,
                      reference_sha256=hashlib.sha256(args.reference.read_bytes()).hexdigest(),
                      strict_numerical_acceptance_evaluated=False)
    runtime = None
    host_group = None
    state = None
    completed_step = None
    try:
        emit('load_begin', free_bytes=torch.cuda.mem_get_info()[0])
        runtime = load(args.artifact, device=device)
        host_group = dist.new_group(backend='gloo')
        assert runtime.ir_module.metadata['num_hidden_layers'] == 45
        assert runtime.codegen['mesh_device_extent'] == 4
        assert runtime._paged_config.max_sequence_length == 4096
        if prefix is not None:
            assert runtime.ir_module.metadata['num_experts'] == 288
            assert runtime.ir_module.metadata['num_experts_per_tok'] == 8
            revision = reference_versions.get('checkpoint_revision')
            if revision is not None:
                assert runtime.ir_module.metadata['revision'] == revision, 'checkpoint revisions differ'
        report['semantic_hash'] = runtime.ir_module.semantic_hash
        source = args.artifact / runtime.codegen['source']
        report['source_sha256'] = hashlib.sha256(source.read_bytes()).hexdigest()
        emit('load_done', seconds=time.perf_counter() - started, free_bytes=torch.cuda.mem_get_info()[0])
        state = runtime.create_state()
        assert len(state.tensor_states) == 34
        ids = torch.full((1,), args.initial_token, dtype=torch.int32, device=device)
        logits, token = runtime.create_outputs()
        has_request_phase = 'prefill_input' in runtime.ir_module.metadata['composite_abi']
        emit('prepare_begin', free_bytes=torch.cuda.mem_get_info()[0])
        runtime.prepare(ids, state, logits=logits, next_token=token,
                        **({'prefill': True} if has_request_phase else {}))
        report['resources'] = runtime.resource_report
        assert report['resources']['spill_bytes'] == 0
        emit('prepare_done', resources=report['resources'], free_bytes=torch.cuda.mem_get_info()[0])
        for request in range(args.requests):
            reset_request(state)
            ids.fill_(args.initial_token)
            records = []
            for step in range(args.steps):
                reference = None
                if prefix is None:
                    path = args.reference / f'step-{step:03d}.pt'
                    wait_begin = time.monotonic()
                    while not path.is_file():
                        assert time.monotonic() - wait_begin < 1800, f'reference missing: {path}'
                        time.sleep(1)
                    reference = torch.load(path, map_location='cpu', weights_only=False)
                input_token = int(ids.item())
                expected_input = (reference['record']['input_token'] if prefix is None else
                                  args.initial_token if step == 0 else prefix[step - 1])
                assert input_token == expected_input, 'independent greedy histories diverged'
                dist.barrier(group=host_group)
                emit('step_begin', request=request, step=step, input_token=input_token)
                begin = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                begin.record()
                runtime.run_into(logits, token, ids, state,
                                 **({'prefill': step == 0} if has_request_phase else {}))
                end.record()
                torch.cuda.synchronize()
                got = logits.cpu()
                selected = int(token.item())
                completed_step = step
                torch.save({'logits': got, 'token': selected}, output / f'request-{request}-step-{step:03d}.pt')
                emit('step_finished', request=request, step=step, token=selected,
                     device_ms_with_state_carry=begin.elapsed_time(end),
                     logits_finite=bool(torch.isfinite(got).all()))
                assert torch.isfinite(got).all(), 'full logits contain non-finite values'
                if prefix is None:
                    errors = {'logits': compare(got, reference['record']['logits'], label='full logits')}
                    assert selected == reference['record']['token'], (selected, reference['record']['token'])
                else:
                    check_greedy_step(input_token, selected, prefix, args.initial_token, step)
                    errors = {}
                errors.update(check_state(state, reference, step))
                record = {'step': step, 'input_token': input_token, 'token': selected,
                          'device_ms_with_state_carry': begin.elapsed_time(end), 'max_abs_errors': errors}
                records.append(record)
                emit('step_passed', request=request, **record)
                ids.copy_(token)
                completed_step = None
                del reference
            report['requests'].append(records)
            dist.barrier(group=host_group)
        report['passed'] = True
        report['peak_allocated_bytes'] = torch.cuda.max_memory_allocated()
        report['peak_reserved_bytes'] = torch.cuda.max_memory_reserved()
    except BaseException as error:
        report.update(error_type=type(error).__name__, error=str(error), traceback=traceback.format_exc())
        emit('failed', error_type=type(error).__name__, error=str(error))
        if state is not None and completed_step is not None:
            paged = state.paged
            torch.save({
                'step': completed_step,
                'tensor_states': {layer: (conv.cpu(), rec.cpu())
                                  for layer, (conv, rec) in state.tensor_states.items()},
                'active_kv': torch.stack([
                    paged.kv_caches[int(paged.block_table[0, pos // paged.config.block_size]),
                                    :, :, pos % paged.config.block_size].cpu()
                    for pos in range(completed_step + 1)]),
                'paged_metadata': {name: getattr(paged, name).cpu()
                                   for name in ('seq_lens', 'query_start_loc', 'slot_mapping', 'block_table')},
                'dsa_packed': (state.dsa.packed_states[:, :completed_step + 1].cpu()
                               if state.dsa is not None else None),
                'dsa_seq_lens': state.dsa.seq_lens.cpu() if state.dsa is not None else None,
            }, output/'failed-state.pt')
        raise
    finally:
        report['seconds'] = time.perf_counter() - started
        (output / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
        if host_group is not None:
            dist.destroy_process_group(host_group)
        if runtime is not None:
            runtime.close()
        emit('finished', passed=report['passed'], seconds=report['seconds'])


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--artifact', type=Path, required=True)
    p.add_argument('--reference', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--steps', type=int, default=8)
    p.add_argument('--requests', type=int, default=2)
    p.add_argument('--initial-token', type=int, default=11)
    p.add_argument('--acceptance', choices=('strict', 'official-prefix'), default='strict',
                   help='strict: original CPU tensor checks; official-prefix: exact independent vLLM token prefix')
    main(p.parse_args())
