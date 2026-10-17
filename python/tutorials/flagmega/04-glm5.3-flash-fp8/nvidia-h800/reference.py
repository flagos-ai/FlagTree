# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Independent CPU-IR greedy reference for the complete original checkpoint."""
import argparse
import hashlib
import json
import os
import time
from pathlib import Path

import torch
from triton.flagmega.evaluator import CheckpointWeightResolver, TorchEvaluator
from triton.flagmega.evaluator import torch_backend
from triton.flagmega.importer import DirectoryCheckpoint
from triton.flagmega.importer.glm5_moe import Glm5MoeModelImporter
from triton.flagmega.ir import TensorType, load_module
from triton.flagmega.ir.ops.math import _block_scaled
from triton.flagmega.ir.ops.nn import _sparse_experts
from triton.flagmega.ir.ops.nn._dsa_indexer_state import create_dsa_indexer_state
from triton.flagmega.ir.ops.nn._paged_attention_state import create_paged_attention_state
from triton.flagmega.runtime.module import _torch_dtype

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--input', type=Path, required=True)
p.add_argument('--checkpoint', type=Path, required=True)
p.add_argument('--output', type=Path, required=True)
p.add_argument('--steps', type=int, default=8)
p.add_argument('--initial-token', type=int, default=11)
p.add_argument('--threads', type=int, default=16)
a = p.parse_args()
torch.set_num_threads(a.threads)
a.output.mkdir(parents=True, exist_ok=False)


def save(value, path):
    temporary = path.with_suffix(path.suffix + '.tmp')
    torch.save(value, temporary)
    os.replace(temporary, path)


def rss_bytes():
    return next(int(line.split()[1]) * 1024 for line in Path('/proc/self/status').read_text().splitlines()
                if line.startswith('VmRSS:'))


t = time.perf_counter()
model = load_module(a.input)
checkpoint = DirectoryCheckpoint(a.checkpoint)
importer = Glm5MoeModelImporter(checkpoint)
assert model.metadata['num_hidden_layers'] == 45
values = {}
entry = model.function_map[model.entry]
for node_id in entry.parameters:
    node = model.node_map[node_id]
    name = node.attrs['name']
    if isinstance(node.type, TensorType):
        shape = tuple(dim.fixed_value for dim in node.type.shape)
        values[name] = torch.zeros(shape, dtype=_torch_dtype(torch, node.type.dtype))
    elif name == 'paged_attention_state':
        values[name] = create_paged_attention_state(importer.paged_config)
    elif name == 'dsa_indexer_state':
        values[name] = create_dsa_indexer_state(importer.dsa_config)
    else:
        raise ValueError(f'Unsupported reference parameter {name}: {node.type}')
values['input_ids'].fill_(a.initial_token)
evaluator = TorchEvaluator(CheckpointWeightResolver(checkpoint))
evaluator_sources = {
    Path(source.__file__).name: hashlib.sha256(Path(source.__file__).read_bytes()).hexdigest()
    for source in (torch_backend, _block_scaled, _sparse_experts)
}
print(json.dumps({'loaded_seconds': time.perf_counter() - t,
                  'semantic_hash': model.semantic_hash, 'layers': 45, 'steps': a.steps,
                  'threads': torch.get_num_threads(), 'torch_version': torch.__version__,
                  'initial_token': a.initial_token, 'oracle': 'CPU IR, not native vLLM',
                  'evaluator_source_sha256': evaluator_sources}), flush=True)
records = []
for step in range(a.steps):
    begin = time.perf_counter()
    current = int(values['input_ids'].item())
    with torch.no_grad():
        outputs = evaluator.run(model, values)
    logits, token = outputs[:2]
    assert torch.isfinite(logits).all(), 'reference produced non-finite logits'
    record = {'step': step, 'input_token': current, 'token': int(token.item()),
              'logits': logits.clone(), 'seconds': time.perf_counter() - begin,
              'rss_bytes': rss_bytes(), 'threads': torch.get_num_threads()}
    records.append(record)
    for i, layer in enumerate(importer.kda_layers):
        values[f'kda_conv_state_{layer}'], values[f'kda_recurrent_state_{layer}'] = outputs[2+2*i:4+2*i]
    assert values['paged_attention_state'].sequence_length == step + 1
    paged = values['paged_attention_state']
    active_kv = torch.stack([
        paged.kv_caches[int(paged.block_table[0, position // paged.config.block_size]),
                        :, :, position % paged.config.block_size].clone()
        for position in range(step + 1)
    ])
    save({
        'step': step, 'record': record,
        'evaluator_source_sha256': evaluator_sources,
        'tensor_states': {layer: (values[f'kda_conv_state_{layer}'],
                                   values[f'kda_recurrent_state_{layer}'])
                          for layer in importer.kda_layers},
        'active_kv': active_kv,
        'paged_metadata': {name: getattr(paged, name).clone()
                           for name in ('seq_lens', 'query_start_loc', 'slot_mapping', 'block_table')},
        'dsa_packed': values['dsa_indexer_state'].packed_states[:, :step + 1].clone()
                      if importer.dsa_config else None,
        'dsa_seq_lens': values['dsa_indexer_state'].seq_lens.clone()
                        if importer.dsa_config else None,
    }, a.output / f'step-{step:03d}.pt')
    values['input_ids'].copy_(token.reshape_as(values['input_ids']))
    save({'records': records}, a.output/'tokens.pt')
    print(json.dumps({k:v for k,v in record.items() if k != 'logits'}), flush=True)
save({
    'records': records,
    'evaluator_source_sha256': evaluator_sources,
    'tensor_states': {layer: (values[f'kda_conv_state_{layer}'], values[f'kda_recurrent_state_{layer}'])
                      for layer in importer.kda_layers},
    'dsa_packed': values['dsa_indexer_state'].packed_states[:, :a.steps].clone()
                   if importer.dsa_config else None,
}, a.output/'reference.pt')
(a.output/'summary.json').write_text(json.dumps({
    'layers': 45, 'steps': a.steps, 'initial_token': a.initial_token,
    'tokens': [r['token'] for r in records], 'elapsed_seconds': time.perf_counter() - t,
    'threads': torch.get_num_threads(), 'torch_version': torch.__version__,
    'evaluator_source_sha256': evaluator_sources,
    'oracle': 'CPU IR semantics; not native serving-runtime acceptance',
}, indent=2)+'\n')
