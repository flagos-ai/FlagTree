# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: MIT
"""The fused serving preparation preserves native scalar/page/token buffers."""
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch


TUTORIAL = (Path(__file__).resolve().parents[3] / 'tutorials' / 'flagmega' /
            '01-qwen3-1.7b-bf16' / 'iluvator-bi-v150')


def buffer(shape, dtype):
    cpu = torch.zeros(shape, dtype=dtype, pin_memory=True)
    return SimpleNamespace(cpu=cpu, np=cpu.numpy(),
                           gpu=torch.zeros(shape, dtype=dtype, device='cuda'))


@pytest.mark.skipif(not torch.cuda.is_available(), reason='Requires a GPU')
@pytest.mark.parametrize('position', [31, 255, 256, 511, 512, 3071])
@pytest.mark.parametrize('previous', [False, True])
def test_fused_decode_inputs_preserve_native_contract(position, previous):
    spec = importlib.util.spec_from_file_location('decode_inputs_test', TUTORIAL / 'decode_inputs.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    pages = np.array([9, 3, 14, 7, 2, 11, 4, 15, 6, 1, 13, 5], dtype=np.int32)
    table = SimpleNamespace(block_table=buffer((1, 12), torch.int32),
                            slot_mapping=buffer((3072,), torch.int64),
                            block_size=256, pcp_world_size=1, use_hybrid_blocks=False)
    table.block_table.np[0] = pages
    tokens = np.full((1, 3072), -1, dtype=np.int32)
    tokens[0, position] = 42
    batch = SimpleNamespace(block_table=[table], num_reqs=1,
                            num_computed_tokens_cpu=np.array([position]),
                            num_prompt_tokens=np.array([31]), req_ids=['request'],
                            prev_req_id_to_index={'request': 0} if previous else {},
                            prev_sampled_token_ids=(torch.tensor([[73]], dtype=torch.int32,
                                                                device='cuda') if previous else None),
                            token_ids_cpu=tokens)
    runner = SimpleNamespace(input_batch=batch, max_num_reqs=1, max_num_tokens=3072,
                             speculative_config=None, uses_mrope=False, uses_xdrope_dim=0,
                             enable_prompt_embeds=False, dcp_world_size=1,
                             num_accepted_tokens_event=None, mamba_prev_last_scheduled_idx=None,
                             pin_memory=True, device=torch.device('cuda'),
                             requests={'request': SimpleNamespace(num_tokens=position + 1)})
    for name, shape, dtype in [('input_ids', (3072,), torch.int32),
                               ('query_start_loc', (2,), torch.int32),
                               ('prev_positions', (1,), torch.int64),
                               ('discard_request_mask', (1,), torch.bool),
                               ('num_accepted_tokens', (1,), torch.int32),
                               ('req_indices', (3072,), torch.int64),
                               ('query_pos', (3072,), torch.int64),
                               ('num_scheduled_tokens', (1,), torch.int32)]:
        setattr(runner, name, buffer(shape, dtype))
    runner.positions = torch.zeros(3072, dtype=torch.int64, device='cuda')
    runner.seq_lens = torch.zeros(1, dtype=torch.int32, device='cuda')
    runner.num_computed_tokens = torch.zeros(1, dtype=torch.int32, device='cuda')
    runner.optimistic_seq_lens_cpu = torch.zeros(1, dtype=torch.int32, pin_memory=True)
    prepare = module.FusedDecodeInputs(runner)
    scheduler = SimpleNamespace(total_num_scheduled_tokens=1)
    indices, spec_metadata = prepare.prepare(scheduler, np.array([1], dtype=np.int32))
    torch.cuda.synchronize()
    assert spec_metadata is None and indices.tolist() == [0]
    assert runner.input_ids.gpu[0].item() == (73 if previous else 42)
    assert runner.positions[0].item() == position
    assert runner.seq_lens.tolist() == [position + 1]
    assert runner.num_computed_tokens.tolist() == [position]
    assert runner.query_start_loc.gpu.tolist() == [0, 1]
    assert runner.optimistic_seq_lens_cpu.tolist() == [position + 1]
    assert runner.prev_positions.np.tolist() == ([0] if previous else [-1])
    assert runner.discard_request_mask.gpu.tolist() == [False]
    assert runner.num_accepted_tokens.gpu.tolist() == [1]
    assert runner.req_indices.gpu[0].item() == runner.query_pos.gpu[0].item() == 0
    assert runner.num_scheduled_tokens.gpu.tolist() == [1]
    assert table.block_table.gpu.cpu().numpy().tolist() == [pages.tolist()]
    assert table.slot_mapping.gpu[0].item() == pages[position // 256] * 256 + position % 256
    assert torch.all(table.slot_mapping.gpu[1:] == -1).item()
    assert prepare.calls == 1
    # A new prefill must go through native preparation, including one-token
    # prompts; decode selection must depend on phase as well as token count.
    batch.num_computed_tokens_cpu[0] = 0
    assert not prepare.supports(scheduler)
