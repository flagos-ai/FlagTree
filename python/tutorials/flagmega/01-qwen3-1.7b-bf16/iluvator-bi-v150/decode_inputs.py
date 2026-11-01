# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Prepare vLLM's one-request decode buffers with one upload and one kernel.

The scheduler's CPU position/page table remain authoritative. Under async
scheduling the previous sampled token stays on the GPU. Every native buffer
written by _prepare_inputs for this ABI retains its meaning, so metadata,
sampling and the next native prefill still use the same runner state.
"""
import torch
import triton
import triton.language as tl


@triton.jit
def prepare_decode_inputs(Stage, Prev, Input, Position, QueryStart, Seq, Computed,
                          ReqIndices, QueryPos, Scheduled, Accepted, Discard,
                          BlockTable, Slots, LogitsIndices, WIDTH: tl.constexpr,
                          PAGE: tl.constexpr, MAX_TOKENS: tl.constexpr,
                          TABLE_BLOCK: tl.constexpr, BLOCK: tl.constexpr):
    computed = tl.load(Stage)
    token = tl.load(Stage + 1)
    use_prev = tl.load(Stage + 2)
    previous = tl.load(Prev, mask=use_prev != 0, other=0)
    tl.store(Input, tl.where(use_prev != 0, previous, token))
    tl.store(Position, computed)
    tl.store(QueryStart, 0)
    tl.store(QueryStart + 1, 1)
    tl.store(Seq, computed + 1)
    tl.store(Computed, computed)
    tl.store(ReqIndices, 0)
    tl.store(QueryPos, 0)
    tl.store(Scheduled, 1)
    tl.store(Accepted, 1)
    tl.store(Discard, tl.load(Stage + 3))
    tl.store(LogitsIndices, 0)
    cols = tl.arange(0, TABLE_BLOCK)
    pages = tl.load(Stage + 4 + cols, cols < WIDTH, other=0)
    tl.store(BlockTable + cols, pages, cols < WIDTH)
    page = tl.load(Stage + 4 + computed // PAGE)
    slot = page * PAGE + computed % PAGE
    for start in range(tl.cdiv(MAX_TOKENS, BLOCK)):
        offsets = start * BLOCK + tl.arange(0, BLOCK)
        tl.store(Slots + offsets, tl.where(offsets == 0, slot, -1), offsets < MAX_TOKENS)


class FusedDecodeInputs:
    def __init__(self, runner):
        self.runner = runner
        self.table = runner.input_batch.block_table[0]
        self.width = self.table.block_table.gpu.shape[1]
        # These are the integration's existing single-request, text-only ABI
        # requirements, rather than assumptions inferred from one prompt.
        if (runner.max_num_reqs != 1 or runner.speculative_config is not None
                or runner.uses_mrope or runner.uses_xdrope_dim != 0
                or runner.enable_prompt_embeds or runner.dcp_world_size != 1
                or self.table.pcp_world_size != 1 or self.table.use_hybrid_blocks
                or runner.num_accepted_tokens_event is not None
                or runner.mamba_prev_last_scheduled_idx is not None):
            raise ValueError('Fused decode inputs require one text request and ordinary paged attention')
        self.stage_cpu = torch.empty(4 + self.width, dtype=torch.int64,
                                     pin_memory=runner.pin_memory)
        self.stage_np = self.stage_cpu.numpy()
        self.stage_gpu = torch.empty_like(self.stage_cpu, device=runner.device)
        self.logits_indices = torch.zeros(1, dtype=torch.int64, device=runner.device)
        self.calls = 0

    def supports(self, scheduler):
        batch = self.runner.input_batch
        return (scheduler.total_num_scheduled_tokens == 1 and batch.num_reqs == 1
                and batch.num_computed_tokens_cpu[0] >= batch.num_prompt_tokens[0])

    def prepare(self, scheduler, num_scheduled_tokens):
        r, batch = self.runner, self.runner.input_batch
        if not self.supports(scheduler) or tuple(num_scheduled_tokens) != (1,):
            raise ValueError('Expected one scheduled decode token')
        pos = int(batch.num_computed_tokens_cpu[0])
        if not 0 <= pos < self.width * self.table.block_size:
            raise ValueError('Decode position exceeds the native page table')
        prev = batch.prev_req_id_to_index.get(batch.req_ids[0], -1)
        use_prev = batch.prev_sampled_token_ids is not None and prev == 0
        token = int(batch.token_ids_cpu[0, pos])
        discard = pos + 1 < r.requests[batch.req_ids[0]].num_tokens
        self.stage_np[:4] = (pos, token, int(use_prev), int(discard))
        self.stage_np[4:] = self.table.block_table.np[0]
        # execute_model's synchronize_input_prep protects this pinned staging
        # allocation just as it protects the original native CPU buffers.
        r.input_ids.np[0] = token
        r.query_start_loc.np[:] = (0, 1)
        r.optimistic_seq_lens_cpu[0] = pos + 1
        r.prev_positions.np[0] = prev
        r.discard_request_mask.np[0] = discard
        r.num_accepted_tokens.np[0] = 1
        r.req_indices.np[0] = 0
        r.query_pos.np[0] = 0
        r.num_scheduled_tokens.np[0] = 1
        self.stage_gpu.copy_(self.stage_cpu, non_blocking=True)
        prepare_decode_inputs[(1,)](
            self.stage_gpu, batch.prev_sampled_token_ids if use_prev else r.input_ids.gpu,
            r.input_ids.gpu, r.positions, r.query_start_loc.gpu, r.seq_lens, r.num_computed_tokens,
            r.req_indices.gpu, r.query_pos.gpu, r.num_scheduled_tokens.gpu, r.num_accepted_tokens.gpu,
            r.discard_request_mask.gpu, self.table.block_table.gpu, self.table.slot_mapping.gpu,
            self.logits_indices, WIDTH=self.width, PAGE=self.table.block_size,
            MAX_TOKENS=r.max_num_tokens, TABLE_BLOCK=triton.next_power_of_2(self.width),
            BLOCK=256, num_warps=4)
        self.calls += 1
        return self.logits_indices, None
