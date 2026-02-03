import triton
import triton.language as tl
import triton.language.extra.cann.extension as extension


@triton.jit
def sample_recovered_tokens_kernel(
    output_token_ids_ptr,  # [num_tokens]
    cu_num_draft_tokens_ptr,  # [batch_size]
    draft_token_ids_ptr,  # [num_tokens]
    draft_probs_ptr,  # [num_tokens, vocab_size] or None
    target_probs_ptr,  # [num_tokens, vocab_size]
    q_ptr,  # [batch_size, vocab_size]
    vocab_size,
    PADDED_VOCAB_SIZE: tl.constexpr,
    NO_DRAFT_PROBS: tl.constexpr,
    SUB_BLOCK: tl.constexpr,
):
    req_idx = tl.program_id(0)
    start_idx = 0 if req_idx == 0 else tl.load(cu_num_draft_tokens_ptr + req_idx - 1)
    end_idx = tl.load(cu_num_draft_tokens_ptr + req_idx)
    num_draft_tokens = end_idx - start_idx

    # Early exit for out-of-range positions.
    pos = tl.program_id(1)
    if pos >= num_draft_tokens:
        return

    loop = (vocab_size + SUB_BLOCK - 1) // SUB_BLOCK
    global_recovered_id = -1
    global_max_p = -1.0
    if NO_DRAFT_PROBS:
        draft_token_id = tl.load(draft_token_ids_ptr + start_idx + pos)
        orig_prob = tl.load(target_probs_ptr + (start_idx + pos) * vocab_size + draft_token_id)
        # Temporarily zero out the probability of the draft token.
        # This is essentially the same as target_prob - draft_prob, except that
        # n-gram does not have draft_prob. We regard it as 1.
        tl.store(target_probs_ptr + (start_idx + pos) * vocab_size + draft_token_id, 0)
        for loop_i in range(loop):
            vocab_start = loop_i * SUB_BLOCK
            vocab_offset = vocab_start + tl.arange(0, SUB_BLOCK)
            prob = tl.load(target_probs_ptr + (start_idx + pos) * vocab_size + vocab_offset, mask=vocab_offset
                           < vocab_size, other=0)
            q = tl.load(q_ptr + req_idx * vocab_size + vocab_offset, mask=vocab_offset < vocab_size,
                        other=float("-inf"))
            new_p = prob / q
            recovered_id = tl.argmax(new_p, axis=-1)
            max_p = extension.get_element(new_p, (recovered_id, ))
            if max_p > global_max_p:
                global_max_p = max_p
                global_recovered_id = vocab_start + recovered_id
    else:
        for loop_i in range(loop):
            vocab_start = loop_i * SUB_BLOCK
            vocab_offset = vocab_start + tl.arange(0, SUB_BLOCK)
            draft_prob = tl.load(draft_probs_ptr + (start_idx + pos) * vocab_size + vocab_offset, mask=vocab_offset
                                 < vocab_size, other=0)
            target_prob = tl.load(target_probs_ptr + (start_idx + pos) * vocab_size + vocab_offset, mask=vocab_offset
                                  < vocab_size, other=0)
            prob = tl.maximum(target_prob - draft_prob, 0)
            # NOTE(woosuk): We don't need `prob = prob / tl.sum(prob)` here because
            # `tl.argmax` will select the maximum value.

            q = tl.load(q_ptr + req_idx * vocab_size + vocab_offset, mask=vocab_offset < vocab_size,
                        other=float("-inf"))
            new_p = prob / q
            recovered_id = tl.argmax(new_p, axis=-1)
            max_p = extension.get_element(new_p, (recovered_id, ))
            if max_p > global_max_p:
                global_max_p = max_p
                global_recovered_id = vocab_start + recovered_id

    tl.store(output_token_ids_ptr + start_idx + pos, global_recovered_id)

    if NO_DRAFT_PROBS:
        # Restore the original probability.
        tl.store(target_probs_ptr + (start_idx + pos) * vocab_size + draft_token_id, orig_prob)
