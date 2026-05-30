import triton
import triton.language as tl
import triton.language.extra.cann.extension as extension


@triton.jit(do_not_specialize=["max_spec_len"])
def bonus_renew_1(
    bonus_token_ids_ptr,
    position,
    output_token_ids_ptr,
):
    bonus_token_id = tl.load(bonus_token_ids_ptr + position)
    tl.store(output_token_ids_ptr + position * 2 + 1, bonus_token_id)


@triton.jit(do_not_specialize=["max_spec_len"])
def rejection_greedy_sample_spec_len_1_triton(
    output_token_ids_ptr,  # [batch_size, 2]
    draft_token_ids_ptr,  # [num_tokens]
    target_argmax_ptr,  # [num_tokens]
    bonus_token_ids_ptr,
    vec_len,
    BLOCK_SIZE: tl.constexpr,
):
    block_idx = tl.program_id(0)
    offset = block_idx * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offset < vec_len

    draft_token_id = tl.load(draft_token_ids_ptr + offset, mask)
    target_argmax_id = tl.load(target_argmax_ptr + offset, mask)
    tl.store(output_token_ids_ptr + offset * 2, target_argmax_id, mask)

    for pos in tl.range(0, BLOCK_SIZE):
        draft_token_id1 = extension.get_element(draft_token_id, (pos, ))
        target_argmax1 = extension.get_element(target_argmax_id, (pos, ))
        position = block_idx * BLOCK_SIZE + pos
        if draft_token_id1 == target_argmax1:
            bonus_renew_1(
                bonus_token_ids_ptr,
                position,
                output_token_ids_ptr,
            )
