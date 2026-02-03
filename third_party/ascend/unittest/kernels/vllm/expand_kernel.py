import triton
import triton.language as tl
import triton.language.extra.cann.extension as extension


@triton.jit(do_not_specialize=["replace_from", "replace_to"])
def expand_kernel(
    output_ptr,  # [num_tokens]
    input_ptr,  # [batch_size]
    cu_num_tokens_ptr,  # [batch_size]
    replace_from,
    replace_to,
    vec_len,
    MAX_NUM_TOKENS: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    req_idx = tl.program_id(0)
    offset = req_idx * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    len_mask = offset < vec_len

    start_idx = tl.where(offset == 0, 0, tl.load(cu_num_tokens_ptr + offset - 1, len_mask))
    end_idx = tl.load(cu_num_tokens_ptr + offset, len_mask)
    num_tokens = end_idx - start_idx

    src_val = tl.load(input_ptr + offset, len_mask)
    src_val = tl.where(src_val == replace_from, replace_to, src_val)

    for i in tl.range(0, BLOCK_SIZE):
        num_tokens1 = extension.get_element(num_tokens, (i, ))
        start_idx1 = extension.get_element(start_idx, (i, ))
        src_val1 = extension.get_element(src_val, (i, ))
        offset1 = tl.arange(0, MAX_NUM_TOKENS)
        tl.store(output_ptr + start_idx1 + offset1, src_val1, mask=offset1 < num_tokens1)
