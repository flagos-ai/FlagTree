import sys
import pytest
import triton
import torch
import triton.language as tl

sys.path.append("..")
import test_common

#source: python\sglang\srt\speculative\eagle_utils.py


@triton.jit
def assign_draft_cache_locs(
    req_pool_indices,
    req_to_token,
    seq_lens,
    extend_lens,
    num_new_pages_per_topk,
    out_cache_loc,
    pool_len: tl.constexpr,
    topk: tl.constexpr,
    speculative_num_steps: tl.constexpr,
    page_size: tl.constexpr,
    bs_upper: tl.constexpr,
    iter_upper: tl.constexpr,
):
    BLOCK_SIZE: tl.constexpr = 128
    pid = tl.program_id(axis=0)

    if page_size == 1 or topk == 1:
        copy_len = topk * speculative_num_steps
        out_cache_ptr = out_cache_loc + pid * topk * speculative_num_steps
    else:
        bs_offset = tl.arange(0, bs_upper)
        copy_len = tl.load(extend_lens + pid)
        cum_copy_len = tl.sum(tl.load(extend_lens + bs_offset, mask=bs_offset < pid))
        out_cache_ptr = out_cache_loc + cum_copy_len

    # Part 1: Copy from out_cache_loc to req_to_token
    kv_start = tl.load(seq_lens + pid)
    token_pool = req_to_token + tl.load(req_pool_indices + pid) * pool_len
    num_loop = tl.cdiv(copy_len, BLOCK_SIZE)
    for i in range(num_loop):
        copy_offset = tl.arange(0, BLOCK_SIZE) + i * BLOCK_SIZE
        mask = copy_offset < copy_len
        data = tl.load(out_cache_ptr + copy_offset, mask=mask)
        tl.store(token_pool + kv_start + copy_offset, data, mask=mask)

    if page_size == 1 or topk == 1:
        return

    # Part 2: Copy the indices for the last partial page
    prefix_len = tl.load(seq_lens + pid)
    last_page_len = prefix_len % page_size
    offsets = tl.arange(0, page_size)
    mask = offsets < last_page_len
    num_new_pages_per_topk_ = tl.load(num_new_pages_per_topk + pid)
    prefix_base = token_pool + prefix_len - last_page_len

    for topk_id in range(topk):
        value = tl.load(prefix_base + offsets, mask=mask)
        tl.store(
            prefix_base + topk_id * num_new_pages_per_topk_ * page_size + offsets,
            value,
            mask=mask,
        )

    # Part 3: Remove the padding in out_cache_loc
    iter_offest = tl.arange(0, iter_upper)
    for topk_id in range(topk):
        indices = tl.load(
            prefix_base
            + topk_id * num_new_pages_per_topk_ * page_size
            + last_page_len
            + iter_offest,
            mask=iter_offest < speculative_num_steps,
        )
        tl.store(
            out_cache_loc
            + pid * topk * speculative_num_steps
            + topk_id * speculative_num_steps
            + iter_offest,
            indices,
            mask=iter_offest < speculative_num_steps,
        )


def test_assign_draft_cache_locs(ptfile_path):
    try:
        data = torch.load(ptfile_path, map_location=torch.device('cpu'), weights_only=False)
    except Exception as e:
        pytest.fail(f"load file {ptfile_path} failed: {str(e)}")

    # ptfile format:
    # [input_data] (dict):
    #     key : value
    # [gpu_output] (dict):
    #     key : value
    # [grid] :
    #     (1,)
    input_data = test_common.convert_tensor_with_device_type(data["input_data"], device_type='npu')

    assign_draft_cache_locs[data["grid"]](**input_data)

    # compare the results of GPU and NPU.
    try:
        test_common.compare_data_precision(data["gpu_output"], input_data, device_type='cpu')
    except ValueError as e:
        pytest.fail(f"The testcase failed")