import sys
import pytest
import triton
import torch
import triton.language as tl

sys.path.append("..")
import test_common


# source: python\sglang\srt\mem_cache\memory_pool.py
@triton.jit
def copy_all_layer_kv_cache(
    data_ptrs,
    strides,
    tgt_loc_ptr,
    src_loc_ptr,
    num_locs,
    num_locs_upper: tl.constexpr,
):
    BLOCK_SIZE: tl.constexpr = 128

    bid = tl.program_id(0)
    stride = tl.load(strides + bid)

    data_ptr = tl.load(data_ptrs + bid)
    data_ptr = tl.cast(data_ptr, tl.pointer_type(tl.uint8))

    num_locs_offset = tl.arange(0, num_locs_upper)
    tgt_locs = tl.load(tgt_loc_ptr + num_locs_offset, mask=num_locs_offset < num_locs)
    src_locs = tl.load(src_loc_ptr + num_locs_offset, mask=num_locs_offset < num_locs)

    # NOTE: we cannot parallelize over the tgt_loc_ptr dim with cuda blocks
    # because this copy is an inplace operation.

    num_loop = tl.cdiv(stride, BLOCK_SIZE)
    for i in range(num_loop):
        copy_offset = tl.arange(0, BLOCK_SIZE) + i * BLOCK_SIZE
        mask = (num_locs_offset < num_locs)[:, None] and (copy_offset < stride)[None, :]
        value = tl.load(
            data_ptr + src_locs[:, None] * stride + copy_offset[None, :], mask=mask
        )
        tl.store(
            data_ptr + tgt_locs[:, None] * stride + copy_offset[None, :],
            value,
            mask=mask,
        )


def test_copy_all_layer_kv_cache(ptfile_path):
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
    #     (2,)

    input_data = data['input_data']
    device_type = 'npu'

    num_layers = input_data['num_layers']
    max_seq_len = input_data['max_seq_len']
    hidden_size = input_data['hidden_size']
    num_locs = input_data['num_locs']
    num_locs_upper = input_data['num_locs_upper']

    src_loc = input_data['src_loc_ptr'].to(device_type)
    tgt_loc = input_data['tgt_loc_ptr'].to(device_type)

    strides = input_data['strides'].to(device_type)

    kv_caches = []
    data_ptrs = []

    for i in range(num_layers):
        cache = torch.zeros((max_seq_len, hidden_size), dtype=torch.float16).to(device_type)
        for pos in range(max_seq_len):
            cache[pos, :] = (i * 100 + pos) * torch.ones(hidden_size, dtype=torch.float16).to(device_type)
        kv_caches.append(cache)
        data_ptrs.append(cache.data_ptr())

    data_ptrs_tensor = torch.tensor(data_ptrs, dtype=torch.uint64).to(device_type)

    copy_all_layer_kv_cache[data['grid']](
        data_ptrs=data_ptrs_tensor,
        strides=strides,
        tgt_loc_ptr=tgt_loc,
        src_loc_ptr=src_loc,
        num_locs=num_locs,
        num_locs_upper=num_locs_upper,
    )

    # compare the results of GPU and NPU.
    gpu_out = {'kv_caches': torch.cat(data['gpu_output']['kv_caches'], dim=0)}
    npu_out = {'kv_caches': torch.cat(kv_caches, dim=0)}
    try:
        test_common.compare_data_precision(gpu_out, npu_out, device_type='cpu')
    except ValueError as e:
        pytest.fail(f"The testcase failed")