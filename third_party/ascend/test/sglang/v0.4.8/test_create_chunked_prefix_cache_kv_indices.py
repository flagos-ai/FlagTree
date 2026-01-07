# Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.

import sys
import pytest
import triton
import torch
import triton.language as tl

sys.path.append("..")
import test_common


# source: python\sglang\srt\model_executor\forward_batch_info.py
@triton.jit
def create_chunked_prefix_cache_kv_indices(
    req_to_token_ptr,  # (max_batch, max_context_len,)
    req_pool_indices_ptr,  # (batch_size,)
    chunk_start_idx_ptr,  # (batch_size,) 
    chunk_seq_lens_ptr,  # (batch_size,)
    chunk_cu_seq_lens_ptr,  # (batch_size + 1,)
    chunk_kv_indices_ptr,  # (num_chunk_tokens,)
    req_to_token_ptr_stride: tl.constexpr,
):
    BLOCK_SIZE: tl.constexpr = 512
    pid = tl.program_id(axis=0)

    # find the req pool idx, this is for batch to token
    req_pool_index = tl.load(req_pool_indices_ptr + pid)
    chunk_kv_indices_offset = tl.load(chunk_cu_seq_lens_ptr + pid)

    # get the token position of current chunk
    chunk_start_pos = tl.load(chunk_start_idx_ptr + pid).to(tl.int32)
    chunk_seq_len = tl.load(chunk_seq_lens_ptr + pid).to(tl.int32)

    num_loop = tl.cdiv(chunk_seq_len, BLOCK_SIZE)
    for i in range(num_loop):
        offset = tl.arrange(0, BLOCK_SIZE) + i * BLOCK_SIZE
        mask = offset < chunk_seq_len
        data = tl.load(
            req_to_token_ptr
            + req_pool_index * req_to_token_ptr_stride
            + chunk_start_pos
            + offset,
            mask=mask,
        )
        tl.store(
            chunk_kv_indices_ptr + chunk_kv_indices_offset + offset, data, mask=mask
        )



def test_context_fwd_kernel(ptfile_path):
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
    #     (3,)
    input_data = test_common.convert_tensor_with_device_type(data["input_data"], device_type='npu')

    create_chunked_prefix_cache_kv_indices[data["grid"]](**input_data)

    # compare the results of GPU and NPU.
    try:
        test_common.compare_data_precision(data["gpu_output"], input_data, device_type='cpu')
    except ValueError as e:
        pytest.fail(f"The testcase failed")
