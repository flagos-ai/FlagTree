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


# source: sgl-kernel/benchmark/bench_moe_align_block_size.py
@triton.jit
def moe_align_block_size_stage1(
    topk_ids_ptr,
    tokens_cnts_ptr,
    num_experts: tl.constexpr,
    numel: tl.constexpr,
    tokens_per_thread: tl.constexpr,
):
    pid = tl.program_id(0)
    start_idx = pid * tokens_per_thread
    off_c = (pid + 1) * num_experts

    for i in range(tokens_per_thread):
        if start_idx + i < numel:
            idx = tl.load(topk_ids_ptr + start_idx + i)
            token_cnt = tl.load(tokens_cnts_ptr + off_c + idx)
            tl.store(tokens_cnts_ptr + off_c + idx, token_cnt + 1)


def test_context_fwd_kernel(ptfile_path):
    try:
        data = torch.load(ptfile_path, map_location=torch.device('cpu'), weights_only=False)
    except Exception as e:
        pytest.fail(f"load file {ptfile_path} failed: {str(e)}")

    # ptfile format:
    # [input_data] (dict):
    # [gpu_output] (dict):
    # [grid] :
    input_data = test_common.convert_tensor_with_device_type(data["input_data"], device_type='npu')

    moe_align_block_size_stage1[data["grid"]](**input_data)

    # compare the results of GPU and NPU
    try:
        test_common.compare_data_precision(data["gpu_output"], input_data, device_type='cpu')
    except ValueError as e:
        pytest.fail(f"The testcase failed")
