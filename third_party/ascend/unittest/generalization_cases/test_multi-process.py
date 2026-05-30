# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
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
"""
testing the compilation of multi-process operators
=============
"""
import multiprocessing
import os
import shutil

import psutil
import torch
import torch_npu
import triton
import triton.backends.ascend.runtime
import triton.language as tl


@triton.jit
def add_kernel(
    x_ptr,
    y_ptr,
    output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    output = x + y
    tl.store(output_ptr + offsets, output, mask=mask)


def add_torch(x, y):
    return x + y


def add_triton(x, y):
    output = torch.empty_like(x)
    n_elements = output.numel()
    BLOCK_SIZE = 1024
    add_kernel[
        triton.cdiv(n_elements, BLOCK_SIZE),
    ](
        x,
        y,
        output,
        n_elements,
        BLOCK_SIZE,
    )
    return output


def case_add(size: int):
    x = torch.rand(size, device="npu")
    y = torch.rand(size, device="npu")
    try:
        output_torch = add_torch(x, y)
        output_triton = add_triton(x, y)
        assert torch.allclose(output_triton, output_torch)
    except Exception:
        import traceback
        traceback.print_exc()
        return f"{multiprocessing.current_process().name}:failed"
    return f"{multiprocessing.current_process().name}:success"


def test_multi_process():
    # init env
    TA_CACHE_DEFAULT_PATH = os.environ.get("TRITON_CACHE_DIR", "~/.triton/cache")
    TA_CACHE_PATH = os.path.join(TA_CACHE_DEFAULT_PATH, "multi_process")
    if os.path.exists(TA_CACHE_PATH):
        shutil.rmtree(TA_CACHE_PATH)
    os.environ["TRITON_CACHE_DIR"] = TA_CACHE_PATH
    # multi process
    process_num = min(len(psutil.Process().cpu_affinity()), 64)

    multiprocessing.set_start_method("spawn", force=True)
    results = []
    with multiprocessing.Pool(processes=process_num) as pool:
        results = pool.map(case_add, [(98432, )] * process_num)
    success_num = 0
    for result in results:
        if result and "success" in result:
            success_num += 1
    assert success_num == len(results), f"multi-process failed,failed {len(results) - success_num}"
