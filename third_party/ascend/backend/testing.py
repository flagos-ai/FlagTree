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

import builtins
import multiprocessing
import os
from datetime import datetime, timezone

import triton.runtime as runtime


def do_bench_npu(funcs, warmup=5, active=30, clear_l2_cache=False, prof_dir=None, keep_res=False):
    import torch
    import torch_npu

    if not isinstance(funcs, list):
        funcs = [funcs]

    # warmup kernel
    for fn in funcs:
        fn()
        torch.npu.synchronize()

    experimental_config = torch_npu.profiler._ExperimentalConfig(
        aic_metrics=torch_npu.profiler.AiCMetrics.PipeUtilization,
        profiler_level=torch_npu.profiler.ProfilerLevel.Level1,
        l2_cache=False,
        data_simplification=False,
    )

    if prof_dir is not None:
        torch_path = prof_dir
    else:
        process = multiprocessing.current_process()
        pid = process.pid
        process_name = process.name
        timestamp = datetime.now(tz=timezone.utc).strftime("%Y%m%d_%H%M%S")
        base_path = os.path.join(runtime.cache.get_home_dir(), ".triton", "profile_results")
        torch_path = os.path.join(base_path, f"prof_{timestamp}_{process_name}-{pid}")

    if clear_l2_cache:
        buffer = runtime.driver.active.get_empty_cache_for_benchmark()
        buffer.zero_()
        torch.npu.synchronize()  # shake out of any npu error

    total = warmup + active
    with torch_npu.profiler.profile(
            activities=[torch_npu.profiler.ProfilerActivity.NPU],
            on_trace_ready=torch_npu.profiler.tensorboard_trace_handler(torch_path),
            record_shapes=False,
            profile_memory=False,
            with_stack=False,
            with_flops=False,
            with_modules=False,
            experimental_config=experimental_config,
    ) as prof:
        for fn in funcs:
            for _ in builtins.range(total):
                if clear_l2_cache:
                    buffer.zero_()
                fn()
                torch.npu.synchronize()
    if clear_l2_cache:
        del buffer

    time_cost = _collect_prof_result(torch_path, funcs, warmup, active)
    _rm_dic(keep_res, torch_path)
    return time_cost


def _rm_dic(keep_res, torch_path):
    if keep_res:
        return
    import shutil

    if os.path.exists(torch_path):
        shutil.rmtree(torch_path)


def _collect_prof_result(base_dir: str, funcs, num_warmup: int, num_active: int, key: str = None):
    """
    Collect kernel performance from kernel_details.csv, returned in millisecond.
    The first `num_warmup` rows of each function are warmup data and will be ignored, the next `num_active` rows will be averaged.

    :param base_dir: the profiler path
    :type base_dir: str
    :param funcs: a list of Callable being profiled
    :type funcs: List[Callable]
    :param num_warmup: warmup count in kernel_details.csv of each fn
    :type num_warmup: int
    :param num_active: active count in kernel_details.csv of each fn
    :type num_active: int
    :param key: filter key for kernel name
    :type key: str
    """

    import numpy as np
    import pandas as pd

    kernel_details_file = None
    for root, _, files in os.walk(base_dir):
        for file in files:
            if file == "kernel_details.csv":
                kernel_details_file = os.path.join(root, file)
                break
    num_funcs = len(funcs)
    if kernel_details_file is None:
        if num_funcs == 1:
            return float("inf")
        else:
            return [float("inf")] * num_funcs

    df = pd.read_csv(kernel_details_file)
    # filter out l2 cache clearing operation
    filter_cond = ~df["Type"].str.contains(r"^ZerosLike$", case=False, na=False)
    filter_df = df[filter_cond]
    if key is not None:
        key_rows = filter_df[filter_df["Name"].str.contains(key, na=False)]
    else:
        key_rows = filter_df
    time_cost = [0] * num_funcs
    for func_idx in np.arange(0, num_funcs):
        for active_index in np.arange(0, num_active):
            row_index = func_idx * (num_warmup + num_active) + num_warmup + active_index
            time_cost[func_idx] += key_rows.iloc[row_index]["Duration(us)"]
    time_cost = [x / num_active / 1e3 for x in time_cost]

    if num_funcs == 1:
        return time_cost[0]
    else:
        return time_cost
