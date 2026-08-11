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
from typing import Optional
import fnmatch
import torch
import torch_npu

import triton.runtime as runtime
from triton.knobs import cache

try:
    from triton._C.clear_l2 import do_bench_clear
except ImportError:

    def do_bench_clear(*args, **kwargs):
        raise RuntimeError("triton._C.clear_l2 is not available; ClearL2Cache was skipped at build time. "
                           "Rebuild with ASC support to use clear_l2_cache=True.")


class ProfilerResultMismatchError(RuntimeError):

    def __init__(self, target_kernel_name: str, expected_rows: int, actual_rows: int):
        self.target_kernel_name = target_kernel_name
        self.expected_rows = expected_rows
        self.actual_rows = actual_rows
        super().__init__(
            "Profiler rows filtered by target kernel name do not match the expected count. "
            f"target_kernel_name={target_kernel_name!r}, expected_rows={expected_rows}, actual_rows={actual_rows}")


def do_bench_npu_profiler(
    funcs,
    warmup=5,
    active=30,
    clear_l2_cache=False,
    prof_dir=None,
    keep_res=False,
    target_kernel_name: Optional[str] = None,
):
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
        base_path = cache.get_triton_dir("profile_results")
        torch_path = os.path.join(base_path, f"prof_{timestamp}_{process_name}-{pid}")

    if clear_l2_cache:
        buffer = runtime.driver.active.get_empty_cache_for_benchmark()
        buffer = buffer.float()  # to avoid type cast
        buffer.sum()
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
                    buffer.sum()  # use buffer read to clear l2 cache
                    torch.npu.synchronize()
                fn()
                torch.npu.synchronize()
    if clear_l2_cache:
        del buffer

    try:
        return _collect_prof_result(
            torch_path,
            funcs,
            warmup,
            active,
            target_kernel_name=target_kernel_name,
            clear_l2_cache=clear_l2_cache,
        )
    finally:
        _rm_dic(keep_res, torch_path)


def _rm_dic(keep_res, torch_path):
    if keep_res:
        return
    import shutil

    if os.path.exists(torch_path):
        shutil.rmtree(torch_path)


def _collect_prof_result(
    base_dir: str,
    funcs,
    num_warmup: int,
    num_active: int,
    target_kernel_name: Optional[str] = None,
    clear_l2_cache: bool = False,
):
    """
    Collect kernel performance from task_time*.csv or kernel_details.csv, returned in millisecond.
    Uses task_time*.csv by default. If target_kernel_name is provided, scans for kernel_details.csv as a fallback for accuracy.
    The first `num_warmup` rows of each function are warmup data and will be ignored, the next `num_active` rows will be averaged.

    :param base_dir: the profiler path
    :type base_dir: str
    :param funcs: a list of Callable being profiled
    :type funcs: List[Callable]
    :param num_warmup: warmup count in task_time*.csv or kernel_details.csv of each fn
    :type num_warmup: int
    :param num_active: active count in task_time*.csv or kernel_details.csv of each fn
    :type num_active: int
    :param target_kernel_name: target triton kernel name reported by profiler
    :type target_kernel_name: Optional[str]
    """

    import numpy as np
    import pandas as pd
    use_task_time = (target_kernel_name is None)
    kernel_details_file = None
    for root, _, files in os.walk(base_dir):
        for file in files:
            if use_task_time and fnmatch.fnmatch(file, "task_time*.csv"):
                kernel_details_file = os.path.join(root, file)
                break
            elif not use_task_time and file == "kernel_details.csv":
                kernel_details_file = os.path.join(root, file)
                break
    num_funcs = len(funcs)
    if kernel_details_file is None:
        if num_funcs == 1:
            return float("inf")
        else:
            return [float("inf")] * num_funcs

    df = pd.read_csv(kernel_details_file)
    if use_task_time:
        # The first and last lines of the task_time*.csv file are PROFILING_DISABLE, which should be deleted.
        df = df[1:-1]
        col_time = "task_time(us)"
        filter_cond = (not clear_l2_cache) | ~df["kernel_name"].str.contains(r"^ReduceSum", case=False, na=False)
    else:
        col_time = "Duration(us)"
        filter_cond = (not clear_l2_cache) | ~df["Type"].str.contains(r"^ReduceSum$", case=False, na=False)
    # filter out l2 cache clearing operation
    filter_df = df[filter_cond]
    if target_kernel_name is not None:
        filter_df = filter_df[filter_df["Name"] == target_kernel_name]

    expected_rows = num_funcs * (num_warmup + num_active)
    actual_rows = len(filter_df)
    if target_kernel_name is not None and actual_rows != expected_rows:
        raise ProfilerResultMismatchError(target_kernel_name, expected_rows, actual_rows)

    mul = 1
    if num_funcs == 1:
        if actual_rows % expected_rows != 0:
            return float("inf")
        mul = actual_rows // expected_rows
        num_warmup = num_warmup * mul
        num_active = num_active * mul

    time_cost = [0] * num_funcs
    for func_idx in np.arange(0, num_funcs):
        for active_index in np.arange(0, num_active):
            row_index = func_idx * (num_warmup + num_active) + num_warmup + active_index
            time_cost[func_idx] += filter_df.iloc[row_index][col_time]
    time_cost = [x * mul / num_active / 1e3 for x in time_cost]

    if num_funcs == 1:
        return time_cost[0]
    else:
        return time_cost


try:
    from mspti import KernelMonitor
except ImportError:
    KernelMonitor = None


# If the CANN version is earlier than 9.1.0, it needs to set libmspti.so in LD_PRELOAD to use mspti.
def do_bench_npu_mspti(
    funcs,
    warmup=5,
    active=30,
    clear_l2_cache=False,
    target_kernel_name: Optional[str] = None,
):
    if not isinstance(funcs, list):
        funcs = [funcs]

    for fn in funcs:
        fn()
        torch.npu.synchronize()

    if clear_l2_cache:
        buffer = runtime.driver.active.get_empty_cache_for_benchmark()
    else:
        buffer = None

    all_kernel_durations = []

    def callback(data):
        if clear_l2_cache and ('zero' in data.name.lower() or 'zeroslike' in data.name.lower()):
            return
        if target_kernel_name is not None and target_kernel_name not in data.name:
            return
        all_kernel_durations.append(data.end - data.start)

    monitor = KernelMonitor()
    torch.npu.synchronize()

    monitor.start(callback)

    try:
        total = warmup + active
        for fn in funcs:
            for _ in builtins.range(total):
                if clear_l2_cache:
                    buffer.zero_()
                fn()
    finally:
        torch.npu.synchronize()
        monitor.stop()

    num_funcs = len(funcs)
    duration_per_kernel = []

    expected_rows = num_funcs * total
    actual_rows = len(all_kernel_durations)
    if actual_rows < expected_rows:
        if num_funcs == 1:
            return float("inf")
        return [float("inf")] * num_funcs

    mul = 1
    if num_funcs == 1:
        if actual_rows % expected_rows != 0:
            return float("inf")
        mul = actual_rows // expected_rows
        warmup = warmup * mul
        total = actual_rows

    current_idx = 0
    for i in range(num_funcs):
        current_func_records = all_kernel_durations[current_idx:current_idx + total]
        current_idx += total
        current_active_records = current_func_records[warmup:total]
        avg_time = sum(current_active_records) * mul / len(current_active_records)
        avg_time_ms = avg_time / 1000000.0
        duration_per_kernel.append(avg_time_ms)

    if num_funcs == 1:
        return duration_per_kernel[0]
    else:
        return duration_per_kernel


def do_bench_npu(
    funcs,
    warmup=5,
    active=30,
    clear_l2_cache=False,
    prof_dir=None,
    keep_res=False,
    target_kernel_name: Optional[str] = None,
):
    import math
    mspti_available = True
    if KernelMonitor is None:
        mspti_available = False
        print(f"[WARNING] mspti package not found. Falling back to torch_npu.profiler.")
    if not isinstance(funcs, list):
        funcs = [funcs]
    results = None
    need_fallback = True
    if mspti_available and target_kernel_name is None:
        try:
            results = do_bench_npu_mspti(funcs, warmup, active, clear_l2_cache, target_kernel_name)
            first_val = results[0] if isinstance(results, list) else results
            if not math.isinf(first_val):
                need_fallback = False
        except Exception:
            pass
    if need_fallback:
        results = do_bench_npu_profiler(funcs, warmup, active, clear_l2_cache, prof_dir, keep_res, target_kernel_name)
    return results
