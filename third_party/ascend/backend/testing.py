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
import triton
from datetime import datetime, timezone

import triton.runtime as runtime
from triton._C.clear_l2 import do_bench_clear


def do_bench_npu(funcs, warmup=25, active=100, prof_dir=None, clear_l2_cache=False, keep_res=False, collect_prof=True,
                 return_mode="mean", quantiles=None):
    import torch
    import torch_npu

    assert return_mode in ["min", "max", "mean", "median", "all"]

    if not isinstance(funcs, list):
        funcs = [funcs]

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
        device = triton.runtime.driver.active.get_current_device()
        stream = triton.runtime.driver.active.get_current_stream(device)
        buffer = runtime.driver.active.get_empty_cache_for_benchmark()
        do_bench_clear(buffer.data_ptr(), buffer.numel(), stream)
        torch.npu.synchronize()  # shake out of any npu error

    # cal warmup num
    di = runtime.driver.active.get_device_interface()
    for fn in funcs:
        fn()
        di.synchronize()

    cache = runtime.driver.active.get_empty_cache_for_benchmark()
    start_event = di.Event(enable_timing=True)
    end_event = di.Event(enable_timing=True)
    start_event.record()
    for fn in funcs:
        for _ in range(5):
            cache.zero_()
            fn()
    end_event.record()
    di.synchronize()
    estimate_ms = start_event.elapsed_time(end_event) / 5

    n_warmup = min(5, max(1, int(warmup / estimate_ms)))
    n_repeat = min(30, max(1, int(active / estimate_ms)))

    total = n_warmup + n_repeat
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
        # Run for 300 μs to raise the frequency to 800.
        mat_a = torch.randn(4096, 4096).to(dtype=torch.bfloat16)
        mat_b = torch.randn(4096, 4096).to(dtype=torch.bfloat16)
        mat_c = torch.matmul(mat_a, mat_b)
        mat_c.cpu()

        for fn in funcs:
            # execute
            for _ in builtins.range(total):
                if clear_l2_cache:
                    do_bench_clear(buffer.data_ptr(), buffer.numel(), stream)
                fn()
                torch.npu.synchronize()

    if collect_prof:
        time_cost = _collect_prof_result(torch_path, funcs, n_warmup, n_repeat, return_mode=return_mode,
                                         quantiles=quantiles)  # read kernel_details.csv
    else:
        time_cost = _collect_single(torch_path, return_mode=return_mode)  # read op_static.csv
    _rm_dic(keep_res, torch_path)
    return time_cost


# keep the original behavior to get the statistics for the specified kernel func
def _collect_single(base_dir: str, key: str = None, print_flag=True, return_mode="mean") -> float:
    if not os.path.exists(base_dir):
        print("if not os.path.exits(base_dir)")
        return float("inf")

    import pandas as pd

    for root, _, files in os.walk(base_dir):
        for file in files:
            if file != "op_statistic.csv":
                continue
            target_file = os.path.join(root, file)
            df = pd.read_csv(target_file)
            if print_flag:
                print(df)
            if key is not None:
                key_rows = df[df["OP Type"].str.startswith(key, na=False)]
                if not key_rows.empty:
                    if return_mode == "all":
                        return key_rows["Total Time(us)"].values[0]
                    elif return_mode == "min":
                        return key_rows["Min Time(us)"].values[0]
                    elif return_mode == "max":
                        return key_rows["Max Time(us)"].values[0]
                    elif return_mode == "mean":
                        return key_rows["Avg Time(us)"].values[0]
                    else:
                        assert return_mode in ["min", "max", "mean", "all"], "not support other mode"
                return float("inf")
            else:
                # default: read the first row except header
                # return df.loc[0, "Avg Time(us)"]
                # default: extract clear L2 cache time (L2 cache clear operation)
                filter_cond = df["OP Type"].str.contains(r"do_bench_clear", case=False, na=False)
                filter_df = df[filter_cond]
                if not filter_df.empty:
                    clear_l2_time = filter_df.iloc[0]['Avg Time(us)']
                    if print_flag:
                        print("Clear L2 cache time:", clear_l2_time)

                # Calculate total time of all operators excluding clear l2 cache
                non_clearL2_df = df[~df["OP Type"].str.contains(r"do_bench_clear", case=False, na=False)]
                if return_mode == "all":
                    ops_time = non_clearL2_df["Total Time(us)"].sum()
                elif return_mode == "min":
                    ops_time = non_clearL2_df["Min Time(us)"].sum()
                elif return_mode == "max":
                    ops_time = non_clearL2_df["Max Time(us)"].sum()
                elif return_mode == "mean":
                    ops_time = non_clearL2_df["Avg Time(us)"].sum()
                else:
                    assert return_mode in ["min", "max", "mean", "all"], "not support other mode"
                # all_ops_total_time = non_clearL2_df['Avg Time(us)'].sum()
                ops_time = round(ops_time, 3)
                if print_flag:
                    print(f"All ops {return_mode} time:", ops_time)

                return ops_time

    return float("inf")


def _rm_dic(keep_res, torch_path):
    if keep_res:
        return
    import shutil

    if os.path.exists(torch_path):
        shutil.rmtree(torch_path)


def _collect_prof_result(base_dir: str, funcs, num_warmup: int, num_active: int, key: str = None, return_mode="mean",
                         quantiles=None):
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
    :param return mode
    :type return_mode: str
    :param quantiles: Performance percentile to return in addition to the median.
    :type quantiles: list[float], optional
    """

    import numpy as np
    import pandas as pd
    import torch

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
    filter_cond = ~df["Type"].str.contains(r"do_bench_clear", case=False, na=False)
    filter_df = df[filter_cond]
    if key is not None:
        key_rows = filter_df[filter_df["Name"].str.contains(key, na=False)]
    else:
        key_rows = filter_df

    times = []
    for func_idx in np.arange(0, num_funcs):
        for active_index in np.arange(0, num_active):
            row_index = func_idx * (num_warmup + num_active) + num_warmup + active_index
            times.append(float(key_rows.iloc[row_index]["Duration(us)"]))

    if quantiles is not None and len(quantiles) > 0:
        if not all(0.0 <= q <= 1.0 for q in quantiles):
            raise ValueError("quantiles values must be in range [0.0, 1.0].")
        # 0.0~1.0 to 10~100
        q_values = np.percentile(times, [q * 100 for q in quantiles])
        return torch.tensor(q_values, dtype=torch.float32)

    if return_mode == "all":
        return sum(times)
    elif return_mode == "min":
        return min(times)
    elif return_mode == "max":
        return max(times)
    elif return_mode == "median":
        import statistics
        return statistics.median(times)
    elif return_mode == "mean":
        import statistics
        return statistics.mean(times)
    else:
        raise ValueError(f"return_mode '{return_mode}' not supported. Use 'min', 'max', 'mean', 'median', or 'all'.")
