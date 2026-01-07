import torch
import os
import multiprocessing
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import logging
import builtins
from triton import runtime

def is_do_bench_npu():
    enable_bench_npu = os.getenv("TRITON_BENCH_METHOD", 'default').lower() == 'npu'
    if torch.npu.is_available() and enable_bench_npu:
        return True
    return False


def collect_files(base_dir):
    import pandas as pd
    for root, dirs, files in os.walk(base_dir):
        for file in files:
            if file != 'op_statistic.csv':
                continue
            target_file = os.path.join(root, file)
            df = pd.read_csv(target_file)
            triton_rows = df[df['OP Type'].str.startswith('triton', na=False)]
            if not triton_rows.empty:
                return triton_rows['Avg Time(us)'].values[0]
            return float('inf')
    return float('inf')


def collect_single(base_dir: str, key: str = None) -> float:
    if not os.path.exists(base_dir):
        return float('inf')

    import pandas as pd
    for root, _, files in os.walk(base_dir):
        for file in files:
            if file != 'op_statistic.csv':
                continue
            target_file = os.path.join(root, file)
            df = pd.read_csv(target_file)
            if key is not None:
                key_rows = df[df['OP Type'].str.startswith(key, na=False)]
                if not key_rows.empty:
                    return key_rows['Avg Time(us)'].values[0]
                return float('inf')
            else:
                # default: read the first row except header
                return df.loc[0, 'Avg Time(us)']

    return float('inf')

def _rm_dic(keep_res, torch_path):
    if keep_res:
        return
    import shutil
    if os.path.exists(torch_path):
        shutil.rmtree(torch_path)

def _collect_mul_prof_result(base_dir: str, kernel_dict, total, key: str = None):
    import numpy as np
    import pandas as pd
    tiling_dict = {}
    kernel_details_file = None
    for root, _, files in os.walk(base_dir):
        for file in files:
            if file == "kernel_details.csv":
                kernel_details_file = os.path.join(root, file)
                break
    num_funcs = len(kernel_dict)
    if kernel_details_file is None or os.path.exists(kernel_details_file) is False:
        for config, _ in kernel_dict.items():
            tiling_dict[config] = [float('inf')]
        return tiling_dict
    df = pd.read_csv(kernel_details_file)
    # filter out l2 cache clear operation
    filter_cond = ~df["Name"].str.contains(r"zero|ZerosLike", case=False, na=False)
    filter_df = df[filter_cond]
    if key is not None:
        key_rows = filter_df[filter_df["Name"].str.contains(key, na=False)]
    else:
        key_rows = filter_df
    time_cost = [0] * num_funcs
    for func_idx in np.arange(0, num_funcs):
        for active_index in np.arange(0, total):
            row_index = active_index + func_idx * total
            time_cost[func_idx] += key_rows.iloc[row_index]["Duration(us)"]
    time_cost = [x / total for x in time_cost]
    for (config, avg_time) in zip(kernel_dict.keys(), time_cost):
        tiling_dict[config] = [avg_time]
    return tiling_dict

def do_bench_npu(fn, warmup=5, active=30, prof_dir=None, keep_res=False):
    import torch_npu
    import multiprocessing
    from triton import runtime

    # warmup kernel
    fn()
    torch.npu.synchronize()

    experimental_config = torch_npu.profiler._ExperimentalConfig(
        aic_metrics=torch_npu.profiler.AiCMetrics.PipeUtilization,
        profiler_level=torch_npu.profiler.ProfilerLevel.Level1,
        l2_cache=False,
        data_simplification=False
    )
    skip_first = 1
    wait = 0
    repeat = 1
    total = skip_first + (wait + warmup + active) * repeat

    if prof_dir is not None:
        torch_path = prof_dir
    else:
        process = multiprocessing.current_process()
        pid = process.pid
        process_name = process.name
        timestamp = datetime.now(tz=timezone.utc).strftime("%Y%m%d_%H%M%S")
        base_path = os.path.join(runtime.cache.get_home_dir(), ".triton", "profile_results")
        torch_path = os.path.join(base_path, f"prof_{timestamp}_{process_name}-{pid}")
    with torch_npu.profiler.profile(
        activities=[
            torch_npu.profiler.ProfilerActivity.NPU
        ],
        schedule=torch_npu.profiler.schedule(wait=wait, warmup=warmup, active=active, repeat=repeat, skip_first=skip_first),
        on_trace_ready=torch_npu.profiler.tensorboard_trace_handler(torch_path),
        record_shapes=False,
        profile_memory=False,
        with_stack=False,
        with_flops=False,
        with_modules=False,
        experimental_config=experimental_config,
    ) as prof:
        for _ in builtins.range(total):
            fn()
            prof.step()
            torch.npu.synchronize()

    time = collect_single(torch_path)
    _rm_dic(keep_res, torch_path)
    return time

def do_bench_multiple_kernel_npu(kernel_dict, active=30, prof_dir=None, keep_res=False):
    import torch
    import torch_npu

    from .compiler.errors import CompileTimeAssertionFailure, MLIRCompilationError, CompilationError
    assert len(kernel_dict) > 0, f"ERROR: length of kernel_dict is {len(kernel_dict)}, no kernel is profiling."

    # warmup kernel
    def run_fn(fn):
        try:
            fn()
        except (CompileTimeAssertionFailure, MLIRCompilationError, CompilationError) as ex:
            raise ex

    def run_all_fns():
        import psutil
        max_workers = psutil.cpu_count(logical=False)
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = []
            for _, fn in kernel_dict.items():
                future = executor.submit(run_fn, fn)
                futures.append(future)
            for future in futures:
                try:
                    future.result()
                except Exception as ex:
                    logging.info(f"Exception raised while benchmarking function.{ex}")
    run_all_fns()

    experimental_config = torch_npu.profiler._ExperimentalConfig(
        aic_metrics=torch_npu.profiler.AiCMetrics.PipeUtilization,
        profiler_level=torch_npu.profiler.ProfilerLevel.Level1,
        l2_cache=False,
        data_simplification=False
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

    l2_cache_size = 192 * (1 << 20)
    buffer = torch.empty(l2_cache_size // 4, dtype=torch.int, device="npu")
    buffer.zero_()
    torch.npu.synchronize()  # shake out of any npu error

    with torch_npu.profiler.profile(
            activities=[
                torch_npu.profiler.ProfilerActivity.NPU
            ],
            on_trace_ready=torch_npu.profiler.tensorboard_trace_handler(torch_path),
            record_shapes=False,
            profile_memory=False,
            with_stack=False,
            with_flops=False,
            with_modules=False,
            experimental_config=experimental_config,
    ) as prof:
        for _, fn in kernel_dict.items():
            for _ in builtins.range(active):
                buffer.zero_()
                fn()
                torch.npu.synchronize()
    del buffer

    tiling_dict = _collect_mul_prof_result(base_dir=torch_path, kernel_dict=kernel_dict, total=active)
    _rm_dic(keep_res, torch_path)
    return tiling_dict

def ext_do_bench_npu(fn, warmup, rep, quantiles, return_mode):
    import torch
    from triton.testing import _summarize_statistics
    avg_time = do_bench_npu(fn, warmup=max(5, warmup), active=max(30, rep))
    return _summarize_statistics(torch.tensor([avg_time], dtype=torch.float), quantiles, return_mode)

def testing_spec_range(num):
    return builtins.range(num)

testing_ext_spec_func_list = [
    "do_bench_npu",
    "do_bench_multiple_kernel_npu"
]
