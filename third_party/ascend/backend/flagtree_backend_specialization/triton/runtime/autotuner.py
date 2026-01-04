import os
import threading
from concurrent.futures import ThreadPoolExecutor
import logging

def set_Autotuner_auto_profile_dir(autotuner, auto_profile_dir):
    autotuner.auto_profile_dir = auto_profile_dir

def ext_Autotuner_do_bench_MLIRCompilationError():
    from ..compiler.errors import MLIRCompilationError
    return (MLIRCompilationError,)

def _tiling_kernel(self, *args, config, **meta):
    # check for conflicts, i.e. meta-parameters both provided
    # as kwargs and by the autotuner
    conflicts = meta.keys() & config.kwargs.keys()
    if conflicts:
        raise ValueError(f"Conflicting meta-parameters: {', '.join(conflicts)}."
                            " Make sure that you don't re-define auto-tuned symbols.")
    # augment meta-parameters with tunable ones
    current = dict(meta, **config.all_kwargs())
    full_nargs = {**self.nargs, **current}

    def kernel_call():
        if config.pre_hook:
            config.pre_hook(full_nargs)
        self.pre_hook(full_nargs)
        try:
            self.fn.run(
                *args,
                **current,
            )
        except Exception as e:
            try:
                self.post_hook(full_nargs, exception=e)
            finally:
                # Throw exception raised by `self.fn.run`
                raise

        self.post_hook(full_nargs, exception=None)
    return kernel_call

def _batch_benchmark(self, kernel_dict, rep=10, quantiles=None):
    """
        Benchmark the runtime of the provided function.
        By default, return the median runtime of :code:`fn` along with
        the 20-th and 80-th performance percentile.

        :param kernel_dict: Function to benchmark
        :type kernel_dict: Callable
        :param rep: Repetition time (in ms)
        :type rep: int
        :param quantiles: Performance percentile to return in addition to the median.
        :type quantiles: list[float], optional
    """
    assert len(kernel_dict) > 0, f"ERROR: length of kernel_dict is empty."
    kernel_dict_temp_lock = threading.Lock()
    tiling_dict_lock = threading.Lock()
    tiling_dict = {}
    kernel_dict_temp = {}
    from triton.compiler.errors import CompileTimeAssertionFailure, CompilationError
    from triton.runtime.errors import OutOfResources
    from ..compiler.errors import MLIRCompilationError

    def run_fn(config, fn):
        try:
            with kernel_dict_temp_lock:
                fn()
                kernel_dict_temp[config] = fn
        except (CompileTimeAssertionFailure, MLIRCompilationError, CompilationError) as ex:
            with tiling_dict_lock:
                tiling_dict[config] = [float('inf')]
            raise ex

    def run_all_fns():
        import psutil
        max_workers = psutil.cpu_count(logical=False)
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = []
            for config, fn in kernel_dict.items():
                future = executor.submit(run_fn, config, fn)
                futures.append(future)
            for future in futures:
                try:
                    future.result()
                except Exception as ex:
                    logging.info(f"Exception raised while benchmarking function.{ex}")

    run_all_fns()

    if self.do_bench.__module__ == "triton.testing":
        enable_bench_npu = os.getenv("TRITON_BENCH_METHOD", 'default').lower() == 'npu'
        import torch
        if torch.npu.is_available() and enable_bench_npu:
            from triton.testing import do_bench_multiple_kernel_npu
            tiling_dict_temp = do_bench_multiple_kernel_npu(kernel_dict_temp, active=max(30, rep), prof_dir=None, keep_res=False)
            tiling_dict.update(tiling_dict_temp)
            return tiling_dict
    for config, kernel_call in kernel_dict_temp.items():
        try:
            tiling_dict[config] = self.do_bench(kernel_call, quantiles=quantiles)
        except (OutOfResources, CompileTimeAssertionFailure, MLIRCompilationError) as ex:
            tiling_dict[config] = [float("inf"), float("inf"), float("inf")]
    return tiling_dict

def _profile(autotuner, *args, config, **meta):
    from ..testing import do_bench_npu
    kernel_call = _tiling_kernel(*args, config=config, **meta)
    do_bench_npu(
        kernel_call, prof_dir=autotuner.auto_profile_dir, keep_res=True
    )

def _batch_bench(self, *args, configs, **kwargs):
    kernel_dict = {config: _tiling_kernel(*args, config=config, **kwargs) for config in configs}
    return _batch_benchmark(kernel_dict=kernel_dict, quantiles=(0.5, 0.2, 0.8))

def ext_Autotuner_batch_bench(autotuner, args, configs, kwargs):
    return _batch_bench(autotuner, args, configs, kwargs)

def ext_Autotuner_profile(autotuner, used_cached_result, args, kwargs):
    if not used_cached_result and autotuner.auto_profile_dir is not None:
        _profile(autotuner, *args, config=autotuner.best_config, **kwargs)

def default_Config_arg_is_none():
    return True

def set_Config_extra_options(config, bishengir_options):
    # BiShengIR Options allowed for autotune
    config.multibuffer = bishengir_options.get("multibuffer", None) # Compiler Default True
    config.sync_solver = bishengir_options.get("sync_solver", None) # Compiler Default False
    config.unit_flag = bishengir_options.get("unit_flag", None) # Compiler Default False
    config.limit_auto_multi_buffer_only_for_local_buffer = bishengir_options.get("limit_auto_multi_buffer_only_for_local_buffer", None) # Compiler Default False
    config.limit_auto_multi_buffer_of_local_buffer = bishengir_options.get("limit_auto_multi_buffer_of_local_buffer", None) # Compiler Default no-limit
    config.set_workspace_multibuffer = bishengir_options.get("set_workspace_multibuffer", None) # Compiler Default 1
    config.enable_hivm_auto_cv_balance = bishengir_options.get("enable_hivm_auto_cv_balance", None) # Compiler Default True
    config.tile_mix_vector_loop = bishengir_options.get("tile_mix_vector_loop", None) # Compiler Default 1
    config.tile_mix_cube_loop = bishengir_options.get("tile_mix_cube_loop", None) # Compiler Default 1

def ext_Config_all_kwargs(config):
    return (
        ("force_simt_template", config.force_simt_template),
        ("enable_linearize", config.enable_linearize),
        ("multibuffer", config.multibuffer),
        ("enable_hivm_auto_cv_balance", config.enable_hivm_auto_cv_balance),
        ("sync_solver", config.sync_solver),
        ("unit_flag", config.unit_flag),
        ("limit_auto_multi_buffer_only_for_local_buffer", \
            config.limit_auto_multi_buffer_only_for_local_buffer),
        ("limit_auto_multi_buffer_of_local_buffer", config.limit_auto_multi_buffer_of_local_buffer),
        ("set_workspace_multibuffer", config.set_workspace_multibuffer),
        ("tile_mix_vector_loop", config.tile_mix_vector_loop),
        ("tile_mix_cube_loop", config.tile_mix_cube_loop)
    )

def ext_Config_to_str(res, config):
    res.append(f"multibuffer: {config.multibuffer}")
    res.append(f"enable_hivm_auto_cv_balance: {config.enable_hivm_auto_cv_balance}")
    res.append(f"sync_solver: {config.sync_solver}")
    res.append(f"unit_flag: {config.unit_flag}")
    res.append(f"limit_auto_multi_buffer_only_for_local_buffer: \
        {config.limit_auto_multi_buffer_only_for_local_buffer}")
    res.append(f"limit_auto_multi_buffer_of_local_buffer: {config.limit_auto_multi_buffer_of_local_buffer}")
    res.append(f"set_workspace_multibuffer: {config.set_workspace_multibuffer}")
    res.append(f"tile_mix_vector_loop: {config.tile_mix_vector_loop}")
    res.append(f"tile_mix_cube_loop: {config.tile_mix_cube_loop}")
    res.append(f"force_simt_template: {config.force_simt_template}")

def new_AutoTilingTuner(hints, fn, configs, key, reset_to_zero, restore_value, pre_hook,
                        post_hook, prune_configs_by, warmup, rep,
                        use_cuda_graph, do_bench, auto_profile_dir):
    if hints is not None and hints.get("enable_ascend_autotune"):
        from triton.runtime.autotiling_tuner import AutoTilingTuner
        return AutoTilingTuner(fn, fn.arg_names, configs, key, reset_to_zero, restore_value, pre_hook=pre_hook,
                               post_hook=post_hook, prune_configs_by=prune_configs_by, warmup=warmup, rep=rep,
                               use_cuda_graph=use_cuda_graph, do_bench=do_bench, auto_profile_dir=auto_profile_dir)
    return None
