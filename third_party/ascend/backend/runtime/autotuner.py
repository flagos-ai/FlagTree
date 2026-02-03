# Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
# Copyright 2018-2020 Philippe Tillet
# Copyright 2020-2022 OpenAI
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

from __future__ import annotations

import builtins
import os
import time
import copy
from typing import Dict, List
from torch import Tensor

from triton.runtime.autotuner import Autotuner, Config

from .utils import get_byte_per_numel, is_valid_axis_name, valid_axis_names
from .autoparser import SplitAxesParser, TilingAxesParser, ReductionAxesParser, LowDimsAxesParser, PtrNumsParser


class AutoTilingTuner(Autotuner):
    """
    Automatic generateing candidate tiling configs and evaluating their performance to get the best config.
    """

    def __init__(
        self,
        fn,
        arg_names,
        configs,
        key,
        reset_to_zero,
        restore_value,
        pre_hook=None,
        post_hook=None,
        prune_configs_by: Dict = None,
        warmup=None,
        rep=None,
        use_cuda_graph=False,
        do_bench=None,
        auto_profile_dir=None,
        hints=None,
    ):
        """
        :param key: a list of argument name, where the change of arguments in value will triger re-generating candidates configs and evaluating.
            The parameters in the list will be assigned axis names in sequence, with the axis name being in
            {'x','y','z','w','v','t','rx','ry','rz','rw','rv','rt}, where the prefix 'r' means a reduction axis.
            Only the axis name in this param should add perfix 'r' if it's a reduction axis.
        :type key: List[str]
        """
        super().__init__(
            fn,
            arg_names,
            configs,
            key,
            reset_to_zero,
            restore_value,
            pre_hook,
            post_hook,
            prune_configs_by,
            warmup,
            rep,
            use_cuda_graph,
            do_bench,
        )
        if not hints:
            self.hints = {}
        else:
            self.hints = hints
        split_params = self.hints.get("split_params", None)
        tiling_params = self.hints.get("tiling_params", None)
        low_dim_axes = self.hints.get("low_dim_axes", None)
        reduction_axes = self.hints.get("reduction_axes", None)
        self._init_axis_params(key, split_params, tiling_params, low_dim_axes, reduction_axes)

        self.auto_gen_config = not configs or self.hints.get("auto_gen_config", False)
        self.gen_configs = []  # generated configs from TileGenerator
        self.auto_profile_dir = auto_profile_dir
        if not configs:
            self.user_configs = []
        else:
            self.user_configs = configs
        self.is_simt_mode = False
        self.user_specified_warps = None
        self.print_autotuning = os.getenv("TRITON_PRINT_AUTOTUNING", None) == "1"

    def _init_axis_params(self, key, split_params, tiling_params, low_dim_axes, reduction_axes):
        if isinstance(key, list):
            if (split_params or tiling_params or low_dim_axes or reduction_axes):
                raise ValueError(
                    "If any axis-related parameters (split_params, tiling_params, low_dim_axes, reduction_axes)"
                    " are provided, 'key' must be a dict, not a list.")
            if len(key) > len(valid_axis_names):
                raise ValueError("Number of parameters exceeds the number of available axes.")
            self.keys = {axis: param for axis, param in zip(valid_axis_names, key)}
        elif isinstance(key, dict):
            if not set(key.keys()).issubset(set(valid_axis_names)):
                raise ValueError("All keys in 'key' must be valid axis names. Got unexpected keys.")
            self.keys = key
            if any([split_params, tiling_params, low_dim_axes, reduction_axes]) is None:
                raise ValueError(
                    "If 'key' is a dict, all axis-related parameters (split_params, tiling_params, low_dim_axes,"
                    " reduction_axes) must be provided.")
            if not isinstance(split_params, dict):
                raise ValueError("split_params must be a dict, got: {}".format(type(split_params)))
            if not isinstance(tiling_params, dict):
                raise ValueError("tiling_params must be a dict, got: {}".format(type(tiling_params)))
            if not isinstance(low_dim_axes, list):
                raise ValueError("low_dim_axes must be a list, got: {}".format(type(low_dim_axes)))
            if not isinstance(reduction_axes, list):
                raise ValueError("reduction_axes must be a list, got: {}".format(type(reduction_axes)))

            used_axes = set(split_params.keys()).union(
                tiling_params.keys(),
                low_dim_axes,
                reduction_axes,
            )
            if not used_axes.issubset(self.keys.keys()):
                raise ValueError(
                    "The following axes are used but not present in the 'key': {}".format(used_axes -
                                                                                          set(self.keys.keys())))

        self.split_params = split_params
        self.tiling_params = tiling_params
        self.low_dim_axes = low_dim_axes
        self.reduction_axes = reduction_axes
        self.dual_reduction = False
        self.persistent_reduction = False
        self.num_buffers = -1

    def _autoparse_axis_params(self, all_args):
        miss_params = [arg for arg in self.arg_names if arg not in all_args.keys()]
        # parse pointer params nums
        if self.num_buffers == -1:
            self.num_buffers = self._autoparse_ptr_nums(all_args)

        # parse autotiling axes
        # reduction axis must be parsed before other axes. it will alter the key
        if not self.reduction_axes:
            self.reduction_axes = self._autoparse_reduction_axes()
        if len(self.reduction_axes) >= 2:
            self.dual_reduction = True

        if not self.low_dim_axes:
            self.low_dim_axes = self._autoparse_low_dim_axes()

        if len(self.reduction_axes) == 1 and \
            self.reduction_axes[0] == self.low_dim_axes[0] and \
            all_args.get(self.keys[self.reduction_axes[0]], float("inf")) < 1024:
            self.persistent_reduction = True

        if not self.split_params:
            self.split_params = self._autoparse_split_params(miss_params)
        miss_params = [arg for arg in miss_params if arg not in self.split_params.values()]
        if not self.tiling_params:
            self.tiling_params = self._autoparse_tiling_params(miss_params)
        miss_params = [arg for arg in miss_params if arg not in self.tiling_params.values()]
        if miss_params:
            raise ValueError(f"Missing required arguments: {miss_params}. "
                             f"These arguments must be explicitly provided and cannot be automatically tuned. "
                             f"Please ensure that these arguments are passed when calling the function.")

    def _gen_tile_configs(self, kv_dict: Dict[str, int], dtype: torch.dtype) -> List[Config]:
        from .tile_generator import KernelMeta, TileGenerator

        axis_sizes = {}
        for k, v in kv_dict.items():
            if not is_valid_axis_name(k):
                continue
            if not isinstance(v, int):
                raise ValueError(f"Not supported dim type: {type(v)}, `int` is the only supported type")
            axis_sizes[k] = v

        kernel_meta = KernelMeta(
            axis_sizes,
            self.split_params,
            self.tiling_params,
            self.low_dim_axes,
            dtype,
            self.persistent_reduction,
            self.dual_reduction,
            self.num_buffers,
            self.is_simt_mode,
        )
        tile_gen = TileGenerator(kernel_meta=kernel_meta)
        tile_gen.descend_split_tiling()

        self.gen_configs.clear()
        self.gen_configs = tile_gen.configs

        if self.is_simt_mode:
            _default_cand_num_warps = [8, 16, 32, 64]
            cand_num_warps = (_default_cand_num_warps
                              if self.user_specified_warps is None else [self.user_specified_warps])
            simt_configs = []
            for base_cfg in self.gen_configs:
                for num_warps in cand_num_warps:
                    new_cfg = copy.deepcopy(base_cfg)
                    new_cfg.num_warps = num_warps
                    simt_configs.append(new_cfg)

            if self.print_autotuning:
                print(f"Triton autotuning: Expanded to {len(simt_configs)} SIMT configs (with warps: {cand_num_warps})")

            self.gen_configs = simt_configs

        if len(self.gen_configs) == 0:
            print("[WARNING] The generated candidate tiling configs are empty based on provided parameters!")

        if self.print_autotuning:
            print("Generated configs number: {}".format(len(self.gen_configs)))

    def generate_key_and_configs(self, *args, **kwargs):
        self.nargs = dict(zip(self.arg_names, args))
        self.is_simt_mode = kwargs.get('force_simt_only', False)
        if 'num_warps' in kwargs and kwargs['num_warps'] is not None:
            self.user_specified_warps = kwargs['num_warps']

        # generate key
        all_args = {**self.nargs, **kwargs}
        _args = {k: v for (k, v) in all_args.items() if k in self.arg_names}
        key = [_args[v] for _, v in self.keys.items() if v in _args]

        # Currently, we use the dtype with maximum byte length
        dtype = None
        for _, arg in _args.items():
            if hasattr(arg, "dtype"):
                key.append(str(arg.dtype))
                dtype = (arg.dtype if get_byte_per_numel(arg.dtype) >= get_byte_per_numel(dtype) else dtype)
        if dtype is None:
            raise NotImplementedError("Not support for non-Tensor inputs")

        key = tuple(key)
        if key not in self.cache:
            if self.auto_gen_config:
                self._autoparse_axis_params(all_args)
                _kv_dict = {k: _args[v] for k, v in self.keys.items() if v in _args}
                self._gen_tile_configs(_kv_dict, dtype)
            if len(self.gen_configs) == 0 and len(self.user_configs) == 0:
                self.configs = [
                    Config(
                        {},
                        num_warps=4,
                        num_stages=2,
                        num_ctas=1,
                        num_buffers_warp_spec=0,
                        num_consumer_groups=0,
                        reg_dec_producer=0,
                        reg_inc_consumer=0,
                    )
                ]
            else:
                self.configs = self.gen_configs + self.user_configs
        return key

    def run(self, *args, **kwargs):
        key = self.generate_key_and_configs(*args, **kwargs)
        used_cached_result = True
        if key not in self.cache:
            # prune configs
            pruned_configs = self.prune_configs(kwargs)
            if len(pruned_configs) > 1:
                used_cached_result = False
                bench_start = time.time()
                timings = self._batch_bench(*args, configs=pruned_configs, **kwargs)
                bench_end = time.time()
                self.bench_time = bench_end - bench_start
                self.cache[key] = builtins.min(timings, key=timings.get)
                full_nargs = {**self.nargs, **kwargs, **self.cache[key].all_kwargs()}
                self.pre_hook(full_nargs, reset_only=True)
                self.configs_timings = timings
                config = self.cache[key]
            else:
                config = pruned_configs[0]
        else:
            config = self.cache[key]

        self.best_config = config
        if self.print_autotuning and not used_cached_result:
            print(f"Triton autotuning for function {self.base_fn.__name__} finished after "
                  f"{self.bench_time:.2f}s; best config selected: {self.best_config};")

        if not used_cached_result and self.auto_profile_dir is not None:
            self._profile(*args, config=self.best_config, **kwargs)
        if config.pre_hook is not None:
            full_nargs = {**self.nargs, **kwargs, **config.all_kwargs()}
            config.pre_hook(full_nargs)
        final_kwargs = dict(config.all_kwargs(), **kwargs)
        ret = self.fn.run(
            *args,
            **final_kwargs,
        )
        self.nargs = None
        return ret

    def _batch_bench(self, *args, configs, **kwargs):
        from triton.compiler.errors import CompileTimeAssertionFailure, MLIRCompilationError
        from triton.runtime.errors import OutOfResources

        kernels_call = {config: self._make_kernel_call(*args, config=config, **kwargs) for config in configs}
        run_fns = {}
        exc = None
        exc_stack = ""

        for config, fn in kernels_call.items():
            try:
                fn()
                run_fns[config] = fn
            except (CompileTimeAssertionFailure, MLIRCompilationError, OutOfResources) as e:
                import traceback
                exc_stack = traceback.format_exc()
                exc = e

        if len(run_fns) == 0:
            raise RuntimeError(f"No valid triton configs. {type(exc).__name__}: {exc} \nStack trace: {exc_stack}")

        if len(run_fns) == 1:
            # we ignore expensive profiling method when only single config is left
            return {config: self.do_bench(fn) for config, fn in run_fns.items()}

        use_profiling = os.getenv("TRITON_BENCH_METHOD", "default").lower() == "npu"
        if use_profiling:
            from ..testing import do_bench_npu

            time_cost = do_bench_npu(list(run_fns.values()), clear_l2_cache=False)
            assert len(time_cost) == len(run_fns)
            return {config: cost for config, cost in zip(run_fns.keys(), time_cost)}
        else:
            return {config: self.do_bench(fn) for config, fn in run_fns.items()}

    def _make_kernel_call(self, *args, config, **meta):
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

    def warmup(self, *args, **kwargs):
        _ = self.generate_key_and_configs(*args, **kwargs)
        pruned_configs = self.prune_configs(kwargs)
        ret = []
        for config in pruned_configs:
            ret.append(self.fn.warmup(*args, **kwargs, **config.all_kwargs()))
        self.nargs = None
        return ret

    def _profile(self, *args, config, **meta):
        from ..testing import do_bench_npu

        kernel_call = self._make_kernel_call(*args, config=config, **meta)
        do_bench_npu(kernel_call, prof_dir=self.auto_profile_dir, keep_res=True)

    def _autoparse_split_params(self, candidates_params: List[str]) -> Dict[str, str]:
        """
        Extracts the split axis parameters from triton kernel code.
        """
        func_ast = self.fn.parse()
        parser = SplitAxesParser(func_ast, self.keys, candidates_params)
        split_axes = parser.parse()
        if self.print_autotuning:
            print(f"Ascend autotuning parse split axes: {split_axes}")
        return split_axes

    def _autoparse_tiling_params(self, candidates_params: List[str]) -> Dict[str, str]:
        """
        Extracts the tiling axis parameters from triton kernel code.
        """
        func_ast = self.fn.parse()
        parser = TilingAxesParser(func_ast, self.keys, candidates_params)
        tiling_axes = parser.parse()
        if self.print_autotuning:
            print(f"Ascend autotuning parse tiling axes: {tiling_axes}")
        return tiling_axes

    def _autoparse_reduction_axes(self) -> List[str]:
        """
        Extracts the reduction axis parameters from triton kernel code.
        """
        func_ast = self.fn.parse()
        parser = ReductionAxesParser(func_ast, self.keys)
        reduction_axes = parser.parse()
        for axis in reduction_axes:
            self.keys[f"r{axis}"] = self.keys.pop(axis)
        reduction_axes = [f"r{axis}" for axis in reduction_axes]

        if self.print_autotuning:
            print(f"Ascend autotuning parse keys: {self.keys} \n"
                  f"Ascend autotuning parse reduction axes: {reduction_axes}")
        return reduction_axes

    def _autoparse_low_dim_axes(self) -> List[str]:
        """
        Extracts the low dimension axis from triton kernel code.
        """
        func_ast = self.fn.parse()
        parser = LowDimsAxesParser(func_ast, self.keys)
        low_dim_axes = parser.parse()
        if len(low_dim_axes) < 1:
            raise ValueError(f"Failed to parse low-dimensional axes.")
        if self.print_autotuning:
            print(f"Ascend autotuning parse low dimensional axes: {low_dim_axes}")
        return low_dim_axes

    def _autoparse_ptr_nums(self, all_args: dict) -> int:
        """
        Counts the number of pointer parameters from triton kernel code.
        """
        ptr_nums = 0
        ptr_params = list()
        for k, v in all_args.items():
            if isinstance(v, Tensor):
                ptr_nums += 1
                ptr_params.append(k)

        if self.print_autotuning:
            print(f"Ascend autotuning parse pointer params: {ptr_params}, pointer nums: {ptr_nums}")
        return ptr_nums


def autotune(configs, key, prune_configs_by=None, reset_to_zero=None, restore_value=None, pre_hook=None, post_hook=None,
             warmup=None, rep=None, use_cuda_graph=False, do_bench=None, *, auto_prof_dir=None, hints=None):
    """
    Decorator for auto-tuning a :code:`triton.jit`'d function.

    .. highlight:: python
    .. code-block:: python

        @triton.autotune(configs=[
            triton.Config(kwargs={'BLOCK_SIZE': 128}, num_warps=4),
            triton.Config(kwargs={'BLOCK_SIZE': 1024}, num_warps=8),
          ],
          key=['x_size'] # the two above configs will be evaluated anytime
                         # the value of x_size changes
        )
        @triton.jit
        def kernel(x_ptr, x_size, **META):
            BLOCK_SIZE = META['BLOCK_SIZE']
    :note: When all the configurations are evaluated, the kernel will run multiple times.
           This means that whatever value the kernel updates will be updated multiple times.
           To avoid this undesired behavior, you can use the `reset_to_zero` argument, which
           resets the value of the provided tensor to `zero` before running any configuration.

    If the environment variable :code:`TRITON_PRINT_AUTOTUNING` is set to
    :code:`"1"`, Triton will print a message to stdout after autotuning each
    kernel, including the time spent autotuning and the best configuration.

    :param configs: a list of :code:`triton.Config` objects
    :type configs: list[triton.Config]
    :param key: a list of argument names whose change in value will trigger the evaluation of all provided configs.
    :type key: list[str]
    :param prune_configs_by: a dict of functions that are used to prune configs, fields:
        'perf_model': performance model used to predicate running time with different configs, returns running time
        'top_k': number of configs to bench
        'early_config_prune'(optional): a function used to do early prune (eg, num_stages). It takes configs:List[Config] as its input, and returns pruned configs.
    :param reset_to_zero: a list of argument names whose value will be reset to zero before evaluating any configs.
    :type reset_to_zero: list[str]
    :param restore_value: a list of argument names whose value will be restored after evaluating any configs.
    :type restore_value: list[str]
    :param pre_hook: a function that will be called before the kernel is called.
        This overrides the default pre_hook used for 'reset_to_zero' and 'restore_value'.
        'kwargs': a dict of all arguments passed to the kernel.
        'reset_only': a boolean indicating whether the pre_hook is called to reset the values only, without a corresponding post_hook.
    :type pre_hook: lambda args, reset_only
    :param post_hook: a function that will be called after the kernel is called.
        This overrides the default post_hook used for 'restore_value'.
        'kwargs': a dict of all arguments passed to the kernel.
        'exception': the exception raised by the kernel in case of a compilation or runtime error.
    :type post_hook: lambda args, exception
    :param warmup: warmup time (in ms) to pass to benchmarking (deprecated).
    :type warmup: int
    :param rep: repetition time (in ms) to pass to benchmarking (deprecated).
    :type rep: int
    :param do_bench: a benchmark function to measure the time of each run.
    :type do_bench: lambda fn, quantiles
    :param auto_prof_dir: the specified directory to store the profiling results of the best config.
        If this parameter is None or the best config is retrieved from cache, the profiling process will be ignored.
    :type auto_prof_dir: str
    :param hints: a dict of autotune hint auguments passed to AutoTilingTuner.
    """

    def decorator(fn):
        return AutoTilingTuner(fn, fn.arg_names, configs, key, reset_to_zero, restore_value, pre_hook=pre_hook,
                               post_hook=post_hook, prune_configs_by=prune_configs_by, warmup=warmup, rep=rep,
                               use_cuda_graph=use_cuda_graph, do_bench=do_bench, auto_profile_dir=auto_prof_dir,
                               hints=hints)

    return decorator
