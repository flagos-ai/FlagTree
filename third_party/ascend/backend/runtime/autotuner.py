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
import copy
import functools
import ast
import os
import time
from concurrent.futures import ThreadPoolExecutor
import itertools
from typing import Dict, List

from torch import Tensor

import triton
from triton.runtime.autotuner import Autotuner, Config
from triton.tools.get_ascend_devices import is_compile_on_910_95

from .autoparser import (LowDimsAxesParser, PtrNumsParser, ReductionAxesParser, SplitAxesParser, TilingAxesParser)
from .utils import get_byte_per_numel, is_valid_axis_name, valid_axis_names


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
        self.user_defined_do_bench = do_bench is not None
        if not hints:
            self.hints = {}
        else:
            self.hints = hints
        split_params = self.hints.get("split_params", None)
        tiling_params = self.hints.get("tiling_params", None)
        low_dim_axes = self.hints.get("low_dim_axes", None)
        reduction_axes = self.hints.get("reduction_axes", None)
        self._init_axis_params(
            key,
            split_params,
            tiling_params,
            low_dim_axes,
            reduction_axes,
        )

        self.auto_gen_config = not configs or self.hints.get("auto_gen_config", False)
        self.gen_configs = []  # generated configs from TileGenerator
        self.auto_profile_dir = auto_profile_dir
        if not configs:
            self.user_configs = []
        else:
            self.user_configs = configs
        self.is_simt_mode = False
        self.simt_stack_limit = 8192
        self.user_specified_warps = None
        self.user_specified_multibuffer = None
        self.default_multibuffer = not is_compile_on_910_95
        self.print_autotuning = os.getenv("TRITON_PRINT_AUTOTUNING", None) == "1"
        # Compile kernels in parallel by default for triton.runtime.JITFunction,
        # but not for others, e.g., LibEntry, since it's not compatible with AsyncCompileMode
        self.compile_parallel = (isinstance(self.fn, triton.runtime.JITFunction)
                                 and os.getenv("TRITON_AUTOTUNE_PARALLEL_COMPILE", "1") == "1")

    def _expand_simt_num_warps_configs(self, base_configs: List[Config]) -> List[Config]:
        _default_cand_num_warps = [8, 16, 32, 64]
        cand_num_warps = (_default_cand_num_warps if self.user_specified_warps is None else [self.user_specified_warps])

        simt_configs = []
        for base_cfg in base_configs:
            for num_warps in cand_num_warps:
                new_cfg = copy.deepcopy(base_cfg)
                new_cfg.num_warps = num_warps
                simt_configs.append(new_cfg)

        if self.print_autotuning:
            print(f"Triton autotuning: Expanded to {len(simt_configs)} SIMT configs (with warps: {cand_num_warps})")
        return simt_configs

    def _expand_simd_multibuffer_configs(self, base_configs: List[Config]) -> List[Config]:
        if self.user_specified_multibuffer is not None:
            if self.print_autotuning:
                print("Triton autotuning: Skip SIMD multibuffer expansion because user "
                      f"specified multibuffer={self.user_specified_multibuffer}")
            return base_configs

        opposite_default_multibuffer = not self.default_multibuffer
        simd_configs = []
        for base_cfg in base_configs:
            simd_configs.append(base_cfg)
            new_cfg = copy.deepcopy(base_cfg)
            new_cfg.kwargs["multibuffer"] = opposite_default_multibuffer
            simd_configs.append(new_cfg)

        if self.print_autotuning:
            print("Triton autotuning: Expanded to "
                  f"{len(simd_configs)} SIMD configs (toggle multibuffer={opposite_default_multibuffer})")
        return simd_configs

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
        self.all_split_params = {}
        self.fixed_split_params = {}
        self.tiling_params = tiling_params
        self.low_dim_axes = low_dim_axes
        self.reduction_axes = reduction_axes
        self.fixed_grid_dims = set()
        self.fixed_grid_dim_values = {}
        self.split_axis_pid_dims = {}
        self.axis_pid_dims = {}
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

        if len(self.reduction_axes) == 1:
            reduction_axis = self.reduction_axes[0]
            reduction_param = self.keys.get(reduction_axis, None)
            reduction_numel = all_args.get(reduction_param, float("inf"))
            persistent_threshold = self._get_persistent_reduction_threshold(reduction_axis)
            if reduction_numel <= persistent_threshold:
                self.persistent_reduction = True

        if not self.split_params:
            all_split_params = self._autoparse_split_params(self._get_constexpr_candidates())
            self.all_split_params = dict(all_split_params)
            self.fixed_split_params = {}
            self.fixed_grid_dim_values = self._get_fixed_grid_dim_values(
                all_args.get("grid", None),
                all_args,
            )
            self.fixed_grid_dims = set(self.fixed_grid_dim_values.keys())

            fixed_grid_axes = {axis for axis, pid_dim in self.axis_pid_dims.items() if pid_dim in self.fixed_grid_dims}

            # Only missing constexpr params are tunable, and fixed-grid axes
            # should not be tuned on split.
            self.split_params = {
                axis: param
                for axis, param in all_split_params.items()
                if param in miss_params and axis not in fixed_grid_axes
            }

            # Fixed split is inferred only from fixed grid dims.
            for axis, pid_dim in self.axis_pid_dims.items():
                if pid_dim not in self.fixed_grid_dims:
                    continue
                core_num = self.fixed_grid_dim_values.get(pid_dim, 0)
                axis_len_name = self.keys.get(axis, None)
                axis_len = all_args.get(axis_len_name, None)
                if not isinstance(core_num, int) or core_num <= 0:
                    continue
                if not isinstance(axis_len, int) or axis_len <= 0:
                    continue

                self.fixed_split_params[axis] = (axis_len + core_num - 1) // core_num
        elif not self.axis_pid_dims:
            # When split axes are provided by hints, parse axis->program_id mapping
            # independently for fixed-grid semantics and diagnostics.
            self._autoparse_axis_pid_dims()
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
            self.fixed_split_params,
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
            self.gen_configs = self._expand_simt_num_warps_configs(self.gen_configs)
        else:
            self.gen_configs = self._expand_simd_multibuffer_configs(self.gen_configs)

        if len(self.gen_configs) == 0:
            print("[WARNING] The generated candidate tiling configs are empty based on provided parameters!")

        if self.print_autotuning:
            print("Generated configs number: {}".format(len(self.gen_configs)))

    def generate_key_and_configs(self, *args, **kwargs):
        self.nargs = dict(zip(self.arg_names, args))
        self.is_simt_mode = kwargs.get('force_simt_only', False)
        if 'num_warps' in kwargs and kwargs['num_warps'] is not None:
            self.user_specified_warps = kwargs['num_warps']
        else:
            self.user_specified_warps = None
        if 'multibuffer' in kwargs and kwargs['multibuffer'] is not None:
            self.user_specified_multibuffer = kwargs['multibuffer']
        else:
            self.user_specified_multibuffer = None

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
        if self.is_simt_mode and kwargs.get('simt_stack_limit', None) is None:
            kwargs['simt_stack_limit'] = self.simt_stack_limit
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

        if self.compile_parallel:
            import psutil

            max_workers = min(psutil.cpu_count(logical=False) // 2, len(kernels_call))
            future_kernels = []
            try:
                with (
                        ThreadPoolExecutor(max_workers=max_workers) as executor,
                        triton.AsyncCompileMode(executor),
                ):
                    for config, fn in kernels_call.items():
                        future_kernels.append((config, fn(warmup=True)))

                    for config, fut in future_kernels:
                        try:
                            if hasattr(fut, "result"):
                                fut = fut.result()
                            run_fns[config] = functools.partial(kernels_call[config], warmup=False)
                        except (CompileTimeAssertionFailure, MLIRCompilationError) as e:
                            import traceback
                            exc_stack = traceback.format_exc()
                            exc = e
            except Exception as e:
                # ignore exception from __exit__() of AsyncCompileMode
                triton.runtime._async_compile.active_mode.set(None)
        else:
            for config, fn in kernels_call.items():
                try:
                    fn(warmup=False)
                    run_fns[config] = functools.partial(fn, warmup=False)
                except (CompileTimeAssertionFailure, MLIRCompilationError, OutOfResources) as e:
                    import traceback
                    exc_stack = traceback.format_exc()
                    exc = e

        if len(run_fns) == 0:
            raise RuntimeError(f"No valid triton configs. {type(exc).__name__}: {exc} \nStack trace: {exc_stack}")

        if len(run_fns) == 1:
            # we ignore expensive profiling method when only single config is left
            return {config: self.do_bench(fn, quantiles=(0.5, 0.2, 0.8)) for config, fn in run_fns.items()}

        use_profiling = os.getenv("TRITON_BENCH_METHOD", "default").lower() == "npu"
        # Respect user-provided benchmarkers even when NPU profiling mode is enabled.
        use_npu_profiling = use_profiling and not self.user_defined_do_bench
        if use_npu_profiling:
            from ..testing import do_bench_npu

            time_cost = do_bench_npu(list(run_fns.values()), clear_l2_cache=False)
            assert len(time_cost) == len(run_fns)
            return {config: cost for config, cost in zip(run_fns.keys(), time_cost)}
        else:
            return {config: self.do_bench(fn, quantiles=(0.5, 0.2, 0.8)) for config, fn in run_fns.items()}

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

        def kernel_call(warmup):
            if config.pre_hook:
                config.pre_hook(full_nargs)
            self.pre_hook(full_nargs)
            try:
                current.update({"warmup": warmup})
                res = self.fn.run(
                    *args,
                    **current,
                )
                if warmup:
                    return res
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
        if self.compile_parallel:
            import psutil

            max_workers = min(psutil.cpu_count(logical=False) // 2, len(pruned_configs))
            with (
                    ThreadPoolExecutor(max_workers=max_workers) as executor,
                    triton.AsyncCompileMode(executor),
            ):
                for config in pruned_configs:
                    ret.append(self.fn.warmup(*args, **kwargs, **config.all_kwargs()))
        else:
            for config in pruned_configs:
                ret.append(self.fn.warmup(*args, **kwargs, **config.all_kwargs()))
        self.nargs = None
        return ret

    def _profile(self, *args, config, **meta):
        from ..testing import do_bench_npu

        kernel_call = self._make_kernel_call(*args, config=config, **meta)
        fn = functools.partial(kernel_call, warmup=False)
        do_bench_npu(fn, prof_dir=self.auto_profile_dir, keep_res=True)

    def _autoparse_split_params(self, candidates_params: List[str]) -> Dict[str, str]:
        """
        Extracts the split axis parameters from triton kernel code.
        """
        func_ast = self.fn.parse()
        parser = SplitAxesParser(func_ast, self.keys, candidates_params)
        split_axes = parser.parse()
        self.split_axis_pid_dims = dict(getattr(parser, "split_axis_pid_dims", {}))
        self.axis_pid_dims = dict(getattr(parser, "axis_pid_dims", {}))
        if self.print_autotuning:
            print(f"Ascend autotuning parse split axes: {split_axes}, "
                  f"split axis pid dims: {self.split_axis_pid_dims}, "
                  f"axis pid dims: {self.axis_pid_dims}")
        return split_axes

    def _autoparse_axis_pid_dims(self) -> Dict[str, int]:
        """
        Extract axis -> program_id dim mapping without relying on split-parameter
        classification, so fixed-grid semantics can always consume it.
        """
        func_ast = self.fn.parse()
        parser = SplitAxesParser(
            func_ast,
            self.keys,
            self._get_constexpr_candidates(),
        )
        _ = parser.parse()
        self.axis_pid_dims = dict(getattr(parser, "axis_pid_dims", {}))
        self.split_axis_pid_dims = dict(getattr(parser, "split_axis_pid_dims", {}))
        if self.print_autotuning:
            print("Ascend autotuning parse axis pid dims (independent): "
                  f"{self.axis_pid_dims}")
        return self.axis_pid_dims

    def _get_constexpr_candidates(self) -> List[str]:
        """
        Returns all constexpr parameter names from the kernel function definition.
        """
        func_ast = self.fn.parse()
        constexpr_names = []
        for node in ast.walk(func_ast):
            if not isinstance(node, ast.FunctionDef):
                continue
            if not isinstance(node.args, ast.arguments):
                continue
            for arg in node.args.args:
                if not isinstance(arg, ast.arg):
                    continue
                ann = arg.annotation
                if (isinstance(ann, ast.Attribute) and isinstance(ann.value, ast.Name) and ann.value.id == "tl"
                        and ann.attr == "constexpr"):
                    constexpr_names.append(arg.arg)
            break
        return constexpr_names

    def _get_fixed_grid_dim_values(self, grid, all_args: Dict[str, object] = None) -> Dict[int, int]:
        """
        Returns fixed grid dim -> value.
        - Static tuple/list grid: direct extraction
        - Callable grid: infer fixed dims by perturbing missing constexpr params
        """
        if grid is None:
            return {}
        if callable(grid):
            return self._infer_fixed_dims_from_callable_grid(grid, all_args or {})
        return self._extract_fixed_grid_dims(grid)

    def _extract_fixed_grid_dims(self, grid) -> Dict[int, int]:
        if isinstance(grid, int):
            grid = (grid, )
        if not isinstance(grid, (tuple, list)):
            return {}
        fixed_dims = {}
        for idx, dim in enumerate(grid):
            if isinstance(dim, int) and dim > 0:
                fixed_dims[idx] = dim
        return fixed_dims

    def _normalize_grid_tuple(self, grid_out):
        if isinstance(grid_out, int):
            return (grid_out, )
        if isinstance(grid_out, (tuple, list)):
            return tuple(grid_out)
        return None

    def _infer_fixed_dims_from_callable_grid(self, grid_fn, all_args: Dict[str, object]) -> Dict[int, int]:
        constexpr_candidates = self._get_constexpr_candidates()
        base_meta = dict(all_args or {})

        # Fill missing constexpr with stable probe defaults so grid(meta) can execute.
        for name in constexpr_candidates:
            if name not in base_meta:
                base_meta[name] = 128

        try:
            base_grid_raw = grid_fn(dict(base_meta))
        except Exception:
            return {}

        base_grid = self._normalize_grid_tuple(base_grid_raw)
        if base_grid is None:
            return {}

        dynamic_dims = set()
        # Missing constexpr are tunable candidates.
        tunable_probe_names = [name for name in constexpr_candidates if name not in (all_args or {})]
        probe_values = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]

        for name in tunable_probe_names:
            baseline = base_meta.get(name, 128)
            for probe in probe_values:
                if probe == baseline:
                    continue
                probe_meta = dict(base_meta)
                probe_meta[name] = probe
                try:
                    probe_grid_raw = grid_fn(probe_meta)
                except Exception:
                    continue
                probe_grid = self._normalize_grid_tuple(probe_grid_raw)
                if probe_grid is None:
                    continue
                if len(probe_grid) != len(base_grid):
                    dynamic_dims.update(range(min(len(probe_grid), len(base_grid))))
                    continue
                for idx, (base_dim, probe_dim) in enumerate(zip(base_grid, probe_grid)):
                    if not (isinstance(base_dim, int) and isinstance(probe_dim, int)):
                        dynamic_dims.add(idx)
                        continue
                    if base_dim != probe_dim:
                        dynamic_dims.add(idx)

        fixed_dims = {}
        for idx, dim in enumerate(base_grid):
            if idx in dynamic_dims:
                continue
            if isinstance(dim, int) and dim > 0:
                fixed_dims[idx] = dim
        return fixed_dims

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
            if self.print_autotuning:
                print("[WARNING] Failed to parse low-dimensional axes, fallback to empty low_dim_axes.")
            return []
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

    def _get_persistent_reduction_threshold(self, reduction_axis: str) -> int:
        # Keep this heuristic aligned with inductor-style policy:
        # inner reduction axis uses a larger threshold than other axes.
        if self.low_dim_axes and reduction_axis == self.low_dim_axes[0]:
            return 1024
        return 64


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


_ALL_PARAMS = {
    "num_stages",
    "unit_flag",
    "limit_auto_multi_buffer_only_for_local_buffer",
    "limit_auto_multi_buffer_of_local_buffer",
    "set_workspace_multibuffer",
    "enable_hivm_auto_cv_balance",
    "tile_mix_vector_loop",
    "tile_mix_cube_loop",
    "enable_ubuf_saving",
}

_DEFAULTS = {
    "num_stages": [2],
    "unit_flag": [False],
    "limit_auto_multi_buffer_only_for_local_buffer": [False],
    "limit_auto_multi_buffer_of_local_buffer": ["no-l0c"],
    "set_workspace_multibuffer": [2, 4],
    "enable_hivm_auto_cv_balance": [True],
    "tile_mix_vector_loop": [2, 4],
    "tile_mix_cube_loop": [2, 4],
    "enable_ubuf_saving": [True],
}

_VALID_VALUES = {
    "num_stages": [1, 2],
    "limit_auto_multi_buffer_of_local_buffer": ["no-limit", "no-l0c"],
    "set_workspace_multibuffer": [2, 4],
    "tile_mix_vector_loop": [2, 4, 8],
    "tile_mix_cube_loop": [2, 4, 8],
}

_CUBE_PARAMS = {"num_stages", "unit_flag", "limit_auto_multi_buffer_of_local_buffer"}

_MIXCV_PARAMS = {
    "num_stages",
    "unit_flag",
    "limit_auto_multi_buffer_only_for_local_buffer",
    "limit_auto_multi_buffer_of_local_buffer",
    "set_workspace_multibuffer",
    "enable_hivm_auto_cv_balance",
    "tile_mix_vector_loop",
    "tile_mix_cube_loop",
    "enable_ubuf_saving",
}

_VECTOR_PARAMS = {
    "num_stages",
    "enable_ubuf_saving",
}


def _check_boolean_list(val, param_name):
    return isinstance(val, (list, tuple)) and len(val) > 0 and all(isinstance(x, bool) for x in val)


def _check_string_in_set(val, valid_set, param_name):
    return isinstance(val, (list, tuple)) and len(val) > 0 and all(v in valid_set for v in val)


def _check_int_in_set(val, valid_set, param_name):
    return isinstance(val, (list, tuple)) and len(val) > 0 and all(isinstance(v, int) and v in valid_set for v in val)


_VALIDATION_RULES = {
    "num_stages": {
        "desc": f"must be one or more of: {_VALID_VALUES['num_stages']}", "check":
        lambda val, p: _check_int_in_set(val, _VALID_VALUES['num_stages'], p)
    },
    "unit_flag": {"desc": "must be non-empty list/tuple of boolean values", "check": _check_boolean_list},
    "limit_auto_multi_buffer_only_for_local_buffer":
    {"desc": "must be non-empty list/tuple of boolean values", "check": _check_boolean_list},
    "limit_auto_multi_buffer_of_local_buffer": {
        "desc": f"must be one or more of: {_VALID_VALUES['limit_auto_multi_buffer_of_local_buffer']}", "check":
        lambda val, p: _check_string_in_set(val, _VALID_VALUES['limit_auto_multi_buffer_of_local_buffer'], p)
    },
    "set_workspace_multibuffer": {
        "desc": f"must be one or more of: {_VALID_VALUES['set_workspace_multibuffer']}", "check":
        lambda val, p: _check_int_in_set(val, _VALID_VALUES['set_workspace_multibuffer'], p)
    },
    "enable_hivm_auto_cv_balance":
    {"desc": "must be non-empty list/tuple of boolean values", "check": _check_boolean_list},
    "tile_mix_vector_loop": {
        "desc": f"must be one or more of: {_VALID_VALUES['tile_mix_vector_loop']}", "check":
        lambda val, p: _check_int_in_set(val, _VALID_VALUES['tile_mix_vector_loop'], p)
    },
    "tile_mix_cube_loop": {
        "desc": f"must be one or more of: {_VALID_VALUES['tile_mix_cube_loop']}", "check":
        lambda val, p: _check_int_in_set(val, _VALID_VALUES['tile_mix_cube_loop'], p)
    },
    "enable_ubuf_saving": {"desc": "must be non-empty list/tuple of boolean values", "check": _check_boolean_list},
}


class BaseAutotuner:
    """
    Base class for generating auto-tuning configurations without block dimensions.
    Users must provide fixed dimension parameters when calling the kernel.
    """

    def __init__(self, operator_name, supported_params, default_params, validation_rules):
        self.operator_name = operator_name
        self.supported_params = supported_params
        self.default_params = default_params
        self.validation_rules = validation_rules

    def validate_parameters(self, **kwargs):
        # Check for unsupported parameters
        invalid_params = [k for k in kwargs.keys() if k not in _ALL_PARAMS]
        if invalid_params:
            print(f"[ERROR] Invalid parameters for {self.operator_name}: {invalid_params}")
            return False

        for param, rule in self.validation_rules.items():
            if param in kwargs:
                if not rule["check"](kwargs[param], param):
                    print(f"[ERROR] Invalid value for '{param}' in {self.operator_name}: {kwargs[param]}")
                    print(f"        Expected: {rule['desc']}")
                    return False
        return True

    def get_configs(self, **kwargs):
        """
        Generate a list of Config objects.
        Each parameter must be provided as a list (even for a single value).
        The function produces the Cartesian product of all parameter lists.
        - num_stages: each value will be set as Config.num_stages (not placed in kwargs)
        - other parameters: each value will be placed in Config.kwargs
        Returns a list of Config objects.
        """
        if not self.validate_parameters(**kwargs):
            return []

        # Collect parameter values, using defaults for missing ones
        param_values = {}
        for p in sorted(self.supported_params):
            if p in kwargs:
                param_values[p] = kwargs[p]
            else:
                param_values[p] = self.default_params.get(p, [None])

        keys = list(param_values.keys())
        values = list(param_values.values())
        combos = list(itertools.product(*values))

        configs = []
        for combo in combos:
            config_kwargs = {}
            num_stages_val = None
            for i, pname in enumerate(keys):
                val = combo[i]
                if pname == "num_stages":
                    num_stages_val = val
                else:
                    config_kwargs[pname] = val

            configs.append(Config(kwargs=config_kwargs, num_stages=num_stages_val if num_stages_val is not None else 2))
        return configs


CubeAutotuner = BaseAutotuner(operator_name="cube", supported_params=_CUBE_PARAMS, default_params=_DEFAULTS,
                              validation_rules=_VALIDATION_RULES)

MixcvAutotuner = BaseAutotuner(operator_name="mixcv", supported_params=_MIXCV_PARAMS, default_params=_DEFAULTS,
                               validation_rules=_VALIDATION_RULES)

VectorAutotuner = BaseAutotuner(operator_name="vector", supported_params=_VECTOR_PARAMS, default_params=_DEFAULTS,
                                validation_rules=_VALIDATION_RULES)


def get_autotune_cube_config(**kwargs: Any) -> List[triton.Config]:
    """
    Generate autotune configuration for the cube operator.
    Supported parameters: num_stages, unit_flag, limit_auto_multi_buffer_of_local_buffer.
    """
    import triton
    return CubeAutotuner.get_configs(**kwargs)


def get_autotune_cv_config(**kwargs: Any) -> List[triton.Config]:
    """
    Generate autotune configuration for the mixcv operator.
    Supported parameters: num_stages, unit_flag, limit_auto_multi_buffer_only_for_local_buffer,
                limit_auto_multi_buffer_of_local_buffer, set_workspace_multibuffer,
                enable_hivm_auto_cv_balance, tile_mix_vector_loop, tile_mix_cube_loop, enable_ubuf_saving
    """
    import triton
    return MixcvAutotuner.get_configs(**kwargs)


def get_autotune_vector_config(**kwargs: Any) -> List[triton.Config]:
    """
    Generate autotune configuration for the vector operator.
    Supported parameters: num_stages, enable_ubuf_saving
    """
    import triton
    return VectorAutotuner.get_configs(**kwargs)


def get_max_configs(config, kernel_type="mixcv", **kwargs):
    """
    Expand a single base Config by combining it with tuning parameters.

    :param config: A triton.Config object serving as the base.
    :param kernel_type: Operator type, one of "cube", "mixcv", "vector". Default "mixcv".
    :param kwargs: Tuning parameters, each provided as a list (e.g., enable_hivm_auto_cv_balance=[True, False]).
                   If a parameter is not provided, its value is taken from the base config (if present)
                   or from the defaults.
    :return: List of expanded Config objects.
    """
    # Determine the set of parameters supported by the current kernel_type
    if kernel_type == "cube":
        supported = _CUBE_PARAMS
    elif kernel_type == "vector":
        supported = _VECTOR_PARAMS
    else:
        supported = _MIXCV_PARAMS

    # Warn about unsupported parameters provided in kwargs
    unsupported = [k for k in kwargs if k not in supported and k in _ALL_PARAMS]
    if unsupported:
        print(
            f"[WARNING] The following parameters are not supported for kernel_type '{kernel_type}': {unsupported}. They will be ignored."
        )

    # Build value lists for each parameter (priority: kwargs > base config > defaults)
    param_values = {}
    base_kwargs = config.kwargs
    base_num_stages = config.num_stages

    for param in sorted(supported):
        if param in kwargs:
            # User-provided list via tuning_params takes precedence
            val_list = kwargs[param]
        elif param == "num_stages":
            # num_stages is an attribute of Config, not part of kwargs.
            # If not provided in tuning_params, use the value from the base config as a fixed single-element list.
            val_list = [base_num_stages]
        elif param in base_kwargs:
            # Parameter present in base config's kwargs -> fix to that single value
            val_list = [base_kwargs[param]]
        else:
            # Otherwise fall back to defaults
            val_list = _DEFAULTS.get(param, [None])

        # Validate the value list
        if param in _VALIDATION_RULES:
            rule = _VALIDATION_RULES[param]
            if not rule["check"](val_list, param):
                raise ValueError(f"Invalid value for '{param}': {val_list}. Expected: {rule['desc']}")
        param_values[param] = val_list

    # Cartesian product of all parameter lists
    keys = list(param_values.keys())
    values = list(param_values.values())
    combos = list(itertools.product(*values))

    new_configs = []
    for combo in combos:
        # Start with a copy of the original config's kwargs
        new_kwargs = config.kwargs.copy()
        num_stages_val = None

        for i, pname in enumerate(keys):
            val = combo[i]
            if pname == "num_stages":
                num_stages_val = val
            else:
                # Overwrite or add the parameter to kwargs
                new_kwargs[pname] = val

        new_config = Config(kwargs=new_kwargs, num_warps=config.num_warps,
                            num_stages=num_stages_val if num_stages_val is not None else config.num_stages,
                            num_ctas=config.num_ctas, num_buffers_warp_spec=config.num_buffers_warp_spec,
                            num_consumer_groups=config.num_consumer_groups, reg_dec_producer=config.reg_dec_producer,
                            reg_inc_consumer=config.reg_inc_consumer, maxnreg=config.maxnreg, pre_hook=config.pre_hook)
        new_configs.append(new_config)

    return new_configs


def max_autotune(configs, key, kernel_type="mixcv", prune_configs_by=None, reset_to_zero=None, restore_value=None,
                 pre_hook=None, post_hook=None, warmup=None, rep=None, use_cuda_graph=False, do_bench=None,
                 **tuning_params):
    """
    Decorator that expands each base Config with tuning parameters before auto-tuning.

    Usage is similar to @triton.autotune, but allows automatic expansion of
    additional tuning parameters (e.g., enable_hivm_auto_cv_balance, tile_mix_vector_loop, ...)
    for each provided base configuration.

    :param configs: List of base triton.Config objects.
    :param key: List of argument names whose change triggers re-tuning.
    :param kernel_type: Operator type, one of "cube", "mixcv", "vector". Default "mixcv".
    :param prune_configs_by: Same as in autotune.
    :param reset_to_zero: Same as in autotune.
    :param restore_value: Same as in autotune.
    :param pre_hook: Same as in autotune.
    :param post_hook: Same as in autotune.
    :param warmup: Deprecated.
    :param rep: Deprecated.
    :param use_cuda_graph: Deprecated.
    :param do_bench: Same as in autotune.
    :param tuning_params: Additional tuning parameters as keyword arguments.
                          Each value must be a list; the Cartesian product of these lists
                          will be combined with each base config.
    """

    def decorator(fn):
        if not configs or len(configs) == 0:
            raise ValueError("[max_autotune] The argument 'configs' cannot be empty. "
                             "Please provide at least one base config. ")
        # Expand each base config with the provided tuning parameters
        expanded_configs = []
        for cfg in configs:
            expanded = get_max_configs(cfg, kernel_type=kernel_type, **tuning_params)
            expanded_configs.extend(expanded)

        # Call the original autotune decorator with the expanded configs
        return autotune(configs=expanded_configs, key=key, prune_configs_by=prune_configs_by,
                        reset_to_zero=reset_to_zero, restore_value=restore_value, pre_hook=pre_hook,
                        post_hook=post_hook, warmup=warmup, rep=rep, use_cuda_graph=use_cuda_graph,
                        do_bench=do_bench)(fn)

    return decorator
