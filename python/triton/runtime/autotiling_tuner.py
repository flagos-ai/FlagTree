from __future__ import annotations

import builtins
import os
import time
from typing import Dict, List

from .autotuner import Autotuner, Config
from .utils import get_byte_per_numel, is_valid_axis_name, valid_axis_names
from .autoparser import SplitAxesParser, TilingAxesParser, LowDimsAxesParser, PtrNumsParser


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
        auto_profile_dir=None
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
            auto_profile_dir,
        )

        if not configs:
            self.user_configs = []
        else:
            self.user_configs = configs
        self.gen_configs = []  # generated configs from TileGenerator

        self.split_params = None
        self.tiling_params = None
        self.low_dims = None
        self.dual_reduction = False
        self.persistent_reduction = False
        self.input_ptr_num = 0
        if len(key) > len(valid_axis_names):
            raise ValueError("Number of parameters exceeds the number of available axes.")
        self.keys = {axis: param for axis, param in zip(valid_axis_names, key)}
        self.print_autotuning = os.getenv("TRITON_PRINT_AUTOTUNING", None) == "1"

    def _gen_tile_configs(
        self, kv_dict: Dict[str, int], dtype: torch.dtype
    ) -> List[Config]:
        from .tile_generator import KernelMeta, TileGenerator

        axis_sizes = {}
        for k, v in kv_dict.items():
            if not is_valid_axis_name(k):
                continue
            if not isinstance(v, int):
                raise ValueError(
                    f"Not supported dim type: {type(v)}, `int` is the only supported type"
                )
            axis_sizes[k] = v

        kernel_meta = KernelMeta(
            axis_sizes,
            self.split_params,
            self.tiling_params,
            self.low_dims,
            dtype,
            self.persistent_reduction,
            self.dual_reduction,
            self.input_ptr_num,
        )
        tile_gen = TileGenerator(kernel_meta=kernel_meta)
        tile_gen.descend_split_tiling()

        self.gen_configs.clear()
        self.gen_configs = tile_gen.configs
        if len(self.gen_configs) == 0:
            print(
                "[WARNING] The generated candidate tiling configs are empty based on provided parameters!"
            )

        if len(self.gen_configs) == 0 and len(self.user_configs) == 0:
            return [
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
            return self.gen_configs + self.user_configs

    def run(self, *args, **kwargs):
        self.nargs = dict(zip(self.arg_names, args))
        used_cached_result = True

        # generate key
        all_args = {**self.nargs, **kwargs}
        _args = {k: v for (k, v) in all_args.items() if k in self.arg_names}
        _kv_dict = {k: _args[v] for k, v in self.keys.items() if v in _args}
        key = list(_kv_dict.values())

        # Currently, we use the dtype with maximum byte length
        dtype = None
        for _, arg in _args.items():
            if hasattr(arg, "dtype"):
                key.append(str(arg.dtype))
                dtype = (
                    arg.dtype
                    if get_byte_per_numel(arg.dtype) >= get_byte_per_numel(dtype)
                    else dtype
                )
        if dtype is None:
            raise NotImplementedError("Not support for non-Tensor inputs")

        key = tuple(key)
        if key not in self.cache:
            miss_params = [arg for arg in self.arg_names if arg not in all_args.keys()]
            # parse pointer params nums
            self.input_ptr_nums = self.autoparse_ptr_nums(miss_params)

            # parse autotiling axes
            if not self.low_dims:
                self.low_dims = self.autoparse_low_dims()
            if not self.split_params:
                self.split_params = self.autoparse_split_params(miss_params)
            miss_params = [arg for arg in miss_params if arg not in self.split_params.values()]
            if not self.tiling_params:
                self.tiling_params = self.autoparse_tiling_params(miss_params)
            miss_params = [arg for arg in miss_params if arg not in self.tiling_params.values()]
            if miss_params:
                raise ValueError(
                    f"Missing required arguments: {miss_params}. "
                    f"These arguments must be explicitly provided and cannot be automatically tuned. "
                    f"Please ensure that these arguments are passed when calling the function."
                )

            # prune configs
            self.configs = self._gen_tile_configs(_kv_dict, dtype)
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
        if os.getenv("TRITON_PRINT_AUTOTUNING", None) == "1" and not used_cached_result:
            print(
                f"Triton autotuning for function {self.base_fn.__name__} finished after "
                f"{self.bench_time:.2f}s; best config selected: {self.best_config};"
            )

        if not used_cached_result and self.auto_profile_dir is not None:
            self._profile(*args, config=self.best_config, **kwargs)
        if config.pre_hook is not None:
            full_nargs = {**self.nargs, **kwargs, **config.all_kwargs()}
            config.pre_hook(full_nargs)
        ret = self.fn.run(
            *args,
            **kwargs,
            **config.all_kwargs(),
        )
        self.nargs = None
        return ret

    def autoparse_split_params(self, candidates_params: List[str]) -> Dict[str, str]:
        """
        Extracts the split axis parameters from triton kernel code. 
        """
        if self.print_autotuning:
            print(f"Triton autotuning: Starting split params parsing...")
        func_ast = self.fn.parse()
        parser = SplitAxesParser(func_ast, self.keys, candidates_params)
        split_axes = parser.parse()
        if self.print_autotuning:
            print(
                f"Triton autotuning: Split params parsing complete. "
                f"Split params: {split_axes}"
            )
        return split_axes
    
    def autoparse_tiling_params(self, candidates_params: List[str]) -> Dict[str, str]:
        """
        Extracts the tiling axis parameters from triton kernel code.
        """
        if self.print_autotuning:
            print(f"Triton autotuning: Starting tiling params parsing...")
        func_ast = self.fn.parse()
        parser = TilingAxesParser(func_ast, self.keys, candidates_params)
        tiling_axes = parser.parse()
        if self.print_autotuning:
            print(
                f"Triton autotuning: Tiling params parsing complete. "
                f"Tiling params: {tiling_axes}"
            )
        return tiling_axes
    
    def autoparse_low_dims(self) -> List[str]:
        """
        Extracts the low dimension axis from triton kernel code.
        """
        if self.print_autotuning:
            print(f"Triton autotuning: Starting Low dims axes parsing...")
        func_ast = self.fn.parse()
        parser = LowDimsAxesParser(func_ast, self.keys)
        low_dims = parser.parse()
        if self.print_autotuning:
            print(
                f"Triton autotuning: Low dims axes parsing complete. "
                f"Keys: {self.keys}, Low dims: {low_dims}"
            )
        return low_dims
    
    def autoparse_ptr_nums(self, miss_params: List[str]) -> int:
        """
        Counts the number of pointer parameters from triton kernel code.
        """
        if self.print_autotuning:
            print(f"Triton autotuning: Starting ptr nums parsing...")
        func_ast = self.fn.parse()
        parser = PtrNumsParser(func_ast, miss_params)
        ptr_nums, ptr_params = parser.parse()
        if self.print_autotuning:
            print(
                f"Triton autotuning: Pointer nums parsing complete. "
                f"Pointer params: {ptr_params}, pointer nums: {ptr_nums}"
            )
        return ptr_nums