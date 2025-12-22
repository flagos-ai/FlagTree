from __future__ import annotations

import builtins
import os
import time
from typing import Dict, List

from .autotuner import Autotuner, Config
from .utils import get_byte_per_numel, is_valid_axis_name


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
        split_params=None,
        tiling_params=None,
        low_dims=None,
        dual_reduction=False,
        persistent_reduction=False,
    ):
        """
        :param key: a dict of axis name: argument name, where the change of arguments in value will triger re-generating candidates configs and evaluating.
            The axis name should be in {'x','y','z','w','v','t','rx','ry','rz','rw','rv','rt}, where the prefix 'r' means a reduction axis.
            Only the axis name in this param should add perfix 'r' if it's a reduction axis.
        :type key: Dict[str, str]
        :param split_params: a dict of axis name: argument name, the argument is an adjustable parameter in a split axis, such as 'XBLOCK'.
            The axis name must be in key's axis names. Do not add prefix 'r' before the axis name.
            This param can be empty. Note that the auto tiling feature will be disabled when the split_params and tiling_params are both empty.
            The split axis can usually be identified according to `tl.program_id()` expression.
        :type split_params: Dict[str, str]
        :param tiling_params: a dict of axis name: argument name, the argument is an adjustable parameter in a tiling axis, such as 'XBLOCK_SUB'.
            The axis name must be in key's axis names. Do not add prefix 'r' before the axis name.
            This param can be empty. Note that the auto tiling feature will be disabled when the split_params and tiling_params are both empty.
            The tiling axis can usually be identified according to `tl.arange()` expression.
        :type tiling_params: Dict[str, str]
        :param low_dims: a list of axis name in which the corresponding axis is low dim aixs.
            The axis name must be in key's axis names. Do not add prefix 'r' before the axis name.
        :type low_dims: List[str]
        :param dual_reduction: performing reduction on more than one axis.
        :param persistent_reduction: there is no splitting in reduction axis.
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

        self.split_params = split_params
        self.tiling_params = tiling_params
        self.low_dims = low_dims
        self.dual_reduction = dual_reduction
        self.persistent_reduction = persistent_reduction

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
        )
        tile_gen = TileGenerator(kernel_meta=kernel_meta)
        tile_gen.descend_split_tiling()

        self.gen_configs.clear()
        self.gen_configs = list(tile_gen.configs.values())
        if len(self.gen_configs) == 0:
            print(
                "[WARNING] The generated candidate tiling configs are empty based on provided parameters!"
            )

        if len(self.gen_configs) == 0 and len(self.user_configs) == 0:
            return [
                Config({})
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
            # prune configs
            self.configs = self._gen_tile_configs(_kv_dict, dtype)
            pruned_configs = self.prune_configs(kwargs)
            if len(pruned_configs) > 1:
                used_cached_result = False
                bench_start = time.time()
                timings = {
                    config: self._bench(*args, config=config, **kwargs)
                    for config in pruned_configs
                }
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

        # flagtree backend specialization
        from triton.runtime.driver import flagtree_backend_specialization
        flagtree_backend_specialization('ext_Autotuner_profile', self, used_cached_result, args, kwargs)
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
