from __future__ import annotations

import builtins
import time
import inspect
import hashlib
import json
from functools import cached_property
from typing import Dict, Tuple, List, Optional

from .. import knobs
from .jit import KernelInterface, JITFunction
from .errors import OutOfResources, PTXASError, AutotunerError
from .driver import driver
from .cache import get_cache_manager, triton_key
from .adjust_kernel_param import auto_adjust_block_sizes
from triton._C.libtriton import get_cache_invalidating_env_vars


class Autotuner(KernelInterface):

    def __init__(self, fn, arg_names, configs, key, reset_to_zero, restore_value, pre_hook=None, post_hook=None,
                 prune_configs_by: Optional[Dict] = None, warmup=None, rep=None, use_cuda_graph=False, do_bench=None,
                 cache_results=False, generate_configs=None, op_affiliation="", row_sign="", col_sign="",
                 n_elem_sign=""):
        """
        :param prune_configs_by: a dict of functions that are used to prune configs, fields:
            'perf_model': performance model used to predicate running time with different configs, returns running time
            'top_k': number of configs to bench
            'early_config_prune': a function used to prune configs. It should have the signature
                `prune_configs_by( configs: List[triton.Config], named_args: Dict[str, Any], **kwargs: Dict[str, Any]) -> List[triton.Config]:`
                and return pruned configs. It should return at least one config.
        """
        if not configs:
            self.configs = [Config({}, num_warps=4, num_stages=2, num_ctas=1)]
        else:
            self.configs = configs
        # XPU-specific (Triton 3.0 compat) auto-config generation hooks. When
        # generate_configs is set with an empty configs list, block_size_candidates
        # synthesises shape-dependent configs at run() time.
        self.no_configs = (not configs)
        self.generate_configs = generate_configs
        self.op_affiliation = op_affiliation
        self.row_sign = row_sign
        self.col_sign = col_sign
        self.n_elem_sign = n_elem_sign
        if self.configs and (len(self.configs) > 0):
            self.shared_config_pre_hook = self.configs[0].pre_hook  # flagtree aabs
        self.keys = key
        self.cache: Dict[Tuple, Config] = {}
        self.arg_names = arg_names
        self.cache_results = (cache_results or knobs.autotuning.cache) and not knobs.runtime.interpret

        # Reset to zero or restore values
        self.reset_to_zero = []
        if reset_to_zero is not None:
            self.reset_to_zero = list(reset_to_zero)
        self.restore_value = []
        if restore_value is not None:
            self.restore_value = list(restore_value)

        # Hook to reset or restore for required tensors
        self.pre_hook = lambda kwargs, reset_only=False: 0
        self.post_hook = lambda kwargs, exception: 0
        self.user_defined_pre_hook = False
        self.user_defined_post_hook = False
        if pre_hook:
            self.pre_hook = pre_hook
            self.user_defined_pre_hook = True
        elif (len(self.reset_to_zero) > 0 or len(self.restore_value) > 0):

            def _pre_hook(kwargs, reset_only=False):
                for name in self.reset_to_zero:
                    kwargs[name].zero_()
                if not reset_only:
                    self.restore_copies = {name: kwargs[name].clone() for name in self.restore_value}

            self.pre_hook = _pre_hook

        if post_hook:
            self.post_hook = post_hook
            self.user_defined_post_hook = True
        elif len(self.restore_value) > 0:

            def _post_hook(kwargs, exception):
                for name in self.restore_value:
                    kwargs[name].copy_(self.restore_copies[name])
                self.restore_copies = {}

            self.post_hook = _post_hook

        self.perf_model = None
        self.configs_top_k = 1.0
        self.early_config_prune = None
        if prune_configs_by:
            self.perf_model = prune_configs_by.get("perf_model", self.perf_model)
            self.configs_top_k = prune_configs_by.get("top_k", self.configs_top_k)
            self.early_config_prune = prune_configs_by.get("early_config_prune", self.early_config_prune)

        self.fn = fn
        self.base_fn = fn
        while not inspect.isfunction(self.base_fn):
            self.base_fn = self.base_fn.fn

        self._do_bench = do_bench
        self.num_warmups = warmup
        self.num_reps = rep
        self.use_cuda_graph = use_cuda_graph
        self.seen_tuned_metas = {}  # flagtree aabs: deduplicate tuned meta

        # If we got explicitly called via the old interface, raise a warning
        # and proceed with the old behavior.
        if warmup is not None or rep is not None or use_cuda_graph:
            import warnings
            warnings.warn(("warmup, rep, and use_cuda_graph parameters are deprecated. See "
                           "https://github.com/triton-lang/triton/pull/4496 for details."), DeprecationWarning,
                          stacklevel=1)
            if use_cuda_graph:
                from ..testing import do_bench_cudagraph
                self._do_bench = lambda kernel_call, quantiles: do_bench_cudagraph(
                    kernel_call,
                    rep=rep if rep is not None else 100,
                    quantiles=quantiles,
                )
                return

            import triton.testing
            self._do_bench = lambda kernel_call, quantiles: triton.testing.do_bench(
                kernel_call,
                warmup=warmup if warmup is not None else 25,
                rep=rep if rep is not None else 100,
                quantiles=quantiles,
            )
            return

    @cached_property
    def do_bench(self):
        if self._do_bench is None:
            return driver.active.get_benchmarker()
        return self._do_bench

    def _bench(self, *args, config, **meta):
        from ..compiler.errors import CompileTimeAssertionFailure

        verbose = knobs.autotuning.print
        if verbose:
            print(f"Autotuning kernel {self.base_fn.__name__} with config {config}")

        # check for conflicts, i.e. meta-parameters both provided
        # as kwargs and by the autotuner
        conflicts = meta.keys() & config.kwargs.keys()
        if conflicts:
            raise ValueError(f"Conflicting meta-parameters: {', '.join(conflicts)}."
                             " Make sure that you don't re-define auto-tuned symbols.")
        # augment meta-parameters with tunable ones
        current = dict(meta, **config.all_kwargs())
        # flagtree aabs: auto_adjust_block_sizes
        if knobs.autotuning.adjust_block_size:

            def _unwrap_to_jitfunction(fn):
                from triton.runtime.jit import JITFunction
                while not isinstance(fn, JITFunction):
                    if not hasattr(fn, 'fn'):
                        return None
                    fn = fn.fn
                return fn

            jit_fn = _unwrap_to_jitfunction(self.fn)
            if jit_fn is not None:
                auto_adjust_block_sizes(self.nargs, jit_fn, self.configs, current, config)
        meta_key = tuple(sorted(current.items()))
        if meta_key in self.seen_tuned_metas:
            return self.seen_tuned_metas[meta_key]  # flagtree aabs: deduplicate tuned meta
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

        try:
            rett = self.do_bench(kernel_call, quantiles=(0.5, 0.2, 0.8))
        except (OutOfResources, CompileTimeAssertionFailure, PTXASError) as e:
            if verbose:
                print(f"Autotuning failed with {e}")
            rett = [float("inf"), float("inf"), float("inf")]

        self.seen_tuned_metas[meta_key] = rett  # flagtree aabs: deduplicate tuned meta
        return rett

    def check_disk_cache(self, tuning_key, configs, bench_fn):
        # We can't serialize prehooks, so just give up and run the benchmarks.
        if not tuning_key or any(cfg.pre_hook for cfg in configs):
            bench_fn()
            return False

        from triton.compiler.compiler import make_backend

        fn = self.fn
        while not isinstance(fn, JITFunction):
            fn = fn.fn

        env_vars = get_cache_invalidating_env_vars()
        cache_key = [
            triton_key(),
            make_backend(driver.active.get_current_target_inside()).hash(),
            fn.cache_key,
            str(sorted(env_vars.items())),
            str(tuning_key),
        ] + [str(c) for c in configs]
        cache_key = hashlib.sha256("-".join(cache_key).encode("utf-8")).hexdigest()
        cache = get_cache_manager(cache_key)
        file_name = f"{fn.__name__[:150]}.autotune.json"
        path = cache.get_file(file_name)
        if path:
            with open(path, "r") as cached_configs:
                timings = json.load(cached_configs)["configs_timings"]
                timings = {Config(**config): timing for config, timing in timings}
                self.cache[tuning_key] = builtins.min(timings, key=timings.get)
                self.configs_timings = timings
            return True

        bench_fn()
        cache.put(
            json.dumps({
                "key":
                tuning_key,
                "configs_timings":
                [(config.__dict__, timings) for config, timings in self.configs_timings.items() if not config.pre_hook],
            }), file_name, binary=False)
        return False

    def run(self, *args, **kwargs):
        self.seen_tuned_metas = {}  # flagtree aabs: deduplicate tuned meta
        self.nargs = dict(zip(self.arg_names, args))
        # XPU (Triton 3.0 compat): synthesise configs on every invocation when
        # the user passed `configs=[]` together with `generate_configs=...`.
        # block_size_candidates is shape-dependent, so it MUST be recomputed for
        # each new shape (freezing it to the first shape would mis-tile later).
        if self.no_configs and self.generate_configs is not None:
            self.configs = block_size_candidates(self.nargs, self.generate_configs, self.op_affiliation, self.row_sign,
                                                 self.col_sign, self.n_elem_sign)
            if not self.configs:
                self.configs = [Config({}, num_warps=4, num_stages=2, num_ctas=1)]
        used_cached_result = True
        if len(self.configs) > 1:
            all_args = {**self.nargs, **kwargs}
            _args = {k: v for (k, v) in all_args.items() if k in self.arg_names}
            key = [_args[key] for key in self.keys if key in _args]
            for _, arg in _args.items():
                if hasattr(arg, "dtype"):
                    key.append(str(arg.dtype))
            key = tuple(key)
            if key not in self.cache:
                used_cached_result = False
                pruned_configs = self.prune_configs(kwargs)

                def benchmark():
                    bench_start = time.time()
                    timings = {config: self._bench(*args, config=config, **kwargs) for config in pruned_configs}
                    bench_end = time.time()
                    self.bench_time = bench_end - bench_start
                    self.cache[key] = builtins.min(timings, key=timings.get)
                    full_nargs = {**self.nargs, **kwargs, **self.cache[key].all_kwargs()}
                    self.pre_hook(full_nargs, reset_only=True)
                    self.configs_timings = timings

                if self.cache_results:
                    used_cached_result = self.check_disk_cache(key, pruned_configs, benchmark)
                else:
                    benchmark()

            config = self.cache[key]
        else:
            config = self.configs[0]
        self.best_config = config
        if knobs.autotuning.print and not used_cached_result:
            print(f"Triton autotuning for function {self.base_fn.__name__},\nwith key as {key},\n"
                  f"finished after {self.bench_time:.2f}s,\nbest config selected: {self.best_config};")
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

    def prune_configs(self, kwargs: Dict) -> List[Config]:
        # flagtree aabs: use deepcopy to prevent modification of the original configs
        import copy
        pruned_configs = copy.deepcopy(self.configs)
        # pruned_configs = self.configs
        if self.early_config_prune:
            pruned_configs = self.early_config_prune(self.configs, self.nargs, **kwargs)
            if not pruned_configs:
                raise AutotunerError(
                    "No valid autotuner configs after pruning. `early_config_prune` should return at least one config.")
        if self.perf_model:
            top_k = self.configs_top_k
            if isinstance(top_k, float) and top_k <= 1.0:
                top_k = int(len(self.configs) * top_k)
            elif not isinstance(top_k, int):
                # Slice index must be an integer
                raise TypeError("Error while pruning configs, top_k must be either 1) a float <= 1.0 or 2) an int")

            if len(pruned_configs) > top_k:
                est_timing = {
                    config: self.perf_model(
                        **self.nargs,
                        **kwargs,
                        **config.all_kwargs(),
                    )
                    for config in pruned_configs
                }
                pruned_configs = sorted(est_timing.keys(), key=lambda x: est_timing[x])[:top_k]
        return pruned_configs

    def warmup(self, *args, **kwargs):
        self.nargs = dict(zip(self.arg_names, args))
        ret = []
        for autotune_config in self.prune_configs(kwargs):
            ret.append(self.fn.warmup(
                *args,
                **kwargs,
                **autotune_config.all_kwargs(),
            ))
        self.nargs = None
        return ret


class Config:
    """
    An object that represents a possible kernel configuration for the auto-tuner to try.

    :ivar kwargs: a dictionary of meta-parameters to pass to the kernel as keyword arguments.
    :type kwargs: dict[Str, Any]
    :ivar num_warps: the number of warps to use for the kernel when compiled for GPUs. For example, if
                      `num_warps=8`, then each kernel instance will be automatically parallelized to
                      cooperatively execute using `8 * 32 = 256` threads.
    :type num_warps: int
    :ivar num_stages: the number of stages that the compiler should use when software-pipelining loops.
                       Mostly useful for matrix multiplication workloads on SM80+ GPUs.
    :type num_stages: int
    :ivar num_ctas: number of blocks in a block cluster. SM90+ only.
    :type num_ctas: int
    :type maxnreg: Optional[int]
    :ivar maxnreg: maximum number of registers one thread can use.  Corresponds
                       to ptx .maxnreg directive.  Not supported on all platforms.
    :ivar pre_hook: a function that will be called before the kernel is called. Parameters of this
                    function are args.
    :ivar ir_override: filename of a user-defined IR (*.{ttgir|llir|ptx|amdgcn}).
    """

    def __init__(self, kwargs, num_warps=4, num_stages=2, num_ctas=1, maxnreg=None, pre_hook=None, ir_override=None):
        self.kwargs = kwargs
        self.num_warps = num_warps
        self.num_ctas = num_ctas
        self.num_stages = num_stages
        self.maxnreg = maxnreg
        self.pre_hook = pre_hook
        self.ir_override = ir_override

    def __setstate__(self, state):
        self.kwargs = state.get("kwargs", {})
        self.num_warps = state.get("num_warps", 4)
        self.num_stages = state.get("num_stages", 2)
        self.num_ctas = state.get("num_ctas", 1)
        self.maxnreg = state.get("maxnreg", None)
        self.pre_hook = state.get("pre_hook", None)
        self.ir_override = state.get("ir_override", None)

    def all_kwargs(self):
        return {
            **self.kwargs, **{
                k: v
                for (k, v) in (
                    ("num_warps", self.num_warps),
                    ("num_ctas", self.num_ctas),
                    ("num_stages", self.num_stages),
                    ("maxnreg", self.maxnreg),
                    ("ir_override", self.ir_override),
                ) if v is not None
            }
        }

    def __str__(self):
        res = []
        for k, v in self.kwargs.items():
            res.append(f"{k}: {v}")
        res.append(f"num_warps: {self.num_warps}")
        res.append(f"num_ctas: {self.num_ctas}")
        res.append(f"num_stages: {self.num_stages}")
        res.append(f"maxnreg: {self.maxnreg}")
        return ", ".join(res)

    def __hash__(self):
        return hash((*self.all_kwargs().items(), self.pre_hook))

    def __eq__(self, other):
        self_tuple = tuple((
            *self.all_kwargs().items(),
            self.pre_hook,
        ))
        other_tuple = tuple((
            *other.all_kwargs().items(),
            other.pre_hook,
        ))
        return self_tuple == other_tuple


def autotune(configs, key, prune_configs_by=None, reset_to_zero=None, restore_value=None, pre_hook=None, post_hook=None,
             warmup=None, rep=None, use_cuda_graph=False, do_bench=None, cache_results=False, generate_configs=None,
             op_affiliation="sdnn", row_sign=None, col_sign=None, n_elem_sign=None):
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
        def kernel(x_ptr, x_size, BLOCK_SIZE: tl.constexpr):
            ...
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
        'early_config_prune': a function used to prune configs. It should have the signature
                `prune_configs_by( configs: List[triton.Config], named_args: Dict[str, Any], **kwargs: Dict[str, Any]) -> List[triton.Config]:`
                and return pruned configs. It should return at least one config.
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
    :param cache_results: whether to cache autotune timings to disk.  Defaults to False.
    "type cache_results: bool
    """

    def decorator(fn):
        return Autotuner(fn, fn.arg_names, configs, key, reset_to_zero, restore_value, pre_hook=pre_hook,
                         post_hook=post_hook, prune_configs_by=prune_configs_by, warmup=warmup, rep=rep,
                         use_cuda_graph=use_cuda_graph, do_bench=do_bench, cache_results=cache_results,
                         generate_configs=generate_configs, op_affiliation=op_affiliation, row_sign=row_sign,
                         col_sign=col_sign, n_elem_sign=n_elem_sign)

    return decorator


class Heuristics(KernelInterface):

    def __init__(self, fn, arg_names, values) -> None:
        self.fn = fn
        self.values = values
        self.arg_names = arg_names

    def run(self, *args, **kwargs):
        for v, heur in self.values.items():
            kwargs[v] = heur({**dict(zip(self.arg_names, args)), **kwargs})
        return self.fn.run(*args, **kwargs)


def heuristics(values):
    """
    Decorator for specifying how the values of certain meta-parameters may be computed.
    This is useful for cases where auto-tuning is prohibitively expensive, or just not applicable.

    .. highlight:: python
    .. code-block:: python

        # smallest power-of-two >= x_size
        @triton.heuristics(values={'BLOCK_SIZE': lambda args: triton.next_power_of_2(args['x_size'])})
        @triton.jit
        def kernel(x_ptr, x_size, BLOCK_SIZE: tl.constexpr):
            ...
    :param values: a dictionary of meta-parameter names and functions that compute the value of the meta-parameter.
                   each such function takes a list of positional arguments as input.
    :type values: dict[str, Callable[[dict[str, Any]], Any]]
    """

    def decorator(fn):
        return Heuristics(fn, fn.arg_names, values)

    return decorator


# ---------------------------------------------------------------------------
# XPU-only auto-config generators, ported from Triton 3.0's
# `python/triton/runtime/autotuner.py`. These are invoked from
# `Autotuner.run` when the user passed `configs=[], generate_configs=...`.
# Helpers and constants below intentionally mirror the 3.0 implementation so
# behaviour is identical for kernels written against the 3.0 fork.
# ---------------------------------------------------------------------------
import os  # noqa: E402  (kept here to localise XPU helpers)


def cdiv(x: int, y: int):
    return (x + y - 1) // y


def floordiv(x: int, y: int):
    return x // y


def aligned(x: int, y: int):
    return cdiv(x, y) * y


def next_power_of_2(n: int):
    """Return the smallest power of 2 greater than or equal to n."""
    n -= 1
    n |= n >> 1
    n |= n >> 2
    n |= n >> 4
    n |= n >> 8
    n |= n >> 16
    n |= n >> 32
    n += 1
    return n


def find_next_multiple_of_12(n):
    if n <= 0:
        return 12
    remainder = n % 12
    if remainder == 0:
        return n
    return n + (12 - remainder)


def append_candidate(candicates, target_candicate):
    for item in candicates:
        if item.all_kwargs() == target_candicate.all_kwargs():
            return
    candicates.append(target_candicate)


def check_out_of_mem(block_size_m, block_size_n, block_size_k, mem, ele_bytes, dotout_ele_bytes, bias, buffer_num,
                     a_trans, b_trans, dcache, wcache, hp_mode):
    am_layout, ak_layout = block_size_m, block_size_k
    bk_layout, bn_layout = block_size_k, block_size_n
    if a_trans:
        am_layout, ak_layout = block_size_k, block_size_m
    if b_trans:
        bn_layout, bk_layout = block_size_k, block_size_n
    min_load_a_size = 80 * aligned(block_size_k * ele_bytes, mem[1]) if not a_trans else block_size_k * (128 *
                                                                                                         ele_bytes)
    min_load_b_size = block_size_k * 64 * ele_bytes if not b_trans else 64 * aligned(block_size_k * ele_bytes, mem[1])
    cmem = mem[0] < aligned(block_size_n * dotout_ele_bytes, mem[1]) * \
        (block_size_m * (1 + hp_mode) + bias * (1.6 * block_size_m)) * buffer_num + \
        (min_load_a_size + min_load_b_size) * (2 + 2 * hp_mode)
    bmem = bn_layout * bk_layout > wcache
    amem = am_layout * ak_layout > dcache
    # Full-tile check: ensure the actual tile sizes fit in uniSram when allocated
    # by TritonSDNNMultiBuffer pass (which allocates full BM*BK, BK*BN, BM*BN tiles).
    # Also account for an output store buffer (ele_bytes, same shape as C), which some
    # kernels (e.g., fused MoE) emit as a separate loopGridForOp-level allocation.
    full_a_size = block_size_m * block_size_k * ele_bytes
    full_b_size = block_size_k * block_size_n * ele_bytes
    full_c_size = block_size_m * block_size_n * dotout_ele_bytes * (1 + hp_mode)
    full_out_size = block_size_m * block_size_n * ele_bytes  # output store (bf16/fp16)
    # At minimum numStages=1: C + out + A + B must fit; with multi-buffer add (A+B)*(buffer_num-1)
    full_tile_mem = mem[0] < full_c_size + full_out_size + (full_a_size + full_b_size) * buffer_num
    return cmem or bmem or amem or full_tile_mem


def find_optimal_block_size_k(current_k, min_block_k, ele_bytes, fn_check_out_of_mem):
    candidates = []
    for k in range(min_block_k, current_k + 1):
        if current_k % k == 0:
            candidates.append(k)
    for k in list([1024 // ele_bytes, 512 // ele_bytes]):
        if k in candidates:
            candidates.remove(k)
            candidates.append(k)
    candidates.sort()
    valid_block_k = None
    for candidate_k in candidates:
        if not fn_check_out_of_mem(candidate_k):
            valid_block_k = candidate_k
            break
    return valid_block_k


def add_candidate_for_workload_not_balanced(configs, block_size_m, block_size_n, block_size_k, buffer_num, meta_info):
    input_size = meta_info['input_size']
    mem = meta_info['mem']
    ele_bytes = meta_info['ele_bytes']
    dotout_ele_bytes = meta_info['dotout_ele_bytes']
    bias = meta_info['bias']
    block_names = meta_info['block_names']
    grid_aligned = meta_info['grid_aligned']
    aligned_size = meta_info['aligned_size']
    a_trans = meta_info['a_trans']
    b_trans = meta_info['b_trans']
    min_block_k = meta_info['min_block_k']
    dcache = meta_info["dcache"]
    wcache = meta_info["wcache"]
    hp_mode = meta_info["hp_mode"]

    grid_m_aligned = cdiv(input_size[0], block_size_m)
    grid_n_aligned = cdiv(input_size[1], block_size_n)

    if min_block_k > block_size_k:
        min_block_k = block_size_k

    top_p = 3
    block_size_m = max(2, min(block_size_m, input_size[0]))
    block_size_n = max(2, min(block_size_n, input_size[1]))

    valid_block_k = find_optimal_block_size_k(
        block_size_k, min_block_k, ele_bytes,
        lambda k: check_out_of_mem(block_size_m, block_size_n, k, mem, ele_bytes, dotout_ele_bytes, bias, buffer_num,
                                   a_trans, b_trans, dcache, wcache, hp_mode))
    if valid_block_k is None or valid_block_k < min_block_k:
        return
    block_size_k = valid_block_k

    if (grid_m_aligned * grid_n_aligned) < grid_aligned:
        tmp_grid_m = cdiv(input_size[0], block_size_m)
        tmp_grid_n = cdiv(input_size[1], block_size_n)
        append_candidate(
            configs, Config({block_names[0]: block_size_m, block_names[1]: block_size_n, block_names[2]: block_size_k}))
        for i in range(2, 13):
            tmp_block_size_m = block_size_m // i
            if tmp_block_size_m < 2:
                break
            tmp_grid_m = cdiv(input_size[0], tmp_block_size_m)
            if (tmp_grid_m * tmp_grid_n) % grid_aligned == 0:
                append_candidate(
                    configs,
                    Config(
                        {block_names[0]: tmp_block_size_m, block_names[1]: block_size_n, block_names[2]: block_size_k}))
        for i in range(2, 13):
            tmp_block_size_n = block_size_n // i
            if tmp_block_size_n < 2:
                break
            tmp_grid_n = cdiv(input_size[1], tmp_block_size_n)
            if (tmp_grid_m * tmp_grid_n) % grid_aligned == 0:
                append_candidate(
                    configs,
                    Config(
                        {block_names[0]: block_size_m, block_names[1]: tmp_block_size_n, block_names[2]: block_size_k}))
    else:
        append_candidate(
            configs, Config({block_names[0]: block_size_m, block_names[1]: block_size_n, block_names[2]: block_size_k}))
        if input_size[0] % block_size_m != 0:
            for i in range(block_size_m, 1, -1):
                _block_size_m = i
                if (cdiv(input_size[0], _block_size_m) * grid_n_aligned) % grid_aligned == 0:
                    if _block_size_m % 64 == 0:
                        append_candidate(
                            configs,
                            Config({
                                block_names[0]: _block_size_m, block_names[1]: block_size_n, block_names[2]:
                                block_size_k
                            }))
                    break
        elif input_size[1] % block_size_n != 0:
            for i in range(block_size_n, 1, -1):
                _block_size_n = i
                if (cdiv(input_size[1], _block_size_n) * grid_m_aligned) % grid_aligned == 0:
                    append_candidate(
                        configs,
                        Config(
                            {block_names[0]: block_size_m, block_names[1]: _block_size_n, block_names[2]:
                             block_size_k}))
                    break
        else:
            for i in range(block_size_m, (grid_m_aligned - 1) * aligned_size["m_aligned"] + 1, -1):
                _block_size_m = i
                for j in range(block_size_n, (grid_n_aligned - 1) * aligned_size["n_aligned"] + 1, -1):
                    _block_size_n = j
                    tmp_grid_m = cdiv(input_size[0], _block_size_m)
                    tmp_grid_n = cdiv(input_size[1], _block_size_n)
                    if (tmp_grid_m * tmp_grid_n) % grid_aligned == 0:
                        top_p -= 1
                        append_candidate(
                            configs,
                            Config({
                                block_names[0]: _block_size_m, block_names[1]: _block_size_n, block_names[2]:
                                block_size_k
                            }))
                        break
                    if top_p == 0:
                        break


def add_candidate_for_workload_balanced(configs, block_size_m, block_size_n, block_size_k, buffer_num, meta_info):
    input_size = meta_info['input_size']
    mem = meta_info['mem']
    ele_bytes = meta_info['ele_bytes']
    dotout_ele_bytes = meta_info['dotout_ele_bytes']
    bias = meta_info['bias']
    block_names = meta_info['block_names']
    a_trans = meta_info['a_trans']
    b_trans = meta_info['b_trans']
    dcache = meta_info["dcache"]
    wcache = meta_info["wcache"]
    min_block_k = meta_info["min_block_k"]
    hp_mode = meta_info["hp_mode"]
    grid_m_aligned = cdiv(input_size[0], block_size_m)
    grid_n_aligned = cdiv(input_size[1], block_size_n)

    if min_block_k > block_size_k:
        min_block_k = block_size_k
    if input_size[0] % grid_m_aligned == 0:
        block_size_m = max(2, floordiv(input_size[0], grid_m_aligned))
    if input_size[1] % grid_n_aligned == 0:
        block_size_n = max(2, floordiv(input_size[1], grid_n_aligned))

    valid_block_k = find_optimal_block_size_k(
        block_size_k, min_block_k, ele_bytes,
        lambda k: check_out_of_mem(block_size_m, block_size_n, k, mem, ele_bytes, dotout_ele_bytes, bias, buffer_num,
                                   a_trans, b_trans, dcache, wcache, hp_mode))
    if valid_block_k is None or valid_block_k < min_block_k:
        return
    block_size_k = valid_block_k
    append_candidate(configs,
                     Config({block_names[0]: block_size_m, block_names[1]: block_size_n, block_names[2]: block_size_k}))


def get_input_ele_bytes(args):
    ele_bytes = 4
    if "a_ptr" in args.keys():
        A = args["a_ptr"]
    elif "inp" in args.keys():
        A = args["inp"]
    else:
        A = args["A"]
    if A.dtype.__str__() == "torch.float16":
        ele_bytes = 2
    return ele_bytes


def balance_grid(block_size_m, block_size_n, input_size):
    grid_x = cdiv(input_size[0], block_size_m)
    grid_y = cdiv(input_size[1], block_size_n)
    total_grid = grid_x * grid_y
    next_multiple_of_12 = find_next_multiple_of_12(total_grid)
    grid_y = cdiv(next_multiple_of_12, grid_x)
    block_size_n = cdiv(input_size[1], grid_y)
    return block_size_m, block_size_n


def block_size_candidates_cluster(args, generate_configs, op_affiliation, row_sign, col_sign, n_elem_sign):
    configs = []
    if "BLOCK_SIZE" in args.keys():
        if n_elem_sign is None:
            raise RuntimeError("Failed to tune block size. Miss n_elem_sign")
        n_elements = args[n_elem_sign]
        block_size = cdiv(n_elements, 12)
        append_candidate(configs, Config({"BLOCK_SIZE": block_size}))
        block_size = next_power_of_2(cdiv(n_elements, 12))
        append_candidate(configs, Config({"BLOCK_SIZE": block_size}))
        return configs

    ele_bytes = get_input_ele_bytes(args)
    grid_aligned = 12
    BLOCK_M = "BLOCK_M"
    BLOCK_N = "BLOCK_N"
    block_names = (BLOCK_M, BLOCK_N)
    if row_sign is None or col_sign is None:
        raise RuntimeError("Failed to tune block_m/block_n size. Miss row_sign/col_sign")
    m = args[row_sign]
    n = args[col_sign]
    input_size = (m, n)
    mem = (8192, 64)
    aligned_size = {"m_aligned": 64, "n_aligned": 64}
    core_num = 64
    buffer_size_upper = 512
    if "buffer_size" in args.keys():
        buffer_size_upper = args["buffer_size"]
    buffer_size_elem_cnt = cdiv(buffer_size_upper, ele_bytes)
    experimental_fine_tune = bool(os.getenv("TRITON_FINE_AUTOTUNE", False))

    block_size_m = input_size[0]
    block_size_n = input_size[1]
    if buffer_size_elem_cnt != next_power_of_2(buffer_size_elem_cnt):
        raise RuntimeError("buffer_size should be power of two")

    if buffer_size_elem_cnt * core_num >= block_size_n:
        block_size_m = next_power_of_2(cdiv(input_size[0], 12))
        block_size_n = input_size[1]
        append_candidate(configs, Config({block_names[0]: block_size_m, block_names[1]: block_size_n}))
        block_size_m = cdiv(input_size[0], 12)
        block_size_n = input_size[1]
        append_candidate(configs, Config({block_names[0]: block_size_m, block_names[1]: block_size_n}))
        if experimental_fine_tune:
            block_size_m = next_power_of_2(cdiv(input_size[0], 12))
            block_size_n = next_power_of_2(input_size[1])
            append_candidate(configs, Config({block_names[0]: block_size_m, block_names[1]: block_size_n}))
            block_size_m = cdiv(input_size[0], 12)
            block_size_n = next_power_of_2(input_size[1])
            append_candidate(configs, Config({block_names[0]: block_size_m, block_names[1]: block_size_n}))
        return configs

    block_size_m = next_power_of_2(cdiv(input_size[0], 12))
    block_size_n = buffer_size_elem_cnt * core_num
    append_candidate(configs, Config({block_names[0]: block_size_m, block_names[1]: block_size_n}))

    for block_size_n in range(buffer_size_elem_cnt * core_num, 0, -aligned_size["n_aligned"]):
        if len(configs) == 5:
            break
        grid_x = cdiv(input_size[0], block_size_m)
        grid_y = cdiv(input_size[1], block_size_n)
        total_grid = grid_x * grid_y
        if total_grid % grid_aligned != 0:
            (block_size_m, block_size_n) = balance_grid(block_size_m, block_size_n, input_size)
            append_candidate(configs, Config({block_names[0]: block_size_m, block_names[1]: block_size_n}))
        else:
            append_candidate(configs, Config({block_names[0]: block_size_m, block_names[1]: block_size_n}))

    block_size_m = cdiv(input_size[0], 12)
    block_size_n = buffer_size_elem_cnt * core_num
    append_candidate(configs, Config({block_names[0]: block_size_m, block_names[1]: block_size_n}))

    for block_size_n in range(buffer_size_elem_cnt * core_num, 0, -aligned_size["n_aligned"]):
        if len(configs) == 5:
            break
        grid_x = cdiv(input_size[0], block_size_m)
        grid_y = cdiv(input_size[1], block_size_n)
        total_grid = grid_x * grid_y
        if total_grid % grid_aligned != 0:
            (block_size_m, block_size_n) = balance_grid(block_size_m, block_size_n, input_size)
            append_candidate(configs, Config({block_names[0]: block_size_m, block_names[1]: block_size_n}))
        else:
            append_candidate(configs, Config({block_names[0]: block_size_m, block_names[1]: block_size_n}))

    return configs


def _build_2d_configs(arch_meta, generate_configs, args, block_names, ele_bytes, dotout_ele_bytes, bias, hp_mode,
                      min_block_k, a_trans, b_trans, int8_w8a8, TRITON_i4_AUTOTUNING):
    """Common 2-D MM tile-search body shared by xpu3 (arch=3) and mars (arch=4)."""
    mem = arch_meta["mem"]
    wcache = arch_meta["wcache"]
    dcache = arch_meta["dcache"]
    aligned_size = arch_meta["aligned_size"]
    grid_aligned = arch_meta["grid_aligned"]
    max_m_aglined = arch_meta["max_m_aglined"]
    max_n_aglined = arch_meta["max_n_aglined"]

    input_size = (args["M"], args["N"], args["K"])
    block_size_m = max(2, input_size[0])
    block_size_n = max(2, input_size[1])
    if arch_meta["arch"] == 4:
        block_size_k = int(os.getenv("TRITONXPU_QUANT_LEN", input_size[2]))
    else:
        block_size_k = input_size[2]

    meta_info = {
        "ele_bytes": ele_bytes, "dotout_ele_bytes": dotout_ele_bytes, "bias": bias, "grid_aligned": grid_aligned,
        "block_names": block_names, "input_size": input_size, "aligned_size": aligned_size, "mem": mem, "dcache":
        dcache, "wcache": wcache, "a_trans": a_trans, "b_trans": b_trans, "min_block_k": min_block_k, "hp_mode": hp_mode
    }

    configs = []
    for buffer_num in range(2, 0, -1):
        n_loop_num = 4
        for i in range(min(max_m_aglined, cdiv(input_size[0], aligned_size["m_aligned"])), 0, -1):
            if n_loop_num == 0:
                break
            n_loop_num -= 1
            tmp_block_size_m = min(i * aligned_size["m_aligned"], block_size_m)
            for j in range(min(max_n_aglined, cdiv(input_size[1], aligned_size["n_aligned"])), 0, -1):
                tmp_block_size_n = min(j * aligned_size["n_aligned"], block_size_n)
                grid_m_aligned = cdiv(input_size[0], tmp_block_size_m)
                grid_n_aligned = cdiv(input_size[1], tmp_block_size_n)
                total_grid = grid_m_aligned * grid_n_aligned
                if total_grid % grid_aligned != 0:
                    add_candidate_for_workload_not_balanced(configs, tmp_block_size_m, tmp_block_size_n, block_size_k,
                                                            buffer_num, meta_info)
                else:
                    add_candidate_for_workload_balanced(configs, tmp_block_size_m, tmp_block_size_n, block_size_k,
                                                        buffer_num, meta_info)

    if TRITON_i4_AUTOTUNING and arch_meta["arch"] == 3:
        m_aligned = 64
        k_aligned = 64
        for m in range(1, 2):
            for n in range(1, 13):
                for k in range(4, 12):
                    tmp_block_size_n = cdiv(block_size_n, n)
                    tmp_block_size_k = cdiv(block_size_k, k)
                    if tmp_block_size_n % m_aligned == 0 and tmp_block_size_k % k_aligned == 0:
                        if check_out_of_mem(64, tmp_block_size_n, tmp_block_size_k, meta_info['mem'],
                                            meta_info['ele_bytes'], meta_info['dotout_ele_bytes'], meta_info['bias'],
                                            buffer_num, meta_info['a_trans'], meta_info['b_trans'], meta_info["dcache"],
                                            meta_info["wcache"], meta_info["hp_mode"]):
                            append_candidate(
                                configs,
                                Config({
                                    block_names[0]: m_aligned, block_names[1]: tmp_block_size_n, block_names[2]:
                                    tmp_block_size_k
                                }))

    if int8_w8a8:
        for config in list(configs):
            append_candidate(configs, Config(kwargs=config.kwargs, num_stages=3))
    return configs


def block_size_candidates(args, generate_configs, op_affiliation, row_sign, col_sign, n_elem_sign):
    if op_affiliation == "cluster":
        return block_size_candidates_cluster(args, generate_configs, op_affiliation, row_sign, col_sign, n_elem_sign)

    BLOCK_M = "BLOCK_M"
    BLOCK_N = "BLOCK_N"
    BLOCK_K = "BLOCK_K"
    bias = 0
    hp_mode = 0

    if generate_configs == "bmm":
        BLOCK_M, BLOCK_N, BLOCK_K = "TILE_M", "TILE_N", "TILE_K"
    elif generate_configs == "addmm":
        BLOCK_M, BLOCK_N, BLOCK_K = "BLOCK_SIZE_M", "BLOCK_SIZE_N", "BLOCK_SIZE_K"
        bias = 1

    a_trans = False
    b_trans = False
    if "stride_ak" in args.keys():
        a_trans = args["stride_ak"] != 1
    if "stride_bn" in args.keys():
        b_trans = args["stride_bn"] != 1
    block_names = (BLOCK_M, BLOCK_N, BLOCK_K)

    A = args["a_ptr"] if "a_ptr" in args.keys() else args["A"]
    B = args["b_ptr"] if "b_ptr" in args.keys() else args["B"]
    ele_bytes = A.element_size()
    int8_w8a8 = ele_bytes == 1
    min_block_k = 128
    matmul_mode = int(os.getenv("XMLIR_MATMUL_FAST_MODE", "0"))
    if str(A.dtype) == "torch.bfloat16" and matmul_mode == 0:
        ele_bytes = ele_bytes * 2

    TRITON_i4_AUTOTUNING = 0
    if ele_bytes == 2:
        min_block_k = 320
    elif ele_bytes == 1:
        min_block_k = 640
        double_k = False
        if len(A.shape) == 2 and len(B.shape) == 2:
            double_k = ((not a_trans and not b_trans and A.shape[-1] == 2 * B.shape[-1])
                        or (not a_trans and b_trans and A.shape[-1] == 2 * B.shape[0])
                        or (a_trans and not b_trans and A.shape[0] == 2 * B.shape[-1])
                        or (a_trans and b_trans and A.shape[0] == 2 * B.shape[0]))
        elif len(A.shape) == 2 and len(B.shape) == 3:
            double_k = ((not a_trans and not b_trans and A.shape[-1] == 2 * B.shape[1])
                        or (not a_trans and b_trans and A.shape[-1] == 2 * B.shape[-1])
                        or (a_trans and not b_trans and A.shape[0] == 2 * B.shape[-1])
                        or (a_trans and b_trans and A.shape[0] == 2 * B.shape[1]))
        if double_k:
            TRITON_i4_AUTOTUNING = 1
            min_block_k = 1280

    dotout_ele_bytes = 4
    if "dot_out_type" in args.keys():
        import torch  # local import; only needed when XPU MM kernel uses dot_out_type
        if args["dot_out_type"] != torch.float32:
            dotout_ele_bytes = 2
            min_block_k = min_block_k // 2

    xpu_hp_mode = int(os.getenv("TRITONXPU_HP_MODE", "0"))
    if str(A.dtype) == "torch.bfloat16" and xpu_hp_mode == 1:
        hp_mode = 1

    arch = driver.active.get_current_target().arch
    if arch == 3:
        max_m_aglined, max_n_aglined = 6, 8
        if hp_mode == 1:
            max_m_aglined, max_n_aglined = 5, 5
        if matmul_mode == 1 and ele_bytes == 2 and args["K"] > 32768:
            max_m_aglined = 4
        arch_meta = {
            "arch": 3, "mem": (1605632, 128), "wcache": 1310720, "dcache": 786432, "aligned_size":
            {"m_aligned": 80, "n_aligned": 64, "k_aligned":
             128}, "grid_aligned": 12, "max_m_aglined": max_m_aglined, "max_n_aglined": max_n_aglined
        }
        return _build_2d_configs(arch_meta, generate_configs, args, block_names, ele_bytes, dotout_ele_bytes, bias,
                                 hp_mode, min_block_k, a_trans, b_trans, int8_w8a8, TRITON_i4_AUTOTUNING)

    if arch == 4:
        max_m_aglined, max_n_aglined = 6, 8
        if hp_mode == 1:
            max_m_aglined, max_n_aglined = 5, 5
        arch_meta = {
            "arch": 4, "mem": (1048576, 128), "wcache": 768 * 1024, "dcache": 320 * 1024, "aligned_size":
            {"m_aligned": 96, "n_aligned": 64, "k_aligned":
             128}, "grid_aligned": 6, "max_m_aglined": max_m_aglined, "max_n_aglined": max_n_aglined
        }
        return _build_2d_configs(arch_meta, generate_configs, args, block_names, ele_bytes, dotout_ele_bytes, bias,
                                 hp_mode, min_block_k, a_trans, b_trans, int8_w8a8, TRITON_i4_AUTOTUNING)

    return []
