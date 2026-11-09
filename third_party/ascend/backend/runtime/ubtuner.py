# Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
#in the Software without restriction, including without limitation the rights
#to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
#copies of the Software, and to permit persons to whom the Software is
#furnished to do so, subject to the following conditions:
#
#The above copyright notice and this permission notice shall be included in
#all copies or substantial portions of the Software.
#
#THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
#IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
#FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
#AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
#LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
#OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
#THE SOFTWARE.
"""
UB Tuner for Triton-Ascend

This module provides @ubtuner decorator that automatically tries different
bisheng-compiler options to avoid UB (Unified Buffer) overflow, without actually
running the kernel on the device.

Unlike @autotuner which tunes performance parameters and requires running on
device, @ubtuner only needs to compile the kernel and check if there's a UB
overflow error.
"""

from __future__ import annotations

import os
import functools
from dataclasses import dataclass, field
from itertools import product
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union

from triton.runtime.jit import KernelInterface


# =============================================================================
# Configuration Constants
# =============================================================================
class _Config:
    """Global configuration constants for UB Tuner."""

    # Environment variable names
    ENV_MODE = "TRITON_ENABLE_UBTUNER"  # 调优模式，可选值: "compile"(仅编译)、"run"(编译+运行测量耗时) (默认: 未设置)
    ENV_EANBLE_PRINT = "TRITON_PRINT_AUTOTUNING"  # 调优日志开关, 复用autotune的环境变量，设为"1"时打印调优过程详细信息 (默认: 未设置/不打印)
    ENV_ENABLE_PRINT_UB_BITS = "ENABLE_PRINT_UB_BITS"  # 运行时打印UB占用位数，run模式下临时设为"true"以获取required_ub_bits (默认: 未设置)

    # Default values
    DEFAULT_MODE = "run"
    DEFAULT_UB_MEMORY_LIMIT = 100  # 1 unit = 256KB, so 100 = 25MB effective limit

    # Error patterns
    UB_OVERFLOW_ERROR = "ub overflow"

    # Supported values
    SUPPORTED_ALGORITHMS = {"greedy"}
    SUPPORTED_MODES = {"compile", "run"}

    # Logging format
    LOG_PREFIX = "[UBTuner]"


# =============================================================================
# UB Option Metadata - Single source of truth for all UB tuning options
# =============================================================================
UB_OPTION_METADATA: Dict[str, Dict[str, Any]] = {
    'auto_multi_buffer': {'type': bool, 'default': False, 'npu_option': 'multibuffer'},
    'vf_fusion_mode':
    {'type': str, 'default': 'ub-aware-op', 'npu_option': 'vf_fusion_mode', 'values': ['ub-aware-op', 'max-parallel']},
}

# Derived lists for convenience
BOOL_OPTIONS = [k for k, v in UB_OPTION_METADATA.items() if v['type'] == bool]
INT_OPTIONS = [k for k, v in UB_OPTION_METADATA.items() if v['type'] == int]
STR_OPTIONS = [k for k, v in UB_OPTION_METADATA.items() if v['type'] == str]

# Mapping from UB config keys to NPUOptions keys
UB_TO_NPU_OPTION_MAP = {k: v['npu_option'] for k, v in UB_OPTION_METADATA.items() if v['npu_option'] is not None}

# =============================================================================
# Cost Model for UB Config Selection
# =============================================================================
DEFAULT_UB_MEMORY_LIMIT = _Config.DEFAULT_UB_MEMORY_LIMIT

UB_OPTION_COST_BENEFIT: Dict[str, Dict[str, float]] = {
    'auto_multi_buffer': {'cost': 8, 'benefit': 1.5},

    # Str options: each possible value has its own cost/benefit entry
    'vf_fusion_mode#ub-aware-op': {'cost': 2, 'benefit': 1.0},
    'vf_fusion_mode#max-parallel': {'cost': 3, 'benefit': 2.0},
}


# =============================================================================
# Utility Functions
# =============================================================================
def _log_debug(message: str) -> None:
    """Print debug message if TRITON_PRINT_UBTUNING=1."""
    if os.getenv(_Config.ENV_EANBLE_PRINT) == "1":
        print(f"{_Config.LOG_PREFIX} {message}")


def _log_info(message: str) -> None:
    """Print info message with green color."""
    if os.getenv(_Config.ENV_EANBLE_PRINT) == "1":
        print(f"\033[1;32m{message}\033[0m")


def _get_option_value(option: str, enabled: bool) -> bool:
    """Get the value for an option based on enabled state and default."""
    if option not in UB_OPTION_METADATA:
        return enabled
    default = UB_OPTION_METADATA[option].get('default', False)
    return not default if enabled else default


def get_mode_choice() -> str:
    """Get the mode to use based on environment variable."""
    mode_env = os.getenv(_Config.ENV_MODE, "None").lower()
    if mode_env in _Config.SUPPORTED_MODES:
        return mode_env
    _log_debug(f"Unknown mode: {mode_env}, using '{_Config.DEFAULT_MODE}' as default")
    return _Config.DEFAULT_MODE


def _try_compile_option(option: str, current_config: Dict[str, bool], linalg_ir: str, metadata: Dict[str, Any],
                        npu_options: Any, try_value: Union[bool, str]) -> Tuple[bool, Optional[str]]:
    """Try compiling with a single option enabled/disabled."""
    from triton.backends.ascend.compiler import try_compile_with_config

    test_config = dict(current_config)
    test_config[option] = try_value

    config = UBConfig(kwargs=test_config)
    return try_compile_with_config(linalg_ir, config.all_kwargs(), metadata, npu_options)


def _build_sorted_option_list(benefit_cost: Dict[str, Dict[str, float]], options: List[str]) -> List[Tuple[str, Any]]:
    """
    Build a flat list of (option_key, value) tuples sorted by benefit/cost ratio.

    For bool options: key is the option name, value is 'not Default_value'
    For str options: key format is "option#value", value is the string value
    """
    items = []

    for opt in options:
        opt_meta = UB_OPTION_METADATA.get(opt, {})
        opt_type = opt_meta.get('type', bool)

        if opt_type == str:
            # Str option: add each possible value as a separate entry
            values = opt_meta.get('values', [opt_meta.get('default', '')])
            for val in values:
                key = f"{opt}#{val}"
                bc = benefit_cost.get(key, {'cost': 0, 'benefit': -1e100})
                ratio = bc.get('benefit', 1.0) / max(bc.get('cost', 1), 1)
                items.append((key, val, opt, ratio))
        else:
            # Bool option:
            bc = benefit_cost.get(opt, {'cost': 0, 'benefit': -1e100})
            ratio = bc.get('benefit', 1.0) / max(bc.get('cost', 1), 1)
            items.append((opt, (not opt_meta.get('default', False)), opt, ratio))

    # Sort by benefit/cost ratio descending
    items.sort(key=lambda x: x[3], reverse=True)
    return items


def greedy_optimize(benefit_cost: Dict[str, Dict[str, float]], options: List[str], linalg_ir: str, metadata: Dict[str,
                                                                                                                  Any],
                    npu_options: Any, _memory_limit: Optional[int] = None  # Unused but kept for compatibility
                    ) -> Dict[str, Any]:
    """
    Greedy strategy: sort options by benefit/cost ratio, try each option.

    Args:
        benefit_cost: Dict mapping option (or "option#value" for str) to {'cost': float, 'benefit': float}
        options: List of option names to consider
        linalg_ir: Linalg IR string for compilation test
        metadata: Compilation metadata
        npu_options: NPUOptions object
        memory_limit: Optional memory limit (unused, for compatibility)

    Returns:
        Dict of option_name -> value (bool for bool options, str for str options)
    """
    if not options:
        return {}

    # Build flat sorted list of all options and their values
    sorted_items = _build_sorted_option_list(benefit_cost, options)

    if os.getenv(_Config.ENV_EANBLE_PRINT) == "1":
        print(f"{_Config.LOG_PREFIX} Greedy strategy - sorted options by benefit/cost:")
        for key, _, _, ratio in sorted_items:
            bc = benefit_cost.get(key, {'cost': 0, 'benefit': 1.0})
            print(f"  {key}: cost={bc.get('cost', 0)}, benefit={bc.get('benefit', 1.0)}, ratio={ratio}")

    result: Dict[str, Any] = {}
    applied_options: Set[str] = set()  # Track which options have been applied

    for _, val, opt, _ in sorted_items:
        is_str_opt = opt in STR_OPTIONS
        is_bool_opt = opt in BOOL_OPTIONS
        # Already found a successful value for this str option, skip
        if is_str_opt and opt in applied_options:
            continue

        _log_debug(f"Greedy: trying config with {opt}={val}")
        try:
            success, error_msg = _try_compile_option(opt, result, linalg_ir, metadata, npu_options, val)
            if success:
                result[opt] = val
                applied_options.add(opt)
                _log_debug(f"Greedy: {opt}={val} succeeded, keeping it")
            else:
                if is_bool_opt:
                    result[opt] = _get_option_value(opt, False)
                err_preview = error_msg[:100] if error_msg else "Unknown"
                _log_debug(f"Greedy: {opt}={val} failed: {err_preview}..., skipping")
        except Exception as e:
            _log_debug(f"Greedy: {opt}={val} exception: {e}, skipping")

    return result


def _measure_option_cost_onboard(fn: Callable, args: Tuple[Any, ...], kwargs: Dict[str, Any], opt: str,
                                 opt_value: Any) -> Tuple[float, float]:
    """
    Measure the cost (UB bits) for a single option configuration.

    Returns:
        Tuple of (cost, benefit) where benefit is negative execution time in ms
    """
    test_kwargs = dict(kwargs)
    test_config = UBConfig(kwargs={opt: opt_value})

    for ub_key, npu_key in UB_TO_NPU_OPTION_MAP.items():
        if ub_key in test_config.all_kwargs():
            test_kwargs[npu_key] = test_config.all_kwargs()[ub_key]

    compiled_kernel = fn.run(*args, **test_kwargs)

    metadata = dict(compiled_kernel.metadata._asdict()) if hasattr(compiled_kernel, 'metadata') else None
    if metadata is None:
        raise RuntimeError("No metadata found in kernel")

    from ..testing import do_bench_npu
    test_kwargs.update({"warmup": False})
    bench_fn = functools.partial(fn.run, *args, **test_kwargs)
    opt_time = do_bench_npu(bench_fn, clear_l2_cache=False)

    return metadata.get('required_ub_bits', 0), -opt_time * 1000


def run_mode_get_benefit_cost(available_options: List[str], fn: Callable, args: Tuple[Any, ...],
                              kwargs: Dict[str, Any]) -> Dict[str, Dict[str, float]]:
    """
    Run mode: measure actual execution time for each option.

    Args:
        available_options: List of option names to measure
        fn: The kernel function
        args: Positional arguments for the kernel
        kwargs: Keyword arguments for the kernel

    Returns:
        Dict of {option: {'cost': float, 'benefit': float}}
        For str options, key format is "option#value"
    """
    result = {}

    # Build flat list of (key, opt, value) to measure
    items_to_measure = []
    for opt in available_options:
        opt_meta = UB_OPTION_METADATA.get(opt, {})
        opt_type = opt_meta.get('type', bool)

        if opt_type == str:
            values = opt_meta.get('values', [opt_meta.get('default', '')])
            for val in values:
                items_to_measure.append((f"{opt}#{val}", opt, val))
        else:
            items_to_measure.append((opt, opt, _get_option_value(opt, True)))

    # Measure each option
    for key, opt, opt_value in items_to_measure:
        _log_debug(f"======== {key} ========")
        try:
            cost, benefit = _measure_option_cost_onboard(fn, args, kwargs, opt, opt_value)
            result[key] = {'cost': cost, 'benefit': benefit}
        except Exception as e:
            _log_debug(f"Run-mode {key} failed: {e}")
            result[key] = {'cost': 0, 'benefit': -1e100}

    return result


# Legacy Option weights (kept for backward compatibility)
OPTION_WEIGHTS: Dict[str, float] = {
    'auto_multi_buffer': 5.0,
}


def compute_config_score(config: 'UBConfig') -> float:
    """
    Compute cost model score for a UB config.

    Higher score = more aggressive config, higher UB usage risk.
    Conservative configs get lower scores.
    """
    kwargs = config.all_kwargs()
    score = 0.0

    for opt_name, weight in OPTION_WEIGHTS.items():
        if opt_name in kwargs:
            value = kwargs[opt_name]
            if isinstance(value, bool) and value:
                score += weight
            elif isinstance(value, int):
                score += weight * value

    return score


def _generate_all_configs() -> List[UBConfig]:
    """Generate all possible UB configurations."""
    configs = []
    bool_combinations = product([False, True], repeat=len(BOOL_OPTIONS))

    for bool_vals in bool_combinations:
        kwargs = {opt: val for opt, val in zip(BOOL_OPTIONS, bool_vals)}
        kwargs.update({opt: 0 for opt in INT_OPTIONS})
        configs.append(UBConfig(kwargs=kwargs))

    return configs


def _sort_configs_by_score(configs: List[UBConfig]) -> List[Tuple[UBConfig, float]]:
    """Sort configs by cost model score (descending)."""
    scored_configs = [(cfg, compute_config_score(cfg)) for cfg in configs]
    scored_configs.sort(key=lambda x: x[1], reverse=True)
    return scored_configs


_SORTED_CONFIGS: Optional[List[Tuple[UBConfig, float]]] = None


def get_sorted_configs() -> List[Tuple[UBConfig, float]]:
    """Get all configs sorted by cost model score (cached)."""
    global _SORTED_CONFIGS

    if _SORTED_CONFIGS is None:
        _SORTED_CONFIGS = _sort_configs_by_score(_generate_all_configs())

    return _SORTED_CONFIGS


def print_sorted_configs() -> None:
    """Print all configs sorted by cost model score (for debugging)."""
    print("UB Configs sorted by cost model score:")
    print("-" * 80)
    for i, (cfg, score) in enumerate(get_sorted_configs()):
        print(f"{i+1:3d}. score={score:5.2f} -> {cfg}")
    print("-" * 80)
    print(f"Total: {len(get_sorted_configs())} configs")


def get_origin_fn(fn):
    original_fn = fn
    while hasattr(original_fn, 'fn'):
        original_fn = original_fn.fn
    return original_fn


@dataclass
class UBConfig:
    """
    UB Configuration - similar to triton.Config

    Represents a set of bisheng-compiler options to try for avoiding UB overflow.
    Each field corresponds to a compiler option that affects UB usage.
    """

    # Compiler options that affect UB usage (generated from UB_OPTION_METADATA)
    auto_multi_buffer: Optional[bool] = None
    ubuf_saving: Optional[bool] = None
    auto_bind_sub_block: Optional[bool] = None
    auto_blockify_loop: Optional[bool] = None
    drop_unit_dims: Optional[bool] = None
    tile_mix_vector_loop: Optional[int] = None
    tile_mix_cube_loop: Optional[int] = None
    enable_hivm_auto_cv_balance: Optional[bool] = None
    sync_solver: Optional[bool] = None
    unit_flag: Optional[bool] = None
    vf_fusion_mode: Optional[str] = None

    # Additional kwargs for extensibility
    kwargs: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if self.kwargs is None:
            self.kwargs = {}

        # Collect all non-None fields into kwargs (using UB_OPTION_METADATA keys)
        for field_name in UB_OPTION_METADATA.keys():
            val = getattr(self, field_name, None)
            if val is not None:
                self.kwargs[field_name] = val

    def all_kwargs(self) -> Dict[str, Any]:
        """Return all configuration options as a dictionary."""
        return self.kwargs.copy()

    def __repr__(self):
        return f"UBConfig({self.kwargs})"


class UBTuner(KernelInterface):
    """
    UB Configuration Tuner - similar to AutoTilingTuner

    Automatically tries different bisheng-compiler options to find a configuration
    that doesn't cause UB overflow. Unlike autotuner, this doesn't require running
    on the device - it only needs to compile and check for errors.

    Uses cost model to prioritize configs with lower UB usage risk first.
    """

    def __init__(
        self,
        fn: Callable,
        key: Optional[List[str]] = None,
    ):
        self.fn = fn
        self.arg_names = fn.arg_names if hasattr(fn, 'arg_names') else []
        self.configs: List[UBConfig] = []
        self.key = key or []
        self.cache: Dict[Tuple, UBConfig] = {}
        self._searched_configs: Set[int] = set()
        self.print_ubtuning = os.getenv(_Config.ENV_EANBLE_PRINT) == "1"
        self.config_iter = self._iter_configs() if not self.configs else iter(self.configs)

    def _iter_configs(self):
        """Iterator to generate candidate UB configurations ordered by cost model score."""
        for config, score in get_sorted_configs():
            config_id = hash(frozenset(config.all_kwargs().items()))
            if config_id not in self._searched_configs:
                self._searched_configs.add(config_id)
                if self.print_ubtuning:
                    print(f"{_Config.LOG_PREFIX} Next config: {config}, score={score:.2f}")
                yield config

    def generate_key_and_configs(self, *args: Any, **kwargs: Any) -> Tuple:
        """Generate cache key based on arguments."""
        nargs = dict(zip(self.arg_names, args))
        nargs.update(kwargs)
        key_args = {k: nargs[k] for k in self.key if k in nargs}
        return tuple(sorted(key_args.items()))

    def run(self, *args: Any, **kwargs: Any):
        """Run UB tuning: first try without tuner, if UB overflow, tune and retry."""
        try:
            return self.fn.run(*args, **kwargs)
        except Exception as e:
            # the overflow problem not autotuned will do ubtune
            autotuned = getattr(get_origin_fn(self.fn), '_triton_autotuned', False)
            if _Config.UB_OVERFLOW_ERROR not in str(e).lower() or autotuned:
                raise e
            print(f"{e}, Ub overflow try ub tuner.")

        self.nargs = dict(zip(self.arg_names, args))
        key = self.generate_key_and_configs(*args, **kwargs)

        best_config = self._get_or_find_config(key, *args, **kwargs)

        kwargs_with_ub = self._apply_ub_config(kwargs, best_config)

        _log_debug(f"Running kernel with options: {kwargs_with_ub}")

        ret = self.fn.run(*args, **kwargs_with_ub)
        self.nargs = None
        return ret

    def _set_ub_optimal_config(self, kwargs: Any):
        """set compiler config that use minimum ub space"""
        for option in BOOL_OPTIONS:
            if option not in UB_OPTION_METADATA:
                continue
            default = UB_OPTION_METADATA[option].get('default', False)
            npu_option = UB_OPTION_METADATA[option].get('npu_option', '')
            if npu_option:
                kwargs[npu_option] = default
        for option in STR_OPTIONS:
            if option not in UB_OPTION_METADATA:
                continue
            default = UB_OPTION_METADATA[option].get('default', '')
            npu_option = UB_OPTION_METADATA[option].get('npu_option', '')
            if npu_option:
                kwargs[npu_option] = default

    def _get_or_find_config(self, key: Tuple, *args: Any, **kwargs: Any) -> UBConfig:
        """Get cached config or find a new one."""
        if key in self.cache:
            _log_debug(f"Using cached config for key: {key}")
            return self.cache[key]

        self._set_ub_optimal_config(kwargs)
        best_config = self._find_best_config(*args, **kwargs)
        self.cache[key] = best_config

        _log_debug(f"Cached config for key: {key}, config: {best_config}")

        return best_config

    def _apply_ub_config(self, kwargs: Dict[str, Any], config: UBConfig) -> Dict[str, Any]:
        """Apply UB config to kwargs, mapping UB keys to NPU options."""
        kwargs_with_ub = dict(kwargs)
        ub_kwargs = config.all_kwargs()

        for ub_key, npu_key in UB_TO_NPU_OPTION_MAP.items():
            if ub_key in ub_kwargs:
                kwargs_with_ub[npu_key] = ub_kwargs[ub_key]

        return kwargs_with_ub

    def _find_best_config(self, *args: Any, **kwargs: Any) -> UBConfig:
        """Find the best UB config using configured algorithm."""
        mode = get_mode_choice()
        _log_debug(f"Using cost mode: {mode}")
        os.environ[_Config.ENV_ENABLE_PRINT_UB_BITS] = 'true' if mode == _Config.DEFAULT_MODE else 'false'
        try:
            compiled_kernel = self.fn.run(*args, **kwargs)
            metadata = dict(compiled_kernel.metadata._asdict()) if hasattr(compiled_kernel, 'metadata') else {}
            npu_options = self._create_npu_options(metadata)
            if npu_options is None:
                return UBConfig()

            asm_key = 'bcmlir'
            linalg_ir = compiled_kernel.asm.get(asm_key) if hasattr(compiled_kernel, 'asm') else None
            if linalg_ir is None:
                _log_debug("Could not get linalg IR, using default config")
                return UBConfig()

            available_options = list(UB_OPTION_METADATA.keys())

            benefit_cost = self._get_benefit_cost(mode, available_options, args, kwargs, npu_options)
        finally:
            os.environ.pop(_Config.ENV_ENABLE_PRINT_UB_BITS, None)

        for config in self._try_algorithms(['greedy'], benefit_cost, available_options, linalg_ir, metadata,
                                           npu_options):
            return config

        return UBConfig()

    def _create_npu_options(self, metadata: Dict[str, Any]):
        """Create NPUOptions from metadata."""
        from triton.backends.ascend.compiler import NPUOptions
        try:
            npu_option_keys = {
                field.name
                for field in NPUOptions.__dataclass_fields__.values()
                if field.init and field.name not in {"arch", "compile_on_910_95"}
            }
            opts_dict = {k: v for k, v in metadata.items() if k in npu_option_keys}
            target_arch = metadata.get("target_arch", metadata.get("arch", ""))
            return NPUOptions(arch=target_arch, **opts_dict)
        except Exception as e:
            _log_debug(f"Failed to create NPUOptions: {e}")
            return None

    def _get_benefit_cost(self, mode: str, available_options: List[str], args: Tuple[Any, ...], kwargs: Dict[str, Any],
                          npu_options: Any) -> Dict[str, Dict[str, float]]:
        """Get benefit/cost data based on mode."""
        if mode == "run":
            _log_debug("Running in run-mode, measuring execution time...")
            return run_mode_get_benefit_cost(available_options, self.fn, args, kwargs)
        return UB_OPTION_COST_BENEFIT

    def _try_algorithms(self, algorithm_list: List[str], benefit_cost: Dict[str, Dict[str, float]],
                        available_options: List[str], linalg_ir: str, metadata: Dict[str, Any],
                        npu_options: Any) -> List[UBConfig]:
        """Try each algorithm in order until one succeeds."""
        for algo in algorithm_list:
            config = self._try_single_algorithm(algo, benefit_cost, available_options, linalg_ir, metadata, npu_options)
            if config is not None:
                return [config]
        return []

    def _try_single_algorithm(self, algo: str, benefit_cost: Dict[str, Dict[str, float]], available_options: List[str],
                              linalg_ir: str, metadata: Dict[str, Any], npu_options: Any) -> Optional[UBConfig]:
        """Try a single algorithm and return config if successful."""
        _log_debug(f"Running {algo} optimization...")

        if algo == "greedy":
            config_dict = greedy_optimize(benefit_cost, available_options, linalg_ir, metadata, npu_options)
            _log_info(f"{_Config.LOG_PREFIX} Greedy optimized config: {config_dict}")
            config = UBConfig(kwargs=config_dict)

        success, error_msg = self._try_compile(config, linalg_ir, metadata, npu_options)

        if success:
            _log_debug(f"{algo.capitalize()} config works!")
            return config

        err_preview = error_msg[:200] if error_msg else "Unknown error"
        _log_debug(f"{algo.capitalize()} config failed: {err_preview}...")
        return None

    def _try_compile(self, config: UBConfig, linalg_ir: str, metadata: Dict[str, Any],
                     npu_options: Any) -> Tuple[bool, Optional[str]]:
        """Try to compile the kernel with the given UB config."""
        from triton.backends.ascend.compiler import try_compile_with_config
        try:
            return try_compile_with_config(linalg_ir, config.all_kwargs(), metadata, npu_options)
        except Exception as e:
            _log_debug(f"Exception during try_compile: {e}")
            return (False, str(e))

    def get_best_config(self, *args: Any, compile_opts=None, **kwargs: Any):
        """Get the best UB config for the given arguments."""
        _log_debug(f"kwargs {kwargs}")
        _log_debug(f"compile_opts {compile_opts}")
        if compile_opts:
            kwargs = {**kwargs, **compile_opts}
        self._set_ub_optimal_config(kwargs)
        cfg = self._find_best_config(*args, **kwargs)
        if cfg.kwargs:
            ub_cfg = self._apply_ub_config({}, cfg)
            return ub_cfg
        return {}


def ubtuner(key=None):

    def decorator(fn):
        # Check if fn (or its underlying function) has been decorated with autotuner
        # When autotune is applied, it marks the original function with '_triton_autotuned' attribute
        # Check fn itself and its underlying 'fn' attribute (for decorator chains)
        original_fn = get_origin_fn(fn)
        # Mark the original function so ubtuner can detect autotune was applied
        setattr(original_fn, '_ubtuned', True)
        if getattr(original_fn, '_triton_autotuned', False):
            raise ValueError(f"Cannot apply @ubtuner decorator on a function that already has @autotune decorator. "
                             f"The @autotune decorator should be the outer decorator (applied first), not @ubtuner. "
                             f"Please change the decorator order: put @autotune outside @ubtuner.")
        return UBTuner(fn, key)

    return decorator
