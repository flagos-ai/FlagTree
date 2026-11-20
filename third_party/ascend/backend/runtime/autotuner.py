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
import gc
import inspect
import math
import os
import pprint
import threading
import time
import warnings
from concurrent.futures import ThreadPoolExecutor
import itertools
from collections.abc import Sequence
from dataclasses import asdict, is_dataclass
from typing import Any, Dict, List, Optional, Tuple

from torch import Tensor

import triton
from triton import knobs
from triton._C.libtriton import get_cache_invalidating_env_vars
from triton.runtime.autotuner import Autotuner, Config
from triton.runtime.jit import compute_cache_key
from triton.backends.ascend.utils import (_InternalNPUOptionInt, _remove_deprecated_npu_options,
                                          _RESERVED_NPU_OPTION_NAMES, is_compile_on_910_95, _multibuffer_mode_to_tuple)

from .autoparser import (LowDimsAxesParser, PtrNumsParser, ReductionAxesParser, SplitAxesParser, TilingAxesParser)
from .dsl_analysis.cv_param_parser import parse_cv_params
from .dsl_analysis.entry import analyze_dot_site_mnk, analyze_signature_and_missing_tunable
from .dsl_analysis.kernel_classifier import resolve_kernel_type
from .dsl_analysis.vv_config_adapter import adapt_vv_v2_to_vector_inputs
from .dsl_analysis.vv_param_parser_v2 import parse_vv_axis_info_v2
from .dsl_analysis.vv_parser_options import (resolve_vv_parser_v2_enabled, resolve_vv_parser_v2_mode)
from .utils import get_byte_per_numel, is_valid_axis_name, valid_axis_names
from .ubtuner import UBTuner, get_origin_fn
from .vector_axes import VectorAxes

# Import CV autotune components
from .cv_autotune.generators.cv_tile_generator import CVTileGenerator
from .cv_autotune.hardware_consts import get_default_hardware_constraints

_RESERVED_HINT_KEYS = {
    "split_params",
    "tiling_params",
    "low_dim_axes",
    "reduction_axes",
    "axes",
    "auto_gen_config",
    "kernel_type",
    "tunable_parameter",
    "vv_parser_v2_mode",
}
_DEFAULT_HINT_NUM_STAGES = [1, 2]
_DEFAULT_COMPILE_MODE = "simd_simt_template"
_FAILURE_DETAIL_MAX_LENGTH = 512


def _format_autotune_failure(exc) -> str:
    """Summarize a failure without retaining its exception or traceback."""
    try:
        detail = getattr(exc, "error_message", None) or str(exc)
        detail = " ".join(str(detail).split())
    except Exception:
        detail = ""
    summary = type(exc).__name__ + (f": {detail}" if detail else "")
    if len(summary) > _FAILURE_DETAIL_MAX_LENGTH:
        summary = summary[:_FAILURE_DETAIL_MAX_LENGTH - 3] + "..."
    return summary


class _UserConfigDiagnostics:
    """Track explicit configs for one invocation, including hints expansion."""

    _REPORTED = "reported"
    _PENDING = {"not_evaluated", "compiling", "benchmarking", "running"}
    _NON_FAILURES = {_REPORTED, "benchmarked", "selected", "skipped_cache_hit", "pruned"}

    def __init__(self, configs):
        self.reset_configs(configs)

    def reset_configs(self, configs):
        # An equal generated config may be the key retained by the benchmark dict.
        self.results = [[config, "not_evaluated", "No benchmark result was produced."] for config in configs]

    def record(self, config, status, detail="", exc=None):
        matched = False
        for result in self.results:
            if result[0] is config or result[0] == config:
                if status == "pruned" and result[1] == "rough_benchmark_failed":
                    # Keep the actual error when time-budget pruning follows it.
                    result[2] += f" {detail}"
                    matched = True
                    continue
                result[1] = status
                result[2] = detail + (f" {_format_autotune_failure(exc)}" if exc is not None else "")
                matched = True
        return matched

    def abort(self, exc):
        for result in self.results:
            if result[1] in self._PENDING:
                result[2] += f" Autotuning interrupted: {_format_autotune_failure(exc)}"

    def report(self):
        incomplete = {
            "compiling": "compilation_incomplete",
            "benchmarking": "benchmark_incomplete",
            "running": "evaluation_incomplete",
        }
        for config, status, detail in self.results:
            # Keep existing success logs; only add failures and interrupted work.
            if status in self._NON_FAILURES:
                continue
            status = incomplete.get(status, status)
            print(f"Triton autotuning: config={config}; {status}; reason={detail.strip()}")


def _format_autotune_timing(timing) -> str:
    """Format the timing returned by the active autotune benchmarker."""
    if isinstance(timing, (tuple, list)):
        labels = ("p50", "p20", "p80")
        return ", ".join(f"{label}={value:.4f} ms" for label, value in zip(labels, timing))
    return f"mean={timing:.4f} ms"


def _get_constexpr_candidates_from_fn(fn) -> List[str]:
    """
    Returns all constexpr parameter names from the kernel function definition.
    """
    current_fn = fn
    visited = set()
    while current_fn is not None and id(current_fn) not in visited:
        visited.add(id(current_fn))
        parse = getattr(current_fn, "parse", None)
        if callable(parse):
            func_ast = parse()
            break
        jit_function = getattr(current_fn, "jit_function", None)
        current_fn = jit_function if jit_function is not None else getattr(current_fn, "fn", None)
    else:
        return []

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


def _clone_config_with_kwargs(
    config: Config,
    kwargs: Dict[str, object],
    num_stages: Optional[int] = None,
) -> Config:
    return Config(
        kwargs=kwargs,
        num_warps=config.num_warps,
        num_stages=config.num_stages if num_stages is None else num_stages,
        num_ctas=config.num_ctas,
        maxnreg=config.maxnreg,
        pre_hook=config.pre_hook,
        ir_override=config.ir_override,
    )


def _normalize_user_configs(configs):
    if not configs:
        return configs
    normalized_configs = []
    for config in configs:
        normalized_kwargs = _remove_deprecated_npu_options(config.kwargs)
        if normalized_kwargs == config.kwargs:
            normalized_configs.append(config)
            continue
        normalized_config = copy.copy(config)
        normalized_config.kwargs = normalized_kwargs
        normalized_configs.append(normalized_config)
    return normalized_configs


def _validate_reserved_npu_option_names(arg_names) -> None:
    conflicts = sorted(set(arg_names) & _RESERVED_NPU_OPTION_NAMES)
    if conflicts:
        raise ValueError("Kernel parameters cannot use reserved Ascend compile-option names: {}.".format(
            ", ".join(conflicts)))


def _split_hints(hints: Optional[Dict[str, object]]):
    hints = {} if hints is None else dict(hints)
    reserved_hints = {k: v for k, v in hints.items() if k in _RESERVED_HINT_KEYS}
    config_hints = {k: v for k, v in hints.items() if k not in _RESERVED_HINT_KEYS}
    return reserved_hints, config_hints


def _validate_user_hints(fn, hints: Optional[Dict[str, object]]) -> None:
    if not hints:
        return

    constexpr_names = set(_get_constexpr_candidates_from_fn(fn))
    legal_keys = set(_RESERVED_HINT_KEYS) | set(_ALL_PARAMS) | constexpr_names
    invalid_keys = sorted(key for key in hints.keys() if key not in legal_keys)
    if invalid_keys:
        raise ValueError("Unsupported hints keys: {}. "
                         "Legal keys must be reserved hints, tl.constexpr kernel parameters, "
                         "or supported compile options.".format(invalid_keys))


def _multibuffer_to_num_stages(multibuffer: bool) -> int:
    if not isinstance(multibuffer, bool):
        raise ValueError("multibuffer must be a boolean when mapped to num_stages. "
                         f"Got: {type(multibuffer)}")
    return 2 if multibuffer else 1


def _normalize_config_hints(
    config_hints: Optional[Dict[str, object]],
    inject_default_num_stages: bool = False,
    has_multibuffer_mode: bool = False,
):
    normalized_hints = {} if config_hints is None else dict(config_hints)
    has_explicit_num_stages = "num_stages" in normalized_hints

    has_mode = has_multibuffer_mode or "multibuffer_mode" in normalized_hints
    if has_mode and has_explicit_num_stages:
        raise ValueError("num_stages and multibuffer_mode cannot be specified together")
    multibuffer_values = (normalized_hints.get("multibuffer") if has_mode else normalized_hints.pop(
        "multibuffer", None))
    if multibuffer_values is not None:
        if not isinstance(multibuffer_values, (list, tuple)) or len(multibuffer_values) == 0:
            raise ValueError("hints['multibuffer'] must be a non-empty list/tuple when used for config expansion.")
        if not all(isinstance(value, bool) for value in multibuffer_values):
            raise ValueError("hints['multibuffer'] must contain only boolean values.")
        if not has_explicit_num_stages and not has_mode:
            normalized_hints["num_stages"] = [_multibuffer_to_num_stages(value) for value in multibuffer_values]

    if inject_default_num_stages and "num_stages" not in normalized_hints and not has_mode:
        normalized_hints["num_stages"] = list(_DEFAULT_HINT_NUM_STAGES)

    return normalized_hints


def _validate_config_hint_values(config_hints: Dict[str, object], constexpr_names: List[str]):
    invalid_keys = sorted(key for key in config_hints if key not in constexpr_names and key not in _ALL_PARAMS)
    if invalid_keys:
        raise ValueError(
            "Invalid hints keys for config expansion: {}. "
            "Keys must be tl.constexpr kernel parameters or supported compile options.".format(invalid_keys))

    for key, value in config_hints.items():
        if not isinstance(value, (list, tuple)) or len(value) == 0:
            raise ValueError(f"hints['{key}'] must be a non-empty list/tuple when used for config expansion.")
        if key in _VALIDATION_RULES:
            rule = _VALIDATION_RULES[key]
            if not rule["check"](value, key):
                raise ValueError(f"Invalid value for '{key}': {value}. Expected: {rule['desc']}")


def _expand_configs_with_hints(fn, configs, config_hints: Dict[str, object]):
    if not config_hints:
        return configs

    constexpr_names = _get_constexpr_candidates_from_fn(fn)
    _validate_config_hint_values(config_hints, constexpr_names)

    if not configs:
        return configs

    keys = list(config_hints.keys())
    values = [config_hints[key] for key in keys]
    expanded_configs = []
    for config in configs:
        for combo in itertools.product(*values):
            new_kwargs = config.kwargs.copy()
            num_stages = config.num_stages
            for key, value in zip(keys, combo):
                if key == "num_stages":
                    num_stages = value
                else:
                    new_kwargs[key] = value
            expanded_configs.append(_clone_config_with_kwargs(config, new_kwargs, num_stages))
    return expanded_configs


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
        cache_results=False,
        auto_profile_dir=None,
        hints=None,
    ):
        """
        :param key: a list of argument names whose runtime value changes invalidate the autotune cache
            and trigger re-evaluation of candidate configs.
        :type key: List[str]
        """
        _validate_reserved_npu_option_names(arg_names)
        hints = _remove_deprecated_npu_options(hints or {})
        configs = _normalize_user_configs(configs)
        _validate_user_hints(fn, hints)
        reserved_hints, raw_config_hints = _split_hints(hints)
        config_hints = _normalize_config_hints(
            raw_config_hints,
            inject_default_num_stages=bool(configs) and bool(raw_config_hints),
            has_multibuffer_mode=any(config.kwargs.get("multibuffer_mode") is not None for config in (configs or ())),
        )

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
            cache_results=cache_results,
        )
        self.user_defined_do_bench = do_bench is not None
        self.hints = reserved_hints
        # Retain user hints to reconsider defaults when a launch supplies a mode.
        self._raw_config_hints = raw_config_hints
        self.config_hints = config_hints
        self.vv_parser_v2_mode = resolve_vv_parser_v2_mode(self.hints)
        self.enable_vv_parser_v2 = resolve_vv_parser_v2_enabled(self.hints)
        self.explicit_tunable_params = self._parse_explicit_tunable_params(self.hints.get("tunable_parameter", None))

        split_params = self.hints.get("split_params", None)
        tiling_params = self.hints.get("tiling_params", None)
        low_dim_axes = self.hints.get("low_dim_axes", None)
        reduction_axes = self.hints.get("reduction_axes", None)
        hints_axes = self.hints.get("axes", None)
        self._init_axis_params(
            key,
            split_params,
            tiling_params,
            low_dim_axes,
            reduction_axes,
            hints_axes,
        )

        self.auto_gen_config = not configs or self.hints.get("auto_gen_config", False)
        self.gen_configs = []  # generated configs from TileGenerator
        self.auto_profile_dir = auto_profile_dir
        if not configs:
            self.user_configs = []
        else:
            self.user_configs = configs
        self.is_simt_mode = False
        self.user_specified_warps = None
        self.user_specified_num_stages = None
        self.user_specified_multibuffer_mode = None
        self.user_specified_multibuffer = None
        target_arch = triton.runtime.driver.active.get_current_target().arch
        self.default_multibuffer = not is_compile_on_910_95(target_arch)
        self.print_autotuning = os.getenv("TRITON_PRINT_AUTOTUNING", None) == "1"

        # Mark the original function so ubtuner can detect autotune was applied
        ori_fn = get_origin_fn(fn)
        setattr(ori_fn, '_triton_autotuned', True)

        self.enable_ubtuner = os.getenv("TRITON_ENABLE_UBTUNER", "").lower() in ('compile', 'run') or getattr(
            ori_fn, '_ubtuned', False)
        if self.enable_ubtuner:
            self.ubtuner = UBTuner(fn, key)

        # Compile kernels in parallel by default for triton.runtime.JITFunction or UBTuner,
        # but not for others, e.g., LibEntry, since it's not compatible with AsyncCompileMode
        self.compile_parallel = (isinstance(self.fn, (triton.runtime.JITFunction, UBTuner))
                                 and os.getenv("TRITON_AUTOTUNE_PARALLEL_COMPILE", "1") == "1")
        self._source_module_ast_cache: Optional[ast.Module] = None
        self._source_module_ast_resolved = False

        # Negative cache for deterministic compile failures.  This is keyed by
        # the JIT compilation identity rather than the autotune tuning key, so
        # a failed compile can be reused when only the tuning key changes while
        # the actual compiler inputs remain identical.
        self._compile_failure_cache: Dict[Tuple, Dict[str, str]] = {}
        self._compile_failure_cache_lock = threading.Lock()
        self._cached_compile_failed_configs: List[Config] = []

    @staticmethod
    def _parse_explicit_tunable_params(raw_value) -> List[str]:
        if raw_value is None:
            return []
        if not isinstance(raw_value, list):
            raise ValueError("hints.tunable_parameter must be a list, got: {}".format(type(raw_value)))
        values = list(raw_value)
        parsed = []
        seen = set()
        for value in values:
            if value in seen:
                continue
            parsed.append(value)
            seen.add(value)
        return parsed

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

    def _get_non_simt_num_stages_candidates(self) -> List[int]:
        if self.user_specified_num_stages is not None:
            return [self.user_specified_num_stages]
        if self.user_specified_multibuffer is not None:
            return [_multibuffer_to_num_stages(self.user_specified_multibuffer)]
        return list(_DEFAULT_HINT_NUM_STAGES)

    def _expand_num_stages_configs(
        self,
        base_configs: List[Config],
        num_stages_candidates: List[int],
    ) -> List[Config]:
        expanded_configs = []
        for base_cfg in base_configs:
            for num_stages in num_stages_candidates:
                expanded_configs.append(
                    _clone_config_with_kwargs(
                        base_cfg,
                        base_cfg.kwargs.copy(),
                        num_stages=num_stages,
                    ))

        if self.print_autotuning:
            print("Triton autotuning: Expanded to "
                  f"{len(expanded_configs)} configs (with num_stages: {num_stages_candidates})")
        return expanded_configs

    def _expand_simd_multibuffer_configs(self, base_configs: List[Config]) -> List[Config]:
        if self.user_specified_multibuffer_mode is not None or "multibuffer_mode" in self.config_hints:
            return base_configs
        if (self.user_specified_multibuffer is not None or self.user_specified_num_stages is not None):
            normalized_configs = []
            for base_cfg in base_configs:
                if self.user_specified_num_stages is not None:
                    base_cfg.num_stages = self.user_specified_num_stages
                    # Keep compiler-visible multibuffer consistent when only num_stages is specified.
                    if self.user_specified_multibuffer is None:
                        base_cfg.kwargs["multibuffer"] = (self.user_specified_num_stages != 1)
                else:
                    base_cfg.num_stages = _multibuffer_to_num_stages(self.user_specified_multibuffer)

                normalized_configs.append(base_cfg)
            if self.print_autotuning:
                print("Triton autotuning: Skip SIMD multibuffer expansion because user "
                      f"specified multibuffer={self.user_specified_multibuffer}, "
                      f"num_stages={self.user_specified_num_stages}")
            return normalized_configs

        opposite_default_multibuffer = not self.default_multibuffer
        simd_configs = []
        for base_cfg in base_configs:
            if base_cfg.kwargs.get("multibuffer_mode") is not None:
                simd_configs.append(base_cfg)
                continue
            base_multibuffer = base_cfg.kwargs.get("multibuffer", self.default_multibuffer)
            base_cfg.kwargs["multibuffer"] = base_multibuffer
            base_cfg.num_stages = _multibuffer_to_num_stages(base_multibuffer)
            simd_configs.append(base_cfg)

            new_cfg = _clone_config_with_kwargs(base_cfg, base_cfg.kwargs.copy())
            new_cfg.kwargs["multibuffer"] = opposite_default_multibuffer
            new_cfg.num_stages = _multibuffer_to_num_stages(opposite_default_multibuffer)
            simd_configs.append(new_cfg)

        if self.print_autotuning:
            print("Triton autotuning: Expanded to "
                  f"{len(simd_configs)} SIMD configs (toggle multibuffer={opposite_default_multibuffer})")
        return simd_configs

    def _parse_hints_axes(self, hints_axes):
        if hints_axes is None:
            return {}
        if not isinstance(hints_axes, dict):
            raise ValueError("hints.axes must be a dict, got: {}".format(type(hints_axes)))
        if not set(hints_axes.keys()).issubset(set(valid_axis_names)):
            raise ValueError("All keys in 'hints.axes' must be valid axis names. Got unexpected keys.")
        for axis, arg_name in hints_axes.items():
            if not isinstance(arg_name, str):
                raise ValueError("All values in 'hints.axes' must be non-empty argument names. "
                                 "Got {} for axis '{}'.".format(type(arg_name), axis))
            if not arg_name:
                raise ValueError("All values in 'hints.axes' must be non-empty argument names.")
            if arg_name.isidentifier() and arg_name not in self._get_runtime_arg_names_for_hints_axes():
                raise ValueError("All values in 'hints.axes' must reference runtime non-constexpr argument names. "
                                 "Got '{}' for axis '{}'.".format(arg_name, axis))
        return dict(hints_axes)

    def _normalize_autotuning_debug_value(self, value):
        if is_dataclass(value) and not isinstance(value, type):
            return self._normalize_autotuning_debug_value(asdict(value))
        if isinstance(value, dict):
            return {str(key): self._normalize_autotuning_debug_value(val) for key, val in value.items()}
        if isinstance(value, (list, tuple)):
            return [self._normalize_autotuning_debug_value(item) for item in value]
        if isinstance(value, set):
            return [self._normalize_autotuning_debug_value(item) for item in sorted(value, key=repr)]
        if hasattr(value, "__dict__") and not isinstance(value, type):
            public_attrs = {key: val for key, val in vars(value).items() if not key.startswith("_")}
            if public_attrs:
                return self._normalize_autotuning_debug_value(public_attrs)
        if isinstance(value, (str, int, float, bool)) or value is None:
            return value
        return repr(value)

    def _print_autotuning_parse_result(self, label: str, result) -> None:
        if not self.print_autotuning:
            return
        normalized_result = self._normalize_autotuning_debug_value(result)
        print("{}: {}".format(
            label,
            pprint.pformat(normalized_result, sort_dicts=True),
        ))

    def _collect_vector_parse_debug_result(
        self,
        required_missing_params: List[str],
        missing_tunable_params: List[str],
    ) -> Dict[str, object]:
        vector_axes = getattr(self, "vector_axes", None)
        if self.vv_parse_result_v2 is not None:
            return {"vector_axes": vector_axes}

        debug_result = {"vector_axes": vector_axes}
        fixed_split_params = dict(getattr(self, "fixed_split_params", {}) or {})
        if fixed_split_params:
            debug_result["fixed_split_params"] = fixed_split_params
        if missing_tunable_params:
            debug_result["missing_tunable_params"] = list(missing_tunable_params)
        if required_missing_params:
            debug_result["required_missing_params"] = list(required_missing_params)
        num_buffers = getattr(self, "num_buffers", -1)
        if num_buffers != -1:
            debug_result["num_buffers"] = num_buffers
        if getattr(self, "persistent_reduction", False):
            debug_result["persistent_reduction"] = True
        if getattr(self, "dual_reduction", False):
            debug_result["dual_reduction"] = True
        return debug_result

    def _get_runtime_arg_names_for_hints_axes(self):
        constexpr_names = set()
        if hasattr(self, "_get_constexpr_candidates"):
            try:
                constexpr_names = set(self._get_constexpr_candidates())
            except Exception:
                constexpr_names = set()
        return [arg_name for arg_name in getattr(self, "arg_names", []) if arg_name not in constexpr_names]

    def _rebuild_vector_axes(self) -> VectorAxes:
        vector_axes = VectorAxes.from_hints_axes(self.hints_axes)

        existing_vector_axes = getattr(self, "vector_axes", None)
        if existing_vector_axes is not None:
            vector_axes.apply_semantic_fields(
                axis_length_exprs=self._normalize_axis_name_mapping(
                    getattr(existing_vector_axes, "axis_length_exprs", {}), ),
                fixed_tiling_exprs=self._normalize_axis_name_mapping(
                    getattr(existing_vector_axes, "fixed_tiling_exprs", {}), ),
                axis_dynamic_sources=self._normalize_axis_name_mapping(
                    getattr(existing_vector_axes, "axis_dynamic_sources", {}), ),
            )

        vector_axes.apply_semantic_fields(
            split_params=self.split_params,
            tiling_params=self.tiling_params,
            low_dim_axes=self.low_dim_axes,
            reduction_axes=self.reduction_axes,
            axis_pid_dims=self.axis_pid_dims,
        )
        return vector_axes

    def _refresh_vector_axes(self) -> None:
        self.vector_axes = self._rebuild_vector_axes()
        self.axis_arg_names = self._get_parser_axis_arg_names()

    def _reset_vector_parse_derived_state(self) -> None:
        self.fixed_split_params = {}
        self.fixed_grid_dims = set()
        self.fixed_grid_dim_values = {}
        self.split_axis_pid_dims = {}
        self.axis_pid_dims = {}
        self.dual_reduction = False
        self.persistent_reduction = False

    def _get_parser_axis_arg_names(self) -> Dict[str, str]:
        axis_arg_names = {}
        if self.vector_axes is not None:
            axis_arg_names.update({
                axis: arg_name
                for axis, arg_name in self.vector_axes.axis_length_exprs.items()
                if self._is_direct_runtime_length_arg_name(arg_name)
            })

        return axis_arg_names

    def _promote_axis_arg_name_to_reduction(self, axis):
        axis = self._get_axis_base_name(axis)
        if not isinstance(axis, str) or not axis:
            return
        if self.vector_axes is None:
            self.vector_axes = self._rebuild_vector_axes()
        axis_arg_name = self.vector_axes.axis_length_exprs.get(axis, None)
        self.vector_axes.ensure_axis(axis)
        if axis_arg_name is not None:
            self.vector_axes.ensure_axis(axis).length_expr = axis_arg_name
        self.vector_axes.apply_semantic_fields(reduction_axes=[axis])
        self.axis_arg_names = self._get_parser_axis_arg_names()

    def _init_axis_params(self, key, split_params, tiling_params, low_dim_axes, reduction_axes, hints_axes=None):
        if not isinstance(key, Sequence) or isinstance(key, (str, bytes, dict)):
            raise ValueError("key must be a list, got: {}".format(type(key)))
        self.keys = list(key)
        self.hints_axes = self._parse_hints_axes(hints_axes)

        axis_related_params = [split_params, tiling_params, low_dim_axes, reduction_axes]
        if any(item is not None for item in axis_related_params):
            if not self.hints_axes:
                raise ValueError("hints.axes must be provided when axis metadata "
                                 "(split_params, tiling_params, low_dim_axes, reduction_axes) is supplied.")
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
            if not used_axes.issubset(self.hints_axes.keys()):
                raise ValueError(
                    "The following axes are used in axis metadata but missing from 'hints.axes': {}".format(
                        used_axes - set(self.hints_axes.keys())))
        elif not self.hints_axes:
            defer_legacy_fallback = (getattr(self, "enable_vv_parser_v2", False)
                                     and getattr(self, "vv_parser_v2_mode", "off") in ("assist", "authoritative"))
            if not defer_legacy_fallback:
                # Compatibility fallback for the legacy vector path: infer x/y/z...
                # from the ordered key list when no explicit axis metadata is supplied.
                self.hints_axes = self._infer_hints_axes_from_key()

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
        self.parser_mode = None
        self.missing_tunable_params = []
        self.cv_parse_result = None
        self.vv_parse_result_v2 = None
        self.vv_adapter_result_v2 = None
        self.vv_gap_fill_report_v2 = None
        self.vector_axes = self._rebuild_vector_axes()
        self.axis_arg_names = self._get_parser_axis_arg_names()

    def _autoparse_axis_params(self, all_args):
        if self.parser_mode is None:
            self.parser_mode = self._resolve_kernel_type()

        if self.parser_mode in ("cube", "mix"):
            result = self._autoparse_axis_params_cube_mix(all_args)
            self.cv_parse_result = result
            return result

        return self._autoparse_axis_params_vector(all_args)

    @staticmethod
    def _find_function_by_name(
        tree: ast.AST,
        func_name: Optional[str],
    ) -> Optional[ast.AST]:
        if not func_name:
            return None
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == func_name:
                return node
        return None

    @staticmethod
    def _find_first_function(tree: ast.AST) -> Optional[ast.AST]:
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                return node
        return None

    @staticmethod
    def _collect_direct_name_callees(function_node: ast.AST) -> List[str]:
        callee_names = set()
        for node in ast.walk(function_node):
            if not isinstance(node, ast.Call):
                continue
            if not isinstance(node.func, ast.Name):
                continue
            callee_names.add(node.func.id)
        return sorted(callee_names)

    def _has_direct_importable_callee(self, function_node: ast.AST) -> bool:
        globals_dict = getattr(self.base_fn, "__globals__", {})
        for callee_name in self._collect_direct_name_callees(function_node):
            callee_obj = globals_dict.get(callee_name, None)
            if isinstance(callee_obj, triton.runtime.JITFunction):
                return True
            if inspect.isfunction(callee_obj):
                return True
        return False

    def _should_expand_vv_ast_context(
        self,
        parsed_ast: ast.AST,
        entry_function_name: Optional[str],
    ) -> bool:
        entry_function = self._find_function_by_name(parsed_ast, entry_function_name)
        if entry_function is None:
            if isinstance(parsed_ast, (ast.FunctionDef, ast.AsyncFunctionDef)):
                entry_function = parsed_ast
            elif isinstance(parsed_ast, ast.Module):
                entry_function = self._find_first_function(parsed_ast)

        if not isinstance(entry_function, (ast.FunctionDef, ast.AsyncFunctionDef)):
            return False
        return self._has_direct_importable_callee(entry_function)

    def _augment_module_ast_with_imported_callees(
        self,
        module_ast: ast.Module,
        entry_function_name: Optional[str],
    ) -> ast.Module:
        if not entry_function_name:
            return module_ast

        entry_function = self._find_function_by_name(module_ast, entry_function_name)
        if not isinstance(entry_function, (ast.FunctionDef, ast.AsyncFunctionDef)):
            return module_ast

        known_function_names = {
            node.name
            for node in ast.walk(module_ast)
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        }

        globals_dict = getattr(self.base_fn, "__globals__", {})
        appended = 0
        for callee_name in self._collect_direct_name_callees(entry_function):
            if callee_name in known_function_names:
                continue

            callee_obj = globals_dict.get(callee_name, None)
            if isinstance(callee_obj, triton.runtime.JITFunction):
                callee_obj = callee_obj.fn
            if not inspect.isfunction(callee_obj):
                continue

            source_file = None
            try:
                source_file = inspect.getsourcefile(callee_obj) or inspect.getfile(callee_obj)
            except Exception:
                source_file = None
            if not source_file or not os.path.isfile(source_file):
                continue

            try:
                with open(source_file, "r", encoding="utf-8") as source_fp:
                    source = source_fp.read()
                callee_module_ast = ast.parse(source, filename=source_file)
            except Exception:
                continue

            callee_def = self._find_function_by_name(callee_module_ast, getattr(callee_obj, "__name__", None))
            if not isinstance(callee_def, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            if callee_def.name in known_function_names:
                continue

            module_ast.body.append(callee_def)
            known_function_names.add(callee_def.name)
            appended += 1

        return module_ast

    def _load_source_module_ast(self) -> Optional[ast.Module]:
        if self._source_module_ast_resolved:
            return self._source_module_ast_cache
        self._source_module_ast_resolved = True

        source_file = None
        try:
            source_file = inspect.getsourcefile(self.base_fn) or inspect.getfile(self.base_fn)
        except Exception:
            source_file = None

        if not source_file or not os.path.isfile(source_file):
            return None

        try:
            with open(source_file, "r", encoding="utf-8") as source_fp:
                source = source_fp.read()
            module_ast = ast.parse(source, filename=source_file)
        except Exception:
            return None

        self._source_module_ast_cache = module_ast
        return self._source_module_ast_cache

    def _build_cv_parse_ast_context(
        self,
        parsed_ast: ast.AST,
        entry_function_name: Optional[str],
    ) -> Tuple[ast.AST, Optional[ast.Module], Optional[str], str]:
        module_ast = self._load_source_module_ast()
        if isinstance(module_ast, ast.Module):
            module_ast = self._augment_module_ast_with_imported_callees(
                module_ast,
                entry_function_name,
            )
            entry_from_source = self._find_function_by_name(module_ast, entry_function_name)
            if entry_from_source is not None:
                return entry_from_source, module_ast, entry_function_name, "source_file"

        if isinstance(parsed_ast, (ast.FunctionDef, ast.AsyncFunctionDef)):
            module_ast = ast.Module(body=[parsed_ast], type_ignores=[])
            if not entry_function_name:
                entry_function_name = parsed_ast.name
            return parsed_ast, module_ast, entry_function_name, "jit_parse_function"

        if isinstance(parsed_ast, ast.Module):
            entry_from_parse = self._find_function_by_name(parsed_ast, entry_function_name)
            if entry_from_parse is None:
                entry_from_parse = self._find_first_function(parsed_ast)
            if entry_from_parse is not None and not entry_function_name:
                entry_function_name = getattr(entry_from_parse, "name", None)
            if entry_from_parse is not None:
                return entry_from_parse, parsed_ast, entry_function_name, "jit_parse_module"
            return parsed_ast, parsed_ast, entry_function_name, "jit_parse_module_no_entry"

        return parsed_ast, None, entry_function_name, "jit_parse_unknown"

    def _is_auto_kernel_hint(self) -> bool:
        kernel_type_hint = self.hints.get("kernel_type", "auto")
        if isinstance(kernel_type_hint, str):
            kernel_type_hint = kernel_type_hint.strip().lower()
        return kernel_type_hint == "auto"

    def _resolve_kernel_type(self) -> str:
        parsed_ast = None
        try:
            parsed_ast = self.fn.parse()
        except Exception:
            parsed_ast = None

        kernel_type = resolve_kernel_type(self.hints, parsed_ast)
        if kernel_type != "vector" or not self._is_auto_kernel_hint() or parsed_ast is None:
            return kernel_type

        entry_function_name = getattr(self.base_fn, "__name__", None) or getattr(self.fn, "__name__", None)
        entry_ast, module_ast, entry_function_name, ast_context = self._build_cv_parse_ast_context(
            parsed_ast,
            entry_function_name,
        )
        if not isinstance(module_ast, ast.Module) or not entry_function_name:
            return kernel_type

        try:
            dot_result = analyze_dot_site_mnk(
                func_ast=entry_ast,
                parser_mode="mix",
                module_ast=module_ast,
                entry_function_name=entry_function_name,
            )
        except Exception:
            return kernel_type

        if dot_result.dot_sites:
            return "mix"
        return kernel_type

    def _autoparse_axis_params_cube_mix(self, all_args):
        parsed_ast = self.fn.parse()
        entry_function_name = getattr(self.base_fn, "__name__", None) or getattr(self.fn, "__name__", None)
        func_ast, module_ast, entry_function_name, ast_context = self._build_cv_parse_ast_context(
            parsed_ast,
            entry_function_name,
        )
        cv_parse_result = parse_cv_params(
            func_ast=func_ast,
            parser_mode=self.parser_mode or "mix",
            arg_names=self.arg_names,
            provided_args=all_args,
            explicit_tunable_params=self.explicit_tunable_params,
            module_ast=module_ast,
            entry_function_name=entry_function_name,
        )
        self.cv_parse_result = cv_parse_result
        self.missing_tunable_params = list(cv_parse_result.missing_tunable_params)
        self._print_autotuning_parse_result(
            "Ascend autotuning cv parse result",
            cv_parse_result,
        )
        if cv_parse_result.status != "ok":
            diagnostics = "; ".join(cv_parse_result.diagnostics).strip()
            if not diagnostics:
                diagnostics = "no diagnostics"
            raise RuntimeError("cv parser status {}: {}".format(cv_parse_result.status, diagnostics))
        return cv_parse_result

    def _autoparse_axis_params_vector(self, all_args):
        self.vv_gap_fill_report_v2 = None
        self._reset_vector_parse_derived_state()
        # Normalize vector axis containers for vv-assist flow. In list-key mode
        # these fields are initialized as None, and vv may legitimately fill only
        # split/tiling/low_dim without reduction axes.
        if self.split_params is None:
            self.split_params = {}
        if self.tiling_params is None:
            self.tiling_params = {}
        if self.low_dim_axes is None:
            self.low_dim_axes = []
        if self.reduction_axes is None:
            self.reduction_axes = []
        self._refresh_vector_axes()

        required_missing_params = [arg for arg in self.arg_names if arg not in all_args.keys()]
        vv_injected_split_axes = set()
        vv_injected_tiling_axes = set()
        existing_split_axes = set(self.split_params.keys())
        existing_tiling_axes = set(self.tiling_params.keys())
        vv_result = self._autoparse_axis_info_v2_for_vector(all_args)
        self._apply_assist_legacy_hints_axes_fallback(all_args, vv_result)
        use_vv_semantics = False
        fallback_to_vector_all = self._should_fallback_to_vector_from_v2(vv_result)
        if not fallback_to_vector_all and self._is_vv_semantic_usable(vv_result, required_missing_params):
            use_vv_semantics = self._apply_vv_axis_semantic_result()
            if use_vv_semantics:
                vv_injected_split_axes = set(self.split_params.keys()) - existing_split_axes
                vv_injected_tiling_axes = set(self.tiling_params.keys()) - existing_tiling_axes

        missing_tunable_params = self._detect_missing_tunable_params(
            all_args,
            required_missing_params,
        )
        self._filter_vv_semantic_tunable_params_by_cv_policy(
            required_missing_params=required_missing_params,
            vv_injected_split_axes=vv_injected_split_axes,
            vv_injected_tiling_axes=vv_injected_tiling_axes,
        )
        self.missing_tunable_params = list(missing_tunable_params)
        vector_gap_fill_applied = []
        self._prepare_vv_gap_fill_report(
            required_missing_params=required_missing_params,
            missing_tunable_params=missing_tunable_params,
        )
        # parse pointer params nums
        if self.num_buffers == -1:
            self.num_buffers = self._autoparse_ptr_nums(all_args)

        # parse autotiling axes
        # reduction axis must be parsed before other axes. it will alter the key
        if not self.reduction_axes:
            if self._should_skip_vv_vector_gap_fill():
                self.reduction_axes = []
            else:
                vector_gap_fill_applied.append("reduction_axes")
                self.reduction_axes = self._parse_reduction_axes_with_fallback()
        if len(self.reduction_axes) >= 2:
            self.dual_reduction = True

        if not self.low_dim_axes:
            if self._should_skip_vv_vector_gap_fill():
                self.low_dim_axes = []
            else:
                vector_gap_fill_applied.append("low_dim_axes")
                self.low_dim_axes = self._parse_low_dim_axes_with_fallback()

        if len(self.reduction_axes) == 1:
            reduction_axis = self.reduction_axes[0]
            reduction_param = self._resolve_axis_length_arg_name(reduction_axis)
            reduction_numel = all_args.get(reduction_param, float("inf"))
            persistent_threshold = self._get_persistent_reduction_threshold(reduction_axis)
            if reduction_numel <= persistent_threshold:
                self.persistent_reduction = True

        if not self.split_params:
            if self._should_skip_vv_vector_gap_fill():
                self.all_split_params = {}
            else:
                vector_gap_fill_applied.append("split_params")
                all_split_params = self._parse_split_params_with_fallback(self._get_constexpr_candidates())
                self.all_split_params = dict(all_split_params)
                self.split_params = {
                    axis: param
                    for axis, param in all_split_params.items()
                    if param in missing_tunable_params
                }
        elif not self.axis_pid_dims:
            # When split axes are provided by hints, parse axis->program_id mapping
            # independently for fixed-grid semantics and diagnostics.
            if not self._should_skip_vv_vector_gap_fill():
                vector_gap_fill_applied.append("axis_pid_dims")
                self._parse_axis_pid_dims_with_fallback()
        self._apply_fixed_grid_semantics(all_args)
        missing_tunable_params = [arg for arg in missing_tunable_params if arg not in self.split_params.values()]
        if not self.tiling_params:
            if self._should_skip_vv_vector_gap_fill():
                self.tiling_params = {}
            else:
                vector_gap_fill_applied.append("tiling_params")
                self.tiling_params = self._parse_tiling_params_with_fallback(missing_tunable_params)
        auto_fallback_report = self._fallback_unresolved_tunable_params_to_split_tiling(missing_tunable_params)
        if self.vv_gap_fill_report_v2 is not None:
            self.vv_gap_fill_report_v2["auto_fallback_split_tiling"] = auto_fallback_report
        missing_tunable_params = [arg for arg in missing_tunable_params if arg not in self.split_params.values()]
        missing_tunable_params = [arg for arg in missing_tunable_params if arg not in self.tiling_params.values()]
        required_missing_params = [arg for arg in required_missing_params if arg not in self.split_params.values()]
        required_missing_params = [arg for arg in required_missing_params if arg not in self.tiling_params.values()]
        required_missing_params = self._filter_hard_missing_required_params(required_missing_params)
        self._finalize_vv_gap_fill_report(
            vector_gap_fill_applied=vector_gap_fill_applied,
            required_missing_params=required_missing_params,
            missing_tunable_params=missing_tunable_params,
        )
        self._refresh_vector_axes()
        self._print_autotuning_parse_result(
            "Ascend autotuning vector final parse result",
            self._collect_vector_parse_debug_result(
                required_missing_params=required_missing_params,
                missing_tunable_params=missing_tunable_params,
            ),
        )
        if required_missing_params:
            raise ValueError(f"Missing required arguments: {required_missing_params}. "
                             f"These arguments must be explicitly provided and cannot be automatically tuned. "
                             f"Please ensure that these arguments are passed when calling the function.")

    @staticmethod
    def _require_base_axis_name(axis: Optional[str]) -> Optional[str]:
        if not isinstance(axis, str) or not axis:
            return None
        if axis.startswith("r"):
            raise ValueError("r-prefixed axis names are not supported; "
                             f"use the base axis name instead of '{axis}'.")
        if axis not in valid_axis_names:
            raise ValueError(f"Unsupported axis name '{axis}'.")
        return axis

    @staticmethod
    def _normalize_reduction_axis_name(axis: Optional[str]) -> Optional[str]:
        if not isinstance(axis, str) or not axis:
            return None
        if axis.startswith("r"):
            raise ValueError("r-prefixed axis names are not supported; "
                             f"use the base axis name instead of '{axis}'.")
        if axis not in valid_axis_names:
            raise ValueError(f"Unsupported axis name '{axis}'.")
        return axis

    def _normalize_reduction_axes(self, reduction_axes: List[str]) -> List[str]:
        normalized = []
        for axis in reduction_axes:
            canonical_axis = self._normalize_reduction_axis_name(axis)
            if canonical_axis is not None:
                normalized.append(canonical_axis)
        return normalized

    def _normalize_vv_reduction_axes(self, reduction_axes: List[str]) -> List[str]:
        return self._normalize_reduction_axes(reduction_axes)

    def _normalize_axis_name_mapping(
        self,
        axis_mapping: Optional[Dict[str, Any]],
    ) -> Dict[str, Any]:
        normalized = {}
        for axis, value in dict(axis_mapping or {}).items():
            base_axis = self._get_axis_base_name(axis)
            if not isinstance(base_axis, str) or not base_axis:
                continue
            normalized.setdefault(base_axis, value)
        return normalized

    def _normalize_axis_name_list(
        self,
        axis_names: Optional[List[str]],
    ) -> List[str]:
        normalized = []
        for axis in list(axis_names or []):
            base_axis = self._get_axis_base_name(axis)
            if not isinstance(base_axis, str) or not base_axis:
                continue
            if base_axis not in normalized:
                normalized.append(base_axis)
        return normalized

    @staticmethod
    def _get_axis_base_name(axis: Optional[str]) -> Optional[str]:
        if not isinstance(axis, str) or not axis:
            return None
        if axis.startswith("r"):
            raise ValueError("r-prefixed axis names are not supported; "
                             f"use the base axis name instead of '{axis}'.")
        if axis not in valid_axis_names:
            raise ValueError(f"Unsupported axis name '{axis}'.")
        return axis

    def _apply_vv_axis_semantic_result(self) -> bool:

        def resolve_base_axis(axis_name):
            resolver = getattr(self, "_get_axis_base_name", None)
            if callable(resolver):
                return resolver(axis_name)
            if not isinstance(axis_name, str) or not axis_name:
                return None
            if axis_name.startswith("r"):
                raise ValueError("r-prefixed axis names are not supported; "
                                 f"use the base axis name instead of '{axis_name}'.")
            return axis_name

        def normalize_axis_name_mapping(axis_mapping):
            normalizer = getattr(self, "_normalize_axis_name_mapping", None)
            if callable(normalizer):
                return normalizer(axis_mapping)
            normalized = {}
            for axis, value in dict(axis_mapping or {}).items():
                base_axis = resolve_base_axis(axis)
                if not isinstance(base_axis, str) or not base_axis:
                    continue
                normalized.setdefault(base_axis, value)
            return normalized

        def normalize_axis_name_list(axis_names):
            normalizer = getattr(self, "_normalize_axis_name_list", None)
            if callable(normalizer):
                return normalizer(axis_names)
            normalized = []
            for axis in list(axis_names or []):
                base_axis = resolve_base_axis(axis)
                if not isinstance(base_axis, str) or not base_axis:
                    continue
                if base_axis not in normalized:
                    normalized.append(base_axis)
            return normalized

        if self.vv_adapter_result_v2 is None:
            return False
        adapter_status = getattr(self.vv_adapter_result_v2, "status", "ok")
        if adapter_status == "failed":
            return False

        applied = False
        raw_adapter_reduction_axes = list(getattr(self.vv_adapter_result_v2, "reduction_axes", []) or [])
        adapter_reduction_axes = self._normalize_vv_reduction_axes(raw_adapter_reduction_axes)
        raw_adapter_low_dim_axes = list(getattr(self.vv_adapter_result_v2, "low_dim_axes", []) or [])
        if adapter_reduction_axes or self.reduction_axes:
            self.reduction_axes = []
            for reduction_axis in adapter_reduction_axes:
                base_axis = resolve_base_axis(reduction_axis)
                self._promote_axis_arg_name_to_reduction(base_axis)
            self.reduction_axes = list(adapter_reduction_axes)
            applied = True

        adapter_low_dim_axes = normalize_axis_name_list(raw_adapter_low_dim_axes)
        if adapter_low_dim_axes or self.low_dim_axes:
            self.low_dim_axes = list(adapter_low_dim_axes)
            applied = True

        adapter_split_params = normalize_axis_name_mapping(getattr(self.vv_adapter_result_v2, "split_params", {}) or {})
        if adapter_split_params or self.split_params:
            self.split_params = dict(adapter_split_params)
            applied = True

        adapter_tiling_params = normalize_axis_name_mapping(
            getattr(self.vv_adapter_result_v2, "tiling_params", {}) or {})
        if adapter_tiling_params or self.tiling_params:
            self.tiling_params = dict(adapter_tiling_params)
            applied = True

        adapter_axis_pid_dims = normalize_axis_name_mapping(
            getattr(self.vv_adapter_result_v2, "axis_pid_dims", {}) or {})
        if adapter_axis_pid_dims or self.axis_pid_dims:
            self.axis_pid_dims = dict(adapter_axis_pid_dims)
            applied = True

        if len(self.reduction_axes or []) >= 2:
            self.dual_reduction = True
        return applied

    def _should_fallback_to_vector_from_v2(self, vv_result) -> bool:
        mode = getattr(self, "vv_parser_v2_mode", "off")
        if mode == "off":
            return True
        if mode == "observe":
            return True
        if vv_result is None:
            return True
        vv_status = getattr(vv_result, "status", "ok")
        if vv_status == "failed":
            return True
        if mode == "assist" and not self._vv_result_has_resolved_axes(vv_result):
            return True
        return False

    @staticmethod
    def _vv_result_has_resolved_axes(vv_result) -> bool:
        if vv_result is None:
            return False
        semantic_fields = (
            "axis_length_exprs",
            "inferred_keys",
            "split_params",
            "tiling_params",
            "fixed_tiling_exprs",
            "axis_pid_dims",
        )
        for field_name in semantic_fields:
            field_value = getattr(vv_result, field_name, None)
            if isinstance(field_value, dict) and field_value:
                return True

        for axis_info in list(getattr(vv_result, "axes", []) or []):
            if getattr(axis_info, "length_expr", None):
                return True
            if getattr(axis_info, "split_param", None):
                return True
            if getattr(axis_info, "tiling_param", None):
                return True
            if getattr(axis_info, "fixed_tiling_expr", None):
                return True
            if getattr(axis_info, "tunable_param", None):
                return True

        return False

    def _infer_hints_axes_from_key(
        self,
        all_args: Optional[Dict[str, object]] = None,
        *,
        int_only: bool = False,
    ) -> Dict[str, str]:
        inferred_hints_axes = {}
        for axis, arg_name in zip(valid_axis_names, self.keys):
            if not (isinstance(arg_name, str) and arg_name):
                continue
            if int_only:
                if all_args is None or arg_name not in all_args:
                    continue
                arg_value = all_args[arg_name]
                if isinstance(arg_value, bool):
                    continue
                if not isinstance(arg_value, int):
                    continue
            inferred_hints_axes[axis] = arg_name
        return inferred_hints_axes

    def _apply_assist_legacy_hints_axes_fallback(
        self,
        all_args: Dict[str, object],
        vv_result,
    ) -> None:
        if not self.enable_vv_parser_v2:
            return
        if getattr(self, "vv_parser_v2_mode", "off") != "assist":
            return
        if self.hints_axes:
            return
        if self._vv_result_has_resolved_axes(vv_result):
            return
        legacy_hints_axes = self._infer_hints_axes_from_key(all_args, int_only=True)
        if not legacy_hints_axes:
            return
        self.hints_axes = legacy_hints_axes
        self._refresh_vector_axes()

    def _is_vv_semantic_usable(
        self,
        vv_result,
        required_missing_params: List[str],
    ) -> bool:
        if vv_result is None or self.vv_adapter_result_v2 is None:
            return False

        adapter_status = getattr(self.vv_adapter_result_v2, "status", "ok")
        if adapter_status == "failed":
            return False

        adapter_split_params = dict(getattr(self.vv_adapter_result_v2, "split_params", {}) or {})
        adapter_tiling_params = dict(getattr(self.vv_adapter_result_v2, "tiling_params", {}) or {})
        mapped_params = set(adapter_split_params.values()) | set(adapter_tiling_params.values())
        if not mapped_params:
            return False

        constexpr_candidates = set(self._get_constexpr_candidates())
        missing_constexpr = set(required_missing_params) & constexpr_candidates
        if not missing_constexpr:
            return True
        if mapped_params & missing_constexpr:
            return True

        if "x" in adapter_split_params or "x" in adapter_tiling_params:
            return True
        return False

    def _should_skip_vv_vector_gap_fill(self) -> bool:
        return (self.enable_vv_parser_v2 and (self.parser_mode or "vector") == "vector"
                and getattr(self, "vv_parser_v2_mode", "off") == "authoritative")

    @staticmethod
    def _is_direct_runtime_length_arg_name(expr: Optional[str]) -> bool:
        return isinstance(expr, str) and expr.isidentifier()

    def _resolve_axis_length_arg_name(self, axis: str) -> Optional[str]:
        axis_length_exprs = {}
        if self.vector_axes is not None:
            axis_length_exprs = dict(getattr(self.vector_axes, "axis_length_exprs", {}) or {})
        elif self.vv_adapter_result_v2 is not None:
            axis_length_exprs = dict(getattr(self.vv_adapter_result_v2, "axis_length_exprs", {}) or {})
        axis_expr = axis_length_exprs.get(axis, None)
        if self._is_direct_runtime_length_arg_name(axis_expr):
            return axis_expr
        parser_axis_arg_names = self._get_parser_axis_arg_names()
        return parser_axis_arg_names.get(axis, None)

    def _materialize_vector_tile_inputs(
        self,
        kv_dict: Dict[str, object],
        all_args: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, object]:
        vector_axes = VectorAxes.from_hints_axes(self.hints_axes)

        if self.vector_axes is not None:
            vector_axes.apply_semantic_fields(
                split_params=self.vector_axes.split_params,
                tiling_params=self.vector_axes.tiling_params,
                low_dim_axes=self.vector_axes.low_dim_axes,
                reduction_axes=self.vector_axes.reduction_axes,
                axis_pid_dims=self.vector_axes.axis_pid_dims,
                axis_length_exprs=self.vector_axes.axis_length_exprs,
                fixed_tiling_exprs=self.vector_axes.fixed_tiling_exprs,
                axis_dynamic_sources=self.vector_axes.axis_dynamic_sources,
            )

        if self.split_params is not None:
            for axis in vector_axes.axes.values():
                axis.split_param = None
            vector_axes.apply_semantic_fields(split_params=self.split_params)
        if self.tiling_params is not None:
            for axis in vector_axes.axes.values():
                axis.tiling_param = None
            vector_axes.apply_semantic_fields(tiling_params=self.tiling_params)
        if self.low_dim_axes is not None:
            vector_axes._low_dim_axis_names = []
            for axis in vector_axes.axes.values():
                axis.is_low_dim = False
            vector_axes.apply_semantic_fields(low_dim_axes=self.low_dim_axes)
        if self.reduction_axes is not None:
            vector_axes._reduction_axis_names = []
            for axis in vector_axes.axes.values():
                axis.is_reduction = False
            vector_axes.apply_semantic_fields(reduction_axes=self.reduction_axes)
        if self.axis_pid_dims is not None:
            for axis in vector_axes.axes.values():
                axis.pid_dim = None
            vector_axes.apply_semantic_fields(axis_pid_dims=self.axis_pid_dims)

        runtime_view = dict(all_args or {})
        for axis, value in dict(kv_dict or {}).items():
            if not is_valid_axis_name(axis):
                continue
            if not isinstance(value, int):
                raise ValueError(f"Not supported dim type: {type(value)}, `int` is the only supported type")
            vector_axes.ensure_axis(axis)
            runtime_view[axis] = value

        axis_sizes, diagnostics = vector_axes.materialize_axis_sizes(runtime_view)
        self.vector_axes = vector_axes
        return {
            "axis_sizes": axis_sizes,
            "split_params": vector_axes.split_params,
            "tiling_params": vector_axes.tiling_params,
            "low_dim_axes": vector_axes.low_dim_axes,
            "reduction_axes": vector_axes.reduction_axes,
            "axis_pid_dims": vector_axes.axis_pid_dims,
            "diagnostics": diagnostics,
        }

    def _apply_fixed_grid_semantics(
        self,
        all_args: Dict[str, object],
    ) -> None:
        self.fixed_split_params = {}
        self.fixed_grid_dim_values = self._get_fixed_grid_dim_values(
            all_args.get("grid", None),
            all_args,
        )
        self.fixed_grid_dims = set(self.fixed_grid_dim_values.keys())
        if not self.fixed_grid_dims or not self.axis_pid_dims:
            return

        fixed_grid_axes = {axis for axis, pid_dim in self.axis_pid_dims.items() if pid_dim in self.fixed_grid_dims}
        if not fixed_grid_axes:
            return

        for axis, pid_dim in self.axis_pid_dims.items():
            if pid_dim not in self.fixed_grid_dims:
                continue
            core_num = self.fixed_grid_dim_values.get(pid_dim, 0)
            axis_len_name = self._resolve_axis_length_arg_name(axis)
            axis_len = all_args.get(axis_len_name, None)
            if not isinstance(core_num, int) or core_num <= 0:
                continue
            if not isinstance(axis_len, int) or axis_len <= 0:
                continue
            self.fixed_split_params[axis] = (axis_len + core_num - 1) // core_num

    def _collect_vv_vector_gap_fill_needed(self) -> List[str]:
        needed = []
        if not self.reduction_axes:
            needed.append("reduction_axes")
        if not self.low_dim_axes:
            needed.append("low_dim_axes")
        if not self.split_params:
            needed.append("split_params")
        if not self.tiling_params:
            needed.append("tiling_params")
        if self.split_params and not self.axis_pid_dims:
            needed.append("axis_pid_dims")
        return needed

    def _prepare_vv_gap_fill_report(
        self,
        required_missing_params: List[str],
        missing_tunable_params: List[str],
    ) -> None:
        if not self.enable_vv_parser_v2 or (self.parser_mode or "vector") != "vector":
            return
        if getattr(self, "vv_parser_v2_mode", "off") not in ("observe", "authoritative"):
            return
        self.vv_gap_fill_report_v2 = {
            "mode":
            getattr(self, "vv_parser_v2_mode", "off"),
            "vv_status":
            (None if self.vv_parse_result_v2 is None else getattr(self.vv_parse_result_v2, "status", None)),
            "adapter_status":
            (None if self.vv_adapter_result_v2 is None else getattr(self.vv_adapter_result_v2, "status", None)),
            "required_missing_params_before_vector":
            list(required_missing_params),
            "missing_tunable_params_before_vector":
            list(missing_tunable_params),
            "vector_gap_fill_needed":
            self._collect_vv_vector_gap_fill_needed(),
            "vector_gap_fill_applied": [],
            "vv_diagnostics": ([] if self.vv_parse_result_v2 is None else list(
                getattr(self.vv_parse_result_v2, "diagnostics", []) or [])),
            "adapter_diagnostics": ([] if self.vv_adapter_result_v2 is None else list(
                getattr(self.vv_adapter_result_v2, "diagnostics", []) or [])),
        }

    def _finalize_vv_gap_fill_report(
        self,
        vector_gap_fill_applied: List[str],
        required_missing_params: List[str],
        missing_tunable_params: List[str],
    ) -> None:
        if self.vv_gap_fill_report_v2 is None:
            return
        self.vv_gap_fill_report_v2.update({
            "vector_gap_fill_applied": list(vector_gap_fill_applied),
            "required_missing_params_final": list(required_missing_params),
            "missing_tunable_params_final": list(missing_tunable_params),
            "final_split_params": dict(self.split_params or {}),
            "final_tiling_params": dict(self.tiling_params or {}),
            "final_low_dim_axes": list(self.low_dim_axes or []),
            "final_reduction_axes": list(self.reduction_axes or []),
            "final_axis_pid_dims": dict(self.axis_pid_dims or {}),
        })

    def _run_vector_parser_with_fallback(
        self,
        parser_name: str,
        parser_fn,
        fallback_value,
    ):
        try:
            return parser_fn()
        except Exception:
            if self.print_autotuning:
                print("[WARNING] Failed to parse {}, fallback to {}.".format(parser_name, fallback_value))
            return fallback_value

    def _parse_reduction_axes_with_fallback(self) -> List[str]:
        result = self._run_vector_parser_with_fallback(
            "reduction_axes",
            self._autoparse_reduction_axes,
            [],
        )
        return list(result or [])

    def _parse_low_dim_axes_with_fallback(self) -> List[str]:
        result = self._run_vector_parser_with_fallback(
            "low_dim_axes",
            self._autoparse_low_dim_axes,
            [],
        )
        return list(result or [])

    def _parse_split_params_with_fallback(
        self,
        candidates_params: List[str],
    ) -> Dict[str, str]:
        result = self._run_vector_parser_with_fallback(
            "split_params",
            lambda: self._autoparse_split_params(candidates_params),
            {},
        )
        return dict(result or {})

    def _parse_tiling_params_with_fallback(
        self,
        candidates_params: List[str],
    ) -> Dict[str, str]:
        result = self._run_vector_parser_with_fallback(
            "tiling_params",
            lambda: self._autoparse_tiling_params(candidates_params),
            {},
        )
        return dict(result or {})

    def _parse_axis_pid_dims_with_fallback(self) -> Dict[str, int]:
        result = self._run_vector_parser_with_fallback(
            "axis_pid_dims",
            self._autoparse_axis_pid_dims,
            {},
        )
        if not result:
            self.axis_pid_dims = {}
            self.split_axis_pid_dims = {}
            return {}
        return dict(result)

    def _get_defaulted_argument_names(self) -> List[str]:
        func_ast = self.fn.parse()
        func_node = self._find_first_function(func_ast)
        if not isinstance(func_node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            return []
        if not isinstance(func_node.args, ast.arguments):
            return []

        defaulted = set()

        positional_args = list(func_node.args.args)
        positional_defaults = list(func_node.args.defaults)
        if positional_defaults:
            default_count = len(positional_defaults)
            for arg in positional_args[-default_count:]:
                if isinstance(arg, ast.arg):
                    defaulted.add(arg.arg)

        for kw_arg, kw_default in zip(
                func_node.args.kwonlyargs,
                func_node.args.kw_defaults,
        ):
            if kw_default is None:
                continue
            if isinstance(kw_arg, ast.arg):
                defaulted.add(kw_arg.arg)
        return list(defaulted)

    def _filter_hard_missing_required_params(
        self,
        required_missing_params: List[str],
    ) -> List[str]:
        if not required_missing_params:
            return []
        defaulted_args = set(self._get_defaulted_argument_names())
        return [arg_name for arg_name in required_missing_params if arg_name not in defaulted_args]

    def _resolve_vv_tunable_candidates_by_cv_policy(
        self,
        required_missing_params: List[str],
    ) -> Tuple[set, set, set, set]:
        constexpr_candidates = set(self._get_constexpr_candidates())
        defaulted_args = set(self._get_defaulted_argument_names())
        explicit_tunable_params = set(self.explicit_tunable_params or [])
        candidate_tunable_params = set()

        for arg_name in required_missing_params:
            if arg_name not in constexpr_candidates:
                continue
            if arg_name in defaulted_args and arg_name not in explicit_tunable_params:
                continue
            candidate_tunable_params.add(arg_name)

        return (
            candidate_tunable_params,
            constexpr_candidates,
            defaulted_args,
            explicit_tunable_params,
        )

    def _filter_vv_semantic_tunable_params_by_cv_policy(
        self,
        required_missing_params: List[str],
        vv_injected_split_axes: set,
        vv_injected_tiling_axes: set,
    ) -> None:
        if not vv_injected_split_axes and not vv_injected_tiling_axes:
            return

        (
            candidate_tunable_params,
            constexpr_candidates,
            defaulted_args,
            explicit_tunable_params,
        ) = self._resolve_vv_tunable_candidates_by_cv_policy(required_missing_params)

        removed_split_params = {}
        for axis in list(vv_injected_split_axes):
            param = self.split_params.get(axis, None)
            if not isinstance(param, str):
                continue
            if param in candidate_tunable_params:
                continue
            removed_split_params[axis] = param
            self.split_params.pop(axis, None)

        removed_tiling_params = {}
        for axis in list(vv_injected_tiling_axes):
            param = self.tiling_params.get(axis, None)
            if not isinstance(param, str):
                continue
            if param in candidate_tunable_params:
                continue
            removed_tiling_params[axis] = param
            self.tiling_params.pop(axis, None)

    @staticmethod
    def _axis_name_affinity_score(param_name: str, axis_name: str) -> int:
        if not isinstance(param_name, str) or not isinstance(axis_name, str):
            return 0
        param_upper = param_name.upper()
        if not param_upper:
            return 0

        axis_token = axis_name.upper()
        if not axis_token:
            return 0

        if param_upper.startswith(axis_token):
            return 100
        if "_{}_".format(axis_token) in "_{}_".format(param_upper):
            return 80
        return 0

    @staticmethod
    def _param_anchor_affinity_score(param_name: str, anchor_name: Optional[str]) -> int:
        if not isinstance(param_name, str) or not isinstance(anchor_name, str):
            return 0
        if not param_name or not anchor_name:
            return 0
        param_upper = param_name.upper()
        anchor_upper = anchor_name.upper()
        if param_upper == anchor_upper:
            return 100
        if param_upper.startswith(anchor_upper) or anchor_upper.startswith(param_upper):
            return 90
        if (param_upper.replace("_SUB", "") == anchor_upper or anchor_upper.replace("_SUB", "") == param_upper):
            return 88
        common_prefix = 0
        for lhs, rhs in zip(param_upper, anchor_upper):
            if lhs != rhs:
                break
            common_prefix += 1
        if common_prefix <= 0:
            return 0
        return int((common_prefix * 100) / max(len(param_upper), len(anchor_upper)))

    def _fallback_unresolved_tunable_params_to_split_tiling(
        self,
        missing_tunable_params: List[str],
    ) -> Dict[str, object]:
        report = {
            "mode": getattr(self, "vv_parser_v2_mode", "off"),
            "initial_unresolved_params": [],
            "assigned_tiling_params": {},
            "assigned_split_params": {},
            "remaining_unresolved_params": [],
            "decisions": [],
            "applied": False,
        }
        if not (self.enable_vv_parser_v2 and (self.parser_mode or "vector") == "vector"
                and getattr(self, "vv_parser_v2_mode", "off") in ("assist", "authoritative")):
            return report

        consumed_params = set(self.split_params.values()) | set(self.tiling_params.values())
        unresolved = []
        seen_unresolved = set()
        for param in missing_tunable_params:
            if param in consumed_params or param in seen_unresolved:
                continue
            unresolved.append(param)
            seen_unresolved.add(param)
        if not unresolved:
            return report
        report["initial_unresolved_params"] = list(unresolved)

        ordered_axes = [
            axis for axis in self.axis_arg_names.keys() if isinstance(axis, str) and is_valid_axis_name(axis)
        ]
        if not ordered_axes:
            report["remaining_unresolved_params"] = list(unresolved)
            return report
        axis_order = {axis: idx for idx, axis in enumerate(ordered_axes)}

        empty_axes = [axis for axis in ordered_axes if axis not in self.split_params and axis not in self.tiling_params]
        split_only_axes = [
            axis for axis in ordered_axes if axis in self.split_params and axis not in self.tiling_params
        ]
        tiling_only_axes = [
            axis for axis in ordered_axes if axis in self.tiling_params and axis not in self.split_params
        ]

        def _pick_best_axis(param_name: str, candidate_axes: List[str], anchor_map: Dict[str, str]):
            best_axis = None
            best_score = None
            for axis in candidate_axes:
                axis_name_score = self._axis_name_affinity_score(param_name, axis)
                anchor_score = self._param_anchor_affinity_score(
                    param_name,
                    anchor_map.get(axis, None),
                )
                order_score = -axis_order.get(axis, 0)
                score = (
                    axis_name_score,
                    anchor_score,
                    order_score,
                )
                if best_score is None or score > best_score:
                    best_axis = axis
                    best_score = score
            return best_axis, best_score

        phase_specs = [
            {
                "phase": "empty_axis_to_tiling",
                "role": "tiling",
                "candidate_axes": empty_axes,
                "anchor_map": {},
            },
            {
                "phase": "split_only_axis_to_tiling",
                "role": "tiling",
                "candidate_axes": split_only_axes,
                "anchor_map": dict(self.split_params),
            },
            {
                "phase": "tiling_only_axis_to_split",
                "role": "split",
                "candidate_axes": tiling_only_axes,
                "anchor_map": dict(self.tiling_params),
            },
        ]

        pending = list(unresolved)
        for phase_spec in phase_specs:
            if not pending:
                break
            candidate_axes = phase_spec["candidate_axes"]
            if not candidate_axes:
                continue
            next_pending = []
            for param_name in pending:
                if not candidate_axes:
                    next_pending.append(param_name)
                    continue
                axis, score = _pick_best_axis(
                    param_name,
                    candidate_axes,
                    phase_spec["anchor_map"],
                )
                if axis is None:
                    next_pending.append(param_name)
                    continue
                if phase_spec["role"] == "tiling":
                    if axis in self.tiling_params:
                        next_pending.append(param_name)
                        continue
                    self.tiling_params[axis] = param_name
                    report["assigned_tiling_params"][axis] = param_name
                else:
                    if axis in self.split_params:
                        next_pending.append(param_name)
                        continue
                    self.split_params[axis] = param_name
                    report["assigned_split_params"][axis] = param_name
                candidate_axes.remove(axis)
                report["decisions"].append({
                    "phase": phase_spec["phase"],
                    "axis": axis,
                    "param": param_name,
                    "score": list(score) if score is not None else [],
                })
            pending = next_pending

        report["remaining_unresolved_params"] = list(pending)
        report["applied"] = bool(report["assigned_tiling_params"] or report["assigned_split_params"])
        return report

    def _detect_missing_tunable_params(self, all_args, required_missing_params: List[str]) -> List[str]:
        try:
            func_ast = self.fn.parse()
            dsl_result = analyze_signature_and_missing_tunable(
                func_ast,
                arg_names=self.arg_names,
                provided_args=all_args,
                split_params=self.split_params or {},
                tiling_params=self.tiling_params or {},
                explicit_tunable_params=self.explicit_tunable_params,
                parser_mode=self.parser_mode or "vector",
            )
            return list(dsl_result.missing_tunable)
        except ValueError:
            raise
        except Exception:
            constexpr_names = set(self._get_constexpr_candidates())
            fallback = [arg_name for arg_name in required_missing_params if arg_name in constexpr_names]
            return fallback

    def _autoparse_axis_info_v2_for_vector(self, all_args=None):
        if not self.enable_vv_parser_v2:
            return None
        if (self.parser_mode or "vector") != "vector":
            return None
        try:
            parsed_ast = self.fn.parse()
            entry_function_name = getattr(self.base_fn, "__name__", None) or getattr(self.fn, "__name__", None)
            func_ast = parsed_ast
            module_ast = None
            ast_context = "jit_parse_direct"

            if self._should_expand_vv_ast_context(parsed_ast, entry_function_name):
                (
                    func_ast,
                    module_ast,
                    entry_function_name,
                    ast_context,
                ) = self._build_cv_parse_ast_context(
                    parsed_ast,
                    entry_function_name,
                )

            result = parse_vv_axis_info_v2(
                func_ast,
                provided_args=dict(all_args or {}),
                hints=self.hints,
                module_ast=module_ast,
                entry_function_name=entry_function_name,
            )
            self.vv_parse_result_v2 = result
            self.vv_adapter_result_v2 = adapt_vv_v2_to_vector_inputs(
                key_axis_to_param=self.axis_arg_names,
                vv_parse_result=result,
            )
            self.vector_axes = VectorAxes.from_hints_axes(self.hints_axes)
            if self.vv_adapter_result_v2 is not None:
                self.vector_axes.apply_semantic_fields(
                    split_params=getattr(self.vv_adapter_result_v2, "split_params", None),
                    tiling_params=getattr(self.vv_adapter_result_v2, "tiling_params", None),
                    low_dim_axes=getattr(self.vv_adapter_result_v2, "low_dim_axes", None),
                    reduction_axes=getattr(self.vv_adapter_result_v2, "reduction_axes", None),
                    axis_pid_dims=getattr(self.vv_adapter_result_v2, "axis_pid_dims", None),
                    axis_length_exprs=getattr(self.vv_adapter_result_v2, "axis_length_exprs", None),
                    fixed_tiling_exprs=getattr(self.vv_adapter_result_v2, "fixed_tiling_exprs", None),
                    axis_dynamic_sources=getattr(self.vv_adapter_result_v2, "axis_dynamic_sources", None),
                )
            return result
        except Exception as exc:
            if self.print_autotuning:
                print("Ascend autotuning vector v2 parse failed: {}: {}".format(
                    type(exc).__name__,
                    exc,
                ))
            self.vv_parse_result_v2 = None
            self.vv_adapter_result_v2 = None
            return None

    def _gen_tile_configs(self, kv_dict: Dict[str, int], dtype: torch.dtype,
                          all_args: Dict[str, Any] = None) -> List[Config]:
        # Use CVTileGenerator for cube/mix parser modes.
        if self.parser_mode in ("cube", "mix") and self.cv_parse_result is not None:
            return self._gen_tile_configs_cv(kv_dict, dtype, all_args)

        # Use the legacy TileGenerator for vector mode.
        from .tile_generator import KernelMeta, TileGenerator

        vector_tile_inputs = self._materialize_vector_tile_inputs(kv_dict, all_args)
        if self.print_autotuning:
            print("Ascend autotuning vector tile inputs: "
                  f"axis_sizes={vector_tile_inputs['axis_sizes']}, "
                  f"split_params={vector_tile_inputs['split_params']}, "
                  f"fixed_split_params={self.fixed_split_params}, "
                  f"tiling_params={vector_tile_inputs['tiling_params']}, "
                  f"low_dim_axes={vector_tile_inputs['low_dim_axes']}, "
                  f"reduction_axes={vector_tile_inputs['reduction_axes']}, "
                  f"persistent_reduction={self.persistent_reduction}, "
                  f"dual_reduction={self.dual_reduction}, "
                  f"num_buffers={self.num_buffers}")

        kernel_meta = KernelMeta(
            vector_tile_inputs["axis_sizes"],
            vector_tile_inputs["split_params"],
            self.fixed_split_params,
            vector_tile_inputs["tiling_params"],
            vector_tile_inputs["low_dim_axes"],
            vector_tile_inputs["reduction_axes"],
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

    def _gen_tile_configs_cv(self, kv_dict: Dict[str, int], dtype: torch.dtype,
                             all_args: Dict[str, Any] = None) -> List[Config]:
        """
        Generate candidate configs with CVTileGenerator for cube/mix mode.

        This path primarily consumes data from ``self.cv_parse_result``:
        - ``dot_sites``: source of the M/N/K dimension metadata
        - ``tunable_params``: parameters selected for autotuning
        - ``missing_tunable_params``: tunable parameters still unresolved
        """
        if self.cv_parse_result is None:
            return []

        dot_sites = self.cv_parse_result.dot_sites
        if not dot_sites:
            return []

        cv_tile_gen = CVTileGenerator(
            cv_parse_result=self.cv_parse_result,
            all_args=all_args,
            dtype=dtype,
        )

        configs = cv_tile_gen.generate_configs()
        self.gen_configs = configs

        return configs

    def _rough_bench_once(self, fn, config=None) -> float:
        di = triton.runtime.driver.active.get_device_interface()
        di.synchronize()
        try:
            start_event = di.Event(enable_timing=True)
            end_event = di.Event(enable_timing=True)
            start_event.record()
            fn()
            end_event.record()
            di.synchronize()
            return start_event.elapsed_time(end_event)
        except Exception as exc:
            if config is not None:
                self._record_user_config(config, "rough_benchmark_failed", exc=exc)
            print("Triton autotuning compile debug: "
                  f"one-shot rough benchmark failed, reason={type(exc).__name__}: {exc}")
            return float("inf")

    def _prune_by_time_limit(self, run_fns: Dict[Config, Any]) -> Dict[Config, Any]:
        time_limit = 200

        if len(run_fns) <= 1:
            return run_fns

        rough_timings = {}
        for config, fn in run_fns.items():
            rough_timings[config] = self._rough_bench_once(fn, config=config)

        sorted_configs = sorted(rough_timings.keys(), key=lambda c: rough_timings[c])

        fastest_time = rough_timings[sorted_configs[0]]
        n_warmup = max(1, int(25 / fastest_time))
        n_repeat = max(1, int(100 / fastest_time))
        self.cv_warmup = n_warmup
        self.cv_repeat = n_repeat

        min_valid = min(10, len(sorted_configs))
        valid_configs = []
        cumulative_time = 0.0

        for config in sorted_configs:
            estimate_s = rough_timings[config] / 1000.0 * (n_warmup + n_repeat)
            cumulative_time += estimate_s
            if len(valid_configs) >= min_valid and cumulative_time > time_limit:
                break
            valid_configs.append(config)

        if self.print_autotuning:
            print(f"Triton autotuning: estimate n_warmup={n_warmup}, n_repeat={n_repeat}, "
                  f"cumulative={cumulative_time:.4f}s/{time_limit}s, "
                  f"valid={len(valid_configs)}/{len(run_fns)}")
            for config in sorted_configs:
                is_valid = config in valid_configs
                status = "valid" if is_valid else "pruned"
                print(f"Triton autotuning compile debug: "
                      f"[{status}] config={config}, rough_time={rough_timings[config]:.4f}ms")

        for config in run_fns:
            if config not in valid_configs:
                self._record_user_config(
                    config, "pruned",
                    f"Excluded by the {time_limit}s benchmark time budget; rough_time={rough_timings[config]} ms.")
        return {cfg: run_fns[cfg] for cfg in valid_configs}

    def generate_key_and_configs(self, *args, **kwargs):
        self.nargs = dict(zip(self.arg_names, args))
        compile_mode = kwargs.get("compile_mode", _DEFAULT_COMPILE_MODE)
        self.is_simt_mode = compile_mode == "simt_only"
        if 'num_warps' in kwargs and kwargs['num_warps'] is not None:
            self.user_specified_warps = kwargs['num_warps']
        else:
            self.user_specified_warps = None
        self.user_specified_multibuffer_mode = _multibuffer_mode_to_tuple(kwargs.get("multibuffer_mode"))
        if 'num_stages' in kwargs and kwargs['num_stages'] is not None:
            self.user_specified_num_stages = kwargs['num_stages']
        else:
            self.user_specified_num_stages = None
        if 'multibuffer' in kwargs and kwargs['multibuffer'] is not None:
            self.user_specified_multibuffer = kwargs['multibuffer']
        else:
            self.user_specified_multibuffer = None

        has_mode = self.user_specified_multibuffer_mode is not None or "multibuffer_mode" in self._raw_config_hints
        config_has_mode = any(config.kwargs.get("multibuffer_mode") is not None for config in self.user_configs)
        if (has_mode or config_has_mode) and self.user_specified_num_stages is not None:
            raise ValueError("num_stages and multibuffer_mode cannot be specified together")
        if has_mode and any(config.num_stages is not None for config in self.user_configs):
            raise ValueError("num_stages and multibuffer_mode cannot be specified together")
        self.config_hints = _normalize_config_hints(
            self._raw_config_hints,
            inject_default_num_stages=bool(self.user_configs) and bool(self._raw_config_hints),
            has_multibuffer_mode=has_mode or config_has_mode,
        )

        # generate key
        all_args = {**self.nargs, **kwargs}
        _args = {k: v for (k, v) in all_args.items() if k in self.arg_names}
        key = [_args[arg_name] for arg_name in self.keys if arg_name in _args]

        # Currently, we use the dtype with maximum byte length
        dtype = None
        for _, arg in _args.items():
            if hasattr(arg, "dtype"):
                key.append(str(arg.dtype))
                dtype = (arg.dtype if get_byte_per_numel(arg.dtype) >= get_byte_per_numel(dtype) else dtype)
        if dtype is None:
            raise NotImplementedError("Not support for non-Tensor inputs")
        key.append(("compile_mode", compile_mode))
        if self.user_specified_multibuffer_mode is not None:
            key.append(("multibuffer_mode", self.user_specified_multibuffer_mode))

        key = tuple(key)
        if key not in self.cache:
            if self.auto_gen_config:
                self.cv_parse_result = self._autoparse_axis_params(all_args)
                _kv_dict = {
                    axis: _args[arg_name]
                    for axis, arg_name in self.axis_arg_names.items()
                    if arg_name in _args
                }
                self._gen_tile_configs(_kv_dict, dtype, all_args)
                if has_mode:
                    # Tile generators supply legacy stage defaults, not user options.
                    for config in self.gen_configs:
                        config.num_stages = None
                self.gen_configs = _expand_configs_with_hints(
                    self.fn,
                    self.gen_configs,
                    self.config_hints,
                )
            expanded_user_configs = _expand_configs_with_hints(
                self.fn,
                self.user_configs,
                self.config_hints,
            )
            self._expanded_user_configs = expanded_user_configs
            if getattr(self, "_user_config_diagnostics", None) is not None:
                self._user_config_diagnostics.reset_configs(expanded_user_configs)
            if len(self.gen_configs) == 0 and len(self.user_configs) == 0:
                self.configs = _expand_configs_with_hints(
                    self.fn,
                    [Config({}, num_warps=4, num_stages=None if has_mode else 2, num_ctas=1)],
                    self.config_hints if has_mode else {},
                )
            else:
                self.configs = self.gen_configs + expanded_user_configs
        return key

    @staticmethod
    def _inject_grid_num_tiles(kwargs):
        # ChunkCoalescing hint: when the launch grid is a static tuple, the
        # outermost grid dim is the tile count along the chunk axis. Exposing it
        # as grid_num_tiles lets the compiler safely coalesce small chunks for
        # unmasked kernels (compile-time known bound). Skipped for callable grids
        # (dynamic, unknown at compile time) so the pass safely bails.
        if "grid_num_tiles" in kwargs:
            return
        grid = kwargs.get("grid", None)
        if callable(grid) or not isinstance(grid, (tuple, list)):
            return
        glen = min(len(grid), 3)
        if glen <= 0:
            return
        outermost = grid[glen - 1]
        if isinstance(outermost, int) and outermost > 1:
            kwargs["grid_num_tiles"] = _InternalNPUOptionInt(outermost)

    def _record_user_config(self, config, status, detail="", exc=None):
        diagnostics = getattr(self, "_user_config_diagnostics", None)
        return diagnostics is not None and diagnostics.record(config, status, detail, exc)

    def _is_user_config(self, config):
        configs = getattr(self, "_expanded_user_configs", getattr(self, "user_configs", []))
        return any(candidate is config or candidate == config for candidate in configs)

    def _record_user_timings(self, timings, source="benchmark"):
        if getattr(self, "_user_config_diagnostics", None) is None:
            return
        for config, timing in timings.items():
            if not self._is_user_config(config):
                continue
            values = timing if isinstance(timing, (tuple, list)) else (timing, )
            if values and all(math.isfinite(value) for value in values):
                self._record_user_config(config, "benchmarked", f"source={source}; {_format_autotune_timing(timing)}")
            else:
                self._record_user_config(config, "invalid_timing", f"source={source}; No finite benchmark timing.")

    def run(self, *args, **kwargs):
        configs = getattr(self, "_expanded_user_configs", getattr(self, "user_configs", []))
        self._user_config_diagnostics = _UserConfigDiagnostics(configs) if self.print_autotuning and configs else None
        try:
            return self._run_autotune(*args, **kwargs)
        except Exception as exc:
            if self._user_config_diagnostics is not None:
                self._user_config_diagnostics.abort(exc)
            raise
        finally:
            if self._user_config_diagnostics is not None:
                self._user_config_diagnostics.report()
            self._user_config_diagnostics = None

    def _run_autotune(self, *args, **kwargs):
        kwargs = _remove_deprecated_npu_options(kwargs)
        self._inject_grid_num_tiles(kwargs)
        key = self.generate_key_and_configs(*args, **kwargs)
        cache_miss = key not in self.cache
        did_benchmark = False
        disk_cache_hit = False
        single_config_cache_pending = False
        if cache_miss:
            # prune configs
            pruned_configs = self.prune_configs(kwargs)
            diagnostics = self._user_config_diagnostics
            if diagnostics is not None:
                for config, _, _ in diagnostics.results:
                    if config not in pruned_configs:
                        diagnostics.record(config, "pruned", "Removed by early_config_prune or perf_model/top_k.")
            if self.enable_ubtuner or len(pruned_configs) > 1:

                def benchmark():
                    nonlocal did_benchmark
                    did_benchmark = True
                    bench_start = time.time()
                    timings = self._batch_bench(*args, configs=pruned_configs, **kwargs)
                    bench_end = time.time()
                    self.bench_time = bench_end - bench_start
                    self.cache[key] = builtins.min(timings, key=timings.get)
                    full_nargs = {**self.nargs, **kwargs, **self.cache[key].all_kwargs()}
                    self.pre_hook(full_nargs, reset_only=True)
                    self.configs_timings = timings
                    self._record_user_timings(timings)

                if self.cache_results:
                    disk_cache_hit = self.check_disk_cache(key, pruned_configs, benchmark)
                    if disk_cache_hit and self._user_config_diagnostics is not None:
                        for config in pruned_configs:
                            self._record_user_config(config, "skipped_cache_hit")
                else:
                    benchmark()

                config = self.cache[key]
            else:
                config = pruned_configs[0]
                single_config_cache_pending = True
        else:
            config = self.cache[key]
            if self._user_config_diagnostics is not None:
                for user_config, _, _ in self._user_config_diagnostics.results:
                    self._record_user_config(user_config, "skipped_cache_hit")

        self.best_config = config

        if self.print_autotuning and did_benchmark:
            all_args = {**self.nargs, **kwargs}
            key_args = {name: arg for name, arg in all_args.items() if name in self.arg_names}
            key_names = [name for name in self.keys if name in key_args]
            key_names.extend(f"{name}.dtype" for name, arg in key_args.items() if hasattr(arg, "dtype"))
            key_names.append("compile_mode")
            if self.user_specified_multibuffer_mode is not None:
                key_names.append("multibuffer_mode")
            key_fields = ", ".join(f"{name}={value[1]!r}" if isinstance(value, tuple) and len(value) == 2
                                   and value[0] == name else f"{name}={value!r}" for name, value in zip(key_names, key))
            print(f"Triton autotuning for function {self.base_fn.__name__},\n"
                  f"with key fields as {key_fields},\n"
                  f"finished after {self.bench_time:.2f}s; best config selected: {self.best_config};")
            self._print_benchmark_results(self.configs_timings)

        if did_benchmark and self.auto_profile_dir is not None:
            self._profile(*args, config=self.best_config, **kwargs)
        ub_cfg = dict(getattr(config, "ubtune_cfg", {}))
        final_kwargs = dict(config.all_kwargs(), **kwargs)
        final_kwargs.update(ub_cfg)
        if single_config_cache_pending:
            self._record_user_config(config, "running", "Single-config execution did not complete.")
        if config.pre_hook is not None:
            config.pre_hook({**self.nargs, **final_kwargs})
        try:
            ret = self.fn.run(
                *args,
                **final_kwargs,
            )
            if single_config_cache_pending:
                self.cache[key] = config
                self._record_user_config(config, "selected", "Only one config remains; executed without benchmarking.")
            return ret
        except Exception as exc:
            from triton.compiler.errors import CompileTimeAssertionFailure, MLIRCompilationError
            from triton.runtime.errors import OutOfResources

            status = "compile_failed" if isinstance(exc, (CompileTimeAssertionFailure, MLIRCompilationError,
                                                          OutOfResources)) else "evaluation_failed"
            self._record_user_config(config, status, exc=exc)
            raise
        finally:
            self.nargs = None
            if did_benchmark and not disk_cache_hit:
                # workaround for memory leak when some configs fail to compile
                gc.collect()

    def _try_ubtuner(self, *args, config, excp, run_fns, **kwargs):
        if not (self.enable_ubtuner and "ub overflow" in str(excp).lower()):
            return

        try:
            # Extract compile options from the current config and forward them
            # to ubtuner as the search starting point.
            compile_opts = config.all_kwargs()
            ub_cfg = self.ubtuner.get_best_config(*args, compile_opts=compile_opts, **kwargs)
            if self.print_autotuning:
                print(f"ub_cfg {ub_cfg}")
            if ub_cfg:
                setattr(config, 'ubtune_cfg', ub_cfg)
                ub_fn = self._make_kernel_call(*args, config=config, **kwargs)
                run_fns[config] = functools.partial(ub_fn, warmup=False)
        except Exception as e:
            if self.print_autotuning:
                print(f"[WARN] encounter exception when try ubtune, Details: {e}")

    def _handle_compile_failure(self, *args, config, excp, run_fns, compile_key, **kwargs):
        self._try_ubtuner(*args, config=config, excp=excp, run_fns=run_fns, **kwargs)
        self._compile_failed_configs.append(config)
        if config not in run_fns:
            self._remember_compile_failure(compile_key, excp)
            self._record_user_config(config, "compile_failed", exc=excp)
        else:
            self._record_user_config(config, "benchmarking",
                                     "UBTuner returned a fallback config; no benchmark result was produced.")

    def _bench_config(self, config, fn):
        self._record_user_config(config, "benchmarking", "Benchmark did not complete.")
        try:
            timing = self.do_bench(fn, quantiles=(0.5, 0.2, 0.8))
        except Exception as exc:
            self._record_user_config(config, "benchmark_failed", exc=exc)
            raise
        self._record_user_timings({config: timing})
        return timing

    def _print_benchmark_results(self, timings) -> None:
        if not self.print_autotuning:
            return

        print(f"Triton autotuning benchmark results for function {self.base_fn.__name__}:")
        for config, timing in timings.items():
            selected = " [selected]" if config == self.best_config else ""
            print(f"  config={config}; {_format_autotune_timing(timing)}{selected}")
            values = timing if isinstance(timing, (tuple, list)) else (timing, )
            if values and all(math.isfinite(value) for value in values):
                self._record_user_config(config, "reported")

    def _get_jit_compile_cache_key(self, *args, config, **meta):
        """Return the JIT compilation identity for one autotune Config.

        The autotune tuning key intentionally does not participate in this
        identity.  If a changed tuning key produces the same JIT
        specialization and backend options, a previous deterministic compile
        failure can be reused safely.  If constexpr values or compile options
        change, ``compute_cache_key`` returns a different identity and the
        Config is compiled again.

        User hooks may mutate arguments before JIT specialization, so use a
        conservative fallback and disable negative caching for those Configs.
        """
        if config.pre_hook is not None or getattr(self, "user_defined_pre_hook", False):
            return None

        try:
            current = dict(meta, **config.all_kwargs())
            ub_cfg = dict(getattr(config, "ubtune_cfg", {}))
            if ub_cfg:
                current.update(ub_cfg)

            # Match the first call made by _batch_bench. Heuristics.run sees
            # grid and warmup before forwarding the remaining arguments to
            # JITFunction.run, so resolve heuristics before removing them.
            current["warmup"] = bool(getattr(self, "compile_parallel", False))
            jit_fn = getattr(self, "fn", None)
            visited = set()
            while isinstance(jit_fn, triton.runtime.Heuristics):
                if id(jit_fn) in visited:
                    return None
                visited.add(id(jit_fn))
                for name, heuristic in jit_fn.values.items():
                    heuristic_args = {**dict(zip(jit_fn.arg_names, args)), **current}
                    current[name] = heuristic(heuristic_args)
                jit_fn = jit_fn.fn

            # Unknown wrappers may transform arguments before JITFunction.run.
            # Never bypass them when constructing a negative-cache key.
            if not isinstance(jit_fn, triton.runtime.JITFunction):
                return None
            if getattr(jit_fn, "pre_run_hooks", None):
                return None

            # These two arguments are consumed by JITFunction.run and are not
            # passed to the generated binder.
            current.pop("grid", None)
            current.pop("warmup", None)

            # Keep this in sync with JITFunction.run before it invokes binder.
            current["debug"] = current.get("debug", jit_fn.debug) or knobs.runtime.debug
            current["instrumentation_mode"] = knobs.compilation.instrumentation_mode

            device = triton.runtime.driver.active.get_current_device()
            _, kernel_key_cache, target, _, binder = jit_fn.device_caches[device]
            _, specialization, options = binder(*args, **current)
            jit_key = compute_cache_key(kernel_key_cache, specialization, options)

            env_vars = get_cache_invalidating_env_vars()
            ubtuner_mode = os.environ.get("TRITON_ENABLE_UBTUNER", "")
            if ubtuner_mode:
                env_vars["TRITON_ENABLE_UBTUNER"] = ubtuner_mode

            return (
                jit_fn.cache_key,
                repr(target),
                str(sorted(env_vars.items())),
                device,
                jit_key,
            )
        except Exception:
            # Failure-key calculation must never break normal autotuning.
            return None

    def _get_cached_compile_failure(self, compile_key):
        if compile_key is None:
            return None
        with self._compile_failure_cache_lock:
            return self._compile_failure_cache.get(compile_key)

    def _remember_compile_failure(self, compile_key, exc) -> None:
        if compile_key is None:
            return
        failure = {
            "exception_type": type(exc).__name__,
        }
        with self._compile_failure_cache_lock:
            self._compile_failure_cache.setdefault(compile_key, failure)

    def _filter_cached_compile_failures(self, *args, configs, **kwargs):
        active_configs = []
        cached_configs = []
        compile_keys = {}

        for config in configs:
            compile_key = self._get_jit_compile_cache_key(*args, config=config, **kwargs)
            compile_keys[config] = compile_key
            failure = self._get_cached_compile_failure(compile_key)
            if failure is None:
                active_configs.append(config)
                continue

            cached_configs.append(config)
            if self.print_autotuning:
                print("Triton autotuning: skip cached compile-failed config "
                      f"{config}; previous failure: {failure['exception_type']}")
                self._record_user_config(config, "reported")

        self._cached_compile_failed_configs = cached_configs
        return active_configs, compile_keys

    def _batch_bench(self, *args, configs, **kwargs):
        from triton.compiler.errors import CompileTimeAssertionFailure, MLIRCompilationError
        from triton.runtime.errors import OutOfResources

        kernels_call = {}
        for config in configs:
            try:
                kernels_call[config] = self._make_kernel_call(*args, config=config, **kwargs)
            except Exception as exc:
                self._record_user_config(config, "configuration_failed", exc=exc)
                raise
        active_configs, compile_keys = self._filter_cached_compile_failures(
            *args,
            configs=configs,
            **kwargs,
        )
        kernels_call = {config: kernels_call[config] for config in active_configs}

        if not kernels_call:
            details = []
            for config in configs:
                failure = self._get_cached_compile_failure(compile_keys.get(config))
                if failure is not None:
                    details.append(f"config={config}: {failure['exception_type']}")
            raise RuntimeError("All triton configs are cached compile failures.\n" + "\n".join(details))

        run_fns = {}
        self._compile_failed_configs = []
        exc = None
        exc_stack = ""

        if self.compile_parallel:
            import psutil

            max_workers = min(psutil.cpu_count(logical=False) * 3 // 4, len(kernels_call))
            future_kernels = []
            failed_submission = None
            try:
                with (
                        ThreadPoolExecutor(max_workers=max_workers) as executor,
                        triton.AsyncCompileMode(executor),
                ):
                    for config, fn in kernels_call.items():
                        self._record_user_config(config, "compiling", "Compilation did not complete.")
                        try:
                            future_kernels.append((config, fn(warmup=True)))
                        except (CompileTimeAssertionFailure, MLIRCompilationError, OutOfResources) as e:
                            failed_submission = config
                            self._record_user_config(config, "compile_failed", exc=e)
                            raise

                    for config, fut in future_kernels:
                        try:
                            if hasattr(fut, "result"):
                                fut = fut.result()
                            if hasattr(fut, "packed_metadata"):
                                kernels_call[config].target_kernel_name = fut.packed_metadata.get("kernel_name")
                            run_fns[config] = functools.partial(kernels_call[config], warmup=False)
                            self._record_user_config(config, "benchmarking",
                                                     "Compilation completed; no benchmark result was produced.")
                        except (CompileTimeAssertionFailure, MLIRCompilationError, OutOfResources) as e:
                            import traceback
                            exc_stack = traceback.format_exc()
                            exc = e
                            self._handle_compile_failure(*args, config=config, excp=e, run_fns=run_fns,
                                                         compile_key=compile_keys.get(config), **kwargs)
            except Exception as e:
                # ignore exception from __exit__() of AsyncCompileMode
                triton.runtime._async_compile.active_mode.set(None)
                for pending in kernels_call:
                    if (pending not in run_fns and pending not in self._compile_failed_configs
                            and pending is not failed_submission):
                        # __exit__ can re-raise an earlier future's exception;
                        # it cannot identify a failure in this pending config.
                        self._record_user_config(pending, "not_evaluated", "Parallel compilation interrupted:", exc=e)
        else:
            for config, fn in kernels_call.items():
                self._record_user_config(config, "compiling", "Compilation or initial execution did not complete.")
                try:
                    compiled_kernel = fn(warmup=False)
                    if hasattr(compiled_kernel, "packed_metadata"):
                        fn.target_kernel_name = compiled_kernel.packed_metadata.get("kernel_name")
                    run_fns[config] = functools.partial(fn, warmup=False)
                    self._record_user_config(config, "benchmarking",
                                             "Compilation completed; no benchmark result was produced.")
                except (CompileTimeAssertionFailure, MLIRCompilationError, OutOfResources) as e:
                    import traceback
                    exc_stack = traceback.format_exc()
                    exc = e
                    self._handle_compile_failure(*args, config=config, excp=e, run_fns=run_fns,
                                                 compile_key=compile_keys.get(config), **kwargs)
                except Exception as e:
                    self._record_user_config(config, "evaluation_failed", exc=e)
                    raise

        if len(run_fns) == 0:
            raise RuntimeError(f"No valid triton configs. {type(exc).__name__}: {exc} \nStack trace: {exc_stack}")

        if len(run_fns) == 1:
            # we ignore expensive profiling method when only single config is left
            return {config: self._bench_config(config, fn) for config, fn in run_fns.items()}

        parser_mode = getattr(self, "parser_mode", None) or "vector"
        cv_parse_result = getattr(self, "cv_parse_result", None)
        if parser_mode in ("cube", "mix") and cv_parse_result is not None:
            run_fns = self._prune_by_time_limit(run_fns)

        use_profiling = os.getenv("TRITON_BENCH_METHOD", "default").lower() == "npu"
        # Respect user-provided benchmarkers even when NPU profiling mode is enabled.
        use_npu_profiling = use_profiling and not self.user_defined_do_bench
        if use_npu_profiling:
            from ..testing import ProfilerResultMismatchError, do_bench_npu

            cv_mode = parser_mode in ("cube", "mix") and cv_parse_result is not None
            warmup = getattr(self, "cv_warmup", 5) if cv_mode else 5
            active = getattr(self, "cv_repeat", 30) if cv_mode else 30
            target_kernel_name = self._resolve_target_kernel_name(kernels_call, run_fns.keys())
            try:
                time_cost = do_bench_npu(
                    list(run_fns.values()),
                    warmup=warmup,
                    active=active,
                    clear_l2_cache=True,
                    target_kernel_name=target_kernel_name,
                )
                assert len(time_cost) == len(run_fns)
                timings = dict(zip(run_fns.keys(), time_cost))
                self._record_user_timings(timings)
                return timings
            except ProfilerResultMismatchError as exc:
                warnings.warn(
                    "Filtered profiler rows do not match the expected count for autotune benchmarking; "
                    f"target_kernel_name={exc.target_kernel_name!r}, expected_rows={exc.expected_rows}, "
                    f"actual_rows={exc.actual_rows}. fallback to default do_bench.",
                    RuntimeWarning,
                    stacklevel=2,
                )
                return {config: self._bench_config(config, fn) for config, fn in run_fns.items()}
        else:
            return {config: self._bench_config(config, fn) for config, fn in run_fns.items()}

    def _resolve_target_kernel_name(self, kernels_call, configs) -> Optional[str]:
        for config in configs:
            kernel_call = kernels_call.get(config)
            if kernel_call is None:
                continue
            kernel_name = getattr(kernel_call, "target_kernel_name", None)
            if kernel_name:
                return kernel_name
        return None

    def _make_kernel_call(self, *args, config, **meta):
        # check for conflicts, i.e. meta-parameters both provided
        # as kwargs and by the autotuner
        conflicts = meta.keys() & config.kwargs.keys()
        if conflicts:
            raise ValueError(f"Conflicting meta-parameters: {', '.join(conflicts)}."
                             " Make sure that you don't re-define auto-tuned symbols.")
        # augment meta-parameters with tunable ones
        current = dict(meta, **config.all_kwargs())
        ub_cfg = dict(getattr(config, "ubtune_cfg", {}))
        if ub_cfg:
            current.update(ub_cfg)
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
                if isinstance(res, tuple):
                    res = res[0]
                packed_metadata = getattr(res, "packed_metadata", None)
                if isinstance(packed_metadata, dict):
                    kernel_call.target_kernel_name = packed_metadata.get("kernel_name")
            except Exception as e:
                try:
                    self.post_hook(full_nargs, exception=e)
                finally:
                    # Throw exception raised by `self.fn.run`
                    raise

            if not warmup:
                self.post_hook(full_nargs, exception=None)
            return res

        kernel_call.target_kernel_name = None
        return kernel_call

    def warmup(self, *args, **kwargs):
        kwargs = _remove_deprecated_npu_options(kwargs)
        self._inject_grid_num_tiles(kwargs)
        _ = self.generate_key_and_configs(*args, **kwargs)
        pruned_configs = self.prune_configs(kwargs)
        ret = []

        def warmup_config(config):
            compile_options = dict(config.all_kwargs(), **kwargs)
            return self.fn.warmup(*args, **compile_options)

        if self.compile_parallel:
            import psutil

            max_workers = min(psutil.cpu_count(logical=False) // 2, len(pruned_configs))
            with (
                    ThreadPoolExecutor(max_workers=max_workers) as executor,
                    triton.AsyncCompileMode(executor),
            ):
                for config in pruned_configs:
                    ret.append(warmup_config(config))
        else:
            for config in pruned_configs:
                ret.append(warmup_config(config))
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
        parser = SplitAxesParser(func_ast, self._get_parser_axis_arg_names(), candidates_params)
        split_axes = parser.parse()
        self.split_axis_pid_dims = dict(getattr(parser, "split_axis_pid_dims", {}))
        self.axis_pid_dims = dict(getattr(parser, "axis_pid_dims", {}))
        self.split_params = dict(split_axes)
        self._refresh_vector_axes()
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
            self._get_parser_axis_arg_names(),
            self._get_constexpr_candidates(),
        )
        _ = parser.parse()
        self.axis_pid_dims = dict(getattr(parser, "axis_pid_dims", {}))
        self.split_axis_pid_dims = dict(getattr(parser, "split_axis_pid_dims", {}))
        self._refresh_vector_axes()
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
        parser_axis_arg_names = self._normalize_axis_name_mapping(self._get_parser_axis_arg_names())
        parser = TilingAxesParser(func_ast, parser_axis_arg_names, candidates_params)
        tiling_axes = parser.parse()
        self.tiling_params = dict(tiling_axes)
        self._refresh_vector_axes()
        if self.print_autotuning:
            print(f"Ascend autotuning parse tiling axes: {tiling_axes}")
        return tiling_axes

    def _autoparse_reduction_axes(self) -> List[str]:
        """
        Extracts the reduction axis parameters from triton kernel code.
        """
        func_ast = self.fn.parse()
        axis_arg_names_before = self._get_parser_axis_arg_names()
        parser = ReductionAxesParser(func_ast, axis_arg_names_before)
        raw_reduction_axes = parser.parse()
        for axis in raw_reduction_axes:
            base_axis = self._get_axis_base_name(axis)
            self._promote_axis_arg_name_to_reduction(base_axis)
        reduction_axes = self._normalize_reduction_axes(raw_reduction_axes)
        self.reduction_axes = list(reduction_axes)
        self._refresh_vector_axes()

        if self.print_autotuning:
            print(f"Ascend autotuning parse axis arg names: {self.axis_arg_names} \n"
                  f"Ascend autotuning parse reduction axes: {reduction_axes}")
        return reduction_axes

    def _autoparse_low_dim_axes(self) -> List[str]:
        """
        Extracts the low dimension axis from triton kernel code.
        """
        func_ast = self.fn.parse()
        parser_axis_arg_names = self._normalize_axis_name_mapping(self._get_parser_axis_arg_names())
        parser = LowDimsAxesParser(func_ast, parser_axis_arg_names)
        low_dim_axes = parser.parse()
        if len(low_dim_axes) < 1:
            if self.print_autotuning:
                print("[WARNING] Failed to parse low-dimensional axes, fallback to empty low_dim_axes.")
            return []
        self.low_dim_axes = list(low_dim_axes)
        self._refresh_vector_axes()
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
        # Reduction axes are stored as base axis names; compare directly with
        # the low-dim base axis.
        reduction_axis_base = self._get_axis_base_name(reduction_axis)
        low_dim_axis_base = (self._get_axis_base_name(self.low_dim_axes[0]) if self.low_dim_axes else None)
        if reduction_axis_base is not None and reduction_axis_base == low_dim_axis_base:
            return 1024
        return 64


def autotune(configs, key, prune_configs_by=None, reset_to_zero=None, restore_value=None, pre_hook=None, post_hook=None,
             warmup=None, rep=None, use_cuda_graph=False, do_bench=None, cache_results=False, *, auto_prof_dir=None,
             hints=None):
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
    kernel, including the benchmark timing for each valid configuration, the
    time spent autotuning, and the best configuration.
    Additional feedback for user-provided configurations (including hints
    expansion) reports failures or interrupted evaluation without repeating
    successful benchmark output. Pruning, successful single-config execution,
    and memory/disk tuning-cache hits do not add diagnostics.

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
    :param cache_results: whether to cache autotune timings to disk.  Defaults to False.
    :type cache_results: bool
    :param auto_prof_dir: the specified directory to store the profiling results of the best config.
        If this parameter is None or the best config is retrieved from cache, the profiling process will be ignored.
    :type auto_prof_dir: str
    :param hints: a dict of autotune hint auguments passed to AutoTilingTuner.
    """

    def decorator(fn):
        return AutoTilingTuner(fn, fn.arg_names, configs, key, reset_to_zero, restore_value, pre_hook=pre_hook,
                               post_hook=post_hook, prune_configs_by=prune_configs_by, warmup=warmup, rep=rep,
                               use_cuda_graph=use_cuda_graph, do_bench=do_bench, cache_results=cache_results,
                               auto_profile_dir=auto_prof_dir, hints=hints)

    return decorator


_ALL_PARAMS = {
    "num_stages",
    "multibuffer_mode",
    "unit_flag",
    "multibuffer",
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

_CUBE_PARAMS = {"num_stages", "multibuffer_mode", "unit_flag", "limit_auto_multi_buffer_of_local_buffer"}

_MIXCV_PARAMS = {
    "num_stages",
    "multibuffer_mode",
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
    "multibuffer_mode",
    "enable_ubuf_saving",
}


def _check_boolean_list(val, param_name):
    return isinstance(val, (list, tuple)) and len(val) > 0 and all(isinstance(x, bool) for x in val)


def _check_string_in_set(val, valid_set, param_name):
    return isinstance(val, (list, tuple)) and len(val) > 0 and all(v in valid_set for v in val)


def _check_int_in_set(val, valid_set, param_name):
    return isinstance(val, (list, tuple)) and len(val) > 0 and all(isinstance(v, int) and v in valid_set for v in val)


def _check_multibuffer_mode_list(values, param_name):
    if not isinstance(values, (list, tuple)) or not values:
        return False
    try:
        return all(value is not None and _multibuffer_mode_to_tuple(value) is not None for value in values)
    except TypeError:
        return False


_VALIDATION_RULES = {
    "multibuffer_mode": {
        "desc": "must be a non-empty list/tuple of mode dictionaries (or cached pairs) with str keys and int counts",
        "check": _check_multibuffer_mode_list,
    },
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
        if "multibuffer_mode" in kwargs and "num_stages" in kwargs:
            raise ValueError("num_stages and multibuffer_mode cannot be specified together")
        if not self.validate_parameters(**kwargs):
            return []

        # Collect parameter values, using defaults for missing ones
        param_values = {}
        for p in sorted(self.supported_params):
            if p == "multibuffer_mode" and p not in kwargs:
                continue
            if p == "num_stages" and "multibuffer_mode" in kwargs:
                continue
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

            configs.append(Config(kwargs=config_kwargs, num_stages=num_stages_val))
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
    kwargs = _remove_deprecated_npu_options(kwargs)

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

    has_mode = "multibuffer_mode" in kwargs or base_kwargs.get("multibuffer_mode") is not None
    if has_mode and ("num_stages" in kwargs or base_num_stages is not None):
        raise ValueError("num_stages and multibuffer_mode cannot be specified together")
    for param in sorted(supported):
        if param == "multibuffer_mode" and param not in kwargs and param not in base_kwargs:
            continue
        if has_mode and param == "num_stages" and param not in kwargs:
            continue
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
                            num_ctas=config.num_ctas, maxnreg=config.maxnreg, pre_hook=config.pre_hook,
                            ir_override=config.ir_override)
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

    tuning_params = _remove_deprecated_npu_options(tuning_params)

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
