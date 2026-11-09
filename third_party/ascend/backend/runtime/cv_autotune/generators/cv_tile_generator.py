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
"""
CV Tile configuration generator for autotune.

This module implements a tree-based configuration generator that creates
all possible tile configurations for CV fusion operators based on
CvParseResult.
"""

# Types used to describe the search space.
from dataclasses import dataclass
from typing import Any, Dict, List, Literal, Optional

import torch

from triton.runtime.autotuner import Config

from ...dsl_analysis.schema import (AXIS_LENGTH_STATE_TUNABLE, CvAxisLengthInfo, CvParseResult)
from ..hardware_consts import (HardwareConstraints, get_default_hardware_constraints)
from ..pruning.heuristic_pruner import HeuristicPruner

ParamType = Literal["NUM", "ENUM", "BOOL"]


@dataclass
class ParamSpace:
    """Definition of a single autotune parameter search space."""
    name: str
    type: ParamType
    min: int = 0
    max: int = 0
    stride: int = 1
    values: List[Any] = None

    def count_values(self) -> int:
        if self.type == "NUM":
            return (self.max - self.min) // self.stride + 1
        elif self.type == "ENUM":
            return len(self.values) if self.values else 0
        elif self.type == "BOOL":
            return 2
        return 0

    def get_values(self) -> List[Any]:
        if self.type == "NUM":
            return list(range(self.min, self.max + 1, self.stride))
        elif self.type == "ENUM":
            return self.values if self.values else []
        elif self.type == "BOOL":
            return [True, False]
        return []

    def check_constraint(self, config: Dict[str, Any]) -> bool:
        """Check parameter constraints."""
        return True


class CVTileGenerator:
    """
    CV Tile configuration generator.

    Builds the search space from ``CvParseResult`` and enumerates valid
    configurations with a BFS-style traversal.

    Core logic:
    1. Extract M/N/K tunable parameters from ``cv_parse_result.dot_sites``.
    2. Use the real axis length as the upper bound of the search space.
    3. Traverse the search space and prune invalid configs with
       ``HeuristicPruner``.
    """

    # Default values for M/N dimensions.
    _MN_MIN = 16
    _MN_DEFAULT_MAX = 256
    _MN_STRIDE = 16

    # Default values for the K dimension.
    _K_MIN = 16
    _K_DEFAULT_MAX = 128
    _K_STRIDE = 16

    _STEP_SMALL = 16
    _STEP_LARGE = 32
    _STEP_THRESHOLD = 64

    def __init__(self, cv_parse_result: CvParseResult, hardware_constraints: Optional[HardwareConstraints] = None,
                 all_args: Optional[Dict[str, Any]] = None, dtype: Optional[torch.dtype] = None):
        """
        Initialize the generator.

        Args:
            cv_parse_result: CV parse result.
            hardware_constraints: Optional hardware constraints. Defaults to
                the built-in constraint set.
            all_args: Full runtime argument map, including tensors, used to
                resolve tensor shapes from ``length_expr``.
        """
        self.cv_parse_result = cv_parse_result
        self.all_args = all_args or {}
        self.hw_constraints = hardware_constraints or get_default_hardware_constraints()
        self.pruner = HeuristicPruner(self.hw_constraints)
        self.dtype = dtype or torch.float16

        # Build the search space directly from dot-site metadata.
        self.search_space = self._build_param_space()

        # Convert search space to ordered list for deterministic generation
        self.params = list(self.search_space.values())

        # Statistics
        self.stats = {
            'generated': 0,
            'pruned': 0,
            'total_possible': 1,
        }

        # Calculate total possible configurations
        for param in self.params:
            self.stats['total_possible'] *= param.count_values()

    def _resolve_axis_max(self, axis_info: CvAxisLengthInfo, default_max: int) -> int:
        """
        Resolve the maximum tile value from ``CvAxisLengthInfo``.

        Priority:
        1. ``const_value``: compile-time constant, use directly.
        2. ``length_expr`` as a key in ``all_args``.
        3. ``length_expr`` parsed as an integer literal.
        4. Fallback to the provided default.

        Args:
            axis_info: Axis length metadata.
            default_max: Default upper bound when resolution fails.

        Returns:
            int: Maximum tile size for the axis.
        """
        # Prefer the compile-time constant when available.
        if axis_info.const_value is not None:
            return axis_info.const_value

        # Try to resolve length_expr through runtime arguments.
        if axis_info.length_expr is not None and axis_info.length_expr in self.all_args:
            value = self.all_args[axis_info.length_expr]
            if isinstance(value, (int, float)):
                return int(value)

        # Fall back to parsing length_expr as a plain integer literal.
        if axis_info.length_expr is not None:
            try:
                return int(axis_info.length_expr)
            except (ValueError, TypeError):
                pass

        return default_max

    def _make_param(self, name: str, max_val: int, min_val: int, stride: int) -> ParamSpace:
        """
        Build a parameter search space with segmented strides.

        Supports dynamic step sizes:
        - Step = 16 when current value <= 64.
        - Step = 32 when current value > 64.
        Always includes `min_val` and `max_val`. Returns an ENUM ParamSpace.

        Args:
            name: Parameter name.
            max_val: Upper bound.
            min_val: Lower bound.
            stride: (Ignored, kept for API compatibility.)

        Returns:
            ParamSpace: ENUM type containing the generated tile sizes.
        """
        # Case 1: max < stride, so max itself is the only legal value.
        values = []
        current = min_val
        while current <= max_val:
            values.append(current)
            if current < self._STEP_THRESHOLD:
                current += self._STEP_SMALL
            else:
                current += self._STEP_LARGE
        if values[-1] != max_val:
            values.append(max_val)
        values = sorted(set(values))
        return ParamSpace(name=name, type="ENUM", values=values)

    @staticmethod
    def _extract_single_tunable_symbol(symbol: Optional[str]) -> Optional[str]:
        """Extract the unique tunable symbol from a composite token."""
        if not isinstance(symbol, str):
            return None

        candidates = []
        for item in symbol.split("|"):
            token = item.strip()
            if not token or not token.isidentifier():
                continue
            if token in candidates:
                continue
            candidates.append(token)
        if len(candidates) == 1:
            return candidates[0]
        return None

    def _build_param_space(self) -> Dict[str, ParamSpace]:
        """
        Build the M/N/K parameter search space from ``cv_parse_result``.

        Strategy, in priority order:
        1. Extract tunable params and axis lengths from dot sites.
           - Visit every dot site and collect all tunable params.
           - Treat repeated parameter names as the same coupled parameter and
             keep the largest axis length across all sites.
           - Treat different parameter names as independent parameters.
        2. If dot sites do not provide enough data, fall back to
           ``tunable_params``.
        3. If ``tunable_params`` is still insufficient, infer missing
           parameters from ``signature_params``.
        4. If all inference fails, use the default search space.

        Returns:
            Dict[str, ParamSpace]: Mapping from parameter name to search range.
        """
        param_space = {}
        dot_sites = self.cv_parse_result.dot_sites

        # Strategy 1: extract parameters from all dot sites.
        # Deduplicate by parameter name; identical names indicate coupling.
        if dot_sites:
            # Collect the largest resolved axis length for each parameter.
            param_max_values: Dict[str, int] = {}  # param_name -> resolved max
            param_roles: Dict[str, str] = {}  # param_name -> role (M/N/K)

            for site in dot_sites:
                for role, axis_info in [("M", site.m), ("N", site.n), ("K", site.k)]:
                    param_name = axis_info.tunable_param
                    if not param_name and axis_info.state == AXIS_LENGTH_STATE_TUNABLE:
                        param_name = self._extract_single_tunable_symbol(axis_info.axis_symbol)
                        if param_name is None:
                            param_name = self._extract_single_tunable_symbol(axis_info.length_expr)
                    if not param_name:
                        continue
                    if axis_info.state != AXIS_LENGTH_STATE_TUNABLE:
                        continue

                    resolved_max = self._resolve_axis_max(axis_info, self._MN_DEFAULT_MAX)

                    if param_name not in param_max_values:
                        # First occurrence: record the resolved axis length.
                        param_max_values[param_name] = resolved_max
                        param_roles[param_name] = role
                    else:
                        # Coupled occurrence: keep the largest axis length.
                        old_max = param_max_values[param_name]
                        if resolved_max > old_max:
                            param_max_values[param_name] = resolved_max
            for param_name, max_val in param_max_values.items():
                role = param_roles[param_name]
                if role == "K":
                    param_space[param_name] = self._make_param(param_name, max_val, self._K_MIN, self._K_STRIDE)
                else:
                    param_space[param_name] = self._make_param(param_name, max_val, self._MN_MIN, self._MN_STRIDE)

        # Strategy 2: if dot-site extraction produced nothing, fall back to
        # tunable_params.
        if not param_space and self.cv_parse_result.tunable_params:
            for param_info in self.cv_parse_result.tunable_params:
                name = param_info.name
                if name in param_space:
                    continue
                # Infer the axis role from the parameter name.
                role = self._infer_role_from_name(name)
                if role == "K":
                    param_space[name] = self._make_param(name, self._K_DEFAULT_MAX, self._K_MIN, self._K_STRIDE)
                else:
                    param_space[name] = self._make_param(name, self._MN_DEFAULT_MAX, self._MN_MIN, self._MN_STRIDE)

        if not param_space:
            return {}

        return param_space

    @staticmethod
    def _infer_role_from_name(name: str) -> str:
        """Infer the axis role from the parameter name."""
        name_upper = name.upper()
        if "K" in name_upper and "BLOCK_K" in name_upper:
            return "K"
        return "M"  # Default to the M/N class.

    def _get_default_param_space(self) -> Dict[str, ParamSpace]:
        """Return the default parameter search space."""
        return {
            'BLOCK_M':
            ParamSpace(name='BLOCK_M', type='NUM', min=self._MN_MIN, max=self._MN_DEFAULT_MAX, stride=self._MN_STRIDE),
            'BLOCK_N':
            ParamSpace(name='BLOCK_N', type='NUM', min=self._MN_MIN, max=self._MN_DEFAULT_MAX, stride=self._MN_STRIDE),
            'BLOCK_K':
            ParamSpace(name='BLOCK_K', type='NUM', min=self._K_MIN, max=self._K_DEFAULT_MAX, stride=self._K_STRIDE),
        }

    def _is_memory_limit_exceeded(self, reason: str) -> bool:
        """
        Check if pruning reason is memory-related.

        This is used to determine if we should break the parameter loop,
        since larger values will also exceed memory limits.

        Args:
            reason: Pruning reason string

        Returns:
            True if the reason indicates memory limit exceeded
        """
        memory_keywords = [
            "Memory limit",
            "L0c",
            "UB",
            "L0A",
            "L0B",
            "memory",
            "Memory:",
        ]
        return any(keyword in reason for keyword in memory_keywords)

    def generate_configs(self) -> List[Config]:
        """
        Generate the full tile-config space recursively and prune afterward.

        Returns:
            List[Config]: Valid ``Config`` objects after pruning.
        """

        if not self.params:
            return []

        # Step 1: generate the full Cartesian product without pruning.
        raw_configs: List[Dict[str, int]] = []
        self._generate_all_recursive(param_idx=0, partial={}, result=raw_configs)

        configs = []
        for cfg_dict in raw_configs:
            is_valid = True
            for site in self.cv_parse_result.dot_sites:
                site_full_config = cfg_dict.copy()
                for role, axis_info in [("M", site.m), ("N", site.n), ("K", site.k)]:
                    target_key = f"standard_block_{role.lower()}"

                    if target_key in site_full_config:
                        continue

                    resolved_val = None

                    if axis_info.tunable_param and axis_info.tunable_param in cfg_dict:
                        resolved_val = cfg_dict[axis_info.tunable_param]
                    elif axis_info.const_value is not None:
                        resolved_val = axis_info.const_value
                    # TODO: replace this fallback with a parser-derived default.
                    else:
                        resolved_val = 128 if role != "K" else 64

                    site_full_config[target_key] = resolved_val

                should_prune, reason = self.pruner.should_prune(site_full_config, self.dtype)

                if should_prune:
                    is_valid = False
                    self.stats['pruned'] += 1
                    break
            if is_valid:
                configs.append(Config(kwargs=cfg_dict, num_warps=1, num_stages=1))

        return configs

    def _generate_all_recursive(self, param_idx: int, partial: Dict[str, int], result: List[Dict[str, int]]):
        """Recursively enumerate all parameter combinations without pruning."""
        if param_idx >= len(self.params):
            result.append(partial.copy())
            return

        param = self.params[param_idx]
        if param.type == "NUM":
            value = param.min
            while value <= param.max:
                partial[param.name] = value
                self._generate_all_recursive(param_idx + 1, partial, result)
                value += param.stride
            del partial[param.name]

        elif param.type == "ENUM":
            for value in param.values:
                partial[param.name] = value
                self._generate_all_recursive(param_idx + 1, partial, result)
            del partial[param.name]

        elif param.type == "BOOL":
            for value in [True, False]:
                partial[param.name] = value
                self._generate_all_recursive(param_idx + 1, partial, result)
            del partial[param.name]

    def estimate_search_space_size(self) -> Dict[str, int]:
        """
        Estimate the size of the search space.

        Returns:
            Dictionary with size statistics
        """
        total = 1
        param_counts = {}

        for param in self.params:
            count = param.count_values()
            param_counts[param.name] = count
            total *= count

        return {
            'total_combinations': total,
            'num_parameters': len(self.params),
            'parameter_counts': param_counts,
        }
