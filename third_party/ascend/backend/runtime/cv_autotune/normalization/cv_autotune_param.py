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
Unified parameter structure for CV autotune algorithms.

This module defines the CvAutotuneParam class, which serves as the unified
input parameter structure for all CV autotune algorithms.
"""

from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List
from ...dsl_analysis.schema import CvParseResult, CvAxisRole


@dataclass
class CvAutotuneParam:
    """
    Unified parameter container for CV autotune.

    This class gathers all inputs required by the algorithms:
    1. CV parse result (``CvParseResult``)
    2. Runtime parameters (the original ``kv_dict``)
    3. Data type (``dtype``)
    4. Other optional configuration

    Design principles:
    - Provide a single input interface for autotune algorithms
    - Carry all required information in one place
    - Offer convenient conversion helpers
    - Support optional hardware constraints
    """

    # ==================== Core inputs ====================

    cv_parse_result: CvParseResult
    """CV parse result imported from ``schema.py``."""

    runtime_params: Dict[str, Any]
    """Runtime parameter dictionary (the original ``kv_dict``)."""

    # ==================== Optional config ====================

    dtype: Optional[str] = None
    """Data type such as ``float16`` or ``float32``."""

    parser_mode: Optional[str] = None
    """Parser mode (``cube`` or ``mix``). Defaults to ``cv_parse_result.mode``."""

    # ==================== Optional hardware constraints ====================

    hardware_constraints: Optional[Dict[str, Any]] = None
    """Hardware constraints such as memory or cache sizes."""

    # ==================== Optional algorithm config ====================

    enable_pruning: bool = True
    """Whether pruning is enabled."""

    enable_profiling: bool = False
    """Whether profiling is enabled."""

    # ==================== Internal caches (derived automatically) ====================

    _cached_axis_sizes: Dict[str, int] = field(default_factory=dict)
    """Cached axis-size mapping."""

    _cached_tiling_params: Dict[str, str] = field(default_factory=dict)
    """Cached tiling-parameter mapping."""

    _cached_fixed_params: Dict[str, int] = field(default_factory=dict)
    """Cached fixed-parameter mapping."""

    _cached_low_dims: List[str] = field(default_factory=list)
    """Cached low-dimension list."""

    # ==================== Convenience helpers ====================

    def __post_init__(self):
        """Perform post-init normalization."""
        # If parser_mode is omitted, derive it from cv_parse_result.
        if self.parser_mode is None:
            self.parser_mode = self.cv_parse_result.mode

        # Precompute commonly used mappings.
        self._precompute_mappings()

    def _precompute_mappings(self):
        """Precompute commonly used mappings."""
        # 1. Build tiling_params from tunable parameters with M/N/K roles.
        for param_info in self.cv_parse_result.tunable_params:
            if param_info.role in (CvAxisRole.M, CvAxisRole.N, CvAxisRole.K):
                # This should eventually map axis names to parameter names.
                # For now, reuse the parameter name directly and refine later
                # with dot-site information if needed.
                self._cached_tiling_params[param_info.role] = param_info.name

        # 2. Build fixed_params from scalar runtime parameters.
        for param_name, param_value in self.runtime_params.items():
            if isinstance(param_value, (int, float)):
                self._cached_fixed_params[param_name] = param_value

        # 3. Derive axis_sizes from dot-site length expressions.
        if self.cv_parse_result.dot_sites:
            first_site = self.cv_parse_result.dot_sites[0]
            for axis_name, axis_info in [('m', first_site.m), ('n', first_site.n), ('k', first_site.k)]:
                if axis_info.length_expr is not None:
                    try:
                        size = int(axis_info.length_expr)
                        self._cached_axis_sizes[axis_name] = size
                    except (ValueError, TypeError):
                        pass

        # 4. Infer low_dims from the smallest length_expr values.
        if self.cv_parse_result.dot_sites:
            first_site = self.cv_parse_result.dot_sites[0]
            axis_sizes = []
            for axis_name, axis_info in [('m', first_site.m), ('n', first_site.n), ('k', first_site.k)]:
                if axis_info.length_expr is not None:
                    try:
                        size = int(axis_info.length_expr)
                        axis_sizes.append((axis_name, size))
                    except (ValueError, TypeError):
                        pass

            # Use the smallest dimensions as low_dim candidates.
            if axis_sizes:
                axis_sizes.sort(key=lambda x: x[1])
                # Keep the smallest one or two dimensions.
                num_low_dims = min(2, len(axis_sizes))
                self._cached_low_dims = [name for name, _ in axis_sizes[:num_low_dims]]

    # ==================== Conversion helpers ====================

    def to_kernel_meta_dict(self) -> Dict[str, Any]:
        """
        Convert to the dictionary format expected by ``KernelMeta``.

        Returns:
            Dict: All fields required to initialize ``KernelMeta``.
        """
        return {
            'axis_sizes': self.axis_sizes, 'split_params': self.split_params, 'fixed_split_params':
            self.fixed_split_params, 'tiling_params': self.tiling_params, 'low_dims': self.low_dims, 'dtype':
            self.dtype, 'persistent_reduction': False,  # Default value.
            'dual_reduction': False,  # Default value.
            'num_buffers': 3,  # Default value.
            'is_simt_mode': False,  # Default value.
        }

    def to_cv_tile_generator_input(self) -> CvParseResult:
        """
        Convert to the input expected by ``CVTileGenerator``.

        ``CVTileGenerator`` consumes ``CvParseResult`` directly, so return it
        as-is.

        Returns:
            CvParseResult: CV parse result.
        """
        return self.cv_parse_result

    def to_pruner_input(self) -> Dict[str, Any]:
        """
        Convert to the input expected by the pruner.

        Returns:
            Dict: Parameter dictionary consumed by the pruner.
        """
        return {
            'cv_parse_result': self.cv_parse_result,
            'runtime_params': self.runtime_params,
            'hardware_constraints': self.hardware_constraints,
        }

    # ==================== Accessors ====================

    @property
    def axis_sizes(self) -> Dict[str, int]:
        """Return the axis-size mapping."""
        return self._cached_axis_sizes

    @property
    def tiling_params(self) -> Dict[str, str]:
        """Return the tiling-parameter mapping."""
        return self._cached_tiling_params

    @property
    def split_params(self) -> Dict[str, str]:
        """
        Return the split-parameter mapping.

        This should eventually infer axis-to-parameter mapping from dot sites.
        It currently returns an empty dict and is left for future work.
        """
        return {}

    @property
    def fixed_split_params(self) -> Dict[str, int]:
        """Return the fixed split parameters."""
        return self._cached_fixed_params

    @property
    def low_dims(self) -> List[str]:
        """Return the low-dimension list."""
        return self._cached_low_dims

    @property
    def tunable_params(self) -> List[str]:
        """Return the list of tunable parameter names."""
        return [p.name for p in self.cv_parse_result.tunable_params]

    @property
    def missing_tunable_params(self) -> List[str]:
        """Return unresolved tunable parameters."""
        return self.cv_parse_result.missing_tunable_params

    @property
    def dot_sites_count(self) -> int:
        """Return the number of dot sites."""
        return len(self.cv_parse_result.dot_sites)

    @property
    def is_mix_mode(self) -> bool:
        """Whether the parser is in mix mode."""
        return self.parser_mode == 'mix'

    @property
    def is_cube_mode(self) -> bool:
        """Whether the parser is in cube mode."""
        return self.parser_mode == 'cube'

    # ==================== Validation ====================

    def validate(self) -> tuple[bool, List[str]]:
        """
        Validate the parameter bundle.

        Returns:
            tuple: (is_valid, error_messages)
        """
        errors = []

        # 1. Check cv_parse_result.
        if self.cv_parse_result is None:
            errors.append("cv_parse_result cannot be None")
        elif self.cv_parse_result.status != "ok":
            errors.append(f"cv_parse_result status is not OK: {self.cv_parse_result.status}")

        # 2. Check runtime_params.
        if not self.runtime_params:
            errors.append("runtime_params cannot be empty")

        # 3. Check unresolved tunable parameters.
        missing = set(self.missing_tunable_params) - set(self.runtime_params.keys())
        if missing:
            errors.append(f"Missing required tunable parameters: {missing}")

        # 4. Check dot_sites.
        if not self.cv_parse_result.dot_sites:
            errors.append("cv_parse_result does not contain any dot_sites")

        return (len(errors) == 0, errors)

    # ==================== Utilities ====================

    def get_param_value(self, param_name: str, default: Any = None) -> Any:
        """
        Return the runtime value for a parameter.

        Args:
            param_name: Parameter name.
            default: Default value.

        Returns:
            Parameter value, or ``default`` if it is absent.
        """
        return self.runtime_params.get(param_name, default)

    def get_tunable_param_by_role(self, role: CvAxisRole) -> Optional[str]:
        """
        Return the tunable parameter name for a given role.

        Args:
            role: Parameter role (M/N/K/OTHER).

        Returns:
            Parameter name, or ``None`` if it does not exist.
        """
        for param_info in self.cv_parse_result.tunable_params:
            if param_info.role == role:
                return param_info.name
        return None

    def get_dot_site(self, site_id: int) -> Optional[Any]:
        """
        Return the dot site for a given site ID.

        Args:
            site_id: Dot site ID

        Returns:
            DotSiteInfo, or ``None`` if it does not exist.
        """
        for site in self.cv_parse_result.dot_sites:
            if site.site_id == site_id:
                return site
        return None

    def __repr__(self) -> str:
        """Return a readable string representation."""
        return (f"CvAutotuneParam(\n"
                f"  mode={self.parser_mode},\n"
                f"  tunable_params={len(self.tunable_params)},\n"
                f"  dot_sites={self.dot_sites_count},\n"
                f"  runtime_params={len(self.runtime_params)},\n"
                f"  dtype={self.dtype}\n"
                f")")


# ==================== Factory ====================


def create_cv_autotune_param(cv_parse_result: CvParseResult, runtime_params: Dict[str, Any],
                             dtype: Optional[str] = None, parser_mode: Optional[str] = None,
                             hardware_constraints: Optional[Dict[str, Any]] = None, **kwargs) -> CvAutotuneParam:
    """
    Factory for creating ``CvAutotuneParam`` instances.

    Args:
        cv_parse_result: CV parse result.
        runtime_params: Runtime parameter dictionary.
        dtype: Optional data type.
        parser_mode: Optional parser mode.
        hardware_constraints: Optional hardware constraints.
        **kwargs: Additional optional arguments.

    Returns:
        CvAutotuneParam instance.
    """
    return CvAutotuneParam(cv_parse_result=cv_parse_result, runtime_params=runtime_params, dtype=dtype,
                           parser_mode=parser_mode, hardware_constraints=hardware_constraints, **kwargs)
