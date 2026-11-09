# Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, or/or sell
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
Hardware constraints for CV autotune.

This module defines hardware-specific constraints for NPU hardware,
including memory limits, cache sizes, and other resource constraints.
"""

from dataclasses import dataclass, field
from typing import Optional, Dict, Any

from .. import utils as runtime_utils


def _arch_default(legacy_value: int, a5_value: int) -> int:
    # Only called for an omitted constructor argument, never during import.
    arch = runtime_utils.target.arch
    if arch.startswith(("Ascend910_95", "Ascend950")):
        return a5_value
    return legacy_value


@dataclass
class HardwareConstraints:
    """
    Hardware constraint parameters for NPU.

    These constraints are used during configuration generation and pruning
    to ensure generated configurations are feasible on the target hardware.

    Attributes:
        l0a_size: L0A cache size in bytes (for matrix A)
        l0b_size: L0B cache size in bytes (for matrix B)
        l0c_size: L0C cache size in bytes (for matrix C)
        max_l0_size: Maximum L0 cache size in bytes
        max_l1_size: Maximum L1 cache size in bytes
        max_l2_size: Maximum L2 cache size in bytes
        ub_size: Unified Buffer size in bytes
    """

    # Default ArchType : A2/A3
    l0a_size: int = 64 * 1024  # 64 KB - for matrix A
    l0b_size: int = 64 * 1024  # 64 KB - for matrix B
    l0c_size: int = field(default_factory=lambda: _arch_default(128 * 1024, 256 * 1024))

    # Memory limits (in bytes)
    max_l0_size: int = 128 * 1024  # 128 KB
    max_l1_size: int = 512 * 1024  # 512 KB
    max_l2_size: int = field(default_factory=lambda: _arch_default(192 * 1024 * 1024, 128 * 1024 * 1024))
    ub_size: int = field(default_factory=lambda: _arch_default(192 * 1024, 248 * 1024))

    def __post_init__(self):
        """Validate hardware constraints after initialization."""

        if self.l0a_size <= 0:
            raise ValueError(f"Invalid l0a_size: {self.l0a_size}")
        if self.l0b_size <= 0:
            raise ValueError(f"Invalid l0b_size: {self.l0b_size}")
        if self.l0c_size <= 0:
            raise ValueError(f"Invalid l0c_size: {self.l0c_size}")
        if self.max_l0_size <= 0:
            raise ValueError(f"Invalid max_l0_size: {self.max_l0_size}")
        if self.max_l1_size <= 0:
            raise ValueError(f"Invalid max_l1_size: {self.max_l1_size}")
        if self.ub_size <= 0:
            raise ValueError(f"Invalid ub_size: {self.ub_size}")

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert hardware constraints to dictionary.

        Returns:
            Dictionary representation of hardware constraints
        """
        return {
            'l0a_size': self.l0a_size,
            'l0b_size': self.l0b_size,
            'l0c_size': self.l0c_size,
            'max_l0_size': self.max_l0_size,
            'max_l1_size': self.max_l1_size,
            'max_l2_size': self.max_l2_size,
            'ub_size': self.ub_size,
        }


def get_default_hardware_constraints(device_type: str = "ascend",
                                     custom_constraints: Optional[HardwareConstraints] = None) -> HardwareConstraints:
    """
    Get default hardware constraints for the specified device type.

    Args:
        device_type: Type of device (e.g., "ascend", "nvidia")
        custom_constraints: Optional custom constraints to override defaults

    Returns:
        HardwareConstraints instance

    Example:
        >>> from cv_autotune.hardware_consts import get_default_hardware_constraints
        >>> constraints = get_default_hardware_constraints()
        >>> print(f"Max L0 size: {constraints.max_l0_size}")
    """
    if custom_constraints is not None:
        return custom_constraints

    return HardwareConstraints()
