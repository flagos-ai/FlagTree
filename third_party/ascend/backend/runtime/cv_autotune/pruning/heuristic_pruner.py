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
Heuristic pruning for CV fusion operator autotune.

This module implements heuristic pruning strategies to reduce the search space
by filtering out invalid or suboptimal configurations.
"""

from typing import Dict, Tuple

import torch

from ..hardware_consts import HardwareConstraints
from ..normalization.param_normalizer import ParamNameNormalizer


class HeuristicPruner:
    """
    Heuristic pruner for configuration space.

    This class implements conservative pruning strategies that only remove
    configurations that are definitely invalid (memory limits, alignment, etc.).
    It does NOT prune configurations that might be optimal.
    """

    def __init__(self, hardware_constraints: HardwareConstraints):
        """
        Initialize the pruner.

        Args:
            hardware_constraints: Hardware constraint parameters
        """
        self.hw = hardware_constraints
        self.normalizer = ParamNameNormalizer()
        self.L1Stages: int = 2
        self.L0AStages: int = 2
        self.L0BStages: int = 2
        self.L0CStages: int = 1

    def should_prune(self, config: Dict, dtype) -> Tuple[bool, str]:
        """
        Check if a configuration should be pruned.

        This implements conservative pruning:
        - Only prunes configurations that violate hard constraints
        - Does NOT prune based on efficiency thresholds (might miss optimal configs)

        Args:
            config: Configuration dictionary

        Returns:
            (should_prune, reason): Tuple of boolean and reason string
        """
        # Normalize parameter names
        normalized_config = self.normalizer.normalize_config(config)

        # Check : Memory limits
        should_prune, reason = self._check_memory_limits(normalized_config, dtype)
        if should_prune:
            return True, f"Memory limit: {reason}"

        # Configuration passes all checks
        return False, ""

    def _check_memory_limits(self, config: Dict, dtype) -> Tuple[bool, str]:
        """
        Check memory limits (conservative).

        This checks if the tile size exceeds hardware memory limits:
        - L0A/L0B/L0C Buffers
        - L1 Buffer

        Args:
            config: Normalized configuration dictionary

        Returns:
            (exceeded, reason): Tuple of boolean and reason string
        """
        # Get tile dimensions
        tile_m = config.get('BLOCK_M', 128)
        tile_n = config.get('BLOCK_N', 128)
        tile_k = config.get('BLOCK_K', 64)

        try:
            if dtype.is_floating_point:
                bits = torch.finfo(dtype).bits
            else:
                bits = torch.iinfo(dtype).bits
            typesize = bits // 8
        except Exception:
            typesize = 2

        # Calculate tile size (assuming fp16, adjust based on dtype)

        l1_size = getattr(self.hw, 'max_l1_size', None) // self.L1Stages
        l0a_size = getattr(self.hw, 'l0a_size', None) // self.L0AStages
        l0b_size = getattr(self.hw, 'l0b_size', None) // self.L0BStages
        l0c_size = getattr(self.hw, 'l0c_size', None) // self.L0CStages

        # # Check L1 limit
        tile_numel = tile_m * tile_k + tile_k * tile_n
        tile_bytes = tile_numel * typesize
        if l1_size is not None and tile_bytes > l1_size:
            return True, f"L1: {tile_bytes} > {l1_size}"
        # Check L0a limit
        tile_bytes = tile_m * tile_k * typesize
        if l0a_size is not None and tile_bytes > l0a_size:
            return True, f"L0a: {tile_bytes} > {l0a_size}"
        # Check L0b limit
        tile_bytes = tile_k * tile_n * typesize
        if l0b_size is not None and tile_bytes > l0b_size:
            return True, f"L0a: {tile_bytes} > {l0b_size}"
        # Check L0c limit
        tile_bytes = tile_m * tile_n * 4
        if l0c_size is not None and tile_bytes > l0c_size:
            return True, f"L0a: {tile_bytes} > {l0c_size}"

        return False, ""
