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
Parameter name normalization for CV autotune.

This module provides functionality to normalize parameter names and
configurations to a common format for processing and pruning.
"""

from typing import Dict, Any
import copy


class ParamNameNormalizer:
    """
    Normalize parameter names and configurations.

    This class handles conversion between different parameter naming
    conventions and ensures consistent format for processing.
    """

    def __init__(self):
        """Initialize the normalizer with default mappings."""
        # Common parameter name mappings
        self.param_mappings = {
            # Block size mappings
            'BLOCK_M': 'BLOCK_M',
            'BLOCK_N': 'BLOCK_N',
            'BLOCK_K': 'BLOCK_K',
            'block_m': 'BLOCK_M',
            'block_n': 'BLOCK_N',
            'block_k': 'BLOCK_K',
            'standard_block_m': 'BLOCK_M',
            'standard_block_n': 'BLOCK_N',
            'standard_block_k': 'BLOCK_K',

            # Other mappings
            'num_warps': 'num_warps',
            'num_stages': 'num_stages',
            'GROUP_M': 'GROUP_M',
            'GROUP_N': 'GROUP_N',
        }

    def normalize_param_name(self, name: str) -> str:
        """
        Normalize a parameter name to standard format.

        Args:
            name: Parameter name to normalize

        Returns:
            Normalized parameter name

        Example:
            >>> normalizer = ParamNameNormalizer()
            >>> normalizer.normalize_param_name('block_m')
            'BLOCK_M'
        """
        return self.param_mappings.get(name, name)

    def normalize_config(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Normalize a configuration dictionary.

        This method:
        1. Normalizes all parameter names
        2. Creates a new dictionary with normalized names
        3. Preserves the original config (does not modify in-place)

        Args:
            config: Configuration dictionary to normalize

        Returns:
            New configuration dictionary with normalized parameter names

        Example:
            >>> normalizer = ParamNameNormalizer()
            >>> config = {'block_m': 128, 'num_warps': 4}
            >>> normalized = normalizer.normalize_config(config)
            >>> normalized
            {'BLOCK_M': 128, 'num_warps': 4}
        """
        normalized = {}
        for key, value in config.items():
            normalized_key = self.normalize_param_name(key)
            normalized[normalized_key] = value
        return normalized

    def normalize_configs(self, configs: list) -> list:
        """
        Normalize a list of configurations.

        Args:
            configs: List of configuration dictionaries

        Returns:
            List of normalized configuration dictionaries
        """
        return [self.normalize_config(config) for config in configs]
