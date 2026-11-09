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
CV Fusion Operator Autotuning Framework.

This package provides autotuning functionality for CV fusion operators
on NPU hardware.
"""

# Configuration generation
from .generators.cv_tile_generator import (
    CVTileGenerator,
    ParamSpace,
)

# Pruning
from .pruning.heuristic_pruner import HeuristicPruner

# Hardware constraints
from .hardware_consts import HardwareConstraints, get_default_hardware_constraints

# Parameter normalization
from .normalization import (
    ParamNameNormalizer,
    CvAutotuneParam,
    create_cv_autotune_param,
)

__all__ = [
    # Configuration generation
    'CVTileGenerator',
    'ParamSpace',

    # Pruning
    'HeuristicPruner',

    # Hardware
    'HardwareConstraints',
    'get_default_hardware_constraints',

    # Parameter normalization
    'ParamNameNormalizer',
    'CvAutotuneParam',
    'create_cv_autotune_param',
]
