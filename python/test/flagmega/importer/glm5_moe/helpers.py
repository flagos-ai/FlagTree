# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Compatibility exports for the GLM-5.3-Flash synthetic checkpoint fixture."""

from triton.flagmega.importer.glm5_moe._synthetic import (
    PUBLISHED,
    REDUCED,
    checkpoint,
    checkpoint_layer,
    configuration,
    kda_weight_shapes,
)

__all__ = [
    "PUBLISHED",
    "REDUCED",
    "checkpoint",
    "checkpoint_layer",
    "configuration",
    "kda_weight_shapes",
]
