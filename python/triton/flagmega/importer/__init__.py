# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Model importers. Qwen3.5/3.8 support is registered explicitly."""

from triton.flagmega.importer.checkpoint import (
    Checkpoint,
    DirectoryCheckpoint,
    MemoryCheckpoint,
    TensorByteRange,
    TensorInfo,
)
from triton.flagmega.importer.source import (
    ImportSourceLocation,
    SOURCE_LOCATION_SCHEMA,
    attach_import_source_locations,
    source_location_of,
)
from triton.flagmega.importer.model import apply_numerical_profile, import_model, import_model_layer, importer_registry
from triton.flagmega.importer.numerics import VLLM_INDUCTOR_LEVEL3, VLLM_AE10_INDUCTOR_LEVEL3
from triton.flagmega.importer.registry import ModelImporterRegistry, ModelImporterSpec
from triton.flagmega.importer.qwen3 import (
    Qwen3LayerConfig,
    Qwen3LayerImporter,
    Qwen3ModelImporter,
    import_qwen3_layer,
    import_qwen3_model,
)
from triton.flagmega.importer.qwen3_5 import Qwen35Layer0Importer, Qwen35LayerConfig, import_qwen3_8_layer0
from triton.flagmega.importer.qwen3_5_moe import Qwen35MoeConfig, Qwen35MoeImporter

__all__ = [
    "Checkpoint",
    "DirectoryCheckpoint",
    "MemoryCheckpoint",
    "ModelImporterRegistry",
    "ModelImporterSpec",
    "ImportSourceLocation",
    "SOURCE_LOCATION_SCHEMA",
    "Qwen3LayerConfig",
    "Qwen3LayerImporter",
    "Qwen3ModelImporter",
    "Qwen35Layer0Importer",
    "Qwen35LayerConfig",
    "Qwen35MoeConfig",
    "Qwen35MoeImporter",
    "TensorInfo",
    "TensorByteRange",
    "attach_import_source_locations",
    "apply_numerical_profile",
    "VLLM_INDUCTOR_LEVEL3",
    "VLLM_AE10_INDUCTOR_LEVEL3",
    "import_model",
    "import_model_layer",
    "importer_registry",
    "source_location_of",
    "import_qwen3_layer",
    "import_qwen3_model",
    "import_qwen3_8_layer0",
]
