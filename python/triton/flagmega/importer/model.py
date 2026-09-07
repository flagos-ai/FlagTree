# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Architecture-dispatched model import entry point used by the CLI."""

from __future__ import annotations

from triton.flagmega.importer.checkpoint import Checkpoint, DirectoryCheckpoint
from triton.flagmega.importer.qwen3 import Qwen3LayerImporter, Qwen3ModelImporter
from triton.flagmega.importer.qwen3_5 import Qwen35Layer0Importer
from triton.flagmega.importer.registry import ModelImporterRegistry, ModelImporterSpec
from triton.flagmega.ir import IRModule


importer_registry = ModelImporterRegistry()


def _register_builtin_importers() -> None:
    importer_registry.register(ModelImporterSpec(
        "qwen3",
        frozenset({"Qwen3ForCausalLM"}),
        frozenset({"qwen3"}),
        lambda source, layer, revision: Qwen3LayerImporter(
            source, layer=layer, revision=revision
        ).import_module(),
        lambda source, revision: Qwen3ModelImporter(source, revision=revision).import_module(),
    ))
    importer_registry.register(ModelImporterSpec(
        "qwen3.5",
        frozenset({"Qwen3_5ForConditionalGeneration"}),
        frozenset({"qwen3_5"}),
        _import_qwen35_layer,
    ))


def _import_qwen35_layer(source: Checkpoint, layer: int, revision: str | None) -> IRModule:
    if layer != 0:
        from triton.flagmega.errors import ImporterError

        raise ImporterError("Qwen3.5/Qwen3.8 single-layer import currently supports only layer 0.")
    return Qwen35Layer0Importer(source, revision=revision).import_module()


_register_builtin_importers()


def import_model_layer(
    checkpoint: Checkpoint | str,
    *,
    layer: int = 0,
    revision: str | None = None,
) -> IRModule:
    """Import one layer after resolving the architecture from ``config.json``."""

    source = DirectoryCheckpoint(checkpoint) if isinstance(checkpoint, str) else checkpoint
    spec = importer_registry.resolve(source.config, full_model=False)
    return spec.import_layer(source, layer, revision)


def import_model(
    checkpoint: Checkpoint | str,
    *,
    revision: str | None = None,
) -> IRModule:
    """Import a complete supported model after architecture dispatch."""

    source = DirectoryCheckpoint(checkpoint) if isinstance(checkpoint, str) else checkpoint
    spec = importer_registry.resolve(source.config, full_model=True)
    assert spec.import_model is not None
    return spec.import_model(source, revision)


__all__ = ["import_model", "import_model_layer", "importer_registry"]
