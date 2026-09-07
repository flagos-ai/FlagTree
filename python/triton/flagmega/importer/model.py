# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Architecture-dispatched model import entry point used by the CLI."""

from __future__ import annotations

from triton.flagmega.importer.checkpoint import Checkpoint, DirectoryCheckpoint
from triton.flagmega.importer.qwen3 import Qwen3LayerImporter, Qwen3ModelImporter
from triton.flagmega.importer.qwen3_5 import Qwen35Layer0Importer
from triton.flagmega.importer.registry import ModelImporterRegistry, ModelImporterSpec
from triton.flagmega.importer.numerics import VLLM_INDUCTOR_LEVEL3
from triton.flagmega.importer.numerics.qwen3 import apply_qwen3_vllm_profile
from triton.flagmega.ir import IRModule, verify_module
from triton.flagmega.errors import ImporterError


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
        numerical_profiles={VLLM_INDUCTOR_LEVEL3: apply_qwen3_vllm_profile},
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
    numerical_profile: str = "nncase",
) -> IRModule:
    """Import one layer after resolving the architecture from ``config.json``."""

    source = DirectoryCheckpoint(checkpoint) if isinstance(checkpoint, str) else checkpoint
    spec = importer_registry.resolve(source.config, full_model=False)
    transform = spec.numerical_transform(numerical_profile)
    return transform(spec.import_layer(source, layer, revision))


def import_model(
    checkpoint: Checkpoint | str,
    *,
    revision: str | None = None,
    numerical_profile: str = "nncase",
) -> IRModule:
    """Import a complete supported model after architecture dispatch."""

    source = DirectoryCheckpoint(checkpoint) if isinstance(checkpoint, str) else checkpoint
    spec = importer_registry.resolve(source.config, full_model=True)
    assert spec.import_model is not None
    transform = spec.numerical_transform(numerical_profile)
    return transform(spec.import_model(source, revision))


def apply_numerical_profile(module: IRModule, profile: str) -> IRModule:
    """Apply a source-runtime contract to trusted imported Python IR.

    Reapplying the same contract is idempotent. Never silently reinterpret a
    checkpoint from another profile or mutate a lowered/distributed ABI.
    """
    if module.stage != "imported":
        raise ImporterError("Numerical profiles require imported IR; later checkpoints already encode their contract.")
    existing = module.metadata.get("numerical_contract", "nncase")
    if existing != "nncase" and existing != profile:
        raise ImporterError(f"Cannot change numerical contract {existing!r} to {profile!r}; re-import the model.")
    spec = importer_registry.resolve(module.metadata, full_model=False)
    transform = spec.numerical_transform(profile)
    return verify_module(transform(module))


__all__ = ["apply_numerical_profile", "import_model", "import_model_layer", "importer_registry"]
