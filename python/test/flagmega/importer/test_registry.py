# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT

import pytest

from triton.flagmega.errors import ImporterError
from triton.flagmega.importer import ModelImporterRegistry, ModelImporterSpec, importer_registry


def _layer(_source, _layer, _revision):
    return "layer"


def _model(_source, _revision):
    return "model"


def test_importer_registry_resolves_architecture_or_model_type():
    registry = ModelImporterRegistry()
    spec = registry.register(ModelImporterSpec(
        "unit",
        frozenset({"UnitArchitecture"}),
        frozenset({"unit"}),
        _layer,
        _model,
    ))

    assert registry.resolve({"architectures": ["UnitArchitecture"]}, full_model=False) is spec
    assert registry.resolve({"model_type": "unit"}, full_model=True) is spec


def test_importer_registry_rejects_ambiguous_config_instead_of_first_match():
    registry = ModelImporterRegistry()
    registry.register(ModelImporterSpec(
        "architecture-match", frozenset({"ArchitectureA"}), frozenset(), _layer,
    ))
    registry.register(ModelImporterSpec(
        "model-type-match", frozenset(), frozenset({"model_b"}), _layer,
    ))

    with pytest.raises(ImporterError, match="ambiguous.*architecture-match.*model-type-match"):
        registry.resolve(
            {"architectures": ["ArchitectureA"], "model_type": "model_b"},
            full_model=False,
        )


def test_importer_registry_reports_missing_full_model_capability():
    registry = ModelImporterRegistry()
    registry.register(ModelImporterSpec(
        "layer-only", frozenset(), frozenset({"layer_only"}), _layer,
    ))

    with pytest.raises(ImporterError, match="no full-model importer"):
        registry.resolve({"model_type": "layer_only"}, full_model=True)


def test_builtin_registry_rejects_contradictory_qwen_identity():
    with pytest.raises(ImporterError, match="ambiguous.*qwen3.*qwen3.5"):
        importer_registry.resolve(
            {
                "architectures": ["Qwen3ForCausalLM"],
                "model_type": "qwen3_5",
            },
            full_model=False,
        )
