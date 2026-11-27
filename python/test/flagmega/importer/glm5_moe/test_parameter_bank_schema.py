# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Importer bank schemas are metadata-only and reject row drift."""

import pytest
from triton.flagmega.errors import ImporterError
from triton.flagmega.importer.glm5_moe import Glm5MoeModelImporter

from python.test.flagmega.importer.glm5_moe.helpers import REDUCED, checkpoint


def test_repeated_layer_schema_keeps_checkpoint_rows():
    source = checkpoint(dict(REDUCED), with_values=False, full_model=True, expert_layout="separate")
    importer = Glm5MoeModelImporter(source, block_size=8, num_blocks=3)
    schema, rows = importer.parameter_bank_schema((0, 1), ("input_layernorm.weight",), name="linear.norm")
    assert schema.name == "linear.norm"
    assert tuple(field.name for field in schema.fields) == ("input_layernorm.weight",)
    assert rows == tuple(
        (f"{importer.model_prefix}layers.{layer}.input_layernorm.weight",)
        for layer in (0, 1)
    )


def test_repeated_layer_schema_rejects_missing_fields():
    source = checkpoint(dict(REDUCED), with_values=False, full_model=True, expert_layout="separate")
    importer = Glm5MoeModelImporter(source, block_size=8, num_blocks=3)
    with pytest.raises(ImporterError, match="missing"):
        importer.parameter_bank_schema((0, 1), ("does_not_exist.weight",), name="bad")
