# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: MIT
"""The logical-to-physical DSA cache map must survive the complete IR path."""

import pytest

from triton.flagmega.errors import CodegenError, IRSchemaError
from triton.flagmega.ir import emit_module, load_module
from triton.flagmega.ir.ops.nn._dsa_indexer_state import DsaIndexerStateConfig
from triton.flagmega.ir.ops.nn.dsa_indexer import DsaIndexer
from python.test.flagmega.codegen.triton.kernels.dsa_indexer.test_decode_runtime import indexer_module


def test_dsa_layer_map_survives_dump_reload_distribution_and_lowering(tmp_path):
    from triton.flagmega.codegen.triton.tir_package import describe_tir_package
    from triton.flagmega.compiler import Compiler

    config = DsaIndexerStateConfig((3, 7), 8, 4, 5)
    module = indexer_module(config, 7)
    path = emit_module(module, tmp_path / "indexer.py")
    restored = load_module(path)
    assert restored.semantic_hash == module.semantic_hash
    compiled = Compiler().compile(restored).module
    calls = [call for call in describe_tir_package(compiled)["render_calls"] if call["family"] == "dsa_indexer"]
    assert len(calls) == 1
    assert calls[0]["state_layer_ids"] == config.layer_ids


@pytest.mark.parametrize("ids", [(3, 3), (True, 7), (-1, 7), (3, "7"), None])
def test_invalid_layer_maps_are_rejected(ids):
    attrs = next(node.attrs for node in indexer_module(DsaIndexerStateConfig((3, 7), 8, 4, 5), 3).nodes
                 if node.op == "nn.dsa_indexer")
    with pytest.raises(IRSchemaError, match="state_layer_ids"):
        DsaIndexer.normalize_attrs({**attrs, "state_layer_ids": ids})


def test_layer_map_must_cover_every_physical_slot():
    with pytest.raises(IRSchemaError, match="every packed cache layer"):
        indexer_module(DsaIndexerStateConfig((3, 7), 8, 4, 5), 3, state_layer_ids=(3,))


def test_legacy_ir_can_load_but_cannot_silently_guess_the_cache_map(tmp_path):
    from triton.flagmega.compiler import Compiler

    module = indexer_module(DsaIndexerStateConfig((3, 7), 8, 4, 5), 3, state_layer_ids=())
    restored = load_module(emit_module(module, tmp_path / "legacy.py"))
    compiled = Compiler().compile(restored).module
    from triton.flagmega.codegen.triton.tir_package import describe_tir_package
    with pytest.raises(CodegenError, match="explicit state_layer_ids"):
        describe_tir_package(compiled)
